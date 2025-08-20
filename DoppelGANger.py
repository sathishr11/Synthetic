"""
Two-Generator DoppelGANger (variable-length, padding+mask, categorical handling)

Save this file as `twogen_doppelganger_varlen.py` and run.

Notes:
- Preprocess continuous features (e.g., StandardScaler) and categorical features as one-hot
- The critic sees soft categorical probabilities during training; at generation time use `postprocess_samples` to convert to hard labels
- This script includes a minimal toy-data demo when run as `__main__`
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# ========= Utility helpers =========

def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Boolean mask [B, T]: True for valid timesteps, False for padding."""
    ar = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ar < lengths.unsqueeze(1)


def gumbel_softmax_sample(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Differentiable categorical sample (soft one-hot)."""
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-12) + 1e-12)
    return F.softmax((logits + g) / tau, dim=-1)


def flatten_cat_timewise(cat_list: List[torch.Tensor]) -> Optional[torch.Tensor]:
    """Concat list of [B,T,K_i] -> [B,T,sum K_i]; returns None if list empty."""
    if not cat_list:
        return None
    return torch.cat(cat_list, dim=-1)


def pad_cat_sequences(batch_cat_list_per_feature: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Input: for each cat feature f, a list of length B with tensors [L_i, K_f] (one-hot)
    Output: list over features of padded [B, T_max, K_f]
    """
    padded_per_feature = []
    for seqs_for_f in batch_cat_list_per_feature:
        padded = pad_sequence(seqs_for_f, batch_first=True, padding_value=0.0)
        padded_per_feature.append(padded)
    return padded_per_feature


# ========= Config =========

@dataclass
class ModelConfig:
    # Latent & model sizes
    noise_dim: int = 32
    hidden_dim: int = 128

    # Sequential features
    cont_seq_dim: int = 0                  # # continuous seq features
    cat_seq_dims: Tuple[int, ...] = ()     # categories per categorical seq feature

    # Static features
    static_cont_dim: int = 0
    static_cat_dims: Tuple[int, ...] = ()

    # Training hyperparams
    lr: float = 2e-4
    betas: Tuple[float, float] = (0.5, 0.9)
    gp_lambda: float = 10.0
    n_critic: int = 5
    gumbel_tau: float = 0.66


# ========= Auxiliary Generator (distribution params) =========

class AuxGenerator(nn.Module):
    """
    Produces distribution parameters:

      Static:
        - continuous: μ, log σ²  (size = static_cont_dim each)
        - categorical: logits per feature (sizes = static_cat_dims)

      Sequence (per timestep):
        - continuous: μ_t, log σ²_t
        - categorical: logits_t per feature
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Static branch
        self.static_mlp = nn.Sequential(
            nn.Linear(cfg.noise_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        static_param_dim = (2 * cfg.static_cont_dim) + sum(cfg.static_cat_dims)
        self.static_out = nn.Linear(cfg.hidden_dim, static_param_dim)

        # Sequence branch (condition on static embedding each step)
        self.seq_lstm = nn.LSTM(
            input_size=cfg.noise_dim + cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            batch_first=True
        )
        seq_param_dim = (2 * cfg.cont_seq_dim) + sum(cfg.cat_seq_dims)
        self.seq_out = nn.Linear(cfg.hidden_dim, seq_param_dim)

    def forward(self, z_static: torch.Tensor, z_seq: torch.Tensor, lengths: List[int]):
        """
        z_static: [B, Z]
        z_seq:    [B, T, Z]
        lengths:  python list of ints
        Returns:
          static_params: dict {cont_mu, cont_logvar, cat_logits(list)}
          seq_params:    dict {cont_mu, cont_logvar, cat_logits(list)}
        """
        B, T, Z = z_seq.shape

        # Static params
        h_s = self.static_mlp(z_static)                 # [B,H]
        raw_s = self.static_out(h_s)                    # [B, static_param_dim]

        s_idx = 0
        static_params: Dict[str, object] = {}
        if self.cfg.static_cont_dim > 0:
            n = self.cfg.static_cont_dim
            static_params["cont_mu"] = raw_s[:, s_idx:s_idx+n]; s_idx += n
            static_params["cont_logvar"] = raw_s[:, s_idx:s_idx+n]; s_idx += n
        static_cat_logits = []
        for k in self.cfg.static_cat_dims:
            static_cat_logits.append(raw_s[:, s_idx:s_idx+k]); s_idx += k
        static_params["cat_logits"] = static_cat_logits

        # Sequence params
        h_s_expand = h_s.unsqueeze(1).expand(-1, T, -1) # [B,T,H]
        seq_in = torch.cat([z_seq, h_s_expand], dim=-1) # [B,T,Z+H]

        packed = nn.utils.rnn.pack_padded_sequence(seq_in, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.seq_lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        raw_t = self.seq_out(out)                       # [B,T,seq_param_dim]

        t_idx = 0
        seq_params: Dict[str, object] = {}
        if self.cfg.cont_seq_dim > 0:
            n = self.cfg.cont_seq_dim
            seq_params["cont_mu"] = raw_t[:, :, t_idx:t_idx+n]; t_idx += n
            seq_params["cont_logvar"] = raw_t[:, :, t_idx:t_idx+n]; t_idx += n
        seq_cat_logits = []
        for k in self.cfg.cat_seq_dims:
            seq_cat_logits.append(raw_t[:, :, t_idx:t_idx+k]); t_idx += k
        seq_params["cat_logits"] = seq_cat_logits

        return static_params, seq_params


# ========= Data Generator (samples from params) =========

class DataGenerator(nn.Module):
    """Samples soft outputs (Gaussian reparam + Gumbel-Softmax)."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, static_params: dict, seq_params: dict, lengths: List[int], tau: float):
        # Static
        static_cont = None
        if self.cfg.static_cont_dim > 0:
            mu = static_params["cont_mu"]
            logvar = static_params["cont_logvar"]
            eps = torch.randn_like(mu)
            static_cont = mu + torch.exp(0.5 * logvar) * eps    # [B,Cs]

        static_cat_probs: List[torch.Tensor] = []
        for logits in static_params["cat_logits"]:
            static_cat_probs.append(gumbel_softmax_sample(logits, tau))  # [B,K]

        # Sequence
        seq_cont = None
        if self.cfg.cont_seq_dim > 0:
            mu = seq_params["cont_mu"]
            logvar = seq_params["cont_logvar"]
            eps = torch.randn_like(mu)
            seq_cont = mu + torch.exp(0.5 * logvar) * eps        # [B,T,Cq]

        seq_cat_probs: List[torch.Tensor] = []
        for logits in seq_params["cat_logits"]:
            seq_cat_probs.append(gumbel_softmax_sample(logits, tau))     # [B,T,K]

        # Apply mask to zero padded steps (cleaner; packing already skips them in LSTM)
        device = seq_cont.device if seq_cont is not None else seq_cat_probs[0].device
        B = len(lengths)
        T = seq_cont.size(1) if seq_cont is not None else seq_cat_probs[0].size(1)
        mask = lengths_to_mask(torch.tensor(lengths, device=device), T).unsqueeze(-1)  # [B,T,1]

        if seq_cont is not None:
            seq_cont = seq_cont * mask
        for i in range(len(seq_cat_probs)):
            seq_cat_probs[i] = seq_cat_probs[i] * mask

        return static_cont, static_cat_probs, seq_cont, seq_cat_probs


# ========= Critic (WGAN-GP) =========

class Critic(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        seq_in_dim = cfg.cont_seq_dim + sum(cfg.cat_seq_dims)
        static_in_dim = cfg.static_cont_dim + sum(cfg.static_cat_dims)

        self.static_fc = nn.Sequential(
            nn.Linear(static_in_dim if static_in_dim > 0 else 1, cfg.hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=seq_in_dim + cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            batch_first=True
        )
        self.fc_out = nn.Linear(cfg.hidden_dim, 1)

    def forward(self,
                seq_cont: Optional[torch.Tensor],
                seq_cat_probs: List[torch.Tensor],
                static_cont: Optional[torch.Tensor],
                static_cat_probs: List[torch.Tensor],
                lengths: List[int]) -> torch.Tensor:

        # Build static vector
        static_parts = []
        if static_cont is not None:
            static_parts.append(static_cont)
        if static_cat_probs:
            static_parts += static_cat_probs
        if static_parts:
            s_vec = torch.cat(static_parts, dim=-1)  # [B,S]
        else:
            # dummy if no static inputs
            s_vec = torch.ones(len(lengths), 1, device=seq_cont.device if seq_cont is not None else seq_cat_probs[0].device)

        s_emb = self.static_fc(s_vec)                # [B,H]

        # Build sequence matrix
        seq_parts = []
        if seq_cont is not None:
            seq_parts.append(seq_cont)
        if seq_cat_probs:
            seq_parts.append(flatten_cat_timewise(seq_cat_probs))
        x_seq = torch.cat(seq_parts, dim=-1)         # [B,T,D]

        B, T, _ = x_seq.shape
        s_broadcast = s_emb.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x_seq, s_broadcast], dim=-1)  # [B,T,D+H]

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        logit = self.fc_out(h_n[-1])                 # [B,1]
        return logit


# ========= WGAN-GP helper =========

def gradient_penalty(critic: Critic,
                     real_batch: dict,
                     fake_batch: dict,
                     lengths: List[int],
                     lam: float) -> torch.Tensor:
    """Interpolate continuous & categorical probs; compute GP."""
    device = next(critic.parameters()).device
    # figure batch size
    if real_batch["seq_cont"] is not None:
        B = real_batch["seq_cont"].size(0)
    else:
        B = real_batch["seq_cat_probs"][0].size(0)

    def _interp(a, b):
        if a is None and b is None:
            return None
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        alpha = torch.rand([B] + [1] * (a.dim() - 1), device=device)
        out = (alpha * a + (1 - alpha) * b).requires_grad_(True)
        return out

    seq_cont_i = _interp(real_batch["seq_cont"], fake_batch["seq_cont"])
    static_cont_i = _interp(real_batch["static_cont"], fake_batch["static_cont"])

    seq_cat_i = []
    for r, f in zip(real_batch["seq_cat_probs"], fake_batch["seq_cat_probs"]):
        seq_cat_i.append(_interp(r, f))
    static_cat_i = []
    for r, f in zip(real_batch["static_cat_probs"], fake_batch["static_cat_probs"]):
        static_cat_i.append(_interp(r, f))

    d_interp = critic(seq_cont_i, seq_cat_i, static_cont_i, static_cat_i, lengths)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=[t for t in [seq_cont_i, static_cont_i] + seq_cat_i + static_cat_i if t is not None],
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True, only_inputs=True
    )
    gp = 0.0
    for g in grads:
        gp = gp + ((g.view(B, -1).norm(2, dim=1) - 1.0) ** 2).mean()
    return lam * gp


# ========= Full model wrapper =========

class TwoGenDoppelGANger(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.G_aux = AuxGenerator(cfg)
        self.G_data = DataGenerator(cfg)
        self.D = Critic(cfg)

    @torch.no_grad()
    def sample(self, lengths: List[int], device=None, tau: Optional[float] = None):
        """Return soft samples dict with given target lengths."""
        if device is None:
            device = next(self.parameters()).device
        B, T = len(lengths), max(lengths)
        z_s = torch.randn(B, self.cfg.noise_dim, device=device)
        z_t = torch.randn(B, T, self.cfg.noise_dim, device=device)
        s_params, t_params = self.G_aux(z_s, z_t, lengths)
        s_cont, s_cat, t_cont, t_cat = self.G_data(s_params, t_params, lengths, tau or self.cfg.gumbel_tau)
        return dict(
            static_cont=s_cont,
            static_cat_probs=s_cat,
            seq_cont=t_cont,
            seq_cat_probs=t_cat,
            lengths=lengths
        )


# ========= Training (single in-memory batch for simplicity) =========

def train_twogen(
    cfg: ModelConfig,
    # real data (already preprocessed)
    real_seq_cont_list: Optional[List[torch.Tensor]],           # list(B) of [L_i, Cq] or None
    real_seq_cat_list_per_feature: List[List[torch.Tensor]],    # per cat feature f: list(B) of [L_i, K_f]
    real_static_cont: Optional[torch.Tensor],                   # [B, Cs] or None
    real_static_cat_list: List[torch.Tensor],                   # list per static cat feature of [B, K_f]
    n_steps: int = 20000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Pass empty list [] if no categorical features; pass None if no continuous for that block.
    All categorical tensors must be one-hot.
    """
    model = TwoGenDoppelGANger(cfg).to(device)

    opt_D = torch.optim.Adam(model.D.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_G = torch.optim.Adam(list(model.G_aux.parameters()), lr=cfg.lr, betas=cfg.betas)  # only Aux has params; Data has none

    # ---- Prepare real batch (pad) ----
    # Lengths from any available sequence stream
    if real_seq_cont_list is not None and len(real_seq_cont_list) > 0:
        lengths = [x.size(0) for x in real_seq_cont_list]
    else:
        # infer from first categorical feature
        lengths = [seq.size(0) for seq in real_seq_cat_list_per_feature[0]]
    B = len(lengths)
    T = max(lengths)

    real_seq_cont = None
    if real_seq_cont_list is not None and len(real_seq_cont_list) > 0:
        real_seq_cont = pad_sequence(real_seq_cont_list, batch_first=True, padding_value=0.0).to(device)

    real_seq_cat_padded = pad_cat_sequences(real_seq_cat_list_per_feature)  # list of [B,T,K_f]
    real_seq_cat_padded = [x.to(device) for x in real_seq_cat_padded]

    real_static_cont = real_static_cont.to(device) if real_static_cont is not None else None
    real_static_cat_list = [x.to(device) for x in real_static_cat_list]

    def real_batch():
        return dict(
            seq_cont=real_seq_cont,
            seq_cat_probs=real_seq_cat_padded,
            static_cont=real_static_cont,
            static_cat_probs=real_static_cat_list
        )

    step = 0
    while step < n_steps:
        # ----- Critic updates -----
        for _ in range(cfg.n_critic):
            opt_D.zero_grad()

            # Fake
            z_s = torch.randn(B, cfg.noise_dim, device=device)
            z_t = torch.randn(B, T, cfg.noise_dim, device=device)
            s_params, t_params = model.G_aux(z_s, z_t, lengths)
            f_s_cont, f_s_cat, f_t_cont, f_t_cat = model.G_data(s_params, t_params, lengths, tau=cfg.gumbel_tau)

            fake = dict(seq_cont=f_t_cont, seq_cat_probs=f_t_cat, static_cont=f_s_cont, static_cat_probs=f_s_cat)
            real = real_batch()

            real_logits = model.D(real["seq_cont"], real["seq_cat_probs"], real["static_cont"], real["static_cat_probs"], lengths)
            fake_logits = model.D(fake["seq_cont"], fake["seq_cat_probs"], fake["static_cont"], fake["static_cat_probs"], lengths)

            d_loss = -(real_logits.mean() - fake_logits.mean())
            gp = gradient_penalty(model.D, real, fake, lengths, cfg.gp_lambda)
            (d_loss + gp).backward()
            opt_D.step()
            step += 1
            if step >= n_steps:
                break
        if step >= n_steps:
            break

        # ----- Generator (Aux) update -----
        opt_G.zero_grad()
        z_s = torch.randn(B, cfg.noise_dim, device=device)
        z_t = torch.randn(B, T, cfg.noise_dim, device=device)
        s_params, t_params = model.G_aux(z_s, z_t, lengths)
        f_s_cont, f_s_cat, f_t_cont, f_t_cat = model.G_data(s_params, t_params, lengths, tau=cfg.gumbel_tau)
        gen_logits = model.D(f_t_cont, f_t_cat, f_s_cont, f_s_cat, lengths)
        g_loss = -gen_logits.mean()
        g_loss.backward()
        opt_G.step()

        if step % 200 == 0:
            print(f"[step {step}] D={d_loss.item():.4f} GP={gp.item():.4f} G={g_loss.item():.4f}")

    return model


# ========= Postprocess (optional) =========

@torch.no_grad()
def postprocess_samples(
    samples: dict,
    inverse_seq_cont=None,         # e.g., StandardScaler for seq continuous (fit on stacked timesteps)
    inverse_static_cont=None,      # e.g., StandardScaler for static continuous
    static_cat_index_to_label: Optional[List[List[str]]] = None,
    seq_cat_index_to_label: Optional[List[List[str]]] = None,
    hard_cat: bool = True,
):
    """Turn soft probs -> indices/labels; inverse-transform continuous."""
    out = {"lengths": samples["lengths"]}

    # Static continuous
    if samples["static_cont"] is not None:
        sc = samples["static_cont"].cpu().numpy()
        if inverse_static_cont is not None:
            sc = inverse_static_cont.inverse_transform(sc)
        out["static_continuous"] = sc

    # Static categorical
    static_cats = []
    for i, probs in enumerate(samples["static_cat_probs"]):
        if hard_cat:
            idx = probs.argmax(-1).cpu().numpy()
        else:
            idx = torch.multinomial(probs, 1).squeeze(1).cpu().numpy()
        if static_cat_index_to_label:
            static_cats.append([static_cat_index_to_label[i][j] for j in idx])
        else:
            static_cats.append(idx)
    out["static_categorical"] = static_cats

    # Sequence continuous
    if samples["seq_cont"] is not None:
        qc = samples["seq_cont"].cpu().numpy()
        if inverse_seq_cont is not None:
            B, T, C = qc.shape
            qc = inverse_seq_cont.inverse_transform(qc.reshape(-1, C)).reshape(B, T, C)
        out["seq_continuous"] = qc

    # Sequence categorical
    seq_cats = []
    for f, probs in enumerate(samples["seq_cat_probs"]):
        if hard_cat:
            idx = probs.argmax(-1).cpu().numpy()  # [B,T]
        else:
            B, T, K = probs.shape
            idx = torch.multinomial(probs.view(B*T, K), 1).view(B, T).cpu().numpy()
        if seq_cat_index_to_label:
            seq_cats.append([[seq_cat_index_to_label[f][j] for j in row] for row in idx])
        else:
            seq_cats.append(idx)
    out["seq_categorical"] = seq_cats
    return out


# ========= Minimal usage sketch =========
if __name__ == "__main__":
    # Example config — replace with your dataset’s dims
    cfg = ModelConfig(
        noise_dim=32,
        hidden_dim=128,
        cont_seq_dim=2,             # e.g., spend, tc
        cat_seq_dims=(50, 8),       # e.g., merchant_id(50), state_bin(8)
        static_cont_dim=3,          # e.g., age, income, hh_size
        static_cat_dims=(2, 5),     # e.g., gender(2), region(5)
        lr=2e-4, betas=(0.5, 0.9), gp_lambda=10.0, n_critic=5, gumbel_tau=0.66
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Fake toy data (replace with your preprocessed tensors) ----
    B = 6
    lengths = [3, 1, 4, 2, 3, 5]           # variable lengths per user
    T = max(lengths)

    # Continuous sequences: list of [L_i, cont_seq_dim]
    real_seq_cont_list = [torch.randn(l, cfg.cont_seq_dim) for l in lengths]

    # Categorical sequences: per feature f, list(B) of [L_i, K_f] one-hot
    def one_hot(idx, K):
        v = torch.zeros(K); v[idx] = 1.0; return v
    real_seq_cat_list_per_feature = []
    for K in cfg.cat_seq_dims:
        feat_list = []
        for l in lengths:
            inds = torch.randint(0, K, (l,))
            feat_list.append(torch.stack([one_hot(i.item(), K) for i in inds], dim=0))
        real_seq_cat_list_per_feature.append(feat_list)

    # Static continuous: [B, static_cont_dim]
    real_static_cont = torch.randn(B, cfg.static_cont_dim) if cfg.static_cont_dim > 0 else None

    # Static categorical: list per feature of [B, K_f] one-hot
    real_static_cat_list = []
    for K in cfg.static_cat_dims:
        inds = torch.randint(0, K, (B,))
        oh = torch.zeros(B, K); oh[torch.arange(B), inds] = 1.0
        real_static_cat_list.append(oh)

    # ---- Train (toy) ----
    model = train_twogen(
        cfg,
        real_seq_cont_list=real_seq_cont_list,
        real_seq_cat_list_per_feature=real_seq_cat_list_per_feature,
        real_static_cont=real_static_cont,
        real_static_cat_list=real_static_cat_list,
        n_steps=600,   # small for demo
        device=device
    )

    # ---- Sample ----
    samples = model.sample(lengths)
    decoded = postprocess_samples(samples)
    print("Sampled lengths:", decoded["lengths"])

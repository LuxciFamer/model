"""Purely data-driven baselines: LSTM / GRU sequence-to-one regressors (Kratzert et al., 2019 style)."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.metrics import nse


class SeqRegressor(nn.Module):
    def __init__(self, n_feat, hidden=64, dropout=0.2, cell="LSTM"):
        super().__init__()
        rnn = nn.LSTM if cell == "LSTM" else nn.GRU
        self.rnn = rnn(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return nn.functional.softplus(self.head(self.drop(h[:, -1, :])))[:, 0]


def make_windows(Xs: np.ndarray, idx: np.ndarray, L: int) -> np.ndarray:
    return np.stack([Xs[i - L + 1:i + 1] for i in idx])


def train_seq2one(Xs, q, cal_idx, val_idx, cell, cfg_l, budget, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)
    L = cfg_l["seq_len"]
    cal_idx = cal_idx[(cal_idx >= L - 1) & np.isfinite(q[cal_idx])]
    val_idx = val_idx[(val_idx >= L - 1) & np.isfinite(q[val_idx])]
    Xc = torch.tensor(make_windows(Xs, cal_idx, L), dtype=torch.float32, device=device)
    yc = torch.tensor(q[cal_idx], dtype=torch.float32, device=device)
    Xv = torch.tensor(make_windows(Xs, val_idx, L), dtype=torch.float32, device=device)
    model = SeqRegressor(Xs.shape[1], cfg_l["hidden_dim"], cfg_l["dropout"], cell).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg_l["lr"])
    var = yc.var() + 1e-6
    best, state, wait, hist = -np.inf, None, 0, []
    for ep in range(budget["epochs"]):
        model.train()
        perm = torch.randperm(len(yc), device=device)
        for b in range(0, len(yc), cfg_l["batch_size"]):
            j = perm[b:b + cfg_l["batch_size"]]
            opt.zero_grad()
            loss = ((model(Xc[j]) - yc[j]) ** 2).mean() / var
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).cpu().numpy()
        s = nse(q[val_idx], pv)
        hist.append({"epoch": ep, "val_NSE": s})
        if s > best + 1e-4:
            best, state, wait = s, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= budget["patience"]:
                break
    model.load_state_dict(state)
    return model, hist


def predict_seq2one(model, Xs, L, device, batch=512):
    idx = np.arange(L - 1, len(Xs))
    out = np.full(len(Xs), np.nan)
    model.eval()
    with torch.no_grad():
        for b in range(0, len(idx), batch):
            j = idx[b:b + batch]
            xb = torch.tensor(make_windows(Xs, j, L), dtype=torch.float32, device=device)
            out[j] = model(xb).cpu().numpy()
    return out

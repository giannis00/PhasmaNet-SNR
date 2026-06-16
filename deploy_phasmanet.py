#!/usr/bin/env python3

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn


class PhiRefractor(nn.Module):
    def __init__(self, dim: int, num_subspaces: int = 4, dropout: float = 0.1):
        super().__init__()
        base = dim // num_subspaces
        sizes = [base] * num_subspaces
        sizes[0] += dim - base * num_subspaces
        self.sizes = sizes

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(s, s),
                nn.GELU(),
                nn.Linear(s, s),
            )
            for s in sizes
        ])
        self.recombine = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        splits = torch.split(x, self.sizes, dim=-1)
        transformed = [blk(s) for blk, s in zip(self.blocks, splits)]
        cat = torch.cat(transformed, dim=-1)
        weights = torch.softmax(self.recombine(cat), dim=-1)
        return cat + self.dropout(weights * cat)


class PhiStageEncoder(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop1(a))
        f = self.ffn(x)
        x = self.norm2(x + self.drop2(f))
        return x


class PhasmaNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_tokens: int = 8,
        num_stages: int = 3,
        num_feedback: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        num_refractor_subspaces: int = 4,
    ):
        super().__init__()
        self.T = num_tokens
        self.D = hidden_dim
        self.K = num_stages
        self.R = num_feedback

        self.tokenizer = nn.Linear(input_dim, self.T * self.D)

        # Keep legacy names (cascades/prisms/gates) to match saved checkpoints.
        self.cascades = nn.ModuleList([
            PhiStageEncoder(self.D, heads=heads, dropout=dropout)
            for _ in range(self.K)
        ])

        self.prisms = nn.ModuleList([
            PhiRefractor(self.D, num_subspaces=num_refractor_subspaces, dropout=dropout)
            for _ in range(self.K)
        ])

        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(self.D, self.D), nn.Sigmoid())
            for _ in range(self.K - 1)
        ])

        self.out_norm = nn.LayerNorm(self.D)
        self.regressor = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.GELU(),
            nn.Linear(self.D, 1),
        )

    def forward(self, x):
        bsz = x.size(0)
        tok = self.tokenizer(x).view(bsz, self.T, self.D)

        embs = [tok]
        for i in range(self.K):
            h = self.cascades[i](embs[-1])
            h = self.prisms[i](h)
            embs.append(h)

        for _ in range(self.R):
            for lvl in range(self.K, 0, -1):
                high = embs[lvl]
                low = embs[lvl - 1]
                if (lvl - 1) < len(self.gates):
                    gate = self.gates[lvl - 1](high)
                else:
                    gate = torch.sigmoid(high)
                embs[lvl - 1] = low + gate * high

            for i in range(self.K):
                embs[i + 1] = self.prisms[i](self.cascades[i](embs[i]))

        top = self.out_norm(embs[-1])
        pooled = top.mean(dim=1)
        return self.regressor(pooled)


def parse_args():
    p = argparse.ArgumentParser(description="Run deployed PhasmaNet checkpoint inference on a pickle dataset.")
    p.add_argument("--data", required=True, help="Input dataset pickle path")
    p.add_argument("--model-dir", default="PhasmaNet-1M", help="Directory containing best_model.pt")
    p.add_argument("--batch", type=int, default=4096, help="Inference batch size")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    p.add_argument("--gpu-id", type=int, default=0, help="CUDA device index to use when --device is cuda/auto")
    p.add_argument("--cpu-threads", type=int, default=None, help="Optional torch CPU thread cap")
    p.add_argument("--out", default=None, help="Output npz path. Defaults to <model-dir>/preds_deploy.npz")
    p.add_argument("--out-csv", default=None, help="Optional output CSV path")
    p.add_argument("--cal-knots", default=None,
                   help="PCHIP calibrator knots JSON. Defaults to <model-dir>/smooth_cal_knots_extended.json")
    p.add_argument("--no-calibration", action="store_true",
                   help="Skip PCHIP calibration even if knots are available.")
    return p.parse_args()


def resolve_device(args):
    if args.cpu_threads is not None and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    if args.device == "cpu":
        return torch.device("cpu")

    cuda_ok = torch.cuda.is_available()
    n_cuda = torch.cuda.device_count() if cuda_ok else 0

    if args.device == "cuda":
        if not cuda_ok:
            raise RuntimeError(
                "--device cuda requested but CUDA is not available. "
                "Check your conda env PyTorch build, NVIDIA driver, and CUDA visibility."
            )
        if args.gpu_id < 0 or args.gpu_id >= n_cuda:
            raise RuntimeError(f"Invalid --gpu-id {args.gpu_id}; visible CUDA devices: {n_cuda}")
        torch.cuda.set_device(args.gpu_id)
        return torch.device(f"cuda:{args.gpu_id}")

    if cuda_ok:
        chosen = args.gpu_id if 0 <= args.gpu_id < n_cuda else 0
        torch.cuda.set_device(chosen)
        return torch.device(f"cuda:{chosen}")
    return torch.device("cpu")


def load_scaler(model_dir, ckpt):
    if "feat_mean" in ckpt and "feat_std" in ckpt and "features" in ckpt:
        return np.asarray(ckpt["feat_mean"], dtype=np.float32), np.asarray(ckpt["feat_std"], dtype=np.float32), list(ckpt["features"])

    scaler_path = os.path.join(model_dir, "feature_scaler.json")
    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    return np.asarray(scaler["mean"], dtype=np.float32), np.asarray(scaler["std"], dtype=np.float32), list(scaler["features"])


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Map NumPy 2.x pickle module paths to NumPy 1.x equivalents."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def read_pickle_compat(path):
    """Load DataFrame pickle across NumPy major-version module path differences."""
    try:
        return pd.read_pickle(path)
    except ModuleNotFoundError as e:
        if "numpy._core" not in str(e):
            raise
        with open(path, "rb") as f:
            obj = _NumpyCompatUnpickler(f).load()
        if isinstance(obj, pd.DataFrame):
            return obj
        raise TypeError(f"Expected a pandas DataFrame in pickle, got: {type(obj)!r}")


def log10_to_linear_safe(log_values, out_dtype=np.float32):
    max_log = np.log10(np.finfo(out_dtype).max)
    min_log = np.log10(np.finfo(out_dtype).tiny)
    clipped = np.clip(log_values, min_log, max_log)
    lin64 = np.power(np.float64(10.0), clipped.astype(np.float64))
    lin64 = np.clip(lin64, np.finfo(out_dtype).tiny, np.finfo(out_dtype).max)
    return lin64.astype(out_dtype)


def load_calibrator(knots_path):
    """Rebuild the PCHIP monotonic calibrator fitted during training.

    The training script (PhasmaNet_SNR.py) fits a smooth monotonic calibrator on
    the validation split and saves its knots to smooth_cal_knots_extended.json.
    At deploy time we reload those knots and apply the same mapping to linear-SNR
    predictions; we do NOT refit (no validation split is available here).
    """
    if not os.path.exists(knots_path):
        return None

    from scipy.interpolate import PchipInterpolator

    with open(knots_path, "r", encoding="utf-8") as f:
        knots = json.load(f)
    xk = np.asarray(knots["xk"], dtype=np.float64)
    yk = np.asarray(knots["yk"], dtype=np.float64)
    if xk.size < 3 or yk.size != xk.size:
        return None

    pchip = PchipInterpolator(xk, yk, extrapolate=True)

    def calibrate_fn(x):
        return pchip(np.asarray(x, dtype=np.float64))

    return calibrate_fn


def main():
    args = parse_args()
    device = resolve_device(args)

    ckpt_path = os.path.join(args.model_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    feat_mean, feat_std, features = load_scaler(args.model_dir, ckpt)
    feedback = int(ckpt.get("recursions", ckpt.get("feedback", 2)))

    model = PhasmaNet(
        input_dim=len(features),
        hidden_dim=128,
        num_tokens=8,
        num_stages=3,
        num_feedback=feedback,
        heads=4,
        dropout=0.1,
        num_refractor_subspaces=4,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    df = read_pickle_compat(args.data)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Input dataset is missing required feature columns: {missing}")

    x = df[features].to_numpy(dtype=np.float32)
    x = (x - feat_mean) / (feat_std + 1e-9)
    x_t = torch.from_numpy(x)

    preds_log_chunks = []
    with torch.inference_mode():
        for i in range(0, x_t.size(0), args.batch):
            xb = x_t[i:i + args.batch].to(device, non_blocking=True)
            pb = model(xb).squeeze(-1).detach().cpu().numpy()
            preds_log_chunks.append(pb)

    pred_log = np.concatenate(preds_log_chunks).astype(np.float32)
    pred_lin = log10_to_linear_safe(pred_log, out_dtype=np.float32)

    # Smooth monotonic (PCHIP) calibration, reusing the knots fitted on the
    # validation split during training.
    cal_fn = None
    knots_path = args.cal_knots or os.path.join(args.model_dir, "smooth_cal_knots_extended.json")
    if not args.no_calibration:
        cal_fn = load_calibrator(knots_path)
    pred_lin_cal = None
    pred_log_cal = None
    if cal_fn is not None:
        pred_lin_cal = cal_fn(pred_lin).astype(np.float32)
        pred_log_cal = np.log10(np.clip(pred_lin_cal, 1e-12, None)).astype(np.float32)

    out_npz = args.out or os.path.join(args.model_dir, "preds_deploy.npz")
    save_payload = {
        "pred_log": pred_log,
        "pred_lin": pred_lin,
        "features": np.array(features),
    }
    if pred_lin_cal is not None:
        save_payload["pred_lin_cal"] = pred_lin_cal
        save_payload["pred_log_cal"] = pred_log_cal
    if "logSNR" in df.columns:
        true_log = df["logSNR"].to_numpy(dtype=np.float32)
        true_lin = log10_to_linear_safe(true_log, out_dtype=np.float32)
        save_payload["true_log"] = true_log
        save_payload["true_lin"] = true_lin
    np.savez_compressed(out_npz, **save_payload)

    out_csv = args.out_csv or os.path.join(args.model_dir, "preds_deploy.csv")
    out_df = pd.DataFrame({"pred_logSNR": pred_log, "pred_SNR": pred_lin})
    if pred_lin_cal is not None:
        out_df["pred_SNR_cal"] = pred_lin_cal
        out_df["pred_logSNR_cal"] = pred_log_cal
    if "logSNR" in df.columns:
        out_df["true_logSNR"] = df["logSNR"].to_numpy(dtype=np.float32)
    out_df.to_csv(out_csv, index=False)

    print(f"Device: {device}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and device.type == "cuda":
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Using GPU: {torch.cuda.get_device_name(device.index if device.index is not None else 0)}")
        torch.cuda.synchronize(device)
    print(f"Rows inferred: {len(df)}")
    if pred_lin_cal is not None:
        print(f"Calibration: PCHIP applied (knots: {knots_path})")
    elif args.no_calibration:
        print("Calibration: skipped (--no-calibration)")
    else:
        print(f"Calibration: skipped (no usable knots at {knots_path})")
    print(f"Model checkpoint: {ckpt_path}")
    print(f"Saved NPZ: {out_npz}")
    print(f"Saved CSV: {out_csv}")


if __name__ == "__main__":
    main()

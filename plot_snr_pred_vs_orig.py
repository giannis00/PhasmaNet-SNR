#!/usr/bin/env python3

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Plot SNR_pred vs SNR_orig from deployment outputs.")
    p.add_argument("--preds-npz", default="PhasmaNet-1M/preds_deploy.npz", help="NPZ produced by deploy_phasmanet.py")
    p.add_argument("--data", default=None, help="Optional original dataset pickle (used if true values are absent in NPZ)")
    p.add_argument("--out", default="PhasmaNet-1M/snr_pred_vs_orig.png", help="Output figure path")
    p.add_argument("--mode", choices=["hexbin", "scatter"], default="hexbin", help="Plot mode for large point clouds")
    p.add_argument("--max-points", type=int, default=200000, help="Max sampled points for scatter mode")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--log-scale", action="store_true", default=True, help="Use log-log axes")
    p.add_argument("--no-log-scale", dest="log_scale", action="store_false", help="Disable log-log axes")
    p.add_argument("--calibrated", action="store_true",
                   help="Plot PCHIP-calibrated predictions (pred_lin_cal) instead of raw (pred_lin).")
    return p.parse_args()


def log10_to_linear_safe(log_values, out_dtype=np.float64):
    max_log = np.log10(np.finfo(out_dtype).max)
    min_log = np.log10(np.finfo(out_dtype).tiny)
    clipped = np.clip(np.asarray(log_values, dtype=np.float64), min_log, max_log)
    lin = np.power(np.float64(10.0), clipped)
    lin = np.clip(lin, np.finfo(out_dtype).tiny, np.finfo(out_dtype).max)
    return lin.astype(out_dtype)


def load_true_and_pred(args):
    arr = np.load(args.preds_npz, allow_pickle=True)
    pred_key = "pred_lin_cal" if args.calibrated else "pred_lin"
    if args.calibrated and pred_key not in arr:
        raise KeyError(
            "--calibrated requested but 'pred_lin_cal' is not in the NPZ. "
            "Re-run deploy_phasmanet.py with calibration enabled."
        )
    pred = arr[pred_key].astype(np.float64)

    if "true_lin" in arr:
        true = arr["true_lin"].astype(np.float64)
        return true, pred

    if "true_log" in arr:
        true = log10_to_linear_safe(arr["true_log"], out_dtype=np.float64)
        return true, pred

    if args.data is None:
        raise RuntimeError(
            "No true values found in NPZ. Provide --data with a pickle containing logSNR."
        )

    df = pd.read_pickle(args.data)
    if "logSNR" not in df.columns:
        raise KeyError("Column logSNR not found in provided dataset.")

    true = log10_to_linear_safe(df["logSNR"].to_numpy(dtype=np.float64), out_dtype=np.float64)
    n = min(len(true), len(pred))
    return true[:n], pred[:n]


def maybe_sample(true, pred, max_points, seed=1337):
    n = len(true)
    if n <= max_points:
        return true, pred
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return true[idx], pred[idx]


def main():
    args = parse_args()
    true, pred = load_true_and_pred(args)

    # Drop non-finite / non-positive points (log axes need >0) and guard against
    # corrupt outliers in the source labels that would otherwise blow up the
    # axis range and hide the real point cloud.
    valid = np.isfinite(true) & np.isfinite(pred) & (true > 0) & (pred > 0)
    n_dropped = int(valid.size - valid.sum())
    true, pred = true[valid], pred[valid]

    lo_t, hi_t = np.percentile(true, [0.05, 99.95])
    lo_p, hi_p = np.percentile(pred, [0.05, 99.95])
    lo = float(max(min(lo_t, lo_p), 1e-12))
    hi = float(max(hi_t, hi_p))
    keep = (true >= lo) & (true <= hi) & (pred >= lo) & (pred <= hi)
    n_dropped += int(keep.size - keep.sum())
    true, pred = true[keep], pred[keep]

    if n_dropped:
        print(f"Dropped {n_dropped} invalid/outlier points before plotting.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.3, 6.0))

    if args.mode == "scatter":
        t, p = maybe_sample(true, pred, args.max_points)
        ax.scatter(t, p, s=4, alpha=0.25, linewidths=0)
    else:
        hb = ax.hexbin(true, pred, gridsize=180, bins="log", mincnt=1)
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("log10(count)")

    ax.plot([lo, hi], [lo, hi], lw=2)

    if args.log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("SNR_orig")
    ax.set_ylabel("SNR_pred (calibrated)" if args.calibrated else "SNR_pred")
    ax.set_title("Predicted vs Original SNR" + (" (calibrated)" if args.calibrated else ""))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved plot: {args.out}")
    print(f"Points: {len(true)}")
    print(f"Mode: {args.mode}")


if __name__ == "__main__":
    main()

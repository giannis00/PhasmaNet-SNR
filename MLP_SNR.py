#!/usr/bin/env python3
# MLP_SNR.py
# Predict log10(SNR) with an MLP baseline.
#
# Artifacts saved per run (in --outdir):
#   - best_model.pt
#   - feature_scaler.json
#   - split_indices.npz
#   - run_config.json
#   - metrics.json
#   - history.csv
#   - preds_val.npz
#   - preds_test_raw.npz
#   - preds_test_cal.npz
#   - smooth_cal_knots_extended.json
#   - plots (*.png)

import os, sys, json, argparse, time, platform, hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# --------------------------- Args ---------------------------
p = argparse.ArgumentParser()
p.add_argument("--data", type=str, required=True)
p.add_argument("--outdir", type=str, default="runs_logsnr_mlp")
p.add_argument("--batch", type=int, default=128)
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--patience", type=int, default=20)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=1337)

# filtering (optional)
p.add_argument("--snr-cut", type=float, default=200,
               help="We keep rows with logSNR < log10(snr_cut). (i.e., SNR < snr_cut)")

p.add_argument("--acc-pct", type=float, default=1.0,
               help="Tolerance in linear SNR percent for the simple within-tolerance metric (not MAPE).")

# loss options
p.add_argument("--loss", choices=["mse", "huber"], default="huber")
p.add_argument("--huber-delta", type=float, default=0.03,
               help="Huber delta in log10(SNR)")

# weighting options
p.add_argument("--weighting", choices=["none", "by_snr", "by_hist"], default="by_snr",
               help="Loss weighting scheme")
p.add_argument("--alpha", type=float, default=0.5,
               help="Exponent for by_snr weighting (weight ∝ SNR^alpha)")
p.add_argument("--hist-bins", type=int, default=20,
               help="Bins for by_hist weighting over log10(SNR)")
p.add_argument("--wmin", type=float, default=0.2)
p.add_argument("--wmax", type=float, default=5.0)

# MLP knobs
p.add_argument("--width", type=int, default=256)
p.add_argument("--depth", type=int, default=4)
p.add_argument("--dropout", type=float, default=0.10)
p.add_argument("--act", choices=["relu", "gelu"], default="gelu")

args = p.parse_args()

# --------------------------- Setup ---------------------------
RUN_TS = time.strftime("%Y%m%d-%H%M%S")
os.makedirs(args.outdir, exist_ok=True)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def dataset_fingerprint(df: pd.DataFrame, cols, n=256):
    sub = df[cols].head(min(n, len(df))).to_numpy().astype(np.float32)
    h = hashlib.sha256(sub.tobytes()).hexdigest()
    return {"sha256_head": h, "n_head": int(min(n, len(df))), "cols": cols}

def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

# --------------------------- Load & filter ---------------------------
df = pd.read_pickle(args.data)
before = len(df)

if args.snr_cut is not None:
    df = df[df["logSNR"] < np.log10(args.snr_cut)].reset_index(drop=True)
    print(f"Filter applied: kept logSNR < log10({args.snr_cut}) | {before} -> {len(df)} rows")

FEATURES = [
    "ds", "logMc", "logq", "cos_inc",
    "s1x", "s1y", "s1z", "s2x", "s2y", "s2z",
    "Fp", "Fx", "tc"
]
TARGET = "logSNR"

logsnr = df[TARGET].to_numpy().astype(np.float64)
snr_lin = 10.0 ** logsnr

data_stats = {
    "data_path": os.path.abspath(args.data),
    "rows_before_filter": int(before),
    "rows_after_filter": int(len(df)),
    "snr_cut": safe_float(args.snr_cut),
    "filter_rule": "kept logSNR < log10(snr_cut)" if args.snr_cut is not None else "no_filter",
    "logSNR_quantiles": {
        "p50": safe_float(np.quantile(logsnr, 0.50)),
        "p90": safe_float(np.quantile(logsnr, 0.90)),
        "p99": safe_float(np.quantile(logsnr, 0.99)),
    },
    "SNR_quantiles": {
        "p50": safe_float(np.quantile(snr_lin, 0.50)),
        "p90": safe_float(np.quantile(snr_lin, 0.90)),
        "p99": safe_float(np.quantile(snr_lin, 0.99)),
    },
    "fingerprint": dataset_fingerprint(df, FEATURES + [TARGET], n=256),
}

X = df[FEATURES].values.astype(np.float32)
y = df[[TARGET]].values.astype(np.float32)

N = len(df)
if N < 200:
    raise RuntimeError(f"Too few samples after filtering: {N}")

train_size = int(0.70 * N)
val_size   = int(0.15 * N)
test_size  = N - train_size - val_size

X_t = torch.tensor(X)
y_t = torch.tensor(y)
dataset = TensorDataset(X_t, y_t)

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(args.seed)
)

# Save split indices
np.savez_compressed(
    os.path.join(args.outdir, "split_indices.npz"),
    train_idx=np.array(train_set.indices, dtype=np.int64),
    val_idx=np.array(val_set.indices, dtype=np.int64),
    test_idx=np.array(test_set.indices, dtype=np.int64),
)

# --------------------------- Normalization (train only) ---------------------------
def stack_subset(sub):
    Xs, ys = [], []
    for xb, yb in sub:
        Xs.append(xb.numpy())
        ys.append(yb.numpy())
    return np.vstack(Xs), np.vstack(ys)

Xtr_np, _ = stack_subset(train_set)
feat_mean = Xtr_np.mean(axis=0)
feat_std  = Xtr_np.std(axis=0) + 1e-9

feat_mean_t = torch.tensor(feat_mean, dtype=torch.float32)
feat_std_t  = torch.tensor(feat_std, dtype=torch.float32)

def norm(x: torch.Tensor) -> torch.Tensor:
    return (x - feat_mean_t.to(x.device)) / feat_std_t.to(x.device)

class NormSet(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset
    def __len__(self): return len(self.subset)
    def __getitem__(self, i):
        xb, yb = self.subset[i]
        return norm(xb), yb

# --------------------------- Weights (train only) ---------------------------
def make_weights_for_subset(subset, scheme):
    _, y_np = stack_subset(subset)  # y in log10(SNR)
    y_log = y_np.squeeze()

    if scheme == "none":
        w = np.ones_like(y_log)
    elif scheme == "by_snr":
        snr = 10.0 ** y_log
        w = (snr / (np.median(snr) + 1e-12)) ** args.alpha
    elif scheme == "by_hist":
        hist, edges = np.histogram(y_log, bins=args.hist_bins)
        hist = np.maximum(hist, 1)
        bin_idx = np.clip(np.searchsorted(edges, y_log, side="right") - 1, 0, len(hist) - 1)
        freq = hist[bin_idx]
        w = 1.0 / freq
    else:
        raise ValueError("Unknown weighting scheme")

    w = np.clip(w, args.wmin, args.wmax)
    w = w / (np.mean(w) + 1e-12)
    return w.astype(np.float32)

train_weights_np = make_weights_for_subset(train_set, args.weighting)
train_weights_t  = torch.tensor(train_weights_np, dtype=torch.float32)

class NormWeightedSet(torch.utils.data.Dataset):
    def __init__(self, subset, weights_tensor):
        self.subset = subset
        self.w = weights_tensor
    def __len__(self): return len(self.subset)
    def __getitem__(self, i):
        xb, yb = self.subset[i]
        return norm(xb), yb, self.w[i]

train_loader = DataLoader(
    NormWeightedSet(train_set, train_weights_t),
    batch_size=args.batch, shuffle=True,
    num_workers=0, pin_memory=(device.type == "cuda")
)
val_loader   = DataLoader(NormSet(val_set),  batch_size=args.batch, shuffle=False)
test_loader  = DataLoader(NormSet(test_set), batch_size=args.batch, shuffle=False)

# save scaler
save_json(
    os.path.join(args.outdir, "feature_scaler.json"),
    {"mean": feat_mean.tolist(), "std": feat_std.tolist(), "features": FEATURES}
)

# --------------------------- Model: MLP ---------------------------
def make_activation(name: str):
    return nn.ReLU() if name == "relu" else nn.GELU()

class MLPRegressor(nn.Module):
    def __init__(self, d_in: int, width: int, depth: int, dropout: float, act: str):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(depth):
            layers.append(nn.Linear(d, width))
            layers.append(make_activation(act))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

model = MLPRegressor(
    d_in=len(FEATURES),
    width=args.width,
    depth=args.depth,
    dropout=args.dropout,
    act=args.act
).to(device)

param_count = count_params(model)
print("Model: MLPRegressor")
print("Param count:", param_count)
print(f"MLP config | width={args.width} depth={args.depth} dropout={args.dropout} act={args.act}")

# --- OPTIMIZER / SCHED ---
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, verbose=True
)

mse_loss = nn.MSELoss(reduction="none")
huber_loss = nn.SmoothL1Loss(beta=args.huber_delta, reduction="none")

def loss_fn(pred, true, weights=None):
    if args.loss == "mse":
        l = mse_loss(pred, true)
    else:
        l = huber_loss(pred, true)
    if weights is not None:
        l = l.squeeze() * weights.to(l.device)
    return l.mean()

# --------------------------- Metrics ---------------------------
def eval_loader(loader, with_weights=False):
    model.eval()
    total_loss = 0.0
    preds_log, trues_log = [], []
    with torch.no_grad():
        for batch in loader:
            if with_weights:
                xb, yb, wb = batch
                xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
                p = model(xb)
                total_loss += loss_fn(p, yb, wb).item() * xb.size(0)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                p = model(xb)
                total_loss += loss_fn(p, yb, None).item() * xb.size(0)

            preds_log.append(p.cpu().numpy())
            trues_log.append(yb.cpu().numpy())

    total_loss /= len(loader.dataset)
    preds_log = np.vstack(preds_log).squeeze()
    trues_log = np.vstack(trues_log).squeeze()

    pred_lin = 10.0 ** preds_log
    true_lin = 10.0 ** trues_log

    mape = float(np.mean(np.abs(pred_lin - true_lin) / (true_lin + 1e-12))) * 100.0
    score_100_minus_mape = 100.0 - mape

    pct = args.acc_pct / 100.0
    within_tol = float(np.mean(np.abs(pred_lin - true_lin) / (true_lin + 1e-12) <= pct))

    ss_res = np.sum((preds_log - trues_log) ** 2)
    ss_tot = np.sum((trues_log - trues_log.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return total_loss, mape, score_100_minus_mape, within_tol, r2, preds_log, trues_log

# --------------------------- Train ---------------------------
history = {
    "tr_loss": [], "va_loss": [],
    "tr_within": [], "va_within": [],
    "tr_mape": [], "va_mape": [],
    "tr_r2": [], "va_r2": []
}

best_val = np.inf
pat = 0
best_path = os.path.join(args.outdir, "best_model.pt")

for epoch in range(1, args.epochs + 1):
    model.train()
    for xb, yb, wb in train_loader:
        xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb, wb)
        loss.backward()
        optimizer.step()

    tr_loss, tr_mape, tr_score, tr_within, tr_r2, _, _ = eval_loader(train_loader, with_weights=True)
    va_loss, va_mape, va_score, va_within, va_r2, _, _ = eval_loader(val_loader)

    scheduler.step(va_loss)

    history["tr_loss"].append(tr_loss);     history["va_loss"].append(va_loss)
    history["tr_within"].append(tr_within); history["va_within"].append(va_within)
    history["tr_mape"].append(tr_mape);     history["va_mape"].append(va_mape)
    history["tr_r2"].append(tr_r2);         history["va_r2"].append(va_r2)

    print(f"Epoch {epoch:03d} | "
          f"train loss {tr_loss:.5f}  val loss {va_loss:.5f} | "
          f"MAPE tr {tr_mape:.2f}%  val {va_mape:.2f}% | "
          f"Acc(1-MAPE) tr {tr_score:.2f}%  val {va_score:.2f}% | "
          f"R2 tr {tr_r2:.4f}  val {va_r2:.4f}")

    if va_loss + 1e-6 < best_val:
        best_val = va_loss
        pat = 0
        torch.save({
            "model": model.state_dict(),
            "feat_mean": feat_mean, "feat_std": feat_std,
            "features": FEATURES,
            "args": vars(args),
        }, best_path)
    else:
        pat += 1
        if pat >= args.patience:
            print("Early stopping.")
            break

print("Best val loss:", best_val)
print("Saved best model ->", best_path)

# Save training history
hist_df = pd.DataFrame({
    "epoch": np.arange(1, len(history["tr_loss"]) + 1),
    "tr_loss": history["tr_loss"],
    "va_loss": history["va_loss"],
    "tr_mape": history["tr_mape"],
    "va_mape": history["va_mape"],
    "tr_r2": history["tr_r2"],
    "va_r2": history["va_r2"],
    "tr_within_tol": history["tr_within"],
    "va_within_tol": history["va_within"],
})
hist_df.to_csv(os.path.join(args.outdir, "history.csv"), index=False)

# --------------------------- Load best & evaluate (VAL + TEST) ---------------------------
ckpt = torch.load(best_path, map_location=device)
model.load_state_dict(ckpt["model"])

va_loss, va_mape, va_score, va_within, va_r2, pred_log_va, true_log_va = eval_loader(val_loader)
te_loss, te_mape, te_score, te_within, te_r2, pred_log_te, true_log_te = eval_loader(test_loader)

print(f"TEST | loss {te_loss:.5f} | MAPE {te_mape:.2f}% | Acc(1-MAPE) {te_score:.2f}% | R2 {te_r2:.4f}")

# Save raw predictions (VAL + TEST)
pred_lin_va = 10.0 ** pred_log_va
true_lin_va = 10.0 ** true_log_va
np.savez_compressed(
    os.path.join(args.outdir, "preds_val.npz"),
    pred_log=pred_log_va.astype(np.float32),
    true_log=true_log_va.astype(np.float32),
    pred_lin=pred_lin_va.astype(np.float32),
    true_lin=true_lin_va.astype(np.float32),
)

pred_lin_te = 10.0 ** pred_log_te
true_lin_te = 10.0 ** true_log_te
split_npz = np.load(os.path.join(args.outdir, "split_indices.npz"))
test_idx = split_npz["test_idx"]

np.savez_compressed(
    os.path.join(args.outdir, "preds_test_raw.npz"),
    pred_log=pred_log_te.astype(np.float32),
    true_log=true_log_te.astype(np.float32),
    pred_lin=pred_lin_te.astype(np.float32),
    true_lin=true_lin_te.astype(np.float32),
    sample_idx=test_idx.astype(np.int64),
)

# --------------------------- Plots (RAW) ---------------------------
epochs = np.arange(1, len(history["tr_loss"]) + 1)

plt.figure(figsize=(6.2, 4.2))
plt.plot(epochs, history["tr_loss"], label="Train")
plt.plot(epochs, history["va_loss"], label="Val")
plt.xlabel("Epoch"); plt.ylabel("Loss (log10 MSE / Huber)")
plt.title(f"Loss vs. Epoch  (loss={args.loss}, weighting={args.weighting})")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "loss_vs_epoch.png"), dpi=180); plt.close()

plt.figure(figsize=(6.2, 4.2))
plt.plot(epochs, history["tr_within"], label=f"Train (±{args.acc_pct:.1f}% SNR)")
plt.plot(epochs, history["va_within"], label=f"Val (±{args.acc_pct:.1f}% SNR)")
plt.xlabel("Epoch"); plt.ylabel("Fraction within tolerance")
plt.title("Within-tolerance vs. Epoch")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "accuracy_vs_epoch.png"), dpi=180); plt.close()

lims = (min(pred_lin_te.min(), true_lin_te.min()), max(pred_lin_te.max(), true_lin_te.max()))
plt.figure(figsize=(5.5, 5.2))
plt.scatter(true_lin_te, pred_lin_te, s=8, alpha=0.35)
plt.plot(lims, lims, lw=2)
plt.xlabel("True SNR"); plt.ylabel("Pred SNR")
plt.title("Test: Pred vs True SNR (raw)")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "scatter_test_linear.png"), dpi=180)
plt.close()

resid = pred_lin_te - true_lin_te
plt.figure(figsize=(6.4, 4.0))
plt.hist(resid, bins=80, alpha=0.85)
plt.xlabel("Pred − True (SNR)")
plt.ylabel("Count")
plt.title("Test Residuals (raw, linear SNR)")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "residuals_test_linear.png"), dpi=180)
plt.close()

# --------------------------- Smooth monotonic calibration (PCHIP) ---------------------------
from scipy.interpolate import PchipInterpolator

def fit_smooth_monotonic_calibrator_extended(pred_lin_val, true_lin_val,
                                             bins=60, min_bin=20):
    pred_lin_val = np.asarray(pred_lin_val, float)
    true_lin_val = np.asarray(true_lin_val, float)

    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(pred_lin_val, qs))
    if len(edges) < 4:
        return lambda x: np.asarray(x, float)

    xk, yk = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred_lin_val >= lo) & (pred_lin_val < hi)
        if m.sum() >= min_bin:
            xk.append(np.median(pred_lin_val[m]))
            yk.append(np.median(true_lin_val[m]))
    xk = np.asarray(xk, float); yk = np.asarray(yk, float)
    if xk.size < 3:
        return lambda x: np.asarray(x, float)

    yk = np.maximum.accumulate(yk)
    xk, keep = np.unique(xk, return_index=True)
    yk = yk[keep]

    x_min, x_max = pred_lin_val.min(), pred_lin_val.max()
    y_min, y_max = true_lin_val.min(), true_lin_val.max()
    if x_min < xk[0]:
        xk = np.r_[x_min, xk]; yk = np.r_[min(y_min, yk[0]), yk]
    if x_max > xk[-1]:
        xk = np.r_[xk, x_max]; yk = np.r_[yk, max(y_max, yk[-1])]
    yk = np.maximum.accumulate(yk)

    pchip = PchipInterpolator(xk, yk, extrapolate=True)

    def calibrate_fn(x):
        x = np.asarray(x, float)
        return pchip(x)

    calibrate_fn._knots = (xk, yk)
    return calibrate_fn

pred_lin_va = 10.0 ** pred_log_va
true_lin_va = 10.0 ** true_log_va

cal_fn = fit_smooth_monotonic_calibrator_extended(pred_lin_va, true_lin_va, bins=60, min_bin=20)
pred_te_cal_lin = cal_fn(pred_lin_te)

mape_cal = float(np.mean(np.abs(pred_te_cal_lin - true_lin_te) / (true_lin_te + 1e-12))) * 100.0
score_cal = 100.0 - mape_cal
pct = args.acc_pct / 100.0
within_cal = float(np.mean(np.abs(pred_te_cal_lin - true_lin_te) / (true_lin_te + 1e-12) <= pct))

pred_log_te_cal = np.log10(np.clip(pred_te_cal_lin, 1e-12, None))
ss_res = np.sum((pred_log_te_cal - true_log_te) ** 2)
ss_tot = np.sum((true_log_te - true_log_te.mean()) ** 2)
r2_cal = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

print(f"[SMOOTH CAL] Test | MAPE {mape_cal:.2f}% | Acc(1-MAPE) {score_cal:.2f}% | "
      f"Acc(±{args.acc_pct:.1f}%) {within_cal:.3f} | R2(log10) {r2_cal:.4f}")

# Save knots
xk, yk = cal_fn._knots
save_json(os.path.join(args.outdir, "smooth_cal_knots_extended.json"),
          {"xk": xk.tolist(), "yk": yk.tolist()})

# Save calibrated test predictions
np.savez_compressed(
    os.path.join(args.outdir, "preds_test_cal.npz"),
    pred_lin_cal=pred_te_cal_lin.astype(np.float32),
    true_lin=true_lin_te.astype(np.float32),
    pred_log_cal=pred_log_te_cal.astype(np.float32),
    true_log=true_log_te.astype(np.float32),
    sample_idx=test_idx.astype(np.int64),
)

# --------------------------- Plots (CALIBRATED) ---------------------------
lims_cal = (min(pred_te_cal_lin.min(), true_lin_te.min()), max(pred_te_cal_lin.max(), true_lin_te.max()))
plt.figure(figsize=(5.5, 5.2))
plt.scatter(true_lin_te, pred_te_cal_lin, s=8, alpha=0.35)
plt.plot(lims_cal, lims_cal, lw=2)
plt.xlabel("True SNR"); plt.ylabel("Pred SNR (calibrated)")
plt.title("Test: Pred vs True SNR (calibrated)")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "scatter_test_linear_calibrated_smooth.png"), dpi=180)
plt.close()

resid_cal = pred_te_cal_lin - true_lin_te
plt.figure(figsize=(6.4, 4.0))
plt.hist(resid_cal, bins=80, alpha=0.85)
plt.xlabel("Pred − True (SNR)")
plt.ylabel("Count")
plt.title("Test Residuals (calibrated, linear SNR)")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "residuals_test_linear_calibrated_smooth.png"), dpi=180)
plt.close()

plt.figure(figsize=(6.4, 4.0))
plt.scatter(true_lin_te, resid_cal, s=8, alpha=0.35)
plt.axhline(0, color="k", lw=1)
plt.xlabel("True SNR")
plt.ylabel("Residual (calibrated)")
plt.title("Calibration residuals vs True SNR (calibrated)")
plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "calibration_residual_vs_trueSNR_calibrated_smooth.png"), dpi=180)
plt.close()

# --------------------------- Save metrics + run_config ---------------------------
metrics = {
    "raw": {
        "val_loss": safe_float(va_loss),
        "val_mape": safe_float(va_mape),
        "val_score_100_minus_mape": safe_float(va_score),
        "val_within_tol": safe_float(va_within),
        "val_r2_log": safe_float(va_r2),
        "test_loss": safe_float(te_loss),
        "test_mape": safe_float(te_mape),
        "test_score_100_minus_mape": safe_float(te_score),
        "test_within_tol": safe_float(te_within),
        "test_r2_log": safe_float(te_r2),
    },
    "calibrated": {
        "test_mape": safe_float(mape_cal),
        "test_score_100_minus_mape": safe_float(score_cal),
        "test_within_tol": safe_float(within_cal),
        "test_r2_log": safe_float(r2_cal),
        "calibrator": {
            "type": "PCHIP_monotone_extended_knots",
            "fit_split": "validation",
            "apply_split": "test",
            "bins": 60,
            "min_bin": 20,
            "knots_file": "smooth_cal_knots_extended.json",
        }
    },
    "counts": {
        "N_total_after_filter": int(N),
        "N_train": int(train_size),
        "N_val": int(val_size),
        "N_test": int(test_size),
    }
}
save_json(os.path.join(args.outdir, "metrics.json"), metrics)

env_info = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "device": str(device),
    "numpy": np.__version__,
    "pandas": pd.__version__,
}

run_config = {
    "timestamp": RUN_TS,
    "command": " ".join(sys.argv),
    "args": vars(args),
    "features": FEATURES,
    "target": TARGET,
    "env": env_info,
    "data_stats": data_stats,
    "model": {
        "name": "MLPRegressor",
        "param_count": int(param_count),
        "width": int(args.width),
        "depth": int(args.depth),
        "dropout": safe_float(args.dropout),
        "act": str(args.act),
    },
    "optimizer": {
        "type": "AdamW",
        "lr": safe_float(args.lr),
        "weight_decay": 1e-4,
    },
    "scheduler": {
        "type": "ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
    },
}
save_json(os.path.join(args.outdir, "run_config.json"), run_config)

print("Saved artifacts in:", os.path.abspath(args.outdir))

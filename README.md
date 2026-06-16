

## Dataset Generation

The datasets were generated using the script:

```bash
python3 SNR-dataset-BBH-injections-v6-FD.py \
  --ncores 8 \
  --output outv6-m12uniform-100k.pkl \
  --nsamples 100000 \
  --m12uniform
```


## Deployment (Inference)

Run a trained checkpoint over a dataset pickle with `deploy_phasmanet.py`. The
model artifacts live in `--model-dir` (default `PhasmaNet-1M`), which must
contain `best_model.pt` and `feature_scaler.json`. Inference runs on the GPU by
default; use the `pytorch` conda environment so CUDA is available.

The script writes predictions to `<model-dir>/preds_deploy.npz` and
`<model-dir>/preds_deploy.csv`. If the input pickle has a `logSNR` column, the
true values are stored alongside the predictions for evaluation.

### With PCHIP calibration (default)

If `<model-dir>/smooth_cal_knots_extended.json` is present (produced during
training), a smooth monotonic PCHIP calibration — fitted on the validation split
— is applied automatically. This adds `pred_lin_cal` / `pred_log_cal` to the NPZ
and `pred_SNR_cal` / `pred_logSNR_cal` to the CSV.

```bash
conda run -n pytorch python deploy_phasmanet.py \
  --data outv6-m12uniform-1M.pkl \
  --model-dir PhasmaNet-1M \
  --device cuda
```

### Without calibration

Pass `--no-calibration` to emit raw model predictions only:

```bash
conda run -n pytorch python deploy_phasmanet.py \
  --data outv6-m12uniform-1M.pkl \
  --model-dir PhasmaNet-1M \
  --device cuda \
  --no-calibration
```

Useful options: `--cal-knots PATH` to point at a different knots file,
`--out`/`--out-csv` to override the output paths, and `--device cpu` to force CPU.


## Plotting Predicted vs Original SNR

Use `plot_snr_pred_vs_orig.py` to visualize predictions from a deployment NPZ.
The script drops non-finite/non-positive points and uses percentile-based axis
limits so a few corrupt labels cannot blow up the range.

```bash
# Raw predictions, log-log hexbin (default)
conda run -n pytorch python plot_snr_pred_vs_orig.py \
  --preds-npz PhasmaNet-1M/preds_deploy.npz \
  --out PhasmaNet-1M/snr_pred_vs_orig.png

# Calibrated predictions, linear-axis scatter
conda run -n pytorch python plot_snr_pred_vs_orig.py \
  --preds-npz PhasmaNet-1M/preds_deploy.npz \
  --out PhasmaNet-1M/snr_pred_vs_orig_cal.png \
  --mode scatter --no-log-scale --calibrated
```

Key flags: `--calibrated` plots `pred_lin_cal` (requires a calibrated NPZ),
`--mode {hexbin,scatter}` selects the render style, `--no-log-scale` switches to
linear axes, and `--max-points` caps the sampled points in scatter mode.
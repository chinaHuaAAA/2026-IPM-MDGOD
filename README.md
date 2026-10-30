# MDGOD

**Defining Consistency: Multi-Scale Noisy Granular-Ball Space Fusion on Spectral-Enhanced Dual-Feature Graph for Outlier Detection**

## Running the Code

Run `main_MDGOD_parallel_10seeds.py` from the `MDGOD` directory:

```bash
python main_MDGOD_parallel_10seeds.py
```

The script runs MDGOD on the `.mat` datasets in the `Datasets` directory. Each dataset should contain a variable named `trandata`, with features in the first columns and the ground-truth label in the last column (`0` for normal samples and `1` for anomalies).

The input directory, output directory, and number of parallel processes can be modified using `DATA_DIR`, `RESULT_DIR`, and `WORKERS` at the top of the script.

## Parameter Settings

The following parameter settings are used:

| Parameter | Search range / Fixed value | Step |
| --------- | ------------------------- | ---- |
| `sigma`   | 0.0–1.0                   | 0.05 |
| `alpha`   | 0.0–1.0                   | 0.05 |
| `layers`  | 5                         | —    |
| `eta`     | 0.05                      | —    |

Parameters are searched in two stages. First, `alpha` is fixed at `1.0`, and `sigma` is selected by the highest mean AUC. Then, the selected `sigma` is fixed while searching `alpha`.

Each parameter setting is evaluated **10 times**, using random seeds **0–9**. The script reports the mean AUC, AP, and runtime for each parameter setting. After each dataset finishes, it additionally reports the best mean AUC and best mean AP, together with their respective parameters.

Results are saved to `Result/MDGOD`, retaining the metrics, runtimes, and anomaly scores from every run.

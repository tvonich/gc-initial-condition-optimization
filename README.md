# GraphCast Initial Condition Optimization

> **Built on [GraphCast](https://github.com/google-deepmind/graphcast) by Google DeepMind.**
> This repository is a research extension of GraphCast and would not exist without it.
> If you use this code, you must also cite the original GraphCast paper (see [Citations](#citations)).

Gradient-based optimization of atmospheric initial conditions (ICs) using DeepMind's [GraphCast](https://github.com/google-deepmind/graphcast) weather prediction model. This code iteratively perturbs the input state at a given time step to minimize multi-step forecast error, yielding "optimal" ICs that produce lower-error predictions within the model. It is not affiliated with or endorsed by Google DeepMind.

This is research code developed for NCAR HPC systems (Derecho / Casper), but it is designed to run on any system with a CUDA-capable GPU and access to ERA5 data in the GraphCast format.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Quick Start](#quick-start)
4. [Configuration Reference](#configuration-reference)
5. [Output Files](#output-files)
6. [ERA5 Data Format](#era5-data-format)
7. [Design Notes](#design-notes)
8. [Citations](#citations)
9. [For Agents](#for-agents)

---

## Requirements

| Item | Requirement |
|------|-------------|
| GPU | NVIDIA A100 recommended; 40 GB for F32 (≤11 day), 80 GB for F64 (>11 day) |
| RAM | ~50 GB (scales with dataset size) |
| Python | 3.10 |
| CUDA | 12.2 with cuDNN 8.9 |
| Queue time | ~2 min walltime for a 1-day IC optimization |

---

## Setup

### 1. Clone this repository and the GraphCast source

This code requires the DeepMind GraphCast library (the `graphcast/` directory). Clone both:

```bash
# Clone this IC optimization repo
git clone https://github.com/tvonich/gc-initial-condition-optimization
cd gc-initial-condition-optimization

# Clone the DeepMind GraphCast source into the graphcast/ directory
git clone https://github.com/google-deepmind/graphcast.git graphcast
```

The `graphcast/` directory contains the model source (GNN architecture, checkpoint loading, data utilities), cloned from the [official DeepMind GraphCast repository](https://github.com/google-deepmind/graphcast). It is not bundled in this repo — clone it separately as shown above. Cite the DeepMind GraphCast paper if you use it.

> **Important:** Do not use the model loading or inference scripts from the DeepMind GraphCast repo directly. This repo provides its own `batch_modules/load_model.py` and `batch_modules/jitted.py`, which wrap the GraphCast model with the custom normalization, BFloat16 casting, and JIT-compiled gradient functions needed for IC optimization. The DeepMind repo's demo notebooks and `run_graphcast.py` are not compatible with this pipeline.

### 2. Download model weights and normalization statistics

Model weights and normalization statistics are distributed separately by DeepMind via Google Cloud Storage. Download them into the expected directories:

```bash
# GraphCast Small (1.0-degree, 13-level) — required for this pipeline
gsutil cp gs://dm_graphcast/params/GraphCast_small\ -\ ERA5\ 1979-2017\ -\ resolution\ 1.0\ -\ pressure\ levels\ 13\ -\ mesh\ 2to5\ -\ precipitation\ input\ and\ output.npz \
    params/params_GraphCast_small.npz

# Normalization statistics (required)
gsutil cp gs://dm_graphcast/stats/diffs_stddev_by_level.nc stats/
gsutil cp gs://dm_graphcast/stats/mean_by_level.nc stats/
gsutil cp gs://dm_graphcast/stats/stddev_by_level.nc stats/
```

> Note: The DeepMind GCS bucket (`gs://dm_graphcast`) requires a Google Cloud account. See the [GraphCast README](https://github.com/google-deepmind/graphcast#model-weights-and-normalization-statistics) for access instructions.

### 3. Create the conda environment

```bash
conda env create -f environment.yml
conda activate graphcast_ic
```

This installs JAX 0.4.28 with CUDA 12.2 support, dm-haiku, optax, xarray, and all required dependencies.

> **NCAR users:** The `jax_cuda2_derecho` environment on Derecho and `jax_cuda2_update` on Casper are equivalent pre-installed environments. Set `CONDA_ENV` in the submit script accordingly and skip `conda env create`.

### 4. Download a test dataset

DeepMind provides a small ERA5 sample (~293 MB, 12 time steps from 2022-01-01). No account required:

```bash
mkdir -p data
wget -O data/era5_sample.nc \
    https://storage.googleapis.com/dm_graphcast/dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-12.nc
```

For your own data, see [ERA5 Data Format](#era5-data-format) below.

---

## Quick Start

After completing setup steps 1–4:

```bash
# 1. Edit the submit script:
#    - Set your PBS project code (replace YOUR_PROJECT_CODE_HERE)
#    - Set DATA_PATH to your ERA5 file (e.g., data/era5_sample.nc)
#    - Set OUTPUT_PATH and CONDA_ENV for your system
#    - If using the sample dataset, set INIT_DATE to 2022-01-01T00:00:00
nano submit_jobs/run_ic_opt_1day.sh

# 2. Submit from the repo root
mkdir -p a_logs
qsub submit_jobs/run_ic_opt_1day.sh

# 3. Monitor
qstat -u $USER
tail -f a_logs/*.OU
```

The submit script handles JSON config creation, env var exports, and cleanup automatically. A 1-day optimization takes ~5 minutes on an A100.

---

## Configuration Reference

The submit script creates a JSON config file at runtime. To modify parameters beyond the 4 user variables, edit the `get_config(...)` call inside the submit script.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `init_date` | string | set by user | Forecast start time (ISO 8601: `YYYY-MM-DDTHH:MM:SS`). Must exist in the dataset with at least 1 time step before it. |
| `run_type` | string | `"optimize"` | `optimize`: IC optimization loop. `pred`: forward rollout only. `loss`: compute loss only. `grad`: compute gradients only. |
| `pred_steps` | int | `4` | Number of 6-hour steps in the optimization window. 4 = 1 day, 8 = 2 days, 20 = 5 days. |
| `justify` | string | `"left"` | Time alignment for input/target extraction. Use `"left"` (default). |
| `selected_vars` | string or list | `"all"` | Variables to include in loss. `"all"` or e.g. `["2m_temperature", "geopotential"]`. |
| `selected_lvls` | string or list | `"all"` | Pressure levels for loss. `"all"` or e.g. `[500, 850, 1000]`. |
| `selected_region` | string or list | `"all"` | Spatial extent. `"all"` or `[lat_min, lat_max, lon_min, lon_max]`. |
| `selected_times` | string or list | `"all"` | Time steps to include in loss. `"all"` or `[start_idx, size]`. |

**Variable names (GraphCast 1.0-degree, 13-level):**
```
Pressure-level: geopotential, temperature, u_component_of_wind, v_component_of_wind,
                specific_humidity, vertical_velocity
Surface:        2m_temperature, mean_sea_level_pressure,
                10m_u_component_of_wind, 10m_v_component_of_wind,
                total_precipitation_6hr
```

**Pressure levels:** `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]` hPa

---

## Output Files

All outputs are written under `OUTPUT_PATH` (set in the submit script):

```
OUTPUT_PATH/
├── perfect_model_params/{YYYY-MM-DDTHH}/
│   └── {date}_{epoch}_{pred_steps}.npz    # Optimized IC parameters, saved at checkpoint epochs
└── perfect_model_losses/{YYYY-MM-DDTHH}/
    └── {date}_*.pkl                       # Loss history (pickle; list of floats per epoch)
```

**Loading optimized ICs:**
```python
import numpy as np
ckpt = np.load("path/to/{date}_{epoch}_{pred_steps}.npz")
# Arrays keyed by flattened parameter names (e.g., "GraphCast/~/mesh_gnn/...")
```

---

## ERA5 Data Format

A test dataset is included in [Setup step 4](#4-download-a-test-dataset) above. For your own data:

### Minimum requirements for this pipeline:

| Property | Requirement |
|----------|-------------|
| Resolution | 1.0-degree lat/lon (181 x 360 grid) |
| Pressure levels | 13 levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa |
| Time step | 6-hourly |
| Time coverage | At least 1 time step before `init_date` must be present in the file |
| Variables | All 11 prognostic variables listed above |
| Static variables | `geopotential_at_surface`, `land_sea_mask` (can have a time dimension; will be reduced automatically) |
| Forcings | `toa_incident_solar_radiation` at each forecast step; year/day progress sines and cosines (computed internally by GraphCast) |
| Format | NetCDF4 (`.nc`), readable by `xarray.load_dataset()` |
| Coordinate names | Must match GraphCast convention: `lat`, `lon`, `level`, `time`, `datetime` |

---

## Design Notes

### Precision

| Precision | When to use | Set via |
|-----------|-------------|---------|
| BF16 | Default (forward pass cast in `jitted.py`) | Default — no change needed |
| F32 | 7–11 day optimizations | `status64 = False` in `make_optimal_ic.py` |
| F64 | >11 day optimizations (~80 GB VRAM) | `status64 = True` (default); `jax_enable_x64=True` |

Two knobs control precision:

1. **`status64` flag** (`make_optimal_ic.py`, near top of file) — controls whether the IC arrays and gradient computations use F32 or F64. Set `status64 = True` for F64, `False` for F32. This also toggles `jax.config.update("jax_enable_x64", status64)` on the next line.

2. **`casting.Bfloat16Cast`** (`graphcast/casting.py`, called inside `jitted.py`) — controls the precision of the GraphCast forward pass itself. By default the model weights and activations are cast to BF16 before each forward pass, which saves memory and matches DeepMind's training setup. To disable BF16 casting and run the forward pass in full F32, set `enabled=False` in the `casting.Bfloat16Cast(predictor)` call inside `construct_wrapped_graphcast()` in `jitted.py`.

For most use cases (≤11 day optimization windows), the default BF16 forward pass with F64 IC arrays (`status64 = True`) is the correct configuration and requires no changes.

### Noise and Perturbation Size

The optimization minimizes the forecast MSE with no explicit regularization on the magnitude of IC perturbations. This means:

- **Noise is not optimized away.** GraphCast's GNN architecture acts as a low-pass filter during the forward pass, so high-frequency noise in the ICs does not propagate to the loss — the optimizer has no gradient signal to remove it. The resulting optimal ICs may contain residual noise that is invisible to the model but physically unrealistic.
- **Perturbation size is unconstrained by default.** If unrealistically large IC perturbations are a concern (e.g., for studying physically plausible corrections), a proximity penalty can be added to the loss:
  ```python
  # Example: L2 proximity penalty to the original IC (add inside the loss function)
  penalty_weight = 0.01
  loss = forecast_loss + penalty_weight * jnp.mean((inputs - original_inputs)**2)
  ```
  This is not implemented by default because the research focus is on the theoretical lower bound of forecast error, not on physically constrained corrections. Add it in `jitted.py` if needed for your application.

### Optimization Window Strategy

**The problem with a single fixed window.** When optimizing over a long forecast (e.g., 10 days), the MSE loss is an unweighted sum across all lead times. Because forecast errors grow roughly exponentially with lead time, the loss is dominated by the last few time steps. The optimizer therefore focuses almost entirely on getting the end of the forecast right. But physically, the end of the forecast can only be correct if the early part of the trajectory is also correct — the phase-space region that leads to a good 10-day forecast is narrow, and the optimizer cannot find it if it is ignoring the early loss signal.

**The expanding window method.** Rather than optimizing over the full window from the start, we grow the optimization target incrementally:

1. Optimize over a short window (e.g., 1 day / 4 steps) until convergence.
2. Extend the window by `r_step` steps (default: 10) and continue optimizing from the ICs found in step 1.
3. Repeat until the target lead time is reached.

Each short-window stage guides the ICs into the correct part of phase space for that lead time before the next stage asks them to satisfy a longer constraint. The ICs found at each stage become the warm start for the next, so optimization progress is cumulative rather than redundant.

This is implemented in the `optimize` branch of `make_optimal_ic.py` via the `while pred_steps <= step_limit` loop, with `r_step` controlling the window increment. The default submit script uses `pred_steps=4` (1 day) and `step_limit=pred_steps`, so it runs a single window — to use the expanding method, set `step_limit` to your target lead time and `r_step` to the desired increment (e.g., `step_limit=40`, `r_step=10` for 10-step (2.5-day) optimization in 4 stages):

```python
config = get_config(
    init_date='2022-01-01T00:00:00',
    run_type='optimize',
    pred_steps=4,      # starting window
    ...
)
# In make_optimal_ic.py, set:
# r_step = 10
# step_limit = 40    # expands: 4 → 14 → 24 → 34 → 40
```

**Alternatives.** Other approaches to the same problem include time-decaying loss weights (upweighting early lead times) or curriculum schedules that increase the window continuously rather than in steps. The expanding window method was developed for this pipeline because it is simple, requires no hyperparameter tuning of weight schedules, and naturally inherits the warm-start benefit of prior stages.

### Learning Rate

The Adam optimizer LR is set automatically by `pred_steps` in `make_optimal_ic.py`:

| pred_steps | LR |
|---|---|
| 0–59 | 1e-3 |
| ≥ 60 | 1e-4 |

For longer optimizations or if the loss diverges, reduce `base_lr` manually near the top of the `optimize` branch.

---

## Citations

If you use this code in your work, please cite the paper and the GraphCast model:

```bibtex
% Cite the paper this code was developed for
@article{vonich2024predictability,
  author  = {Vonich, P. Trent and Hakim, Gregory J.},
  title   = {Predictability Limit of the 2021 {Pacific Northwest} Heatwave
             From Deep-Learning Sensitivity Analysis},
  journal = {Geophysical Research Letters},
  volume  = {51},
  number  = {19},
  pages   = {e2024GL110651},
  year    = {2024},
  doi     = {10.1029/2024GL110651},
  url     = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2024GL110651}
}

% Required: cite the GraphCast model
@article{lam2023graphcast,
  title     = {{GraphCast}: Learning skillful medium-range global weather forecasting},
  author    = {Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and
               Wirnsberger, Peter and Fortunato, Meire and Alet, Ferran and
               Ravuri, Suman and Ewalds, Timo and Eaton-Rosen, Zach and
               Hu, Weihua and Bezenac, Emmanuel de and Sanchez, Clementine and
               Requeima, James and Schmitt, Corentin and Vinyals, Oriol and
               Mohamed, Shakir and Kastner, Raia and Veness, Joel and
               Beucler, Tom and others},
  journal   = {Science},
  volume    = {382},
  number    = {6677},
  pages     = {1416--1421},
  year      = {2023},
  doi       = {10.1126/science.adi2336},
  publisher = {American Association for the Advancement of Science}
}
```

---

## For Agents

This section provides structured information for automated tools or agents interacting with this codebase.

### Pipeline Execution Order

```
submit script (run_ic_opt_1day.sh)
  → exports REPO_ROOT, GRAPHCAST_DATA_PATH, GRAPHCAST_STATS_PATH,
            GRAPHCAST_PARAMS_PATH, GRAPHCAST_OUTPUT_PATH, j_path
  → writes jsons/{PBS_JOBID}.json
  → python batch_modules/make_optimal_ic.py
      → imports batch_modules/config.py       (reads GRAPHCAST_DATA_PATH)
      → imports batch_modules/load_model.py   (reads GRAPHCAST_STATS_PATH, GRAPHCAST_PARAMS_PATH)
      → imports batch_modules/prep_prediction.py  (reads j_path JSON)
      → imports batch_modules/jitted.py       (JIT-compiles loss/gradient functions)
      → imports graphcast/                    (GNN model, checkpoint, data utilities)
      → runs Adam optimization loop
      → saves .npz checkpoints to GRAPHCAST_OUTPUT_PATH/perfect_model_params/{date}/
      → saves loss history .pkl to GRAPHCAST_OUTPUT_PATH/perfect_model_losses/{date}/
  → deletes jsons/{PBS_JOBID}.json
```

### Environment Variables

| Variable | Set by | Consumed by | Description |
|----------|--------|-------------|-------------|
| `REPO_ROOT` | submit script | `make_optimal_ic.py` (sys.path) | Absolute path to repo root |
| `GRAPHCAST_DATA_PATH` | submit script | `config.py` | ERA5 NetCDF input file path |
| `GRAPHCAST_STATS_PATH` | submit script | `load_model.py` | Directory containing `*_by_level.nc` stats files |
| `GRAPHCAST_PARAMS_PATH` | submit script | `load_model.py` | Path to `.npz` model weights |
| `GRAPHCAST_OUTPUT_PATH` | submit script | `make_optimal_ic.py` | Base directory for all outputs |
| `j_path` | submit script | `prep_prediction.py`, `jitted.py` | Path to the per-job JSON config file |

### Key Files

| File | Role |
|------|------|
| `batch_modules/make_optimal_ic.py` | Main entry point; IC optimization loop |
| `batch_modules/config.py` | `get_config()` — builds JSON config dict |
| `batch_modules/load_model.py` | Loads params + normalization statistics at import time |
| `batch_modules/prep_prediction.py` | Reads JSON config; provides normalize/zero_grads utilities |
| `batch_modules/jitted.py` | JIT-compiled loss and gradient functions |
| `graphcast/graphcast.py` | Core GNN architecture |
| `graphcast/data_utils.py` | `extract_inputs_targets_forcings()` — dataset slicing |
| `submit_jobs/run_ic_opt_1day.sh` | Portable PBS submit script |
| `params/params_GraphCast_small.npz` | Model weights (download separately — see Setup) |
| `stats/*.nc` | Normalization statistics (download separately — see Setup) |

### Success Criteria

A successful run produces:
1. At least one `.npz` file in `GRAPHCAST_OUTPUT_PATH/perfect_model_params/{date}/`
2. Loss values printed to stdout (format: `epoch N loss X.XXX`)
3. No remaining `.json` file in `jsons/` (cleanup ran)
4. PBS job exits with status 0

### Common Failure Modes

| Symptom | Likely Cause |
|---------|-------------|
| `FileNotFoundError: stats/diffs_stddev_by_level.nc` | `GRAPHCAST_STATS_PATH` not set or stats not downloaded |
| `FileNotFoundError: params/params_GraphCast_small.npz` | Model weights not downloaded |
| `KeyError: init_date not found in dataset` | `INIT_DATE` not present in ERA5 file |
| `OOM / CUDA out of memory` | Dataset too large or wrong GPU type; requires A100 80GB for F64 |
| `Multiple values for argument` | Calling JIT-wrapped functions with positional args — use keyword args |

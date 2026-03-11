#!/bin/bash -l
# =============================================================================
# GraphCast 1-Day Initial Condition Optimization — PBS Submit Script
#
# Runs a 24-hour IC optimization (4 x 6-hour steps) using gradient descent
# via the GraphCast Small (1.0-degree, 13-level) model.
#
# USAGE:
#   1. Edit the USER CONFIGURATION section below.
#   2. cd to the repo root (the directory that contains batch_modules/).
#   3. qsub submit_jobs/run_ic_opt_1day.sh
#
# PBS logs are written to a_logs/ relative to the submission directory.
# You MUST qsub from the repo root for this path to resolve correctly.
# Create a_logs/ before submitting:  mkdir -p a_logs
#
# NCAR QUEUE NOTES:
#   develop — short runs (<30 min), fast start, max 2 jobs at once
#   main    — standard allocation queue
#   preempt — low-priority, can be interrupted
# =============================================================================

#PBS -N graphcast_ic_opt
#PBS -A YOUR_PROJECT_CODE_HERE
#PBS -l select=1:ncpus=4:ngpus=1:mem=120GB:gpu_type=a100
#PBS -l walltime=00:30:00
#PBS -q develop
#PBS -o a_logs
#PBS -j oe

# =============================================================================
# USER CONFIGURATION — edit these variables, then qsub
# =============================================================================
INIT_DATE="2020-01-01T00:00:00"               # Forecast start (ISO 8601).
                                               # Must exist in your dataset with
                                               # at least 1 time step before it.

DATA_PATH="/path/to/your/era5_dataset.nc"      # ERA5 NetCDF file in GraphCast format.
                                               # 1-degree resolution, 13 pressure levels.
                                               # See README for required variable list.

OUTPUT_PATH="/path/to/your/output_directory"   # Where optimized ICs and loss logs are saved.
                                               # Subdirectories created automatically:
                                               #   perfect_model_params/{date}/
                                               #   perfect_model_losses/{date}/
                                               # NCAR example: /glade/derecho/scratch/${USER}/graphcast_ic_outputs

CONDA_ENV="graphcast_ic"                       # Conda environment name.
                                               # Default: graphcast_ic (from environment.yml)
                                               # Casper alternative: jax_cuda2_update
# =============================================================================

# Repo root = the directory from which you ran qsub (must be the repo root).
# PBS_O_WORKDIR is set by PBS to the submission directory. BASH_SOURCE does not
# work because PBS copies the script to a spool directory before execution.
REPO_ROOT="${PBS_O_WORKDIR}"

# Unique job number from PBS — avoids race condition when jobs start simultaneously.
number=${PBS_JOBID%%.*}
export j_path="${REPO_ROOT}/jsons/${number}.json"

# Export env vars consumed by batch_modules at import time.
export REPO_ROOT
export GRAPHCAST_DATA_PATH="${DATA_PATH}"
export GRAPHCAST_STATS_PATH="${REPO_ROOT}/stats"
export GRAPHCAST_PARAMS_PATH="${REPO_ROOT}/params/params_GraphCast_small.npz"
export GRAPHCAST_OUTPUT_PATH="${OUTPUT_PATH}"

# Ensure required directories exist.
mkdir -p "${REPO_ROOT}/a_logs" "${REPO_ROOT}/jsons" "${OUTPUT_PATH}"

# Activate conda environment.
# On NCAR systems, 'module load conda' sets up the conda shell function.
# On other systems, replace with: source ~/miniconda3/etc/profile.d/conda.sh
module load conda 2>/dev/null || true
conda activate "${CONDA_ENV}"

# Write job config JSON.
# pred_steps=4 → 4 x 6h = 24h (1-day) optimization.
# To extend lead time: set pred_steps=8 (2-day), pred_steps=12 (3-day), etc.
python -c "
import json, sys, os
sys.path.insert(0, '${REPO_ROOT}/batch_modules')
from config import get_config
config = get_config(
    init_date='${INIT_DATE}',
    run_type='optimize',
    pred_steps=4,
    justify='left',
    selected_region='all',
    selected_vars='all',
    selected_lvls='all',
    selected_times='all'
)
json.dump(config, open(os.environ['j_path'], 'w'))
"

# Run IC optimization from repo root.
cd "${REPO_ROOT}"
start_time=$(date +%s)

python -u batch_modules/make_optimal_ic.py

end_time=$(date +%s)
time_diff=$((end_time - start_time))
hours=$((time_diff / 3600))
minutes=$(( (time_diff % 3600) / 60 ))
seconds=$((time_diff % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"

rm -f "${j_path}"

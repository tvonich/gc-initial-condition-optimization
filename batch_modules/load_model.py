import os
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
from graphcast import xarray_jax
from graphcast import xarray_tree

from xarray import load_dataset as xld

fp = ''  # Optional: path to custom-trained params checkpoint (used in loss branch only)
def load_params(params_file_path=None):
    if params_file_path is None:
        params_file_path = os.environ.get('GRAPHCAST_PARAMS_PATH', 'params/params_GraphCast_small.npz')

    with open(params_file_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
        return ckpt.params, ckpt.model_config, ckpt.task_config, ckpt.description

#NORMALIZATION
# Path to the local directory for normalization data
local_stats_path = os.environ.get('GRAPHCAST_STATS_PATH', 'stats/')

# Load diffs_stddev_by_level from the local file
with open(f"{local_stats_path}/diffs_stddev_by_level.nc", "rb") as f:
    diffs_stddev_by_level = xld(f).compute()

# Load mean_by_level from the local file
with open(f"{local_stats_path}/mean_by_level.nc", "rb") as f:
    mean_by_level = xld(f).compute()

# Load stddev_by_level from the local file
with open(f"{local_stats_path}/stddev_by_level.nc", "rb") as f:
    stddev_by_level = xld(f).compute()


# LOAD MODEL
params, model_config, task_config, model_description = load_params(params_file_path=None)

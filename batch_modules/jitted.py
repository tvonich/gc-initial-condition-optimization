#jitted.py
import functools
from load_model import params, model_config, task_config
from load_model import diffs_stddev_by_level, mean_by_level, stddev_by_level

from graphcast import graphcast
from graphcast import casting
from graphcast import normalization
from graphcast import autoregressive
from graphcast import xarray_tree
from graphcast import xarray_jax

import jax
import xarray as xr
import jax.numpy as jnp
from graphcast import xarray_jax
from optax import apply_updates
from functools import partial
from haiku import transform_with_state as tws
from os import getenv
from json import load as jload
import numpy as np
import haiku as hk

# Access the j_path environment variable and get configs
j_path = getenv('j_path')
with open(j_path, 'r') as f:
    config = jload(f)

selected_times  = config['selected_times']
selected_vars   = config.get('selected_vars',   'all')
selected_lvls   = config.get('selected_lvls',   'all')
selected_region = config.get('selected_region', 'all')

# Per-variable loss weights for the optional selector-aware loss.
# Mirrors the dict in DeepMind's GraphCast (graphcast.py:418-431). Variables
# not listed here default to weight 1.0.
_PER_VARIABLE_WEIGHTS = {
    "2m_temperature":           1.0,
    "10m_u_component_of_wind":  0.1,
    "10m_v_component_of_wind":  0.1,
    "mean_sea_level_pressure":  0.1,
    "total_precipitation_6hr":  0.1,
}

state={} #Needed for final functions on this page

def denormalize_64(dt, stats):
    """
    Denormalize an xarray.Dataset using provided statistics.
    Uses float64 precision for arithmetic operations to maintain numerical stability.
    Parameters:
    - dt (xr.Dataset): The dataset to denormalize.
    - stats (dict): A dictionary containing 'mean' and 'std' for each variable.
    Returns:
    - xr.Dataset: The denormalized dataset in float64.
    """
    dt_copy = dt.copy()

    dynamic_vars = [
        "geopotential", "temperature", "u_component_of_wind", "v_component_of_wind",
        "specific_humidity", "vertical_velocity", "2m_temperature", "mean_sea_level_pressure",
        "10m_v_component_of_wind", "10m_u_component_of_wind"
    ]
    time_vars =['toa_incident_solar_radiation','day_progress_sin','day_progress_cos','year_progress_cos','year_progress_sin'
    ]

    # Include 'total_precipitation_6hr' if present
    if "total_precipitation_6hr" in dt_copy.data_vars:
        dynamic_vars.append("total_precipitation_6hr")

    for var_name in time_vars:
       dt_copy[var_name] = dt_copy[var_name].astype('float64')

    for var_name in dynamic_vars:
      da = dt_copy[var_name]
      mean = stats[var_name]['mean'].astype('float64')
      std = stats[var_name]['std'].astype('float64')

      da_64 = da.astype('float64')
      denormalized = da_64 * std + mean

      dt_copy[var_name] = denormalized

    return dt_copy

def denormalize_32(dt, stats):
    """
    Denormalize an xarray.Dataset using provided statistics.
    Parameters:
    - dt (xr.Dataset): The dataset to denormalize.
    - stats (dict): A dictionary containing 'mean' and 'std' for each variable.
    Returns:
    - xr.Dataset: The denormalized dataset in float32.
    """
    dt_copy = dt.copy()

    dynamic_vars = [
        "geopotential", "temperature", "u_component_of_wind", "v_component_of_wind",
        "specific_humidity", "vertical_velocity", "2m_temperature", "mean_sea_level_pressure",
        "10m_v_component_of_wind", "10m_u_component_of_wind"]

    # Include 'total_precipitation_6hr' if present
    if "total_precipitation_6hr" in dt_copy.data_vars:
        dynamic_vars.append("total_precipitation_6hr")

    for var_name in dynamic_vars:
      da = dt_copy[var_name]
      mean = stats[var_name]['mean'].astype('float32')
      std = stats[var_name]['std'].astype('float32')

      da_32 = da.astype('float32')
      denormalized = da_32 * std + mean

      dt_copy[var_name] = denormalized.astype('float32')

    return dt_copy

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig):
  """Constructs and wraps the GraphCast Predictor."""
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16.
  predictor = casting.Bfloat16Cast(predictor)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
  return predictor

@tws
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)

# =====================================================================
# Vendored selector-aware loss
# =====================================================================
# These helpers and `custom_weighted_mse` are vendored from DeepMind's
# graphcast/losses.py (Apache 2.0). We vendor instead of patching the
# cloned graphcast/ directory so the public release stays compatible with
# a clean DeepMind clone. When all selectors are 'all', the math here is
# numerically identical to graphcast.losses.weighted_mse_per_level.

def _check_uniform_spacing_and_get_delta(vector):
    diff = np.diff(vector)
    if not np.all(np.isclose(diff[0], diff)):
        raise ValueError(f'Vector {diff} is not uniformly spaced.')
    return diff[0]


def _weight_for_latitude_vector_without_poles(latitude):
    """cos(lat) weights for grids that don't include the poles."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if np.sum(latitude % delta_latitude) != 0:
        raise ValueError('Invalid latitude value (decimal not on grid).')
    return np.cos(np.deg2rad(latitude))


def _weight_for_latitude_vector_with_poles(latitude):
    """Area weights for grids that include the poles (uses sin-difference)."""
    delta_latitude = np.abs(_check_uniform_spacing_and_get_delta(latitude))
    if (not np.isclose(np.max(latitude), 90.) or
            not np.isclose(np.min(latitude), -90.)):
        raise ValueError(
            f'Latitude vector {latitude} does not start/end at +- 90 degrees.')
    weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta_latitude/2))
    weights[[0, -1]] = np.sin(np.deg2rad(delta_latitude/4)) ** 2
    return weights


def normalized_latitude_weights(data):
    """Unit-mean cos(lat) area weights, dispatching pole / no-pole."""
    latitude = data.coords['lat']
    if np.any(np.isclose(np.abs(latitude), 90.)):
        weights = _weight_for_latitude_vector_with_poles(latitude)
    else:
        weights = _weight_for_latitude_vector_without_poles(latitude)
    return weights / weights.mean(skipna=False)


def normalized_level_weights(data):
    """Pressure-proportional level weights, normalized to unit mean."""
    level = data.coords['level']
    return level / level.mean(skipna=False)


def _mean_preserving_batch(x):
    return x.mean([d for d in x.dims if d != 'batch'], skipna=False)


def sum_per_variable_losses(per_variable_losses, weights):
    """Weighted sum of per-variable losses → (total, weighted_per_var)."""
    if not set(weights.keys()).issubset(set(per_variable_losses.keys())):
        raise ValueError(
            'Passing a weight that does not correspond to any variable '
            f'{set(weights.keys()) - set(per_variable_losses.keys())}')
    weighted_per_variable_losses = {
        name: loss * weights.get(name, 1)
        for name, loss in per_variable_losses.items()
    }
    total = xr.concat(
        weighted_per_variable_losses.values(), dim='variable', join='exact'
    ).sum('variable', skipna=False)
    return total, weighted_per_variable_losses


def custom_weighted_mse(predictions, targets):
    """Selector-aware lat/level/var-weighted MSE.

    Reads `selected_vars`, `selected_lvls`, `selected_region`, `selected_times`
    from module-level config (set at import time from the j_path JSON).
    Each defaults to 'all' (no filtering).

    `selected_region` semantics: [lat_min, lat_max, lon_min, lon_max] with
    longitude in [0, 359] and lon_max >= lon_min (no 0/360 wrap-around).
    `selected_times`  semantics: [start_idx, size] (positional, 0-indexed).
    """
    # Variable selection.
    if selected_vars == 'all':
        per_variable_weights = _PER_VARIABLE_WEIGHTS
    else:
        vars_list = [selected_vars] if isinstance(selected_vars, str) else list(selected_vars)
        predictions = predictions[vars_list]
        targets     = targets[vars_list]
        per_variable_weights = {v: _PER_VARIABLE_WEIGHTS.get(v, 1.0) for v in vars_list}

    # Region selection (lat/lon slice; rejects wrap-around).
    if selected_region != 'all':
        lat_min, lat_max, lon_min, lon_max = selected_region
        assert lat_max >= lat_min, (
            f"lat_max ({lat_max}) must be >= lat_min ({lat_min}); "
            "selected_region is [lat_min, lat_max, lon_min, lon_max].")
        assert lon_max >= lon_min, (
            f"lon_max ({lon_max}) must be >= lon_min ({lon_min}); "
            "regions wrapping the 0/360 meridian aren't supported. "
            "Either rotate your region into a contiguous lon band or split "
            "into two runs.")
        predictions = predictions.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max))
        targets = targets.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max))

    # Pressure-level selection (only meaningful for vars with a level dim).
    if selected_lvls != 'all':
        if 'level' in predictions.dims and 'level' in targets.dims:
            predictions = predictions.sel(level=selected_lvls)
            targets     = targets.sel(level=selected_lvls)

    # Forecast time-step selection: [start_idx, size], positional.
    if selected_times != 'all':
        start_idx, size = selected_times
        assert start_idx >= 0 and size > 0, (
            f"selected_times must be [start_idx>=0, size>0]; got {selected_times}. "
            "An empty time slice would produce NaN losses.")
        predictions = predictions.isel(time=slice(start_idx, start_idx + size))
        targets     = targets.isel(time=slice(start_idx, start_idx + size))

    def _per_var_loss(prediction, target):
        # NOTE: predictions here come from `predictor(...)` which returns
        # values in physical units (the InputsAndResiduals wrapper
        # un-normalizes before returning). This is a different magnitude
        # convention than the fall-through path's `predictor.loss(...)`,
        # which computes MSE on normalized residuals. The masked-path
        # loss is therefore much larger numerically (especially geopotential
        # which dominates), but the optimization is still correct.
        diff_sq = (prediction - target) ** 2
        diff_sq = diff_sq * normalized_latitude_weights(target).astype(diff_sq.dtype)
        if 'level' in target.dims:
            diff_sq = diff_sq * normalized_level_weights(target).astype(diff_sq.dtype)
        return _mean_preserving_batch(diff_sq)

    losses = xarray_tree.map_structure(_per_var_loss, predictions, targets)
    return sum_per_variable_losses(losses, per_variable_weights)
# =====================================================================


@tws
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))


@tws
def masked_loss_fn(model_config, task_config, inputs, targets, forcings):
  """Selector-aware loss. Falls through to predictor.loss when all selectors
  are 'all', so default behavior is byte-identical to loss_fn."""
  predictor = construct_wrapped_graphcast(model_config, task_config)
  if (selected_vars   == 'all' and selected_lvls  == 'all'
      and selected_region == 'all' and selected_times == 'all'):
      loss, diagnostics = predictor.loss(inputs, targets, forcings)
  else:
      predictions = predictor(
          inputs, targets_template=targets, forcings=forcings)
      loss, diagnostics = custom_weighted_mse(predictions, targets)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

@tws
def timestep_loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  per_timestep_losses, per_timestep_diagnostics = predictor.per_timestep_loss_and_diagnostics(inputs, targets, forcings)
  return per_timestep_losses, per_timestep_diagnostics

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

def norm_grads64_fn(params, state, model_config, task_config, inputs, targets, forcings, stats, rng):
  def _aux(params, state, i, s, t, f):
    denorm_input = denormalize_64(i,s)
    (loss, diagnostics), next_state = masked_loss_fn.apply(
        params, state, rng, model_config, task_config,
        denorm_input, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True, argnums=2)(params, state, inputs, stats, targets, forcings)
  return loss, diagnostics, grads

def norm_grads32_fn(params, state, model_config, task_config, inputs, targets, forcings, stats, rng):
  def _aux(params, state, i, s, t, f):
    denorm_input = denormalize_32(i,s)
    (loss, diagnostics), next_state = masked_loss_fn.apply(
        params, state, rng, model_config, task_config,
        denorm_input, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True, argnums=2)(params, state, inputs, stats, targets, forcings)
  return loss, diagnostics, grads

def update_inputs(opt_update, grads, opt_state, inputs):
    updates, new_opt_state = opt_update(grads, opt_state, inputs)
    new_inputs = apply_updates(inputs, updates)
    return new_inputs, new_opt_state

def update_params(opt_update, grads, opt_state, params):
    """Standard parameter update function (GraphCast style)."""
    updates, new_opt_state = opt_update(grads, opt_state, params)
    new_params = apply_updates(params, updates)
    return new_params, new_opt_state


def _grads_fn_bound(params, state, inputs, targets, forcings):
    return grads_fn(params, state, model_config, task_config, inputs, targets, forcings)


# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.

def with_configs(fn):
  return partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is required by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))
loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
masked_loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(masked_loss_fn.apply))))
timestep_loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(timestep_loss_fn.apply))))

# runtime signature will be: (params, state, inputs, targets, forcings)
grads_fn_jitted = jax.jit(_grads_fn_bound)
import inspect
assert len(inspect.signature(_grads_fn_bound).parameters) == 5

norm_grads32_fn_jitted = with_params(jax.jit(with_configs(norm_grads32_fn)))
norm_grads64_fn_jitted = with_params(jax.jit(with_configs(norm_grads64_fn)))

run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))

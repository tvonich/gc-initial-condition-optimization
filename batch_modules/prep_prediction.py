#prep_prediction.py
from os import getenv
from json import load as jload
from numpy import zeros_like
from xarray import DataArray as xda


j_path = getenv('j_path')
with open(j_path, 'r') as f:
    config = jload(f)

init_date = config['init_date']
run_type = config['run_type']
pred_steps = config['pred_steps']
train_steps = pred_steps
selected_vars = config['selected_vars']
selected_lvls = config['selected_lvls']
selected_times = config['selected_times']
selected_region = config['selected_region']
just = config['justify']

def normalize_64(ds):
    """
    Normalize an xarray.Dataset over specified dimensions while handling variables with various dimension combinations.
    Uses float64 precision for statistical computations to maintain numerical stability.
    Parameters:
    - ds (xr.Dataset): The input dataset.
    Returns:
    - xr.Dataset: The normalized dataset in float64.
    - dict: A dictionary of statistics (mean and standard deviation) for each variable.
    """
    ds_copy = ds.copy()

    stats = {}

    normalization_dims = ['lon', 'lat', 'batch']
    VARS = ds.data_vars.items()
    static = ['year_progress_cos', 'year_progress_sin', 'day', 'toa_incident_solar_radiation',
              'geopotential_at_surface', 'land_sea_mask', 'day_progress_cos', 'day_progress_sin']

    for var_name, da in VARS:
        if var_name not in static:
            dims_to_reduce = [dim for dim in normalization_dims if dim in da.dims]

            mean = da.mean(dim=dims_to_reduce, keep_attrs=True).astype('float64')
            std = da.std(dim=dims_to_reduce, keep_attrs=True).astype('float64')

            da_64 = da.astype('float64')
            normalized = (da_64 - mean) / std

            ds_copy[var_name] = normalized.astype('float64')

            stats[var_name] = {'mean': mean, 'std': std}
        else:
            ds_copy[var_name] = da.astype('float64')

    return ds_copy, stats

def normalize_32(ds):
    """
    Normalize an xarray.Dataset over specified dimensions while handling variables with various dimension combinations.
    Uses float32 precision.
    Parameters:
    - ds (xr.Dataset): The input dataset.
    Returns:
    - xr.Dataset: The normalized dataset in float32.
    - dict: A dictionary of statistics (mean and standard deviation) for each variable.
    """
    ds_copy = ds.copy()

    stats = {}

    normalization_dims = ['lon', 'lat', 'batch']
    VARS = ds.data_vars.items()
    static = ['year_progress_cos', 'year_progress_sin', 'day', 'toa_incident_solar_radiation',
              'geopotential_at_surface', 'land_sea_mask', 'day_progress_cos', 'day_progress_sin']

    for var_name, da in VARS:
        if var_name not in static:
            dims_to_reduce = [dim for dim in normalization_dims if dim in da.dims]

            mean = da.mean(dim=dims_to_reduce, keep_attrs=True).astype('float32')
            std = da.std(dim=dims_to_reduce, keep_attrs=True).astype('float32')

            da_32 = da.astype('float32')
            normalized = (da_32 - mean) / std

            ds_copy[var_name] = normalized

            stats[var_name] = {'mean': mean, 'std': std}
        else:
            ds_copy[var_name] = da.astype('float32')

    return ds_copy, stats

def make_grads(train_inputs): #removing this rather than zeroing could speed up appreciably
    onethirty=xda(zeros_like(train_inputs['toa_incident_solar_radiation'].values),coords=train_inputs['toa_incident_solar_radiation'].coords, dims=train_inputs['toa_incident_solar_radiation'].dims)
    sixfive=xda(zeros_like(train_inputs['land_sea_mask'].values),coords=train_inputs['land_sea_mask'].coords, dims=train_inputs['land_sea_mask'].dims)
    seventwenty=xda(zeros_like(train_inputs['day_progress_cos'].values),coords=train_inputs['day_progress_cos'].coords, dims=train_inputs['day_progress_cos'].dims)
    two=xda(zeros_like(train_inputs['year_progress_cos'].values),coords=train_inputs['year_progress_cos'].coords, dims=train_inputs['year_progress_cos'].dims)

    fixed_grad_values = {
        'toa_incident_solar_radiation': onethirty,
        'land_sea_mask': sixfive,
        'geopotential_at_surface': sixfive,
        'day_progress_sin': seventwenty,
        'day_progress_cos': seventwenty,
        'year_progress_cos': two,
        'year_progress_sin': two
    }
    return fixed_grad_values

def zero_grads(grads,threshold,fixed_grad_values): #removing this rather than zeroing could speed up appreciably
    for grad_name, fixed_value in fixed_grad_values.items():
        grads[grad_name] = fixed_value

    return grads


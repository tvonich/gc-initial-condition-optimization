#prep_prediction.py
from os import getenv
from json import load as jload
from numpy import zeros_like
from xarray import DataArray as xda
import xarray as xr
from jax import jit
import numpy as np
import jax
from logging import info


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

import numpy as np
import xarray as xr
from scipy.fft import rfft, irfft, rfftfreq

def add_band_noise(ds, noise_factor=1, low_wavenumber=1, high_wavenumber=10, seed=None):
    """
    Add band-limited noise (zonal wavenumbers low to high) to each variable in the dataset,
    scaled by its standard deviation over the specified dimensions.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing variables to which noise should be added.
    noise_factor : float
        Factor by which to scale the standard deviation for noise generation.
    low_wavenumber : int
        Minimum zonal wavenumber for the noise band.
    high_wavenumber : int
        Maximum zonal wavenumber for the noise band.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xarray.Dataset
        A new dataset with added band-limited noise.
    """
    dynamic = ["geopotential"]
    dims_to_std = ['lat', 'lon']

    if seed is not None:
        np.random.seed(seed)

    ds_noisy = ds.copy()

    def apply_bandpass(data, low_k, high_k, dlon):
        n = data.shape[-1]
        freqs = rfftfreq(n, d=dlon)
        wavenums = np.abs(freqs) * 360.0
        mask = (wavenums >= low_k) & (wavenums <= high_k)
        fft_data = rfft(data, axis=-1)
        fft_data[..., ~mask] = 0j
        return irfft(fft_data, n=n, axis=-1)

    for var_name in dynamic:
        data_var = ds_noisy[var_name]
        std = data_var.std(dim=dims_to_std, skipna=True)

        # Generate white noise
        white_noise = noise_factor * std * xr.DataArray(
            np.random.randn(*data_var.shape),
            dims=data_var.dims,
            coords=data_var.coords
        )

        # Bandpass filter along lon
        n_lon = len(data_var.lon)
        dlon = 360.0 / n_lon
        band_noise = xr.apply_ufunc(
            apply_bandpass,
            white_noise,
            low_wavenumber,
            high_wavenumber,
            dlon,
            input_core_dims=[['lon']],
            output_core_dims=[['lon']],
            dask='allowed',
            output_dtypes=[white_noise.dtype]
        )

        # Scale to preserve total variance (approximate)
        num_band_modes = 2 * (high_wavenumber - low_wavenumber + 1)
        scale_factor = np.sqrt(n_lon / num_band_modes)
        band_noise *= scale_factor

        # Add noise
        ds_noisy[var_name] = data_var + band_noise
        info(f'{var_name} dtype: {ds_noisy[var_name].dtype}')

    return ds_noisy

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

import numpy as np
import xarray as xr
import logging

logger = logging.getLogger(__name__)

def add_white_noise(
    ds: xr.Dataset,
    vars_to_perturb=['temperature'],
    noise_factor=0,
    noise_factor_dict=None,
    seed=None,
):
    """
    Add white noise to selected variables in an xarray.Dataset.

    Noise is zero-mean and drawn from a uniform distribution, with the
    `noise_factor` interpreted as the *approximate mean absolute amplitude*
    of the perturbation for each variable.

    More precisely, for a given variable with factor `a`, the noise is
    drawn from U(-2a, 2a). The expected absolute value E[|noise|] = a.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset.
    vars_to_perturb : list of str, optional
        Names of variables to which noise will be added. If None, defaults to
        all data variables in the dataset.
    noise_factor : float, optional
        Global amplitude factor. Roughly the desired mean absolute perturbation
        (e.g., 1.0 K for temperature if units are Kelvin).
    noise_factor_dict : dict, optional
        Mapping {var_name: factor} to override `noise_factor` for specific
        variables. For example:
            {"temperature": 1.0, "u_component_of_wind": 0.5}
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xarray.Dataset
        A new dataset with added white noise.
    """
    if vars_to_perturb is None:
        vars_to_perturb = list(ds.data_vars)

    if noise_factor_dict is None:
        noise_factor_dict = {}

    rng = np.random.default_rng(seed)

    ds_noisy = ds.copy()

    for var_name in vars_to_perturb:
        if var_name not in ds_noisy:
            logger.warning(f"Variable {var_name} not found in dataset; skipping.")
            continue

        data_var = ds_noisy[var_name]

        # Choose amplitude for this variable
        a = noise_factor_dict.get(var_name, noise_factor)

        # Draw white noise ~ U(-2a, 2a), so E[|noise|] = a
        noise = xr.DataArray(
            rng.uniform(low=-2 * a, high=2 * a, size=data_var.shape),
            dims=data_var.dims,
            coords=data_var.coords,
        )

        # Add noise and preserve original dtype
        ds_noisy[var_name] = (data_var + noise).astype(data_var.dtype)

        logger.info(
            f"{var_name}: added white noise with mean |amplitude| ~ {a}, "
            f"dtype after noise: {ds_noisy[var_name].dtype}"
        )

    return ds_noisy


def get_init_times(num_inits, start_time=init_date, increment_hours=12):
    """
    Generate a list of initialization times.

    Args:
        start_time (str): Starting time (e.g., '2020-01-01T00').
        increment_hours (int): Hours between init times (default: 12).
        num_inits (int): Number of init times to generate.

    Returns:
        np.ndarray: Array of datetime64[ns] init times.
    """
    start_time_dt = np.datetime64(start_time)
    return np.arange(
        start_time_dt,
        start_time_dt + np.timedelta64(num_inits * increment_hours, 'h'),
        np.timedelta64(increment_hours, 'h'),
        dtype='datetime64[ns]'
    )

def preprocess_dataset(ds):
    """Remove time dimension from static fields."""
    ds = ds.copy()
    if "time" in ds["geopotential_at_surface"].dims:
        ds["geopotential_at_surface"] = ds["geopotential_at_surface"].isel(time=0)
    if "time" in ds["land_sea_mask"].dims:
        ds["land_sea_mask"] = ds["land_sea_mask"].isel(time=0)
    return ds

def to_numpy_dataset(ds):
    """Convert xarray.Dataset to one where all variables are NumPy arrays, not JAX or Dask."""
    ds_np = ds.copy(deep=True)
    for var in ds_np.data_vars:
        data = ds_np[var].data

        # Compute Dask arrays
        if hasattr(data, "compute"):
            data = data.compute()

        # Convert JAX DeviceArray to NumPy
        data = jax.device_get(data)

        # Ensure it's a NumPy array
        ds_np[var].data = np.asarray(data)

    return ds_np


shared_vars={'10m_v_component_of_wind', 'total_precipitation_6hr', 'geopotential', '10m_u_component_of_wind', 'u_component_of_wind', 'temperature', 'specific_humidity', 'mean_sea_level_pressure', '2m_temperature', 'vertical_velocity', 'v_component_of_wind'}

# dynamic=["geopotential","temperature","u_component_of_wind", "v_component_of_wind","specific_humidity","vertical_velocity","2m_temperature","mean_sea_level_pressure", "10m_v_component_of_wind","10m_u_component_of_wind","total_precipitation_6hr"]
# static=['toa_incident_solar_radiation','geopotential_at_surface', 'land_sea_mask', 'day_progress_cos', 'day_progress_sin', 'year_progress_cos','year_progress_sin']

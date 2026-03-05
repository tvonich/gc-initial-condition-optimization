import os
from logging import info
from datetime import datetime, timedelta
from numpy import random as npr
from jax.random import PRNGKey as jaxkey
from random import seed as rseed


def generate_dates(start_date_str, hours_interval, days):
    """
    Generates a list of date strings every `hours_interval` hours for `days` days.

    Parameters:
    - start_date_str (str): The initial date string in the format 'YYYY-MM-DDTHH'.
    - hours_interval (int): The interval in hours between each date. Default is 12.
    - days (int): The number of days for which to generate dates. Default is 10.

    Returns:
    - list of date strings in the format 'YYYY-MM-DDTHH'.
    """
    # Convert the string to a datetime object
    start_date = datetime.strptime(start_date_str, '%Y-%m-%dT%H')

    # List to hold the date strings
    date_list = []

    # Calculate the number of iterations (total steps = hours in days / hours interval)
    total_steps = (days * 24) // hours_interval

    # Generate dates every `hours_interval` hours for the specified number of days
    for i in range(total_steps):
        next_date = start_date + timedelta(hours=hours_interval * i)
        date_list.append(next_date.strftime('%Y-%m-%dT%H'))

    return date_list

def get_config(init_date,run_type, pred_steps, justify, selected_region, selected_vars, selected_lvls, selected_times):
    info(f"Making {pred_steps*6} hour forecast with Graphcast Small. Run Type = {run_type}")
    if run_type in ['loss','grad']:
        info(f"Selected Variables: {selected_vars}")
        info(f"Selected Levels: {selected_lvls}")
        info(f"Selected Region: {selected_region}")
        info(f"Selected Times: {selected_times}")

    return {
        "init_date":init_date,
        "pred_steps": pred_steps, #Number of 6 hour forecast periods [1,4,12,20,40]
        "train_steps": pred_steps, #Number of 6 hour training periods
        "run_type": run_type, #run_type = 'grad' #'pred' 'loss' or 'grad'
        "justify": justify,
        "selected_vars": selected_vars, #scroll_down for documentation
        "selected_lvls": selected_lvls, #scroll_down for documentation
        "selected_region": selected_region, #[47.,47,238.,238.5]
        "selected_times": selected_times   #'all' or [1,4] #[0,2] 0 --> 2 train_steps **Not Working
    }

def set_deterministic(seed=0):
    key = jaxkey(seed)
    rseed(seed)
    npr.seed(seed)
    return key

def drop_time_dim_if_present(ds, var_name):
    if 'time' in ds[var_name].dims:
        return ds[var_name].isel(time=0, drop=True)
    return ds[var_name]



#Data & Param Paths
# Set GRAPHCAST_DATA_PATH env var before submission, or pass DATA_PATH in your submit script.
dataset_file_path = os.environ.get('GRAPHCAST_DATA_PATH', '/path/to/your/era5_dataset.nc')

#ALL 11 VARIABLES
# ["2m_temperature","mean_sea_level_pressure", "10m_v_component_of_wind","10m_u_component_of_wind","total_precipitation_6hr", "geopotential","temperature","u_component_of_wind", "v_component_of_wind","specific_humidity","vertical_velocity"]

#All 13 LEVELS
#[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

#VARIABLE WEIGHTING (can adjust in graphcast.py)
#weights={"2m_temperature": 1.0,"10m_u_component_of_wind": 0.1,"10m_v_component_of_wind": 0.1,"mean_sea_level_pressure": 0.1,"total_precipitation_6hr": 0.1,}

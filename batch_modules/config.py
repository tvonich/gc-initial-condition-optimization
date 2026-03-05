import os
from logging import info


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
        "run_type": run_type, #run_type = 'optimize' #'pred' 'loss' or 'grad'
        "justify": justify,
        "selected_vars": selected_vars, #scroll_down for documentation
        "selected_lvls": selected_lvls, #scroll_down for documentation
        "selected_region": selected_region, #[47.,47,238.,238.5]
        "selected_times": selected_times   #'all' or [1,4] #[0,2] 0 --> 2 train_steps **Not Working
    }


#Data & Param Paths
# Set GRAPHCAST_DATA_PATH env var before submission, or pass DATA_PATH in your submit script.
dataset_file_path = os.environ.get('GRAPHCAST_DATA_PATH', '/path/to/your/era5_dataset.nc')

#ALL 11 VARIABLES
# ["2m_temperature","mean_sea_level_pressure", "10m_v_component_of_wind","10m_u_component_of_wind","total_precipitation_6hr", "geopotential","temperature","u_component_of_wind", "v_component_of_wind","specific_humidity","vertical_velocity"]

#All 13 LEVELS
#[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

#VARIABLE WEIGHTING (can adjust in graphcast.py)
#weights={"2m_temperature": 1.0,"10m_u_component_of_wind": 0.1,"10m_v_component_of_wind": 0.1,"mean_sea_level_pressure": 0.1,"total_precipitation_6hr": 0.1,}

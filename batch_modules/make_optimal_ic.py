#Run Predictions in Rollout (non-differentiable)
from math import inf
from shlex import join
import sys, os, json, warnings
from numpy import random as npr
from numpy import mean, datetime64, timedelta64, float64, savez, asarray
from numpy import nan as npnan
from numpy import arange
from config import dataset_file_path
from random import seed as rseed
from jax.random import PRNGKey as jaxkey
from jax import config as jconfig
from optax import adam, apply_updates
from pickle import dump
from logging import info, basicConfig, INFO, StreamHandler
from xarray import load_dataset as xld
from xarray import open_dataset as xod
from xarray import open_zarr as xzr
from xarray import concat
from dataclasses import asdict
import haiku as hk
from collections.abc import Mapping

path_to_add = os.environ.get('REPO_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if path_to_add not in sys.path:
    sys.path.append(path_to_add)
warnings.filterwarnings("ignore", message="endian-ness of dtype and endian kwarg do not match, using endian kwarg")

def set_deterministic(seed=0):
    key = jaxkey(seed)
    rseed(seed)
    npr.seed(seed)
    return key

def drop_time_dim_if_present(ds, var_name):
    if 'time' in ds[var_name].dims:
        return ds[var_name].isel(time=0, drop=True)
    return ds[var_name]

def preprocess_dataset(ds):
    """Remove time dimension from static fields."""
    ds = ds.copy()
    if "time" in ds["geopotential_at_surface"].dims:
        ds["geopotential_at_surface"] = ds["geopotential_at_surface"].isel(time=0)
    if "time" in ds["land_sea_mask"].dims:
        ds["land_sea_mask"] = ds["land_sea_mask"].isel(time=0)
    return ds

def _to_mutable(tree):
    if hasattr(hk.data_structures, "to_mutable_dict"):
        return hk.data_structures.to_mutable_dict(tree)
    return tree

def _flatten_nested(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, Mapping):
            yield from _flatten_nested(v, prefix + (k,))
        else:
            key = "/".join(prefix + (k,))
            yield key, asarray(v)

key=set_deterministic()
status64 = True
if status64==True: word64='F64'; lr_min=1e-15
else: word64='F32'; lr_min=1e-7

jconfig.update("jax_debug_nans", True)
jconfig.update("jax_enable_x64", status64)

basicConfig(
    level=INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[StreamHandler()],
    force=True
)


from load_model import model_config, task_config, params, load_params, fp
from prep_prediction import init_date, run_type, pred_steps, selected_vars, selected_lvls, selected_times, selected_region, just
from prep_prediction import zero_grads, normalize_32, normalize_64, make_grads, add_white_noise
from graphcast import rollout, data_utils, checkpoint, graphcast
import jitted

#LOAD TARGET
init_str = init_date[:13]  # Extract the first 13 characters (YYYY-MM-DDTHH)
init_date = datetime64(init_date)
name_path = dataset_file_path.split('/')[-1][5:15]
inter = xld(dataset_file_path).compute()

inter = inter.assign(
    geopotential_at_surface=drop_time_dim_if_present(inter, 'geopotential_at_surface'),
    land_sea_mask=drop_time_dim_if_present(inter, 'land_sea_mask')
)
for j,i in enumerate(inter.datetime.values[0]):
    if i==init_date:
        start_index=j-1
        break



example_batch = inter.isel(time=slice(start_index,None))

date_train=str(example_batch.datetime.values[0][1])[:13]
info(date_train)
num_time_steps = len(example_batch.time)
new_time = arange(0, num_time_steps * 6 * 3600 * 10**9, 6 * 3600 * 10**9, dtype='timedelta64[ns]')
example_batch = example_batch.assign_coords(time=new_time)

noise_factor = 0.03
train_steps = pred_steps
train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times = slice("6h", f"{train_steps*6}h"),
    **asdict(task_config), justify=just)
orig_train_inputs = train_inputs

# Access the j_path environment variable and get configs
j_path = os.getenv('j_path')
with open(j_path, 'r') as f:
    config = json.load(f)

log_loss_rate = 1
loss_tracker = []
param_saver = []
counter=0
inv_counter=0
shutdown=0
ens_count=0
step_down=0.5
pred_cutoff=120
min_loss=1e9
learn_cut=6
ens_cutoff=3
downscale=1
cutoff=15
safe_range=1.025
symbol ='='
model_state = {}
init_pred_steps=pred_steps
first_pred_steps=pred_steps
_output_base = os.environ.get('GRAPHCAST_OUTPUT_PATH', os.path.join(path_to_add, 'outputs'))

# Learning rate schedule (per pred_steps window)
if pred_steps >= 60: base_lr = 1e-4
else: base_lr = 1e-3

if selected_region != 'all': grad_threshold = 0
else: grad_threshold = 0

SAVE=True
init_num_epochs = 100;  num_epochs=init_num_epochs
r_step = 10 #Step between optimizations
step_limit = pred_steps #Maximum optimization length
loss_lead_limit = step_limit
epoch_set = [num_epochs - 1]  # Save checkpoint at the final epoch of each window
epoch_set = []

if selected_vars != 'all' or  selected_lvls != 'all' or selected_times != 'all' or selected_region != 'all':
    info(' ,'.join([word64, date_train, just, run_type,str(pred_steps), str(num_epochs),str(selected_vars), str(selected_lvls), str(selected_times),str(selected_region),str(base_lr),str(cutoff),str(grad_threshold)]))
else:
    info(' ,'.join([word64, date_train, just, run_type,str(pred_steps), str(init_num_epochs),str(base_lr),str(cutoff),"ALL",str(grad_threshold)]))

if selected_lvls == 'all' : selected_lvls = ['all']

_, __, full_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch, target_lead_times = slice("6h", f"{loss_lead_limit*6}h"),
    **asdict(task_config), justify=just)

if run_type == 'optimize':
    r_count=0
    repeat=True

    while example_batch.sizes["time"] >= pred_steps + 2 and pred_steps<=step_limit:
        min_loss=1e9
        counter=0
        inv_counter=0
        shutdown=0

        optimizer = adam(learning_rate=base_lr)
        opt_state = optimizer.init(params)
        loss_tracker = []
        info(f'Lead Time: {pred_steps/4} Days    LR:{base_lr:.9f}')

        for epoch in range(num_epochs):
            if status64==False:
                loss, diagnostics, model_state, grads = jitted.grads_fn_jitted(
                    params,
                    model_state,
                    train_inputs,
                    train_targets,
                    train_forcings,
                )

            loss_tracker.append(loss)

            if epoch==learn_cut:
                if loss_tracker[0]<=min(loss_tracker[1:learn_cut]) and base_lr>=1e-8:
                    repeat=True
                    break
                else: repeat=False

            if 1 <= epoch <= num_epochs-1 and loss<min_loss: info(f'{epoch}  {loss:.6f} *' )
            elif epoch % log_loss_rate == 0: info(f'{epoch}  {loss:.6f}')

            if loss < min_loss:
                min_loss=loss
                symbol ='='
                counter=0
                lowest_diagnostics=diagnostics
                lowest_grads=grads
                lowest_state=model_state
                lowest_params=params
                lowest_epoch=epoch
                inv_counter=inv_counter+1
            else:
                counter=counter+1
                inv_counter=0

            if counter>=cutoff and base_lr*(0.75)**(shutdown+1)>1e-7: shutdown+=1; optimizer = adam(learning_rate=base_lr*(0.75)**(shutdown));info(f'Shrink LR {base_lr*(0.75)**(shutdown):.8f}'); counter=0

            params, opt_state = jitted.update_params(optimizer.update, grads, opt_state, params)

            if epoch in epoch_set and SAVE==True:

                save_dir2 = f"{_output_base}/perfect_model_params/{date_train}"
                os.makedirs(save_dir2, exist_ok=True)
                ckpt_path = f"{save_dir2}/{date_train}_{epoch}_{pred_steps}.npz"

                try:
                    flat = hk.data_structures.to_flat_dict(lowest_params)
                    savez(ckpt_path, **{"/".join(k): v for k, v in flat.items()})
                except AttributeError:
                    nested = _to_mutable(lowest_params)
                    flat_pairs = dict(_flatten_nested(nested))
                    savez(ckpt_path, **flat_pairs)

                info(f"Saved parameters to {ckpt_path}")

        if loss_tracker[0]>min(loss_tracker[1:learn_cut]) or base_lr<=lr_min:
            reduction=(1-min_loss/loss_tracker[0])*100
            info(f'Minimum Loss: {min_loss:.3f}   Loss Reduction: {reduction:.1f}%')

            # Save best IC found for this pred_steps window
            if SAVE:
                save_dir2 = f"{_output_base}/perfect_model_params/{date_train}"
                os.makedirs(save_dir2, exist_ok=True)
                ckpt_path = f"{save_dir2}/{date_train}_{lowest_epoch}_{pred_steps}.npz"
                try:
                    flat = hk.data_structures.to_flat_dict(lowest_params)
                    savez(ckpt_path, **{"/".join(k): v for k, v in flat.items()})
                except AttributeError:
                    nested = _to_mutable(lowest_params)
                    flat_pairs = dict(_flatten_nested(nested))
                    savez(ckpt_path, **flat_pairs)
                info(f"Saved parameters to {ckpt_path}")

            pred_steps+=r_step
            train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
                example_batch, target_lead_times=slice("6h", f"{pred_steps*6}h"),
                **asdict(task_config), justify=just)

            r_count+=1

        else:
            if base_lr>1e-8: base_lr*=0.8
            info(f" LR Reduced {base_lr:.9f} ")
            params=lowest_params


elif run_type == 'pred':

    predictions = rollout.chunked_prediction(
        jitted.run_forward_jitted,
        rng=key,
        inputs=train_inputs,
        targets_template=train_targets,
        forcings=train_forcings)

    if status64==True:
        predictions = predictions.astype({var: float64 for var in predictions.data_vars})
    predictions.chunk({'lat': 181, 'lon': 360,'time': 20})
    os.makedirs(f'{_output_base}/perfect_model_forecasts/{date_train}', exist_ok=True)
    save_path = f'{_output_base}/perfect_model_forecasts/{date_train}/{date_train}_{pred_steps}.zarr'
    predictions.to_zarr(save_path, mode='w')
    info(f"Saved forecast to {save_path}")

elif run_type == 'loss':

    save_dir = f'{_output_base}/perfect_model_losses/{date_train}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loss, diagnostics = jitted.timestep_loss_fn_jitted(
        rng=key,
        inputs=train_inputs,
        targets=train_targets,
        forcings=train_forcings)

    os.chdir(save_dir)
    with open(f"{date_train}_{fp.split('-')[-1].split('.')[0]}.pkl", 'wb') as file:
        dump(loss, file)

elif run_type == 'grad':
    loss, diagnostics, grads = jitted.norm_grads_fn_jitted(
        inputs=train_inputs,
        targets=train_targets,
        forcings=train_forcings,
        stats=norm_stats,
        rng=key)

    grad_out_dir = os.path.join(_output_base, 'gradients', date_train)
    os.makedirs(grad_out_dir, exist_ok=True)
    grads.to_netcdf(os.path.join(grad_out_dir, f'{date_train}_{pred_steps}step_grads.nc'))
else: info('Misspelled run_type.')
info('// Complete //')

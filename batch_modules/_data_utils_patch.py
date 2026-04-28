"""Restores justify='left'/'right' support on graphcast.data_utils.

Upstream DeepMind GraphCast (pinned commit 97b9223a) does not expose a
``justify`` keyword on ``extract_inputs_targets_forcings`` /
``extract_input_target_times``. This repo's IC optimization pipeline
anchors ``init_date`` at the *first* input timestep ("left" justify),
which is a superset of upstream behavior (upstream is right-justified
by default).

This shim is the minimum necessary patch: it overrides exactly two
functions on ``graphcast.data_utils``. All helpers (``_DERIVED_VARS``,
``TISR``, ``add_derived_vars``, ``add_tisr_var``,
``_process_target_lead_times_and_get_duration``) are resolved against
the upstream module at call time, so this patch carries no
re-implementation of upstream logic beyond the two functions below.

Import this module exactly once, after ``from graphcast import data_utils``
and before any call site. ``make_optimal_ic.py`` does this near the top.
"""
import pandas as pd
from graphcast import data_utils


def extract_input_target_times(dataset, input_duration, target_lead_times,
                               justify='left'):
    """Drop-in replacement for ``data_utils.extract_input_target_times``
    that accepts ``justify`` ('left' or 'right'). Logic mirrors the
    private fork verbatim; the 'right' branch matches upstream behavior.
    """
    target_lead_times, target_duration = (
        data_utils._process_target_lead_times_and_get_duration(target_lead_times)
    )
    input_duration = pd.Timedelta(input_duration)
    time = dataset.coords["time"]

    if justify == 'left':
        target_start_time = int(input_duration.total_seconds() / 3600 / 6)
        target_end_time = (
            target_start_time
            + int(target_duration.total_seconds() / 3600 / 6)
        )
        inputs = dataset.isel(time=slice(int(target_start_time)))
        inputs['time'] = inputs['time'] - input_duration + time[1]
        targets = dataset.isel(time=slice(target_start_time, target_end_time))
        targets['time'] = targets['time'] - input_duration + time[1]
    elif justify == 'right':
        dataset = dataset.assign_coords(
            time=time + target_duration - time[-1]
        )
        targets = dataset.sel({"time": target_lead_times})
        zero = pd.Timedelta(0)
        epsilon = pd.Timedelta(1, "ns")
        inputs = dataset.sel(
            {"time": slice(-input_duration + epsilon, zero)}
        )
    else:
        raise ValueError("justify must either be 'left' or 'right'")

    return inputs, targets


def extract_inputs_targets_forcings(
        dataset, *,
        input_variables,
        target_variables,
        forcing_variables,
        pressure_levels,
        input_duration,
        target_lead_times,
        justify='left'):
    """Drop-in replacement for ``data_utils.extract_inputs_targets_forcings``
    that threads ``justify`` through to the patched
    ``extract_input_target_times``.
    """
    dataset = dataset.sel(level=list(pressure_levels))

    if set(forcing_variables) & data_utils._DERIVED_VARS:
        data_utils.add_derived_vars(dataset)
    if set(forcing_variables) & {data_utils.TISR}:
        data_utils.add_tisr_var(dataset)

    dataset = dataset.drop_vars("datetime")

    inputs, targets = extract_input_target_times(
        dataset,
        input_duration=input_duration,
        target_lead_times=target_lead_times,
        justify=justify)

    if set(forcing_variables) & set(target_variables):
        raise ValueError(
            f"Forcing variables {forcing_variables} should not "
            f"overlap with target variables {target_variables}."
        )

    inputs = inputs[list(input_variables)]
    forcings = targets[list(forcing_variables)]
    targets = targets[list(target_variables)]

    return inputs, targets, forcings


# Apply the monkey-patch to graphcast.data_utils.
data_utils.extract_input_target_times = extract_input_target_times
data_utils.extract_inputs_targets_forcings = extract_inputs_targets_forcings

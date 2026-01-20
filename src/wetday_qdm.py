import os
import numpy as np
import pandas as pd
import xarray as xr
import cftime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm


# -----------------------------
# Helpers
# -----------------------------
def _ensure_dir(path: str):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _to_noleap_midnight(t):
    """Convert a timestamp (numpy datetime64 or cftime) to cftime.DatetimeNoLeap at 00:00:00."""
    # If it's numpy datetime64, convert via pandas Timestamp
    if isinstance(t, (np.datetime64,)):
        ts = pd.Timestamp(t)
        return cftime.DatetimeNoLeap(ts.year, ts.month, ts.day, 0, 0, 0)
    # If it's cftime-like
    return cftime.DatetimeNoLeap(t.year, t.month, t.day, 0, 0, 0)


def preprocess_standard(ds, var_name=None):
    """
    1) Drop Feb 29
    2) Convert times to NoLeap and normalize to 00:00:00
    3) Convert precipitation units kg m-2 s-1 -> mm/day (if var_name provided)
    """
    # Drop Feb 29 (works with both datetime64 and cftime calendars)
    t = ds["time"]
    mask = ~((t.dt.month == 2) & (t.dt.day == 29))
    ds = ds.sel(time=mask)

    # Convert to NoLeap midnight
    new_times = [_to_noleap_midnight(tt) for tt in ds["time"].values]
    ds = ds.assign_coords(time=("time", new_times))
    ds["time"].encoding["calendar"] = "365_day"

    # Units convert (precip)
    if var_name:
        units = ds[var_name].attrs.get("units", "")
        if units.strip() == "kg m-2 s-1":
            ds[var_name] = ds[var_name] * 86400.0
            ds[var_name].attrs["units"] = "mm/day"

    return ds


# -----------------------------
# QDM core
# -----------------------------
def get_cdf_func(values):
    vals_sorted = np.sort(values)
    n = len(vals_sorted)

    def cdf(x):
        return np.searchsorted(vals_sorted, x, side="right") / n

    return cdf


def get_inv_cdf_func(values, n_quantiles=1000):
    vals_sorted = np.sort(values)
    q_probs = np.linspace(0, 1, n_quantiles)
    q_vals = np.quantile(vals_sorted, q_probs)

    # Remove duplicates for stable interpolation
    u_vals, u_idx = np.unique(q_vals, return_index=True)
    u_probs = q_probs[u_idx]

    return interp1d(
        u_probs,
        u_vals,
        bounds_error=False,
        fill_value=(q_vals[0], q_vals[-1]),
    )


def qdm_wet_days(obs_h, sim_h, sim_f, wet_threshold=0.1):
    """
    Wet-day-only QDM:
    - Only correct wet days (>= wet_threshold)
    - Dry days in sim_f become 0
    """
    if np.all(np.isnan(obs_h)) or np.all(np.isnan(sim_h)):
        return np.full_like(sim_f, np.nan)

    obs_wet = obs_h[obs_h >= wet_threshold]
    sim_h_wet = sim_h[sim_h >= wet_threshold]
    sim_f_wet = sim_f[sim_f >= wet_threshold]

    # Need enough data to build distributions
    if len(obs_wet) < 10 or len(sim_h_wet) < 10 or len(sim_f_wet) < 10:
        return sim_f

    cdf_sim_f_wet = get_cdf_func(sim_f_wet)
    inv_sim_h_wet = get_inv_cdf_func(sim_h_wet)
    inv_obs_h_wet = get_inv_cdf_func(obs_wet)

    corrected = np.empty_like(sim_f, dtype=float)

    for i, val in enumerate(sim_f):
        if np.isnan(val):
            corrected[i] = np.nan
        elif val < wet_threshold:
            corrected[i] = 0.0
        else:
            pf = cdf_sim_f_wet(val)
            s_h_q = float(inv_sim_h_wet(pf))
            o_h_q = float(inv_obs_h_wet(pf))

            ratio = 1.0 if s_h_q < 1e-6 else (val / s_h_q)
            corrected[i] = o_h_q * ratio

    return corrected


# -----------------------------
# Validation
# -----------------------------
def rolling_sum_within_year(da, window):
    """Rolling sum within each year independently (prevents cross-year contamination)."""
    pieces = []
    for _, sub in da.groupby("time.year"):
        pieces.append(sub.rolling(time=window, min_periods=window).sum())
    return xr.concat(pieces, dim="time")


def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


# -----------------------------
# Pipeline runner
# -----------------------------
def run_pipeline(cfg: dict):
    climate_glob = cfg["climate_glob"]
    gpm_path = cfg["gpm_path"]
    var_gpm = cfg.get("var_gpm", "precipitation")
    var_cmip = cfg.get("var_cmip", "pr")

    wet_threshold = float(cfg.get("wet_threshold", 0.1))
    rolling_days = int(cfg.get("rolling_days", 3))
    interp_method = cfg.get("interp_method", "nearest")

    out_nc = cfg.get("output_netcdf", "")
    out_plot = cfg.get("output_plot", "")
    show_plot = bool(cfg.get("show_plot", False))

    # Load data
    climate_data = xr.open_mfdataset(
        climate_glob, combine="by_coords", chunks={"time": -1}, engine="netcdf4"
    )
    gpm = xr.open_dataset(gpm_path).chunk({"time": -1})

    # Preprocess
    gpm_proc = preprocess_standard(gpm, var_gpm)
    clim_proc = preprocess_standard(climate_data, var_cmip)
    clim_proc = clim_proc.rename({var_cmip: "pr_mm_day"})

    # Interpolate climate to GPM grid
    clim_interp = clim_proc.interp(lat=gpm_proc["lat"], lon=gpm_proc["lon"], method=interp_method)

    # Time slicing
    common_times = np.intersect1d(gpm_proc["time"].values, clim_interp["time"].values)

    future_start = pd.to_datetime(cfg["future_start"])
    future_end = pd.to_datetime(cfg["future_end"])
    future_start = cftime.DatetimeNoLeap(future_start.year, future_start.month, future_start.day)
    future_end = cftime.DatetimeNoLeap(future_end.year, future_end.month, future_end.day)

    obs_hist = gpm_proc.sel(time=common_times)
    sim_hist = clim_interp.sel(time=common_times)
    sim_fut = clim_interp.sel(time=slice(future_start, future_end))

    # Stack (time, point)
    obs_stack = obs_hist[var_gpm].stack(point=("lat", "lon")).load()
    sim_h_stack = sim_hist["pr_mm_day"].stack(point=("lat", "lon")).load()
    sim_f_stack = sim_fut["pr_mm_day"].stack(point=("lat", "lon")).load()

    n_points = obs_stack.shape[1]
    corr_h_vals = np.full_like(sim_h_stack.values, np.nan, dtype=float)
    corr_f_vals = np.full_like(sim_f_stack.values, np.nan, dtype=float)

    print(f"Processing {n_points} grid points with Wet-Day QDM (threshold={wet_threshold})...")

    for i in tqdm(range(n_points)):
        o_h = obs_stack.values[:, i]
        s_h = sim_h_stack.values[:, i]
        s_f = sim_f_stack.values[:, i]

        corr_h_vals[:, i] = qdm_wet_days(o_h, s_h, s_h, wet_threshold=wet_threshold)
        corr_f_vals[:, i] = qdm_wet_days(o_h, s_h, s_f, wet_threshold=wet_threshold)


    # Unstack + concat
    bc_hist = xr.DataArray(
        corr_h_vals,
        coords=sim_h_stack.coords,
        dims=sim_h_stack.dims,
        name="bc_pr_mm_day"
    ).unstack("point")

    bc_fut = xr.DataArray(
        corr_f_vals,
        coords=sim_f_stack.coords,
        dims=sim_f_stack.dims,
        name="bc_pr_mm_day"
    ).unstack("point")

    bc_full = xr.concat([bc_hist, bc_fut], dim="time")

    # Save NetCDF
    if out_nc:
        _ensure_dir(out_nc)
        bc_full.to_netcdf(out_nc)
        print(f"Saved NetCDF: {out_nc}")

    # Validation plot (Annual maxima of rolling sums)
    print("Calculating Independent Annual Maxima (validation plot)...")
    obs_roll = rolling_sum_within_year(obs_hist[var_gpm], rolling_days)
    raw_roll = rolling_sum_within_year(sim_hist["pr_mm_day"], rolling_days)
    corr_roll = rolling_sum_within_year(bc_hist, rolling_days)

    obs_ams = obs_roll.groupby("time.year").max("time")
    raw_ams = raw_roll.groupby("time.year").max("time")
    corr_ams = corr_roll.groupby("time.year").max("time")

    obs_vals = obs_ams.values.flatten()
    raw_vals = raw_ams.values.flatten()
    corr_vals = corr_ams.values.flatten()

    obs_vals = obs_vals[~np.isnan(obs_vals)]
    raw_vals = raw_vals[~np.isnan(raw_vals)]
    corr_vals = corr_vals[~np.isnan(corr_vals)]

    xo, yo = ecdf(obs_vals)
    xr_v, yr = ecdf(raw_vals)
    xc, yc = ecdf(corr_vals)

    plt.figure(figsize=(8, 6))
    plt.plot(xo, yo, label="Obs (GPM)", lw=2.5)
    plt.plot(xr_v, yr, label="Raw Model", color="red", alpha=0.5)
    plt.plot(xc, yc, label="Bias Corrected (Wet-Day QDM)", color="black", ls="--")
    plt.title(f"{rolling_days}-Day Annual Maxima (Years Independent)")
    plt.legend()

    if out_plot:
        _ensure_dir(out_plot)
        plt.savefig(out_plot, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {out_plot}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return bc_full

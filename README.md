# Wet-Day QDM Bias Correction (GPM IMERG + CMIP6)

Bias-correct daily CMIP6 precipitation using **Wet-Day-Only Quantile Delta Mapping (QDM)** with **GPM IMERG** as observations.  
Workflow: preprocess (NoLeap + units) → interpolate CMIP6 to GPM grid → apply wet-day QDM → validate with annual-max ECDF.

## What this repository contains
- `src/` : core functions (preprocess, wet-day QDM, validation)
- `scripts/` : runner script
- `config/` : YAML config (paths + parameters)
- `data/` : local data folder (not committed)
- `outputs/` : results (NetCDF + figures)

## Quick start

### 1) Create the environment
```bash
conda env create -f environment.yml
conda activate wetday-qdm


## 2) Put your data here (local)
```python
data/raw/cmip6/pr/*.nc
data/raw/gpm/GPM_IMERG_daily_mecca_25yr.nc
```


## 3) Edit the config

Copy and edit:

- config/config_example.yaml

Key fields:
- climate_glob (CMIP6 files glob)
- gpm_path (GPM file path)
- wet_threshold (default 0.1 mm/day)
- future_start, future_end
- output paths

# 4) Run
```python
python -m scripts.run_bias_correction --config config/config_example.yml
```

## Outputs

- Bias-corrected NetCDF: outputs/netcdf/bias_corrected_wetday_qdm.nc

- Validation figure (ECDF): outputs/figures/ecdf_annual_maxima.png



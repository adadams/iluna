from pathlib import Path

import xarray as xr
from matplotlib import pyplot as plt

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory.parent
dataset_directory: Path = project_directory / "datasets"
plot_output_directory: Path = project_directory / "fit_outputs"

BDWD_dataset_filepath: Path = (
    dataset_directory / "S4_nirspec_PRISM_WD1202_ap3_bg4_LCData.h5"
)

BDWD_dataset: xr.Dataset = xr.open_dataset(BDWD_dataset_filepath)

BDWD_time: xr.DataArray = BDWD_dataset.time
BDWD_data: xr.DataArray = BDWD_dataset.flux_white
BDWD_data_errors: xr.DataArray = BDWD_dataset.err_white

harmonic_lightcurve_dataset_filepath: Path = (
    dataset_directory / "pure_harmonic_lightcurves_phase_adjusted.nc"
)

harmonic_lightcurve_dataset: xr.Dataset = xr.open_dataset(
    harmonic_lightcurve_dataset_filepath
)

harmonic_lightcurve_time: xr.DataArray = harmonic_lightcurve_dataset.time
harmonic_lightcurve_data: xr.DataArray = (
    harmonic_lightcurve_dataset.system_model_lightcurve
)
constant_lightcurve_data: xr.DataArray = harmonic_lightcurve_data[0, :]


figure, axis = plt.subplots(figsize=(10, 5))
axis.plot(BDWD_time, BDWD_data, color="black")

axis.plot(harmonic_lightcurve_time, constant_lightcurve_data, color="C0", alpha=1)

for i in range(1, 10):
    axis.plot(
        harmonic_lightcurve_time,
        (
            harmonic_lightcurve_data[i, :]
            - constant_lightcurve_data
            + constant_lightcurve_data.min()
        )
        * 10
        + constant_lightcurve_data.median(),
        color="C" + str(i),
        alpha=0.66,
    )

figure.savefig(
    plot_output_directory / "constant_lightcurve_vs_white_light_data.pdf",
    bbox_inches="tight",
)

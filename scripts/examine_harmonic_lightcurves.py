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
    dataset_directory / "pure_harmonic_lightcurves_maximum_degree_20.nc"
)

harmonic_lightcurve_dataset: xr.Dataset = xr.open_dataset(
    harmonic_lightcurve_dataset_filepath
)

harmonic_lightcurve_time: xr.DataArray = harmonic_lightcurve_dataset.time
baseline_WD_lightcurve_data: xr.DataArray = (
    harmonic_lightcurve_dataset.WD_model_lightcurve[0, :]
)
harmonic_lightcurve_data: xr.DataArray = harmonic_lightcurve_dataset.BD_model_lightcurve
constant_lightcurve_data: xr.DataArray = harmonic_lightcurve_data[0, :]


figure, axis = plt.subplots(figsize=(10, 5))
axis.plot(BDWD_time, BDWD_data, color="black")

axis.plot(harmonic_lightcurve_time, constant_lightcurve_data, color="C0", alpha=1)

arbitrary_harmonic_scale: float = 1.0e3

for i in range(1, 8):
    axis.plot(
        harmonic_lightcurve_time,
        harmonic_lightcurve_data[i, :]
        / harmonic_lightcurve_dataset.attrs["baseline_BD_nightside_brightness"].item()
        * arbitrary_harmonic_scale
        + baseline_WD_lightcurve_data
        + harmonic_lightcurve_dataset.attrs["baseline_BD_nightside_brightness"].item(),
        color="C" + str(i),
        alpha=0.66,
    )

figure.savefig(
    plot_output_directory / "constant_lightcurve_vs_white_light_data.pdf",
    bbox_inches="tight",
)

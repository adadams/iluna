from pathlib import Path

import xarray as xr
from matplotlib import pyplot as plt
from spectres import spectres

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory.parent
dataset_directory: Path = project_directory / "datasets"
plot_output_directory: Path = project_directory / "fit_outputs"

plt.style.use(project_directory / "arthur.mplstyle")

harmonic_lightcurves_filepath: Path = (
    dataset_directory / "pure_harmonic_lightcurves_maximum_degree_20.nc"
)
harmonic_lightcurves: xr.Dataset = xr.open_dataset(harmonic_lightcurves_filepath)
constant_WD_lightcurve: xr.DataArray = harmonic_lightcurves.WD_model_lightcurve.isel(
    harmonic_index=0
)

fit_result_filepath: Path = dataset_directory / "fit_iterations.nc"
fit_result: xr.Dataset = xr.open_dataset(fit_result_filepath)

data_filepath: Path = dataset_directory / "S4_nirspec_PRISM_WD1202_ap3_bg4_LCData.h5"
data_dataset: xr.Dataset = xr.open_dataset(data_filepath)

constant_WD_lightcurve_binned_to_data: xr.DataArray = xr.DataArray(
    data=spectres(
        data_dataset.time.values,
        constant_WD_lightcurve.time.values,
        constant_WD_lightcurve.values,
    ),
    dims=("time",),
    coords={"time": data_dataset.time},
).isel(time=slice(1, -1))

time: xr.DataArray = data_dataset.time.isel(time=slice(1, -1))
data: xr.DataArray = data_dataset.flux_white.isel(time=slice(1, -1))
data_errors: xr.DataArray = data_dataset.err_white.isel(time=slice(1, -1))

for i in range(26):
    fiducial_fit_result: xr.Dataset = (
        fit_result.best_fit_light_curve.sel(maximum_eigenmode=i)
        + constant_WD_lightcurve_binned_to_data
    )

    residuals: xr.DataArray = (data - fiducial_fit_result) / data_errors

    figure, (lightcurve_axis, residual_axis) = plt.subplots(
        2, 1, figsize=(8, 6), height_ratios=[2, 1]
    )

    plot_time: xr.DataArray = time - harmonic_lightcurves.attrs["time_transit"].item()

    lightcurve_axis.scatter(
        plot_time,
        data,
        s=2,
        color="black",
        label="White light data",
    )
    lightcurve_axis.plot(
        plot_time,
        fiducial_fit_result,
        color="mediumseagreen",
        label="Fiducial fit",
        zorder=2,
    )

    residual_axis.plot(
        plot_time,
        residuals,
        color="crimson",
        label="Residuals",
        zorder=2,
    )

    residual_axis.set_xlabel("Time relative to mid-transit (days)")

    lightcurve_axis.set_ylabel("Flux (e$^{-}$)")
    residual_axis.set_ylabel(r"$\left(F - F_\mathrm{{model}}\right)/\sigma$")

    lightcurve_axis.legend()
    residual_axis.legend()

    figure.savefig(
        plot_output_directory
        / f"best_fit_lightcurve_at_maximum_eigenmode_{i:02}_vs_white_light_data.pdf",
        bbox_inches="tight",
    )

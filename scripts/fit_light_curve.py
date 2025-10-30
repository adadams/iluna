import sys
from itertools import cycle
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from numpy.linalg import lstsq as least_squares_fit
from spectres import spectres

sys.path.append(str(Path(__file__).parent.parent))

from iluna.basic_types import FluxValue
from iluna.case_types import ObservationCondition
from iluna.helper_functions import open_dataset, save_xarray_data_to_file

# Define your custom color palette
custom_colors = ["#7089d4", "#d4707a", "#70d4a9", "#ffe573", "#ffaa6e", "#9676db"]
custom_color_cycler = cycle(custom_colors)

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory.parent
dataset_directory: Path = project_directory / "datasets"
plot_output_directory: Path = project_directory / "fit_outputs"

plt.style.use(project_directory / "arthur.mplstyle")


def calculate_log_likelihood(
    best_fit_light_curve: xr.DataArray,
    white_light_data: xr.DataArray,
    white_light_data_errors: xr.DataArray,
):
    residuals: xr.DataArray = white_light_data - best_fit_light_curve
    data_variance: xr.DataArray = white_light_data_errors**2

    log_likelihood: float = -0.5 * np.sum(
        np.log(2 * np.pi * data_variance) + residuals**2 / data_variance
    )

    return log_likelihood


def calculate_BIC(
    best_fit_light_curve: xr.DataArray,
    white_light_data: xr.DataArray,
    white_light_data_errors: xr.DataArray,
    number_of_eigenmodes: int,
):
    log_likelihood: float = calculate_log_likelihood(
        best_fit_light_curve, white_light_data, white_light_data_errors
    ).item()

    number_of_data_points: int = len(white_light_data)
    number_of_parameters: int = number_of_eigenmodes

    BIC: float = -2 * log_likelihood + number_of_parameters * np.log(
        number_of_data_points
    )

    return BIC


BIC_value: float = 0


def fit_to_eigencurves(
    eigenmode_harmonic_coefficients: xr.DataArray,
    model_light_curves: xr.DataArray,
    white_light_data: xr.DataArray,
    white_light_data_errors: xr.DataArray,
    maximum_fit_eigenmode: int,
):
    fit_coefficient_array, fit_residual, _, _ = least_squares_fit(
        model_light_curves, white_light_data
    )
    fit_residual: float = fit_residual.item()

    fit_coefficients: xr.DataArray = xr.DataArray(
        data=np.r_[
            fit_coefficient_array,
            np.zeros(len(model_light_curves.eigenmode) - len(fit_coefficient_array)),
        ],
        dims=("eigenmode",),
        coords={"eigenmode": model_light_curves.eigenmode},
        name="fit_coefficients",
    )

    best_fit_light_curve: xr.DataArray = xr.dot(fit_coefficients, model_light_curves)

    best_fit_harmonic_components: xr.DataArray = xr.dot(
        fit_coefficients, eigenmode_harmonic_coefficients
    )

    BIC_value = calculate_BIC(
        best_fit_light_curve,
        white_light_data,
        white_light_data_errors,
        maximum_fit_eigenmode,
    )

    return xr.Dataset(
        {
            "fit_coefficients": fit_coefficients,
            "model_light_curves": model_light_curves,
            "best_fit_harmonic_components": best_fit_harmonic_components,
            "best_fit_light_curve": best_fit_light_curve,
            "white_light_data": white_light_data,
            "white_light_data_errors": white_light_data_errors,
        },
        attrs={"BIC_value": BIC_value, "fit_residual": fit_residual},
    )


if __name__ == "__main__":
    fit_condition_name: ObservationCondition = "at_all_phases"

    harmonic_lightcurve_dataset_filepath: Path = (
        dataset_directory / "pure_harmonic_lightcurves_maximum_degree_20.nc"
    )
    harmonic_lightcurve_dataset: xr.Dataset = open_dataset(
        harmonic_lightcurve_dataset_filepath
    )

    eigenmode_fit_filepath: Path = dataset_directory / "at_all_phases_eigenmodes.nc"
    eigenmode_fit_dataset: xr.Dataset = open_dataset(eigenmode_fit_filepath)

    data_filepath: Path = (
        dataset_directory / "S4_nirspec_PRISM_WD1202_ap3_bg4_LCData.h5"
    )
    data_dataset: xr.Dataset = open_dataset(data_filepath)

    model_light_curves: xr.DataArray = eigenmode_fit_dataset.eigencurves
    model_light_curves_binned_to_data: xr.DataArray = xr.DataArray(
        data=spectres(
            data_dataset.time.values,
            model_light_curves.time.values,
            model_light_curves.transpose("eigenmode", "time").values,
        ),
        dims=("eigenmode", "time"),
        coords={"eigenmode": model_light_curves.eigenmode, "time": data_dataset.time},
    )

    fit_condition: xr.DataArray = xr.full_like(
        data_dataset.time, True
    )  # eigenmode_fit_dataset[fit_condition_name]

    baseline_BD_nightside_brightness: FluxValue = harmonic_lightcurve_dataset.attrs[
        "baseline_BD_nightside_brightness"
    ].item()

    constant_BD_lightcurve: xr.DataArray = (
        harmonic_lightcurve_dataset.BD_model_lightcurve.isel(harmonic_index=0)
    ) / baseline_BD_nightside_brightness

    constant_BD_lightcurve_binned_to_data: xr.DataArray = xr.DataArray(
        data=spectres(
            data_dataset.time.values,
            constant_BD_lightcurve.time.values,
            constant_BD_lightcurve.values,
        ),
        dims=("time",),
        coords={"time": data_dataset.time},
    ).expand_dims(dim={"eigenmode": [0]})

    constant_WD_lightcurve: xr.DataArray = (
        harmonic_lightcurve_dataset.WD_model_lightcurve.isel(harmonic_index=0)
    )

    constant_WD_lightcurve_binned_to_data: xr.DataArray = xr.DataArray(
        data=spectres(
            data_dataset.time.values,
            constant_WD_lightcurve.time.values,
            constant_WD_lightcurve.values,
        ),
        dims=("time",),
        coords={"time": data_dataset.time},
    )

    model_light_curves_for_fitting: xr.DataArray = (
        xr.concat(
            [constant_BD_lightcurve_binned_to_data, model_light_curves_binned_to_data],
            dim="eigenmode",
        )
        .fillna(0.0)
        .rename("model_light_curves_for_fitting")
    )
    model_light_curves_for_fitting.to_netcdf(
        dataset_directory / "model_light_curves_for_fitting.nc"
    )

    white_light_data: xr.DataArray = (
        data_dataset.flux_white.where(fit_condition, drop=True)
        - constant_WD_lightcurve_binned_to_data
    )

    is_valid_data: xr.DataArray = np.logical_not(white_light_data.isnull())

    white_light_data_errors: xr.DataArray = data_dataset.err_white.where(
        fit_condition, drop=True
    )

    constant_harmonic_components: np.ndarray = np.r_[
        1.0, np.zeros_like(eigenmode_fit_dataset.eigenmode)
    ]
    harmonic_components_with_constant_index: np.ndarray = np.r_[
        np.zeros_like(eigenmode_fit_dataset.harmonic_index)[np.newaxis, :],
        eigenmode_fit_dataset.harmonic_components.values,
    ]

    eigenmode_harmonic_components: xr.DataArray = xr.DataArray(
        data=np.c_[
            constant_harmonic_components,
            harmonic_components_with_constant_index,
        ],
        dims=("eigenmode", "harmonic_index"),
        coords={
            "eigenmode": np.r_[0, eigenmode_fit_dataset.eigenmode.values],
            "harmonic_index": harmonic_lightcurve_dataset.harmonic_index,
        },
        name="eigenmode_harmonic_component",
    )

    maximum_fit_eigenmode: int = 26

    fit_datasets: list[xr.Dataset] = [
        fit_to_eigencurves(
            eigenmode_harmonic_components.isel(
                eigenmode=slice(None, fit_eigenmode + 1)
            ),
            model_light_curves_for_fitting.transpose("time", "eigenmode")
            .where(is_valid_data, drop=True)
            .isel(eigenmode=slice(None, fit_eigenmode + 1)),
            white_light_data.where(is_valid_data, drop=True),
            white_light_data_errors.where(is_valid_data, drop=True),
            fit_eigenmode,
        )
        for fit_eigenmode in range(maximum_fit_eigenmode)
    ]

    fit_iteration_dataset: xr.Dataset = xr.concat(fit_datasets, dim="maximum_eigenmode")

    save_xarray_data_to_file(
        fit_iteration_dataset,
        filename=dataset_directory / "fit_iterations.nc",
        multi_index_dims="harmonic_index",
    )

    BIC_values: np.ndarray[np.float64] = np.array(
        [fit_dataset.attrs["BIC_value"] for fit_dataset in fit_datasets]
    )

    minimum_BIC: float = np.min(BIC_values)
    incremental_delta_BICs: np.ndarray[np.float64] = np.r_[0, np.diff(BIC_values)]
    overall_delta_BICs: np.ndarray[np.float64] = BIC_values - minimum_BIC

    BIC_figure, BIC_axis = plt.subplots(figsize=(10, 10))
    BIC_axis.semilogy(BIC_values - minimum_BIC, marker="o")

    BIC_figure.savefig(
        plot_output_directory / "BIC_values.pdf", bbox_inches="tight", dpi=300
    )

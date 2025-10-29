import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TypedDict

import jax
import numpy as np
import xarray as xr
from jaxoplanet.orbits.keplerian import Body, Central
from jaxoplanet.starry import Surface, Ylm, orbit
from matplotlib import pyplot as plt
from nautilus import Prior, Sampler
from scipy import stats
from spectres import spectres

sys.path.append(str(Path(__file__).parent.parent))
from iluna.basic_types import (
    DurationValue,
    FluxValue,
    MassValue,
    NormalizedValue,
    RadiusValue,
)
from iluna.structures import TransitFitParameters
from jaxoplanet_source_mods.light_curves.emission import light_curve
from scripts.prepare_data import prepare_white_light_in_transit_data

jax.config.update("jax_enable_x64", True)

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory.parent
dataset_directory: Path = project_directory / "datasets"
plot_output_directory: Path = project_directory / "fit_outputs"


class TransitData(TypedDict):
    data: np.ndarray[FluxValue]
    errors: np.ndarray[FluxValue]
    transit_time: np.ndarray[np.float64]
    model_transit_time: np.ndarray[np.float64]


"""
def prepare_white_light_in_transit_data(
    BDWD_dataset_filepath: Path = dataset_directory
    / "S4_nirspec_PRISM_WD1202_ap3_bg4_LCData.h5",
    approximate_transit_slice: slice = slice(421, 494),
) -> TransitData:
    BDWD_dataset: xr.Dataset = xr.open_dataset(BDWD_dataset_filepath)

    BDWD_time: xr.DataArray = BDWD_dataset.time
    BDWD_data: xr.DataArray = BDWD_dataset.flux_white
    BDWD_data_errors: xr.DataArray = BDWD_dataset.err_white

    number_of_data_points: int = (
        approximate_transit_slice.stop - approximate_transit_slice.start
    )

    transit_time: np.ndarray = BDWD_time.to_numpy()[approximate_transit_slice]
    transit_data: np.ndarray = BDWD_data.to_numpy()[approximate_transit_slice]
    transit_data_errors: np.ndarray = BDWD_data_errors.to_numpy()[
        approximate_transit_slice
    ]

    minimum_transit_time: float = transit_time.min()
    maximum_transit_time: float = transit_time.max()

    model_transit_time: np.ndarray = np.linspace(
        minimum_transit_time, maximum_transit_time, 100 * number_of_data_points
    )

    return {
        "data": transit_data,
        "errors": transit_data_errors,
        "transit_time": transit_time,
        "model_transit_time": model_transit_time,
    }
"""


class NonEclipseData(TypedDict):
    data: np.ndarray[FluxValue]
    errors: np.ndarray[FluxValue]
    light_curve_time: np.ndarray[np.float64]
    model_light_curve_time: np.ndarray[np.float64]


def prepare_white_light_non_eclipse_data(
    BDWD_dataset_filepath: Path = dataset_directory
    / "S4_nirspec_PRISM_WD1202_ap3_bg4_LCData.h5",
    approximate_eclipse_slice: slice = slice(202, 240),
) -> NonEclipseData:
    BDWD_dataset: xr.Dataset = xr.open_dataset(BDWD_dataset_filepath)

    noneclipse_BDWD_time: xr.DataArray = BDWD_dataset.time
    noneclipse_BDWD_data: xr.DataArray = BDWD_dataset.flux_white
    noneclipse_BDWD_data[approximate_eclipse_slice] = np.nan
    noneclipse_BDWD_data_errors: xr.DataArray = BDWD_dataset.err_white
    noneclipse_BDWD_data_errors[approximate_eclipse_slice] = np.nan

    number_of_data_points: int = len(noneclipse_BDWD_time)

    minimum_transit_time: float = noneclipse_BDWD_time.min().item()
    maximum_transit_time: float = noneclipse_BDWD_time.max().item()
    print(f"transit time range: {minimum_transit_time} to {maximum_transit_time}")

    model_light_curve_time: np.ndarray = np.linspace(
        minimum_transit_time, maximum_transit_time, 100 * number_of_data_points
    )

    return {
        "data": noneclipse_BDWD_data.to_numpy(),
        "errors": noneclipse_BDWD_data_errors.to_numpy(),
        "light_curve_time": noneclipse_BDWD_time.to_numpy(),
        "model_light_curve_time": model_light_curve_time,
    }


def calculate_transit_fit_likelihood(
    parameters: TransitFitParameters,
    data: np.ndarray[FluxValue],
    errors: np.ndarray[FluxValue],
    transit_time: np.ndarray[np.float64],
    model_transit_time: np.ndarray[np.float64],
    generate_test_plot: bool = False,
) -> float:
    sqrt_q0: NormalizedValue = np.sqrt(parameters["q0"])
    q1: NormalizedValue = parameters["q1"]
    u0: float = sqrt_q0 * 2 * q1
    u1: float = sqrt_q0 * (1 - 2 * q1)

    RWD: RadiusValue = parameters["RWD"]
    MWD: MassValue = parameters["MWD"]

    baseline_WD_brightness: FluxValue = parameters["baseline_WD_brightness"]
    baseline_BD_nightside_brightness: FluxValue = parameters[
        "baseline_BD_nightside_brightness"
    ]

    surface_central: Surface = Surface(
        y=Ylm({(0, 0): 1}),
        amplitude=baseline_WD_brightness,
        normalize=False,
        u=(u0, u1),
    )
    surface_planet: Surface = Surface(
        y=Ylm({(0, 0): 1}),
        amplitude=baseline_BD_nightside_brightness,
        normalize=False,
        u=(0, 0),
    )
    body_central: Central = Central(radius=RWD, mass=MWD)
    body_planet: Body = Body(
        period=parameters["period"],
        radius=parameters["RBD"],
        impact_param=parameters["impact_param"],
        time_transit=parameters["time_transit"],
    )

    system: orbit.SurfaceSystem = orbit.SurfaceSystem(
        central=body_central,
        central_surface=surface_central,
        bodies=((body_planet, surface_planet),),
    )

    WD_model_transit_lightcurve, BD_model_transit_lightcurve = light_curve(system)(
        model_transit_time
    ).T
    system_model_transit_lightcurve: np.ndarray = (
        WD_model_transit_lightcurve + BD_model_transit_lightcurve
    )

    system_model_transit_lightcurve_binned_to_data: np.ndarray = spectres(
        transit_time, model_transit_time, system_model_transit_lightcurve
    )

    if generate_test_plot:
        figure, axis = plt.subplots(figsize=(8, 6))

        axis.plot(
            model_transit_time,
            system_model_transit_lightcurve,
            label="model transit lightcurve",
            zorder=100,
        )
        axis.plot(transit_time, data, label="data")

        axis.legend()

        figure.savefig(
            plot_output_directory / "MLE_transit_lightcurve.pdf",
            dpi=300,
            bbox_inches="tight",
        )

    return -0.5 * np.nansum(
        ((system_model_transit_lightcurve_binned_to_data - data) / errors) ** 2
        + np.log(2 * np.pi * errors**2)
    )


def construct_truncated_normal_distribution(
    mean_value: float,
    scale_value: float,
    number_of_scales: float = 3,
    absolute_minimum_value: float = 0,
) -> Callable:
    lower_bound: float = np.max(
        [absolute_minimum_value, mean_value - number_of_scales * scale_value]
    )
    upper_bound: float = mean_value + number_of_scales * scale_value
    lower_bound_transformed, upper_bound_transformed = (
        (lower_bound - mean_value) / scale_value,
        (upper_bound - mean_value) / scale_value,
    )
    return stats.truncnorm(
        lower_bound_transformed,
        upper_bound_transformed,
        loc=mean_value,
        scale=scale_value,
    )


MLE_transit_fit_parameters: TransitFitParameters = {
    "period": 0.049465275206039674,
    "time_transit": 60804.46435895566,
    "MWD": 0.4435380802753672,
    "RWD": 0.021453302527833333,
    "RBD": 0.08703384406115107,
    "impact_param": 1.0925409444810594,
    "baseline_WD_brightness": 4379.8823580584785,
    "baseline_BD_nightside_brightness": 161.88713767392053,
    "q0": 0.021265754077815058,
    "q1": 0.9912678380940251,
}


if __name__ == "__main__":
    white_light_transit_data: TransitData = prepare_white_light_in_transit_data()

    test_likelihood: float = calculate_transit_fit_likelihood(
        parameters=MLE_transit_fit_parameters,
        generate_test_plot=True,
        **white_light_transit_data,
    )
    print(f"{test_likelihood=}")

    test_light_curve_data: NonEclipseData = prepare_white_light_non_eclipse_data()

    run_fit: bool = False

    if run_fit:
        rappaport_constrained_period: DurationValue = 0.049_465_262
        rappaport_constrained_period_uncertainty: DurationValue = 2e-8

        minimum_transit_time: float = np.min(white_light_transit_data["transit_time"])
        maximum_transit_time: float = np.max(white_light_transit_data["transit_time"])

        rappaport_constrained_MWD: MassValue = 0.415
        rappaport_constrained_MWD_uncertainty: MassValue = 0.028

        rappaport_constrained_RWD: RadiusValue = 0.021
        rappaport_constrained_RWD_uncertainty: RadiusValue = 0.001

        rappaport_constrained_RBD: RadiusValue = 0.088
        rappaport_constrained_RBD_uncertainty: RadiusValue = 0.005

        prior = Prior()
        prior.add_parameter(
            "period",
            dist=construct_truncated_normal_distribution(
                mean_value=rappaport_constrained_period,
                scale_value=rappaport_constrained_period_uncertainty,
            ),
        )
        prior.add_parameter(
            "time_transit", dist=(minimum_transit_time, maximum_transit_time)
        )
        prior.add_parameter(
            "MWD",
            dist=construct_truncated_normal_distribution(
                mean_value=rappaport_constrained_MWD,
                scale_value=rappaport_constrained_MWD_uncertainty,
            ),
        )
        prior.add_parameter(
            "RWD",
            dist=construct_truncated_normal_distribution(
                mean_value=rappaport_constrained_RWD,
                scale_value=rappaport_constrained_RWD_uncertainty,
            ),
        )
        prior.add_parameter(
            "RBD",
            dist=construct_truncated_normal_distribution(
                mean_value=rappaport_constrained_RBD,
                scale_value=rappaport_constrained_RBD_uncertainty,
            ),
        )
        rappaport_constrained_radius_ratio: float = (
            rappaport_constrained_RBD / rappaport_constrained_RWD
        )
        prior.add_parameter(
            "impact_param",
            dist=(
                0,
                rappaport_constrained_radius_ratio
                + (1 / rappaport_constrained_radius_ratio) * 1.5,
            ),
        )
        prior.add_parameter("baseline_WD_brightness", dist=(1000, 10000))
        prior.add_parameter("baseline_BD_nightside_brightness", dist=(1, 1000))
        prior.add_parameter("q0", dist=(0, 1))
        prior.add_parameter("q1", dist=(0, 1))

        save_name = "jaxoplanet_keplerian_transit_fit"

        calculate_transit_fit_likelihood_with_white_light_data = partial(
            calculate_transit_fit_likelihood, **white_light_transit_data
        )

        sampler = Sampler(
            prior,
            calculate_transit_fit_likelihood_with_white_light_data,
            n_live=1000,
            filepath=str(dataset_directory / f"{save_name}.hdf5"),
            pool=4,
            resume=True,
        )

        sampler.run(discard_exploration=True, timeout=np.inf, verbose=True)

        points, log_w, log_l = sampler.posterior()
        log_z = sampler.log_z

        np.save(dataset_directory / f"{save_name}_points.npy", points)
        np.save(dataset_directory / f"{save_name}_log_w.npy", log_w)
        np.save(dataset_directory / f"{save_name}_log_l.npy", log_l)
        np.save(dataset_directory / f"{save_name}_log_z.npy", log_z)

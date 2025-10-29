import sys
from pathlib import Path

import jax
import numpy as np
import xarray as xr
from cf_xarray import encode_multi_index_as_compress
from jaxoplanet.orbits.keplerian import Body, Central
from jaxoplanet.starry import Surface, Ylm
from jaxoplanet.starry.light_curves import light_curve
from jaxoplanet.starry.orbit import SurfaceSystem
from pandas import MultiIndex

sys.path.append(str(Path(__file__).parent.parent))
from iluna.basic_types import FluxValue, MassValue, NormalizedValue, RadiusValue
from iluna.structures import TransitFitParameters

# from jaxoplanet_source_mods.light_curves.emission import light_curve
from scripts.prepare_data import FullPhaseData, prepare_white_light_full_data

jax.config.update("jax_enable_x64", True)

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory.parent
dataset_directory: Path = project_directory / "datasets"


def construct_system_lightcurves(
    parameters: dict,
    lightcurve_time: xr.DataArray,
    central_map_parameters: dict = {(0, 0): 1},
    companion_map_parameters: dict = {(0, 0): 1},
) -> xr.Dataset:
    sqrt_q0: NormalizedValue = np.sqrt(parameters["q0"])
    q1: NormalizedValue = parameters["q1"]
    u0: float = sqrt_q0 * 2 * q1
    u1: float = sqrt_q0 * (1 - 2 * q1)

    RWD: RadiusValue = parameters["RWD"]
    MWD: MassValue = parameters["MWD"]

    RBD: RadiusValue = parameters["RBD"]

    baseline_WD_brightness: FluxValue = parameters["baseline_WD_brightness"]
    baseline_BD_nightside_brightness: FluxValue = parameters[
        "baseline_BD_nightside_brightness"
    ]

    body_central: Central = Central(radius=RWD, mass=MWD)
    surface_central: Surface = Surface(
        y=Ylm(central_map_parameters),
        amplitude=baseline_WD_brightness,
        normalize=False,
        u=(u0, u1),
    )

    body_planet: Body = Body(
        period=parameters["period"],
        radius=parameters["RBD"],
        impact_param=parameters["impact_param"],
        time_transit=parameters["time_transit"],
    )

    phase_adjustment: float = (
        np.pi
        * (
            1
            - 2
            * (
                MLE_transit_fit_parameters["time_transit"]
                / MLE_transit_fit_parameters["period"]
            )
        )
    ) % (2 * np.pi)  # sets nightside view (phase = pi) at time of mid-transit

    surface_planet: Surface = Surface(
        y=Ylm(companion_map_parameters),
        amplitude=baseline_BD_nightside_brightness,
        period=parameters["period"],  # i.e. tidal locking
        phase=phase_adjustment,
        normalize=False,
        u=(0, 0),
    )

    system: SurfaceSystem = SurfaceSystem(
        central=body_central,
        central_surface=surface_central,
        bodies=((body_planet, surface_planet),),
    )

    WD_model_lightcurve, BD_model_lightcurve = light_curve(system, order=25)(
        lightcurve_time.to_numpy()
    ).T

    relative_position_x, relative_position_y, relative_position_z = (
        system.relative_position(lightcurve_time.to_numpy())
    )
    relative_positions: xr.Dataset = xr.Dataset(
        data_vars={
            "relative_position_x": xr.DataArray(
                relative_position_x.squeeze(),
                dims=("time",),
                coords={"time": lightcurve_time},
                name="relative_position_x",
            ),
            "relative_position_y": xr.DataArray(
                relative_position_y.squeeze(),
                dims=("time",),
                coords={"time": lightcurve_time},
                name="relative_position_y",
            ),
            "relative_position_z": xr.DataArray(
                relative_position_z.squeeze(),
                dims=("time",),
                coords={"time": lightcurve_time},
                name="relative_position_z",
            ),
        }
    )
    lateral_distance: xr.DataArray = (
        relative_positions.relative_position_x**2
        + relative_positions.relative_position_y**2
    ) ** 0.5

    something_is_occulted: xr.DataArray = lateral_distance < (RBD + RWD)
    companion_in_front_of_host: xr.DataArray = (
        relative_positions.relative_position_z > 0
    )
    companion_behind_host: xr.DataArray = relative_positions.relative_position_z < 0

    relative_positions["BD_in_front_of_WD"] = np.logical_and(
        something_is_occulted, companion_in_front_of_host
    )
    relative_positions["BD_completely_occults_WD"] = np.logical_and(
        lateral_distance < (RBD - RWD), companion_in_front_of_host
    )

    relative_positions["WD_in_front_of_BD"] = np.logical_and(
        something_is_occulted, companion_behind_host
    )
    relative_positions["WD_completely_occults_BD"] = np.logical_and(
        lateral_distance < (RWD - RBD), companion_behind_host
    )

    WD_model_lightcurve_dataarray: xr.DataArray = xr.DataArray(
        data=WD_model_lightcurve,
        dims=("time",),
        coords={"time": lightcurve_time},
        name="WD_model_lightcurve",
    )

    BD_model_lightcurve_dataarray: xr.DataArray = xr.DataArray(
        data=BD_model_lightcurve,
        dims=("time",),
        coords={"time": lightcurve_time},
        name="BD_model_lightcurve",
    )

    system_model_lightcurve_dataarray: xr.DataArray = (
        WD_model_lightcurve_dataarray + BD_model_lightcurve_dataarray
    )

    system_model_lightcurves: xr.Dataset = xr.Dataset(
        data_vars={
            "WD_model_lightcurve": WD_model_lightcurve_dataarray,
            "BD_model_lightcurve": BD_model_lightcurve_dataarray,
            "system_model_lightcurve": system_model_lightcurve_dataarray,
            **relative_positions,
        },
    )

    return system_model_lightcurves


def construct_constant_lightcurves(
    parameters: dict, lightcurve_time: xr.DataArray
) -> xr.Dataset:
    return construct_system_lightcurves(parameters, lightcurve_time)


def construct_harmonic_orders_as_coordinate_index(maximum_harmonic_degree: int):
    total_number_of_orders: int = (maximum_harmonic_degree + 1) ** 2 - 1

    harmonic_degrees: np.ndarray[(total_number_of_orders,), np.float64] = (
        np.concatenate(
            [
                [degree] * (2 * degree + 1)
                for degree in range(maximum_harmonic_degree + 1)
            ]
        )
    )
    harmonic_orders: np.ndarray[(total_number_of_orders,), np.float64] = np.concatenate(
        [
            np.arange(-degree, degree + 1)
            for degree in range(maximum_harmonic_degree + 1)
        ]
    )

    harmonic_index: MultiIndex = MultiIndex.from_arrays(
        (harmonic_degrees, harmonic_orders), names=("degree", "order")
    )

    return harmonic_index


def construct_pure_harmonic_lightcurves(
    parameters: dict,
    lightcurve_time: xr.DataArray,
    maximum_harmonic_degree: int,
    central_map_parameters: dict = {(0, 0): 1},
) -> ...:
    harmonic_indices: MultiIndex = construct_harmonic_orders_as_coordinate_index(
        maximum_harmonic_degree
    )

    # harmonic_index_coordinate: xr.Coordinates = xr.Coordinates.from_pandas_multiindex(
    #    harmonic_indices, dim="harmonic_index"
    # )

    harmonic_lightcurve_deviations: xr.Dataset = (
        xr.concat(
            [
                construct_system_lightcurves(
                    parameters,
                    lightcurve_time,
                    central_map_parameters=central_map_parameters,
                    companion_map_parameters={harmonic_index: 1},
                )
                # - constant_lightcurves
                for harmonic_index in harmonic_indices
            ],
            dim="harmonic_index",
        )
        .assign_coords(harmonic_index=harmonic_indices)
        .assign_attrs(parameters)
    )

    return harmonic_lightcurve_deviations


if __name__ == "__main__":
    white_light_data: FullPhaseData = prepare_white_light_full_data()

    model_transit_time_as_coordinate: xr.Variable = xr.Variable(
        data=white_light_data["model_lightcurve_time"],
        dims=("time",),
        attrs={"unit": "day"},
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

    constant_lightcurves: xr.Dataset = construct_constant_lightcurves(
        MLE_transit_fit_parameters, model_transit_time_as_coordinate
    )

    maximum_harmonic_degree: int = 20

    pure_harmonic_lightcurves: xr.Dataset = construct_pure_harmonic_lightcurves(
        MLE_transit_fit_parameters,
        model_transit_time_as_coordinate,
        maximum_harmonic_degree,
    )

    test_lightcurve_deviations_serializable: xr.Dataset = (
        encode_multi_index_as_compress(pure_harmonic_lightcurves, "harmonic_index")
    )
    test_lightcurve_deviations_serializable.to_netcdf(
        dataset_directory
        / f"pure_harmonic_lightcurves_maximum_degree_{maximum_harmonic_degree}.nc"
    )

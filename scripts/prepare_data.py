from pathlib import Path
from typing import TypedDict

import numpy as np
import xarray as xr

from iluna.basic_types import FluxValue

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory.parent
dataset_directory: Path = project_directory / "datasets"


class TransitData(TypedDict):
    data: np.ndarray[FluxValue]
    errors: np.ndarray[FluxValue]
    transit_time: np.ndarray[np.float64]
    model_transit_time: np.ndarray[np.float64]


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


class NonEclipseData(TypedDict):
    data: np.ndarray[FluxValue]
    errors: np.ndarray[FluxValue]
    lightcurve_time: np.ndarray[np.float64]
    model_lightcurve_time: np.ndarray[np.float64]


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

    model_lightcurve_time: np.ndarray = np.linspace(
        minimum_transit_time, maximum_transit_time, 100 * number_of_data_points
    )

    return {
        "data": noneclipse_BDWD_data.to_numpy(),
        "errors": noneclipse_BDWD_data_errors.to_numpy(),
        "lightcurve_time": noneclipse_BDWD_time.to_numpy(),
        "model_lightcurve_time": model_lightcurve_time,
    }


class FullPhaseData(TypedDict):
    data: np.ndarray[FluxValue]
    errors: np.ndarray[FluxValue]
    lightcurve_time: np.ndarray[np.float64]
    model_lightcurve_time: np.ndarray[np.float64]


def prepare_white_light_full_data(
    BDWD_dataset_filepath: Path = dataset_directory
    / "S4_nirspec_PRISM_WD1202_ap3_bg4_LCData.h5",
) -> FullPhaseData:
    BDWD_dataset: xr.Dataset = xr.open_dataset(BDWD_dataset_filepath)

    BDWD_time: xr.DataArray = BDWD_dataset.time
    BDWD_data: xr.DataArray = BDWD_dataset.flux_white
    BDWD_data_errors: xr.DataArray = BDWD_dataset.err_white

    number_of_data_points: int = len(BDWD_time)

    minimum_time: float = BDWD_time.min().item()
    maximum_time: float = BDWD_time.max().item()

    model_lightcurve_time: np.ndarray = np.linspace(
        minimum_time, maximum_time, 100 * number_of_data_points
    )

    return {
        "data": BDWD_data.to_numpy(),
        "errors": BDWD_data_errors.to_numpy(),
        "lightcurve_time": BDWD_time.to_numpy(),
        "model_lightcurve_time": model_lightcurve_time,
    }

from pathlib import Path

import numpy as np
import xarray as xr
from fit_keplerian_system_transit_with_jaxoplanet import TransitFitParameters

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory.parent
dataset_directory: Path = project_directory / "datasets"


def export_results_to_dataset(
    run_name: str, parameter_names: list[str], file_directory: Path = dataset_directory
) -> tuple[xr.Dataset, xr.Dataset]:
    sampled_points_filepath: Path = file_directory / f"{run_name}_points.npy"
    log_likelihoods_filepath: Path = file_directory / f"{run_name}_log_l.npy"
    log_weights_filepath: Path = file_directory / f"{run_name}_log_w.npy"
    evidences_filepath: Path = file_directory / f"{run_name}_log_z.npy"

    sampled_points: np.ndarray = np.load(sampled_points_filepath)
    log_likelihoods: np.ndarray = np.load(log_likelihoods_filepath)
    log_weights: np.ndarray = np.load(log_weights_filepath)
    evidence: float = float(np.load(evidences_filepath))

    sample_coordinate: xr.DataArray = xr.DataArray(
        data=np.arange(len(sampled_points)),
        dims=("sample",),
        coords={"sample": np.arange(len(sampled_points))},
        name="sample",
    )

    sampled_points_by_parameter: dict = {
        parameter_name: {
            "dims": ("sample",),
            "data": parameter_samples.T,
            "name": parameter_names,
        }
        for parameter_name, parameter_samples in zip(
            parameter_names,
            sampled_points.T,
        )
    }

    log_likelihood_dataarray: xr.DataArray = xr.DataArray(
        data=log_likelihoods,
        dims=("sample",),
        coords={"sample": sample_coordinate},
        name="log_likelihood",
    )

    log_weight_dataarray: xr.DataArray = xr.DataArray(
        data=log_weights,
        dims=("sample",),
        coords={"sample": sample_coordinate},
        name="log_weight",
    )

    sample_dataset: xr.Dataset = xr.Dataset.from_dict(
        {
            "data_vars": sampled_points_by_parameter,
            "coords": {"sample": sample_coordinate.to_dict()},
            "dims": {"sample": "sample"},
            "attrs": {"evidence": evidence, "jupiter_radius_in_cm": 7.13552e9},
        }
    )

    results_dataset: xr.Dataset = xr.merge(
        [sample_dataset, log_likelihood_dataarray, log_weight_dataarray]
    )

    results_dataset.to_netcdf(file_directory / f"{run_name}_results.nc")

    MLE_dataset: xr.Dataset = results_dataset.isel(
        sample=np.argmax(log_likelihoods)
    ).drop_vars(["log_likelihood", "log_weight"])
    print(f"{MLE_dataset=}")

    MLE_dataset.to_netcdf(file_directory / f"{run_name}_MLE.nc")

    return results_dataset, MLE_dataset


if __name__ == "__main__":
    run_name: str = "jaxoplanet_keplerian_transit_fit"

    results_filepath: Path = dataset_directory / f"{run_name}_results.nc"

    need_to_generate_results: bool = not Path.exists(results_filepath)

    # get list of typeddict fields from TransitFitParameters
    parameter_names: list[str] = [
        field for field in TransitFitParameters.__annotations__.keys()
    ]

    if need_to_generate_results:
        export_results_to_dataset(run_name, parameter_names)

    results: xr.Dataset = xr.open_dataset(results_filepath)

    MLE_results_filepath: Path = dataset_directory / f"{run_name}_MLE.nc"
    MLE_results: xr.Dataset = xr.open_dataset(MLE_results_filepath)

    # results["inclination"] = (
    #    np.arccos(results["impact_param"] * 0.021 / 0.444) * 180 / np.pi
    # ).assign_attrs(units="deg")

    # MLE_results["inclination"] = (
    #    np.arccos(MLE_results["impact_param"] * 0.021 / 0.444) * 180 / np.pi
    # ).assign_attrs(units="deg")

    MLE_results["u0"] = np.sqrt(MLE_results["q0"]) * 2 * MLE_results["q1"]
    MLE_results["u1"] = np.sqrt(MLE_results["q0"]) * (1 - 2 * MLE_results["q1"])

    # results.to_netcdf(dataset_directory / f"{run_name}_results.nc")

    # MLE_results.to_netcdf(dataset_directory / f"{run_name}_MLE.nc")

    results_credible_intervals: xr.Dataset = xr.Dataset.from_dict(
        {
            parameter_name: {
                "data": np.percentile(
                    parameter_samples,
                    [16, 50, 84],
                    method="inverted_cdf",
                    weights=np.exp(results.log_weight),
                ),
                "dims": ("percentile",),
            }
            for parameter_name, parameter_samples in results.data_vars.items()
        }
    ).assign_coords(
        {
            "percentile": [16, 50, 84],
        }
    )

    print(f"{results_credible_intervals=}")

    for parameter_name, parameter_samples in results.data_vars.items():
        credible_interval = np.percentile(
            parameter_samples,
            [16, 50, 84],
            method="inverted_cdf",
            weights=np.exp(results.log_weight),
        )
        median: float = credible_interval[1]
        lower_bound: float = credible_interval[1] - credible_interval[0]
        upper_bound: float = credible_interval[2] - credible_interval[1]

        print(f"{parameter_name}: {median} +{upper_bound} -{lower_bound}")

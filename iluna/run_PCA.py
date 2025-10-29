import sys
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

sys.path.append(str(Path(__file__).parent.parent))
from iluna.helper_functions import reconstitute_multi_index, save_xarray_data_to_file

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory.parent
dataset_directory: Path = project_directory / "datasets"
plot_output_directory: Path = project_directory / "fit_outputs"


def project_harmonic_lightcurves_onto_eigenbasis(
    harmonic_lightcurves: xr.Dataset, number_of_eigenmodes: int
) -> ...:
    # PCA order for inputs is (time, harmonic_number)
    pca = PCA(n_components=number_of_eigenmodes, whiten=True)

    pca.fit(harmonic_lightcurves)

    return pca.components_, pca.explained_variance_ratio_


if __name__ == "__main__":
    harmonic_lightcurve_dataset_filepath: Path = (
        dataset_directory / "pure_harmonic_lightcurves_maximum_degree_20.nc"
    )

    harmonic_lightcurve_dataset: xr.Dataset = reconstitute_multi_index(
        xr.open_dataset(harmonic_lightcurve_dataset_filepath)
    )

    harmonic_lightcurve_time: xr.DataArray = harmonic_lightcurve_dataset.time

    constant_lightcurve: xr.DataArray = (
        harmonic_lightcurve_dataset.system_model_lightcurve.isel(harmonic_index=0)
    )

    harmonic_lightcurves_minus_constant_lightcurve: xr.DataArray = (
        ((harmonic_lightcurve_dataset.system_model_lightcurve) - constant_lightcurve)
        .isel(harmonic_index=slice(1, None))
        .transpose("time", "harmonic_index")
    )

    WD_in_front_of_BD: xr.DataArray = (
        harmonic_lightcurve_dataset.WD_in_front_of_BD.isel(harmonic_index=0, drop=True)
    )
    BD_in_front_of_WD: xr.DataArray = (
        harmonic_lightcurve_dataset.BD_in_front_of_WD.isel(harmonic_index=0, drop=True)
    )

    in_secondary_eclipse: xr.DataArray = (WD_in_front_of_BD).rename(
        "in_secondary_eclipse"
    )  # WD partially occults BD
    out_of_occultation: xr.DataArray = np.logical_and(
        np.logical_not(WD_in_front_of_BD),
        np.logical_not(BD_in_front_of_WD),
    ).rename("out_of_occultation")  # i.e. phase only
    at_all_phases: xr.DataArray = xr.full_like(harmonic_lightcurve_time, True).rename(
        "at_all_phases"
    )  # placeholder for doing no cuts

    fit_condition: xr.DataArray = out_of_occultation

    nonconstant_harmonics: xr.DataArray = (
        harmonic_lightcurves_minus_constant_lightcurve.where(fit_condition)
    )
    nonconstant_harmonics_cut_to_fit_region: xr.DataArray = (
        nonconstant_harmonics.dropna("time")
    )

    number_of_eigenmodes: int = 25

    pca_components, pca_variance_fraction = (
        project_harmonic_lightcurves_onto_eigenbasis(
            nonconstant_harmonics_cut_to_fit_region.values, number_of_eigenmodes
        )
    )

    harmonic_index_minus_constant: xr.DataArray = (
        harmonic_lightcurves_minus_constant_lightcurve.harmonic_index
    )

    eigenmode_coordinate: xr.Variable = xr.Variable(
        dims=("eigenmode",), data=np.arange(number_of_eigenmodes) + 1
    )

    pca_component_dataarray: xr.DataArray = xr.DataArray(
        pca_components,
        dims=("eigenmode", "harmonic_index"),
        coords={
            "eigenmode": eigenmode_coordinate,
            "harmonic_index": harmonic_index_minus_constant,
        },
        name="harmonic_components",
    )

    pca_variance_fraction_dataarray: xr.DataArray = xr.DataArray(
        pca_variance_fraction,
        dims=("eigenmode",),
        coords={"eigenmode": eigenmode_coordinate},
        name="variance_fraction",
    )

    eigencurve_dataarray: xr.DataArray = (
        xr.dot(nonconstant_harmonics, pca_component_dataarray)
        .rename("eigencurves")
        .transpose("eigenmode", "time")
    )

    eigenmode_dataset: xr.Dataset = xr.merge(
        [
            pca_component_dataarray,
            pca_variance_fraction_dataarray,
            eigencurve_dataarray,
            fit_condition,
        ]
    )

    eigenmode_dataset_encoded_for_saving: xr.DataArray = save_xarray_data_to_file(
        eigenmode_dataset,
        filename=dataset_directory / f"{fit_condition.name}_eigenmodes.nc",
        multi_index_dims="harmonic_index",
    )

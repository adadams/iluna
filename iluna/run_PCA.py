from pathlib import Path

import cf_xarray
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA

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
        dataset_directory / "pure_harmonic_lightcurves_phase_adjusted.nc"
    )

    harmonic_lightcurve_dataset: xr.Dataset = cf_xarray.decode_compress_to_multi_index(
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
    print(f"{np.shape(harmonic_lightcurves_minus_constant_lightcurve.values)=}")

    pca_components, pca_variance_fraction = (
        project_harmonic_lightcurves_onto_eigenbasis(
            harmonic_lightcurves_minus_constant_lightcurve.values,
            number_of_eigenmodes=10,
        )
    )

    print(f"{pca_components=}")
    print(f"{pca_variance_fraction=}")
    print(f"{np.shape(pca_variance_fraction)=}")
    print(f"{np.sum(pca_variance_fraction)=}")

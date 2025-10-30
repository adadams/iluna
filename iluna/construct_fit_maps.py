import sys
from pathlib import Path

import jax
import xarray as xr
from jaxoplanet.starry import Surface
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.starry.visualization import show_surface
from matplotlib import pyplot as plt
from pandas import MultiIndex

sys.path.append(str(Path(__file__).parent.parent))

from datasets.white_light_transit_fit import MLE_transit_fit_parameters
from iluna.case_types import ObservationCondition
from iluna.generate_harmonic_lightcurves import construct_surface_system
from iluna.helper_functions import open_dataset

jax.config.update("jax_enable_x64", True)

current_directory: Path = Path(__file__).parent
project_directory: Path = current_directory.parent
dataset_directory: Path = project_directory / "datasets"
plot_output_directory: Path = project_directory / "fit_outputs"

if __name__ == "__main__":
    fit_condition_name: ObservationCondition = "at_all_phases"

    fit_results_dataset_filepath: Path = dataset_directory / "fit_iterations.nc"

    fit_results_dataset: xr.Dataset = open_dataset(fit_results_dataset_filepath)

    harmonic_indices_as_tuples: MultiIndex = (
        fit_results_dataset.harmonic_index.to_pandas()
    ).values

    eigenmap_coordinates: list[dict[tuple[int, int], float]] = [
        {
            harmonic_index: fit_results_harmonic_coefficient.item()
            for harmonic_index, fit_results_harmonic_coefficient in zip(
                harmonic_indices_as_tuples, fit_results_harmonic_coefficients
            )
        }
        for fit_results_harmonic_coefficients in fit_results_dataset.best_fit_harmonic_components
    ]

    figure, axis_grid = plt.subplots(3, 9, figsize=(24, 8))
    axis_list: list[plt.Axes] = axis_grid.flatten()

    number_of_modes_to_plot: int = len(axis_list) - 1

    fit_results_systems: list[SurfaceSystem] = [
        construct_surface_system(
            parameters=MLE_transit_fit_parameters,
            companion_map_parameters=eigenmap_coordinates[n],
        )
        for n in range(number_of_modes_to_plot)
    ]

    for n, (axis, fit_results_system) in enumerate(zip(axis_list, fit_results_systems)):
        companion_surface: Surface = fit_results_system.body_surfaces[0]

        # theta value is the opposite of the mid-transit phase calculated in "construct_surface_system"
        show_surface(companion_surface, theta=-1.009, n=None, cmap="magma", ax=axis)
        axis.set_title(f"Mode {n}", fontsize=16)

    figure.savefig(
        plot_output_directory
        / f"first_{number_of_modes_to_plot}_{fit_condition_name}_fit_maps.pdf",
        bbox_inches="tight",
    )

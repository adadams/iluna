from collections.abc import Iterable
from pathlib import Path

import xarray as xr
from cf_xarray import decode_compress_to_multi_index, encode_multi_index_as_compress


def save_xarray_data_to_file(
    xarray_data: xr.DataArray | xr.Dataset | xr.DataTree,
    filename: Path | str,
    multi_index_dims: str | Iterable[str] | None = None,
):
    if multi_index_dims is not None:
        xarray_data = encode_multi_index_as_compress(xarray_data, multi_index_dims)

    xarray_data.to_netcdf(filename)


def reconstitute_multi_index(xarray_data: xr.DataArray | xr.Dataset | xr.DataTree):
    return decode_compress_to_multi_index(xarray_data)

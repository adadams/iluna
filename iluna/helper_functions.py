from collections.abc import Iterable
from os import PathLike
from typing import Optional, TypeVar

import xarray as xr
from cf_xarray import decode_compress_to_multi_index, encode_multi_index_as_compress

XarrayData = TypeVar("XarrayDataType", xr.DataArray, xr.Dataset, xr.DataTree)


def save_xarray_data_to_file(
    xarray_data: XarrayData,
    filename: str | PathLike,
    multi_index_dims: Optional[str | Iterable[str]] = None,
) -> XarrayData:
    if multi_index_dims is not None:
        xarray_data = encode_multi_index_as_compress(xarray_data, multi_index_dims)

    xarray_data.to_netcdf(filename)

    return xarray_data


def reconstitute_multi_index(xarray_data: xr.DataArray | xr.Dataset | xr.DataTree):
    return decode_compress_to_multi_index(xarray_data)


def open_dataset(filename: str | PathLike) -> xr.Dataset:
    return decode_compress_to_multi_index(xr.open_dataset(filename))

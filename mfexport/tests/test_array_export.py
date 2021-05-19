from pathlib import Path
import numpy as np
import pytest
from mfexport.array_export import make_levels, export_array
from mfexport.grid import MFexportGrid


def test_make_levels():

    array = np.linspace(-5.5, 5.4, 100)
    levels = make_levels(array, 0.2, maxlevels=1000)

    assert levels[0] >= array.min()
    assert levels[-1] <= array.max()

    levels = make_levels(array, 0.002, maxlevels=1000)
    assert len(levels) == 1000


@pytest.mark.parametrize('pathlib_path', (True, False))
def test_int64_export(test_output_folder, pathlib_path):
    arr = np.ones((2, 2), dtype=np.int64)
    mg = MFexportGrid(delr=np.ones(2), delc=np.ones(2))
    if pathlib_path:
        f = test_output_folder / 'junk.tif'
    else:
        f = str(test_output_folder / 'junk.tif')
    export_array(f, arr, mg)

import numpy as np
from ..array_export import make_levels, export_array
from ..grid import MFexportGrid


def test_make_levels():

    array = np.linspace(-5.5, 5.4, 100)
    levels = make_levels(array, 0.2, maxlevels=1000)

    assert levels[0] >= array.min()
    assert levels[-1] <= array.max()

    levels = make_levels(array, 0.002, maxlevels=1000)
    assert len(levels) == 1000


def test_int64_export(tmpdir):
    arr = np.ones((2, 2), dtype=np.int64)
    mg = MFexportGrid(delr=np.ones(2), delc=np.ones(2))
    export_array('{}/junk.tif'.format(tmpdir),
                 arr,
                 mg
                 )

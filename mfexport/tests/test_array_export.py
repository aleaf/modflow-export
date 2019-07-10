import numpy as np
from ..array_export import make_levels


def test_make_levels():

    array = np.linspace(-5.5, 5.4, 100)
    levels = make_levels(array, 0.2, maxlevels=1000)

    assert levels[0] >= array.min()
    assert levels[-1] <= array.max()

    levels = make_levels(array, 0.002, maxlevels=1000)
    assert len(levels) == 1000

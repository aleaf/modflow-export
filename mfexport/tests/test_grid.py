import copy
from .test_model_export import lpr_modelgrid


def test_grid_eq(lpr_modelgrid):
    grid2 = copy.deepcopy(lpr_modelgrid)
    assert lpr_modelgrid == grid2

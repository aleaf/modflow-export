import copy
import numpy as np
from mfexport.grid import get_kij_from_node3d


def test_grid_eq(lpr_modelgrid):
    grid2 = copy.deepcopy(lpr_modelgrid)
    assert lpr_modelgrid == grid2


def test_get_kij_from_node3d():
    nlay, nrow, ncol = 3, 3, 4
    nodes_expected = [(0, (0, 0, 0)),
                      (2, (0, 0, 2)),
                      (ncol, (0, 1, 0)),
                      (nrow * ncol + 2, (1, 0, 2)),
                      (nrow * ncol*2 + 3, (2, 0, 3)),
                      (nrow*ncol*nlay-1, (2, nrow-1, ncol-1)),
                      (nrow * ncol * nlay, (nlay, 0, 0))
                      ]
    for node3d, expected in nodes_expected:
        k, i, j = get_kij_from_node3d(node3d, nrow, ncol)
        assert (k, i, j) == expected
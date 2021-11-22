import os
import numpy as np
from flopy.utils import binaryfile as bf
from mfexport.utils import get_water_table


def get_water_table2(heads, nodata, per_idx=None):
    """
    Nest loop approach to getting a 2D array representing
    the water table elevation for each
    stress period in heads array.

    Parameters
    ----------
    heads : 3 or 4-D np.ndarray
        Heads array.
    nodata : real
        HDRY value indicating dry cells.
    per_idx : int or sequence of ints
        stress periods to return. If None,
        returns all stress periods (default is None).

    Returns
    -------
    wt : 2 or 3-D np.ndarray of water table elevations
        for each stress period.

    """
    heads = np.array(heads, ndmin=4)
    nper, nlay, nrow, ncol = heads.shape
    if per_idx is None:
        per_idx = list(range(nper))
    elif np.isscalar(per_idx):
        per_idx = [per_idx]
    wt = []
    for per in per_idx:
        wt_per = []
        for i in range(nrow):
            for j in range(ncol):
                for k in range(nlay):
                    if heads[per, k, i, j] != nodata:
                        wt_per.append(heads[per, k, i, j])
                        break
                    elif k == nlay - 1:
                        wt_per.append(nodata)
        assert len(wt_per) == nrow * ncol
        wt.append(np.reshape(wt_per, (nrow, ncol)))
    wt = np.squeeze(wt)
    mask = (wt == nodata)
    wt = np.ma.masked_array(wt, mask)
    return wt


def test_get_water_table(model):
    import matplotlib.pyplot as plt
    m, grid, output_path = model
    head_file = os.path.join(m.model_ws,
                             '{}.hds'.format(m.name))
    hdsobj = bf.HeadFile(head_file)
    kstpkper = hdsobj.get_kstpkper()
    heads = hdsobj.get_alldata()

    # nested loop approach (4D heads)
    wt2 = get_water_table2(heads, nodata=1e30)
    # vectorized approach (4D heads)
    wt = get_water_table(heads, nodata=1e30)
    assert np.allclose(wt2, wt)

    wt2_3d = get_water_table2(heads[0], nodata=1e30)
    wt_3d = get_water_table(heads[0], nodata=1e30)
    assert np.allclose(wt2_3d, wt_3d)
    
    # get valid min/max
    heads[:, 0] = 1e30
    wt3 = get_water_table(heads[0], nodata=-9999)
    # having the top layer all invalid 
    # shouldn't make the whole water table invalid
    assert wt3.data.min() < 1e4
    assert not np.all(wt3.mask)


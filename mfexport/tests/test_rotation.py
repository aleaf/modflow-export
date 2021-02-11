import mfexport
import numpy as np

def test_grid_rotation():
    angrot = 18
    m_lirm_grid = mfexport.MFexportGrid(delr=[1309 for i in range(500)],
                                    delc=[348 for i in range(500)],
                                    xul=1954815, yul=219671,
                                    epsg=4456, angrot=angrot)
    assert m_lirm_grid.angrot == angrot

    m_lirm_ll = mfexport.MFexportGrid(delr=[1309 for i in range(500)],
                            delc=[348 for i in range(500)],
                            xoff=2008584, yoff=54187,
                            epsg=4456, angrot=angrot)

    assert np.allclose([m_lirm_grid.xul, m_lirm_grid.yul], [m_lirm_ll.xul, m_lirm_ll.yul])
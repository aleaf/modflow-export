import mfexport

def test_grid_rotation():
    m_lirm_grid = mfexport.MFexportGrid(delr=[100 for i in range(50)],
                                    delc=[100 for i in range(50)],
                                    xul=1954815, yul=219671,
                                    epsg=4456, angrot=18.0)
    assert m_lirm_grid.angrot == 18.0
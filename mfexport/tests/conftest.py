import os
import shutil
import pytest
import flopy.modflow as fm
from ..grid import load_modelgrid


@pytest.fixture(scope="module", autouse=True)
def tmpdir():
    folder = 'mfexport/tests/tmp'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder


@pytest.fixture(scope='module')
def lpr_model():
    namefile = 'lpr_inset.nam'
    model_ws = 'Examples/data/lpr/'
    m = fm.Modflow.load(namefile,
                        model_ws=model_ws, check=False, forgive=False)
    return m


@pytest.fixture(scope='module')
def lpr_modelgrid():
    grid_file = 'Examples/data/lpr/lpr_grid.json'
    return load_modelgrid(grid_file)

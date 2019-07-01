import os
import flopy.modflow as fm
import pytest
from ..grid import load_modelgrid
from ..mfexport import export


@pytest.fixture(scope='module')
def output_path(tmpdir):
    return os.path.join(tmpdir, 'lpr')


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


def test_model_export(lpr_model, lpr_modelgrid, output_path):

     m = lpr_model
     packages = ['dis', 'rch', 'wel']
     for package in packages:
         export(m, lpr_modelgrid, package, output_path=output_path)
         assert True

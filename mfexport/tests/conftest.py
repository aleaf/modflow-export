import os
import shutil
import pytest
import flopy.modflow as fm
import flopy.mf6 as mf6
from ..grid import load_modelgrid


@pytest.fixture(scope="module", autouse=True)
def tmpdir():
    folder = 'mfexport/tests/tmp'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder


@pytest.fixture(scope="session")
def testdatapath():
    """Smaller datasets for faster test execution."""
    return 'mfexport/tests/data'


@pytest.fixture(scope='module')
def lpr_output_path(tmpdir):
    return os.path.join(tmpdir, 'lpr')


@pytest.fixture(scope='module')
def shellmound_output_path(tmpdir):
    return os.path.join(tmpdir, 'shellmound')


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


@pytest.fixture(scope='module')
def shellmound_simulation(testdatapath):
    sim = mf6.MFSimulation.load('mfsim', 'mf6', 'mf6', sim_ws='{}/shellmound'.format(testdatapath))
    return sim


@pytest.fixture(scope='module')
def shellmound_model(shellmound_simulation):
    return shellmound_simulation.get_model('shellmound')


@pytest.fixture(scope='module')
def shellmound_modelgrid():
    grid_file = 'mfexport/tests/data/shellmound/shellmound_grid.json'
    return load_modelgrid(grid_file)


@pytest.fixture(scope='module')
def shellmound(shellmound_model, shellmound_modelgrid, shellmound_output_path):
    return shellmound_model, shellmound_modelgrid, shellmound_output_path


@pytest.fixture(scope='module')
def lpr(lpr_model, lpr_modelgrid, lpr_output_path):
    return lpr_model, lpr_modelgrid, lpr_output_path


# ugly work-around for fixtures not being supported as test parameters yet
# https://github.com/pytest-dev/pytest/issues/349
@pytest.fixture(params=['shellmound',
                        'lpr'])
def model(request,
          shellmound,
          lpr):
    return {'shellmound': shellmound,
            'lpr': lpr}[request.param]
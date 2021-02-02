"""
Tests for zbud.py module
"""
from subprocess import Popen, PIPE
from pathlib import Path
import shutil
import numpy as np
import pytest
from mfexport.zbud import write_zonebudget6_input


@pytest.mark.parametrize('outname', (None, 'shellmound'))
def test_write_zonebudget6_input(shellmound_model, outname, test_output_folder, mf6_exe):

    nlay, nrow, ncol = shellmound_model.dis.botm.array.shape
    zones2d = np.zeros((nrow, ncol), dtype=int)
    zones2d[10:20, 10:20] = 1
    model_ws = Path(shellmound_model.model_ws)
    budgetfile = model_ws / (shellmound_model.name + '.cbc')
    binary_grid_file = model_ws / (shellmound_model.name + '.dis.grb')
    dest_budgetfile = test_output_folder / budgetfile.name
    dest_binary_grid_file = test_output_folder / binary_grid_file.name
    shutil.copy(budgetfile, dest_budgetfile)
    shutil.copy(binary_grid_file, dest_binary_grid_file)
    if outname is not None:
        outname = test_output_folder / outname
    # delete output files
    (test_output_folder / 'shellmound.zbud.nam').unlink(missing_ok=True)
    (test_output_folder / 'shellmound.zbud.nam').unlink(missing_ok=True)
    (test_output_folder / 'external/budget-zones_000.dat').unlink(missing_ok=True)

    # write zonebudget input
    write_zonebudget6_input(zones2d, budgetfile=dest_budgetfile,
                             binary_grid_file=dest_binary_grid_file,
                             outname=outname)
    assert (test_output_folder / 'shellmound.zbud.nam').exists()
    assert (test_output_folder / 'shellmound.zbud.zon').exists()
    assert (test_output_folder / 'external/budget-zones_000.dat').exists()

    # run zonebudget
    process = Popen([str(mf6_exe), 'shellmound.zbud.nam'], cwd=test_output_folder,
                 stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    assert process.returncode == 0

import os
import shutil
import pytest


@pytest.fixture(scope="module", autouse=True)
def tmpdir():
    folder = 'mfexport/tests/tmp'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    return folder

import os
import numpy as np
import fiona
import rasterio
from shapely.geometry import box
import flopy.modflow as fm
import pytest
from ..gis import shp2df
from ..grid import load_modelgrid
from ..mfexport import export, summarize
from .test_results_export import check_files, compare_polygons

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
    outfiles = export(lpr_model, lpr_modelgrid, output_path=output_path)
    # TODO : add some checks
    assert True


def test_package_export(lpr_model, lpr_modelgrid, output_path):
    packages = ['wel']
    outfiles = export(lpr_model, lpr_modelgrid, packages[0], output_path=output_path)
    # TODO : add some checks
    assert True


def test_summary(lpr_model, output_path):
    df = summarize(lpr_model, output_path=output_path)
    # TODO : add some checks
    assert True


def test_package_list_export(lpr_model, lpr_modelgrid, output_path):
    m = lpr_model
    packages = ['dis', 'rch', 'wel']
    variables = ['botm', 'top', 'thickness', 'rech', 'wel']
    nrow, ncol, nlay, nper = lpr_model.nrow_ncol_nlay_nper
    layers = list(range(lpr_model.nlay))
    outfiles = []
    for package in packages:
        outfiles += export(m, lpr_modelgrid, package, output_path=output_path)
    check_files(outfiles, variables, layers=layers)
    tifs = [f for f in outfiles if f.endswith('.tif')]
    for f in tifs:
        with rasterio.open(f) as src:
            assert src.width == ncol
            assert src.height == nrow
            compare_polygons(lpr_modelgrid.bbox, box(*src.bounds))
    shps = [f for f in outfiles if f.endswith('.shp')]
    for f in shps:
        with fiona.open(f) as src:
            assert box(*src.bounds).within(lpr_modelgrid.bbox)


def test_transient_list_export(lpr_model, lpr_modelgrid, output_path):
    m = lpr_model
    outfiles = export(m, lpr_modelgrid, 'wel', output_path=output_path)
    df = m.wel.stress_period_data.get_dataframe(squeeze=True)
    df2 = shp2df(outfiles[0]).drop('geometry', axis=1)
    assert np.all(df == df2)

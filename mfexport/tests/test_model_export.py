import os
import numpy as np
import fiona
import rasterio
from shapely.geometry import box
import pytest
from gisutils import shp2df
from ..list_export import mftransientlist_to_dataframe
from ..mfexport import export, summarize
from .test_results_export import check_files, compare_polygons


def test_model_export(model):
    m, grid, output_path = model
    outfiles = export(m, grid, output_path=output_path)
    # TODO : add some checks
    assert True


def test_package_export(model):
    m, grid, output_path = model
    packages = ['dis'] # 'wel'
    outfiles = export(m, grid, packages[0], output_path=output_path)
    # TODO : add some checks
    assert True


def get_nlay(model):
    if model.version == 'mf6':
        nlay = model.dis.nlay.array
    else:
        nlay = model.dis.nlay
    return nlay


def get_nrow_ncol_nlay_nper(model):
    if model.version == 'mf6':
        nlay = model.dis.nlay.array
        nrow = model.dis.nrow.array
        ncol = model.dis.ncol.array
        nper = model.nper
    else:
        nrow, ncol, nlay, nper = model.nrow_ncol_nlay_nper
    return nrow, ncol, nlay, nper


def test_variable_export(model):
    m, grid, output_path = model
    variables = ['top', 'thickness']
    layers = list(range(get_nlay(m)))
    outfiles = export(m, grid,
                      variables=variables,
                      output_path=output_path)
    check_files(outfiles, variables, layers=layers)


def test_variable_export_with_package(model):
    m, grid, output_path = model
    variables = ['botm']
    packages = ['dis']
    layers = list(range(get_nlay(m)))
    outfiles = export(m, grid,
                      packages=packages,
                      variables=variables,
                      output_path=output_path)
    check_files(outfiles, variables, layers=layers)


def test_summary(model):
    m, grid, output_path = model
    df = summarize(m, output_path=output_path)
    # TODO : add some checks
    assert True


def test_package_list_export(model):
    m, grid, output_path = model
    packages = ['dis', 'rch'] #, 'wel']
    variables = ['botm', 'top', 'thickness', 'idomain', 'rech', 'recharge'] #, 'wel']
    nrow, ncol, nlay, nper = get_nrow_ncol_nlay_nper(m)
    layers = list(range(nlay))
    outfiles = []
    for package in packages:
        outfiles += export(m, grid, package, output_path=output_path)
    check_files(outfiles, variables, layers=layers)
    tifs = [f for f in outfiles if f.endswith('.tif')]
    for f in tifs:
        with rasterio.open(f) as src:
            assert src.width == ncol
            assert src.height == nrow
            compare_polygons(grid.bbox, box(*src.bounds))
    shps = [f for f in outfiles if f.endswith('.shp')]
    for f in shps:
        with fiona.open(f) as src:
            assert box(*src.bounds).within(grid.bbox)


def test_transient_list_export(model):
    m, grid, output_path = model
    outfiles = export(m, grid, 'wel', output_path=output_path)
    variables = ['wel0_stress_period_data']
    if m.version != 'mf6':
        variables = ['wel_stress_period_data']
    check_files(outfiles, variables=variables)
    df = mftransientlist_to_dataframe(m.wel.stress_period_data, squeeze=True)
    df.index = range(len(df))
    if 'cellid' in df.columns:
        df['cellid'] = df['cellid'].astype(str)
    df2 = shp2df(outfiles[0]).drop('geometry', axis=1)
    assert np.allclose(df.drop('cellid', axis=1),
                       df2.drop('cellid', axis=1))

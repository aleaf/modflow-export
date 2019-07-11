import os
import pytest
from flopy.utils import binaryfile as bf
import numpy as np
import fiona
import rasterio
from shapely.geometry import box
from ..grid import load_modelgrid
from ..results import export_cell_budget, export_heads, export_drawdown, export_sfr_results


@pytest.fixture(scope='module')
def output_path(tmpdir):
    return os.path.join(tmpdir, 'lpr')


def check_files(outfiles, variables, kstpkper=None, layers=None):
    replace = [('model_top', 'top')]
    for f in outfiles:
        assert os.path.getsize(f) > 0
        fname = os.path.split(f)[1]
        for pair in replace:
            fname = fname.replace(*pair)
        props = parse_fname(fname)
        assert props['var'] in variables
        if kstpkper is not None:
            assert (props['stp'], props['per']) in kstpkper
        if props['lay'] is not None:
            assert props['lay'] in layers


def parse_fname(fname):
    info = os.path.splitext(fname)[0].split('_')
    props = {'var': info.pop(0),
             'lay': None,
             'per': None,
             'stp': None,
             'suffix': None}
    for i in range(len(info)):
        item = info.pop(0)
        if 'ctr' in item:
            continue
        for p in ['lay', 'per', 'stp']:
            if p in item:
                props[p] = int(item.strip(p))
    return props


def compare_polygons(p1, p2, **kwargs):
    """Check that two polygons have the same extent"""
    assert np.allclose(p1.area, p2.area, **kwargs)
    assert np.allclose(p1.intersection(p2).area, p1.area, **kwargs)


def test_cell_budget_export(lpr_modelgrid, output_path):
    file = 'Examples/data/lpr/lpr_inset.cbc'
    cbobj = bf.CellBudgetFile(file)
    layers = list(range(cbobj.nlay))
    variables = [bs.decode().strip() for bs in cbobj.textlist]
    nrow, ncol = cbobj.nrow, cbobj.ncol
    cbobj.close()
    kstpkper = [(4, 0)]
    outfiles = export_cell_budget(file, lpr_modelgrid,
                                  kstpkper=kstpkper, output_path=output_path)
    check_files(outfiles, variables, kstpkper)
    tifs = [f for f in outfiles if f.endswith('.tif')]
    for f in tifs:
        with rasterio.open(f) as src:
            assert src.width == ncol
            assert src.height == nrow
            compare_polygons(lpr_modelgrid.bbox, box(*src.bounds))


def test_heads_export(lpr_modelgrid, output_path):
    file = 'Examples/data/lpr/lpr_inset.hds'
    kstpkper = [(4, 0)]
    variables = ['hds', 'wt']
    hdsobj = bf.HeadFile(file)
    layers = list(range(hdsobj.nlay))
    nrow, ncol = hdsobj.nrow, hdsobj.ncol
    hdsobj.close()
    outfiles = export_heads(file, lpr_modelgrid, -1e4, -9999,
                 kstpkper=kstpkper,
                 output_path=output_path)
    check_files(outfiles, variables, kstpkper, layers)
    tifs = [f for f in outfiles if f.endswith('.tif')]
    for f in tifs:
        with rasterio.open(f) as src:
            assert src.width == ncol
            assert src.height == nrow
            compare_polygons(lpr_modelgrid.bbox, box(*src.bounds))
    shps = [f for f in outfiles if f.endswith('.shp')]
    for f in shps:
        with fiona.open(f) as src:
            compare_polygons(lpr_modelgrid.bbox, box(*src.bounds), rtol=0.1)


def test_drawdown_export(lpr_modelgrid, output_path):
    file = 'Examples/data/lpr/lpr_inset.hds'
    kstpkper0 = (4, 4)
    kstpkper1 = (4, 8)
    variables = ['ddn', 'wt-ddn']
    hdsobj = bf.HeadFile(file)
    layers = list(range(hdsobj.nlay))
    nrow, ncol = hdsobj.nrow, hdsobj.ncol
    hdsobj.close()
    outfiles = export_drawdown(file, lpr_modelgrid, -1e4, -9999,
                               kstpkper0=kstpkper0,
                               kstpkper1=kstpkper1,
                               output_path=output_path)
    check_files(outfiles, variables, [kstpkper1], layers)
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


def test_sfr_results_export(lpr_model, lpr_modelgrid, output_path):

    mf2005_sfr_outputfile = 'Examples/data/lpr/lpr_inset.sfr.out'
    kstpkper = [(4, 0)]
    variables = ['sfrout', 'baseflow', 'qaquifer']
    outfiles = export_sfr_results(mf2005_sfr_outputfile=mf2005_sfr_outputfile,
                                  model=lpr_model,
                                  grid=lpr_modelgrid,
                                  kstpkper=kstpkper,
                                  output_length_units='feet',
                                  output_time_units='seconds',
                                  output_path=output_path
                                  )
    check_files(outfiles, variables, kstpkper)

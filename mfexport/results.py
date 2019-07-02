import numpy as np
from flopy.utils import binaryfile as bf
from flopy.utils.postprocessing import get_water_table
from .array_export import export_array, export_array_contours
from .utils import make_output_folders

# TODO: update docstrings

def export_cell_budget(cell_budget_file, grid,
                       kstpkper,
                       text='RECHARGE',
                       idx=0, output_path='postproc', suffix=''):
    """Read a flow component from MODFLOW binary cell budget output;
    write to raster.

    Parameters
    ----------
    cell_budget_file : modflow binary cell budget output
    grid : rOpen.modflow.grid instance
    text : cell budget record to read (e.g. 'STREAM LEAKAGE')
    kstpkper : tuple
        (timestep, stress period) to read
    idx : index of list returned by cbbobj (usually 0)
    outfolder : where to write raster
    """

    if len(kstpkper[0]) == 1:
        kstpkper = [kstpkper]

    pdfs_dir, rasters_dir, shps_dir = make_output_folders(output_path)
    if not isinstance(text, list):
        text = [text]

    cbbobj = bf.CellBudgetFile(cell_budget_file)
    names = [r.decode().strip() for r in cbbobj.get_unique_record_names()]
    names = list(set(text).intersection(names))
    if len(names) == 0:
        print('{} not found in {}'.format(' '.join(text), cell_budget_file))

    outfiles = []
    for kstp, kper in kstpkper:
        for variable in names:
            data = get_surface_bc_flux(cbbobj, variable, kstpkper=(kstp, kper), idx=idx)
            if data is None:
                print('{} not exported.'.format(variable))
                return
            outfile = '{}/{}_per{}_stp{}{}.tif'.format(rasters_dir, variable, kper, kstp, suffix)
            export_array(outfile, data, grid, nodata=0)
            outfiles.append(outfile)
    return outfiles


def export_drawdown(heads_file, grid, hdry, hnflo,
                    kstpkper0=None, kstpkper1=None,
                    levels=None, interval=None,
                    output_path='postproc', suffix=''):
    """Export MODFLOW binary head output to rasters and shapefiles.

    Parameters
    ----------
    modelname : str
        model base name
    grid : rOpen.modflow.grid instance
    hdry : hdry value from UPW package
    hnflo : hnflo value from BAS package
    levels : 1D numpy array
        Values of equal interval contours to write.
    shps_outfolder : where to write shapefiles
    rasters_outfolder : where to write rasters

    Writes
    ------
    * A raster of heads for each layer and a raster of the water table elevation
    * Shapefiles of head contours for each layer and the water table.
    """
    if kstpkper1 == (0, 0):
        print('kstpkper == (0, 0, no drawdown to export')
        return
    kstp, kper = kstpkper1

    pdfs_dir, rasters_dir, shps_dir = make_output_folders(output_path)

    # Heads output
    hdsobj = bf.HeadFile(heads_file)
    hds0 = hdsobj.get_data(kstpkper=kstpkper0)
    wt0 = get_water_table(hds0, nodata=hdry)

    hds1 = hdsobj.get_data(kstpkper=kstpkper1)
    wt1 = get_water_table(hds1, nodata=hdry)

    hds0[(hds0 > 9999) & (hds0 < 0)] = np.nan
    hds1[(hds1 > 9999) & (hds1 < 0)] = np.nan

    ddn = hds0 - hds1
    wt_ddn = wt0 - wt1

    outfiles = []
    outfile = '{}/wt-ddn_per{}_stp{}{}.tif'.format(rasters_dir, kper, kstp, suffix)
    ctr_outfile = '{}/wt-ddn_ctr_per{}_stp{}{}.shp'.format(rasters_dir, kper, kstp, suffix)
    export_array(outfile, wt_ddn, grid, nodata=hnflo)
    export_array_contours(ctr_outfile, wt_ddn, grid, levels=levels, interval=interval,
                               )
    outfiles += [outfile, ctr_outfile]

    for k, d in enumerate(ddn):
        outfile = '{}/ddn_lay{}_per{}_stp{}{}.tif'.format(rasters_dir, k, kper, kstp, suffix)
        ctr_outfile = '{}/ddn_ctr{}_per{}_stp{}{}.shp'.format(rasters_dir, k, kper, kstp, suffix)
        export_array(outfile, d, grid, nodata=hnflo)
        export_array_contours(ctr_outfile, d, grid, levels=levels
                                   )
        outfiles += [outfile, ctr_outfile]
    return outfiles


def export_heads(heads_file, grid, hdry, hnflo,
                 kstpkper=(0, 0), levels=None, interval=None,
                 output_path='postproc', suffix=''):
    """Export MODFLOW binary head output to rasters and shapefiles.

    Parameters
    ----------
    modelname : str
        model base name
    grid : rOpen.modflow.grid instance
    hdry : hdry value from UPW package
    hnflo : hnflo value from BAS package
    levels : 1D numpy array
        Values of equal interval contours to write.
    shps_outfolder : where to write shapefiles
    rasters_outfolder : where to write rasters

    Writes
    ------
    * A raster of heads for each layer and a raster of the water table elevation
    * Shapefiles of head contours for each layer and the water table.
    """

    if len(kstpkper[0]) == 1:
        kstpkper = [kstpkper]

    pdfs_dir, rasters_dir, shps_dir = make_output_folders(output_path)

    outfiles = []
    for kstp, kper in kstpkper:
        # Heads output
        hdsobj = bf.HeadFile(heads_file)
        hds = hdsobj.get_data(kstpkper=(kstp, kper))
        wt = get_water_table(hds, nodata=hdry)
        wt[(wt > 9999) | (wt < 0)] = np.nan

        outfile = '{}/wt_per{}_stp{}{}.tif'.format(rasters_dir, kper, kstp, suffix)
        ctr_outfile = '{}/wt_ctr_per{}_stp{}{}.shp'.format(shps_dir, kper, kstp, suffix)
        export_array(outfile, wt, grid, nodata=hnflo)
        export_array_contours(ctr_outfile, wt, grid, levels=levels, interval=interval)
        outfiles += [outfile, ctr_outfile]

        hds[(hds > 9999) | (hds < 0)] = np.nan

        for k, h in enumerate(hds):
            outfile = '{}/hds_lay{}_per{}_stp{}{}.tif'.format(rasters_dir, k, kper, kstp, suffix)
            ctr_outfile = '{}/hds_ctr_lay{}_per{}_stp{}{}.shp'.format(shps_dir, k, kper, kstp, suffix)
            export_array(outfile, h, grid, nodata=hnflo)
            export_array_contours(ctr_outfile, h, grid, levels=levels,#interval=1,
                                  )
            outfiles += [outfile, ctr_outfile]
    return outfiles


def get_surface_bc_flux(cbbobj, txt, kstpkper=(0, 0), idx=0):
    """Read a flow component from MODFLOW binary cell budget output;

    Parameters
    ----------
    cbbobj : open file handle (instance of flopy.utils.binaryfile.CellBudgetFile
    txt : cell budget record to read (e.g. 'STREAM LEAKAGE')
    kstpkper : tuple
        (timestep, stress period) to read
    idx : index of list returned by cbbobj (usually 0)

    Returns
    -------
    arr : ndarray
    """
    nrow, ncol, nlay = cbbobj.nrow, cbbobj.ncol, cbbobj.nlay
    results = cbbobj.get_data(text=txt, kstpkper=kstpkper, idx=idx)
    # this logic needs some cleanup
    if len(results) > 0:
        results = results[0]
    else:
        print('no data found at {} for {}'.format(kstpkper, txt))
        return
    if isinstance(results, list) and txt == 'RECHARGE':
        results = results[1]
    if results.size == 0:
        print('no data found at {} for {}'.format(kstpkper, txt))
        return
    if results.shape == (nlay, nrow, ncol):
        return results
    elif results.shape == (1, nrow, ncol):
        return results[0]
    elif results.shape == (nrow, ncol):
        return results
    elif len(results.shape) == 1 and \
            len({'node', 'q'}.difference(set(results.dtype.names))) == 0:
        arr = np.zeros(nlay * nrow * ncol, dtype=float)
        arr[results.node - 1] = results.q
        arr = np.reshape(arr, (nlay, nrow, ncol))
        arr = arr.sum(axis=0)
        return arr

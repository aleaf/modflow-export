import numpy as np
from flopy.utils import binaryfile as bf
from flopy.utils.postprocessing import get_water_table
from .array_export import export_array, export_array_contours
from .budget_output import get_surface_bc_flux, read_sfr_output, get_flowja_face
from gisutils import shp2df
from .pdf_export import sfr_baseflow_pdf, sfr_qaquifer_pdf
from .shapefile_export import export_shapefile
from .units import (convert_length_units, convert_time_units,
                    get_length_units, get_time_units, get_unit_text)
from .utils import make_output_folders

# TODO: update docstrings


def export_cell_budget(cell_budget_file, grid,
                       binary_grid_file=None,
                       kstpkper=None, text=None, idx=0,
                       precision='single',
                       output_path='postproc', suffix=''):
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
    print('Exporting cell budget info...')
    print('file: {}'.format(cell_budget_file))
    print('binary grid file: {}'.format(binary_grid_file))

    cbbobj = bf.CellBudgetFile(cell_budget_file, precision=precision)
    if kstpkper is None:
        kstpkper = cbbobj.get_times()[idx]
    if np.isscalar(kstpkper[0]):
        kstpkper = [kstpkper]

    pdfs_dir, rasters_dir, shps_dir = make_output_folders(output_path)
    if text is not None and not isinstance(text, list):
        text = [text]

    names = [r.decode().strip() for r in cbbobj.get_unique_record_names()]
    if text is not None:
        names = list(set(text).intersection(names))
    if len(names) == 0:
        print('{} not found in {}'.format(' '.join(text), cell_budget_file))

    outfiles = []
    for kstp, kper in kstpkper:
        print('stress period {}, timestep {}'.format(kper, kstp))
        for variable in names:
            if variable == 'FLOW-JA-FACE':
                df = get_flowja_face(cbbobj, binary_grid_file=binary_grid_file,
                                       kstpkper=(kstp, kper), idx=idx,
                                       precision=precision)
                # export the vertical fluxes as rasters
                # (in the downward direction; so fluxes between 2 layers
                # would be represented in the upper layer)
                if df is not None and 'kn' in df.columns:
                    vflux = df.loc[(df['kn'] < df['km'])]
                    nlay = vflux['km'].max()
                    _, nrow, ncol = grid.shape
                    vflux_array = np.zeros((nlay, nrow, ncol))
                    vflux_array[vflux['kn'].values,
                                vflux['in'].values,
                                vflux['jn'].values] = vflux.q.values
                    data = vflux_array
            else:
                data = get_surface_bc_flux(cbbobj, variable, kstpkper=(kstp, kper), idx=idx)
            if data is None:
                print('{} not exported.'.format(variable))
                continue
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
    print('Exporting drawdown...')
    print('file: {}'.format(heads_file))
    if kstpkper0 is not None:
        print('from stress period {}, timestep {}'.format(*reversed(kstpkper0)))
    if kstpkper1 is not None:
        print('to stress period {}, timestep {}'.format(*reversed(kstpkper1)))
    print('\n')

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
    ctr_outfile = '{}/wt-ddn_ctr_per{}_stp{}{}.shp'.format(shps_dir, kper, kstp, suffix)
    export_array(outfile, wt_ddn, grid, nodata=hnflo)
    export_array_contours(ctr_outfile, wt_ddn, grid, levels=levels, interval=interval,
                               )
    outfiles += [outfile, ctr_outfile]

    for k, d in enumerate(ddn):
        outfile = '{}/ddn_lay{}_per{}_stp{}{}.tif'.format(rasters_dir, k, kper, kstp, suffix)
        ctr_outfile = '{}/ddn_ctr_lay{}_per{}_stp{}{}.shp'.format(shps_dir, k, kper, kstp, suffix)
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
    if np.isscalar(kstpkper[0]):
        kstpkper = [kstpkper]
    print('Exporting heads...')
    print('file: {}'.format(heads_file))

    pdfs_dir, rasters_dir, shps_dir = make_output_folders(output_path)

    outfiles = []
    for kstp, kper in kstpkper:
        print('stress period {}, timestep {}'.format(kper, kstp))
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
            export_array_contours(ctr_outfile, h, grid, levels=levels, interval=interval,
                                  )
            outfiles += [outfile, ctr_outfile]
    return outfiles


def export_sfr_results(mf2005_sfr_outputfile=None,
                       mf2005_SfrFile_instance=None,
                       mf6_sfr_stage_file=None,
                       mf6_sfr_budget_file=None,
                       model=None,
                       grid=None,
                       kstpkper=(0, 0),
                       sfrlinesfile=None,
                       pointsize=0.5,
                       output_length_units='feet',
                       output_time_units='seconds',
                       gis=True, pdfs=True,
                       output_path='postproc', suffix='',
                       verbose=False):

    pdfs_dir, rasters_dir, shps_dir = make_output_folders(output_path)
    m = model
    if not isinstance(kstpkper, list):
        kstpkper = [kstpkper]
    print('Exporting SFR results...')
    for f in [mf2005_sfr_outputfile, mf6_sfr_stage_file, mf6_sfr_budget_file]:
        if f is not None:
            print('file: {}'.format(f))

    df = read_sfr_output(mf2005_sfr_outputfile=mf2005_sfr_outputfile,
                         mf2005_SfrFile_instance=mf2005_SfrFile_instance,
                         mf6_sfr_stage_file=mf6_sfr_stage_file,
                         mf6_sfr_budget_file=mf6_sfr_budget_file,
                         model=model)
    lmult = convert_length_units(get_length_units(m),
                                 output_length_units)
    tmult = convert_time_units(get_time_units(m),
                               output_time_units)
    unit_text = get_unit_text(output_length_units,
                              output_time_units, 3)

    if 'GWF' in df.columns:
        df['Qaquifer'] = -df.GWF # for consistency with MF2005
    if 'Qmean' not in df.columns:
        df['Qmean'] = df[['Qin', 'Qout']].abs().mean(axis=1)

    # write columns in the output units
    df['Qmean_{}'.format(unit_text)] = df.Qmean * lmult**3/tmult
    df['Qaq_{}'.format(unit_text)] = df.Qaquifer * lmult**3/tmult

    # add model top comparison if available
    if m.dis is not None and 'i' in df.columns and 'j' in df.columns:
        df['model_top'] = m.dis.top.array[df.i.values, df.j.values]
    if 'stage' in df.columns:
        df['above'] = df.stage - df.model_top
    groups = df.groupby('kstpkper')

    outfiles = []
    if gis:
        prj_file = None
        if sfrlinesfile is not None:
            sfrlines = shp2df(sfrlinesfile)
            prj_file = sfrlines[:-4] + '.prj'
            sfrlines.sort_values(by=['iseg', 'ireach'], inplace=True)
            geoms = sfrlines.geometry
        else:
            #assert sr is not None, \
            #    'need SpatialReference instance to locate model grid cells'
            #dfp = groups.get_group((0, 0)).copy()
            geoms = None
            #vertices = sr.get_vertices(dfp.i, dfp.j)
            #geoms = [Polygon(vrt) for vrt in vertices]

        for kstp, kper in kstpkper:
            print('stress period {}, timestep {}'.format(kper, kstp))
            dfp = groups.get_group((kstp, kper)).copy()
            if geoms is not None:
                dfp['geometry'] = geoms
            #dfp = gp.GeoDataFrame(dfp)
            #dfp.crs = sr.proj4_str
            # to use cell polygons instead of lines
            # verts = m.sr.get_vertices(df.i.values, df.j.values)
            #df['geometry'] = [Polygon(v) for v in verts]
            dfp['stp'] = [t[0] for t in dfp['kstpkper']]
            dfp['per'] = [t[1] for t in dfp['kstpkper']]
            dfp.drop('kstpkper', axis=1, inplace=True)  # geopandas doesn't like tuples
            outfile = '{}/sfrout_per{}_stp{}{}.shp'.format(shps_dir, kper, kstp, suffix)

            export_shapefile(outfile, dfp, modelgrid=grid, prj=prj_file)
            outfiles.append(outfile)
            #dfp.to_file(outfile)
            #print('wrote {}'.format(outfile))

    if pdfs:
        # need to add a scale that addresses units
        for kstp, kper in kstpkper:
            print('stress period {}, timestep {}'.format(kper, kstp))
            df = groups.get_group((kstp, kper)).copy()
            bf_outfile = '{}/baseflow_per{}_stp{}{}.pdf'.format(pdfs_dir, kper, kstp, suffix)
            sfr_baseflow_pdf(bf_outfile, df, pointsize=pointsize, verbose=verbose)

            qaq_outfile = '{}/qaquifer_per{}_stp{}.pdf'.format(pdfs_dir, kper, kstp, suffix)
            sfr_qaquifer_pdf(qaq_outfile, df, pointsize=pointsize, verbose=verbose)
            outfiles += [bf_outfile, qaq_outfile]
    return outfiles


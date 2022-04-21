import os
from pathlib import Path
import numpy as np
import pandas as pd
import flopy
mf6 = flopy.mf6
from flopy.datbase import DataType, DataInterface
from flopy.discretization import StructuredGrid
from mfexport.array_export import export_array, export_array_contours, squeeze_3d
from mfexport.list_export import mftransientlist_to_dataframe, get_tl_variables
from mfexport.pdf_export import export_pdf, export_pdf_bar_summary
from mfexport.shapefile_export import export_shapefile
from mfexport.utils import get_flopy_package_fname, make_output_folders


othername = {'model_top': 'top'}


def get_package_list(model):
    skip_packages = ['OC']
    if model.version == 'mf6':
        packages = [p.name[0].upper() for p in model.packagelist]
    else:
        packages = model.get_package_list()
    packages = [p for p in packages if p not in skip_packages]
    return packages


def get_variable_list(variables):
    if isinstance(variables, str):
        variables = [variables.lower()]
    else:
        variables = [v.lower() for v in variables]
    return variables


def get_inactive_cells_mask(model):
    if model.version == 'mf6':
        inactive_cells = np.zeros((model.dis.nlay.array, model.dis.nrow.array, model.dis.ncol.array), dtype=bool)
        inactive_cells[model.dis.idomain.array == 0] = True
    else:
        inactive_cells = np.zeros((model.dis.nlay, model.dis.nrow, model.dis.ncol), dtype=bool)
        inactive_cells[model.bas6.ibound.array == 0] = True
    return inactive_cells


def export(model, modelgrid, packages=None, variables=None, output_path='postproc',
           contours=False, include_inactive_cells=False,
           gis=True, pdfs=True, **kwargs):

    pdfs_dir, rasters_dir, shps_dir = make_output_folders(output_path)

    context = 'model'
    if packages is None:
        if 'package' in kwargs:
            packages = [kwargs.pop('package')]
            context = 'packages'
        else:
            packages = get_package_list(model)
    else:
        context = 'packages'

    if not isinstance(packages, list):
        packages = [packages]

    if variables is not None:
        context = 'variables'
        variables = get_variable_list(variables)
    elif 'variable' in kwargs:
        context = 'variables'
        variables = get_variable_list(kwargs.pop('variable'))

    if not isinstance(modelgrid, StructuredGrid):
        raise NotImplementedError('Unstructured grids not supported')

    inactive_cells = get_inactive_cells_mask(model)
    inactive_cells2d = np.all(inactive_cells, axis=0)  # ij locations where all layers are inactive

    filenames = []
    for package in packages:

        if isinstance(package, str):
            package = getattr(model, package)
        package_name = package.name[0]
        print('\n{} package...'.format(package_name))

        if variables is None and 'sfr' in package_name.lower():
            if 'obs' not in package_name.lower():
                export_sfr(package, modelgrid, gis=gis, pdfs=pdfs,
                        shapefile_outfolder=shps_dir, 
                        pdf_outfolder=pdfs_dir,
                        filenames=filenames)
            # TODO: add SFR obs export
            else:
                print('skipping; not implemented')
            

        if model.version == 'mf6':
            if package.name[0].lower() == 'dis':
                variable_context = context == 'variables' and 'thickness' in variables
                if context in ['model', 'packages'] or variable_context:
                    export_thickness(package.top.array, package.botm.array, modelgrid,
                                     filenames, rasters_dir, shps_dir, pdfs_dir,
                                     gis=gis, pdfs=pdfs, contours=contours,
                                     include_inactive_cells=include_inactive_cells,
                                     inactive_cells=inactive_cells,
                                     **kwargs)

        if variables is not None:
            package_variables = [getattr(package, v, None) for v in variables]
            package_variables = [v for v in package_variables if v is not None]
        else:
            package_variables = package.data_list

        for v in package_variables:
            if isinstance(v, DataInterface):                    
                if v.array is not None:
                    if isinstance(v.name, list):
                        name = v.name[0].strip('_')
                    if isinstance(v.name, str):
                        name = v.name.strip('_')

                    if variables is not None and \
                            othername.get(name.lower(), name.lower()) not in variables:
                        return

                    try:
                        export_variable(v, package, modelgrid,
                                        inactive_cells, inactive_cells2d,
                                        filenames, rasters_dir, shps_dir, pdfs_dir,
                                        gis=gis, pdfs=pdfs, contours=contours,
                                        include_inactive_cells=include_inactive_cells,
                                        **kwargs
                                        )
                    except Exception as e:
                        print('skipped, not implemented yet')

    return filenames


def export_variable(variable, package, modelgrid,
                    inactive_cells, inactive_cells2d,
                    filenames, rasters_dir, shps_dir, pdfs_dir,
                    gis=True, pdfs=True, contours=False,
                    include_inactive_cells=False,
                    **kwargs
                    ):

    v = variable

    # cast output folders to Path instances
    pdfs_dir = Path(pdfs_dir)
    rasters_dir = Path(rasters_dir)
    shps_dir = Path(shps_dir)

    if isinstance(v.name, list):
        name = v.name[0].strip('_')
    if isinstance(v.name, str):
        name = v.name.strip('_')

    if v.data_type == DataType.array2d and len(v.array.shape) == 2 \
            and v.array.shape[1] > 0:
        print('{}:'.format(name))
        array = v.array.copy()
        if not include_inactive_cells:
            array = np.ma.masked_array(array, mask=inactive_cells[0])
        if gis:
            filename = os.path.join(rasters_dir, '{}.tif'.format(name))
            export_array(filename, array, modelgrid, nodata=-9999,
                         **kwargs)
            filenames.append(filename)
            if contours:
                filename = os.path.join(shps_dir, '{}_ctr.shp'.format(name))
                export_array_contours(filename, array, modelgrid,
                                      **kwargs)
                filenames.append(filename)

        if pdfs:
            filename = os.path.join(pdfs_dir, '{}.pdf'.format(name))
            export_pdf(filename, array, nodata=np.nan, text=name,
                       mfarray_type='array2d')
            filenames.append(filename)

    elif v.data_type == DataType.array3d:
        # TODO: add option to export 3d arrays as multiband geotiffs
        print('{}:'.format(name))
        array = v.array.copy()
        if not include_inactive_cells:
            array = np.ma.masked_array(array, mask=inactive_cells)
        for k, array2d in enumerate(array):
            if gis:
                filename = os.path.join(rasters_dir, '{}_lay{}.tif'.format(name, k))
                export_array(filename, array2d, modelgrid, nodata=-9999,
                             **kwargs)
                filenames.append(filename)

                if contours:
                    filename = os.path.join(shps_dir, '{}_lay{}.tif'.format(name, k))
                    export_array_contours(filename, array2d, modelgrid,
                                          **kwargs)
                    filenames.append(filename)

            if pdfs:
                filename = os.path.join(pdfs_dir, '{}_lay{}.pdf'.format(name, k))
                export_pdf(filename, array2d, nodata=np.nan,
                           text='Layer {} {}'.format(k, name),
                           mfarray_type='array2d')
                filenames.append(filename)

    elif v.data_type == DataType.transient2d:
        print('{}:'.format(name))
        array = v.array[:, 0, :, :].copy()
        if not include_inactive_cells:
            array = np.ma.masked_array(array,
                                       mask=np.broadcast_to(inactive_cells2d,
                                                            array.shape))

        # before squeezing, make a bar graph of sums along the first axis
        # skip doing this for some variables
        no_bar = {'irch'}
        if name not in no_bar:
            filename = pdfs_dir / f'{name}_summary.pdf'
            export_pdf_bar_summary(filename, array, title=f'{name} summary')
            filenames.append(str(filename))

        # squeeze the array
        # to only include periods where the stress changes
        unique_arrays = squeeze_3d(array)
        for kper, array2d in unique_arrays.items():

            if gis:
                filename = os.path.join(rasters_dir,
                                        '{}_per{}.tif'.format(name, kper))
                export_array(filename, array2d, modelgrid, nodata=-9999,
                             **kwargs)
                filenames.append(filename)

            if pdfs:
                filename = os.path.join(pdfs_dir,
                                        '{}_per{}.pdf'.format(name, kper))
                export_pdf(filename, array2d, nodata=np.nan,
                           text='Period {} {}'.format(kper, name),
                           mfarray_type='array2d')
                filenames.append(filename)

    elif v.data_type == DataType.transient3d:
        # TODO: need test for transient3d
        # squeeze periods to only those that are different
        print('{}:'.format(name))
        pers = [0] + list(np.where(np.diff(v.array.sum(axis=(1, 2, 3))) != 0)[0])
        array = v.array[pers, :, :, :].copy()

        for kper, array3d in enumerate(array):
            if not include_inactive_cells:
                array3d = np.ma.masked_array(array3d,
                                             mask=np.broadcast_to(inactive_cells2d,
                                                                  array3d.shape))
            for k, array2d in enumerate(array3d):
                if gis:
                    filename = os.path.join(rasters_dir,
                                            '{}_per{}_lay{}.tif'.format(name, kper, k))
                    export_array(filename, array2d, modelgrid, nodata=-9999,
                                 **kwargs)
                    filenames.append(filename)

                if pdfs:
                    filename = os.path.join(pdfs_dir,
                                            '{}_per{}_lay{}.pdf'.format(name, kper, k))
                    export_pdf(filename, array2d, nodata=np.nan,
                               text='Period {} Layer {} {}'.format(kper, k, name),
                               mfarray_type='array2d')
                    filenames.append(filename)

    elif v.data_type == DataType.transientlist:
        packagename = package.name[0].lower().replace('_', '')
        if name in {'perioddata'}:
            print(f'skipping {packagename}.perioddata; efficient export not implemented')
            return
        name = '{}_stress_period_data'.format(packagename)
        if gis:
            filename = os.path.join(shps_dir,
                                    '{}.shp'.format(name)).lower()
            export_shapefile(filename, v, modelgrid,
                             squeeze=True, **kwargs)
            filenames.append(filename)

        if pdfs:
            # skip PDF export of head observations for now
            if isinstance(package, mf6.ModflowUtlobs):
                return
            filename = os.path.join(pdfs_dir,
                                    '{}.pdf'.format(name)).lower()
            df = mftransientlist_to_dataframe(v, squeeze=True)
            tl_variables = get_tl_variables(v)

            for tlv in tl_variables:
                print('{}:'.format(tlv))
                data_cols = [c for c in df.columns if tlv in c]
                period_data = any(any(c.isdigit() for c in s) for s in data_cols)
                if period_data:
                    periods = {int(c.strip(tlv)): c for c in data_cols}
                    array = np.zeros((max(list(periods.keys())) + 1,
                                      df['k'].max() + 1,
                                      modelgrid.nrow,
                                      modelgrid.ncol
                                      ))
                    for per, c in periods.items():
                        array[per, df['k'], df['i'], df['j']] = df[c]
                else:
                    print(('Warning, variable: {}\n'.format(tlv) +
                           'Export of non-period data from transientlists not implemented!')
                          )
                    continue

                text = '{}, {}'.format(name, tlv)
                export_pdf(filename, array, text,
                           mfarray_type='transientlist')
                filenames.append(filename)


def export_sfr(package, modelgrid, gis=True, pdfs=False, 
               shapefile_outfolder='.', pdf_outfolder='.',
               filenames=None):
    """Export an SFR package"""
    
    pdfs_dir = Path(pdf_outfolder)
    shps_dir = Path(shapefile_outfolder)
    package_fname = get_flopy_package_fname(package)
    out_shapefile = shps_dir / (package_fname + '.shp')
    out_pdf = out_shapefile.with_suffix('.pdf')
    if filenames is None:
        filenames = []
    
    if package.parent.version == 'mf6':
        df = pd.DataFrame(package.packagedata.array)
        
        # drop reaches that have no k, i, j info
        df = df.loc[df['cellid'] != 'none']
        
        cols = df.columns.tolist()
        # drop cellid column
        cols = cols[:1] + ['k', 'i', 'j'] + cols[2:]
        k, i, j = zip(*df['cellid'])
        df['k'] = np.array(k, dtype=int)
        df['i'] = np.array(i, dtype=int)
        df['j'] = np.array(j, dtype=int)
        df = df[cols].copy()
        
        # add connections
        out_rno = []
        for row in package.connectiondata.array:
            rno = row[0]
            if rno in df['rno']:
                downstream = [c for c in list(row)[1:] if c < 0]
                if any(downstream):
                    out_rno.append(-downstream[0])
                else:
                    out_rno.append(0)
        df['out_rno'] = np.array(out_rno, dtype=int)
        
        if gis:
            export_shapefile(out_shapefile, df, modelgrid)
            filenames.append(out_shapefile)
        if pdfs:
            # PDF export not implemented yet
            pass
        
    else:
        # mf-2005 not implemented yet
        return
    j=2
    return


def export_thickness(top_array, botm_array, modelgrid,
                     filenames, rasters_dir, shps_dir, pdfs_dir,
                     gis=True, pdfs=True, contours=True,
                     include_inactive_cells=True, inactive_cells=None,
                     **kwargs):

    name = 'thickness'
    nlay, nrow, ncol = botm_array.shape
    all_layers = np.zeros((nlay + 1, nrow, ncol), dtype=float)
    all_layers[0] = top_array
    all_layers[1:] = botm_array
    array = -np.diff(all_layers, axis=0)

    if not include_inactive_cells:
        array = np.ma.masked_array(array, mask=inactive_cells)
    for k, array2d in enumerate(array):
        if gis:
            filename = os.path.join(rasters_dir, '{}_lay{}.tif'.format(name, k))
            export_array(filename, array2d, modelgrid, nodata=-9999,
                         **kwargs)
            filenames.append(filename)

            if contours:
                filename = os.path.join(shps_dir, '{}_lay{}.tif'.format(name, k))
                export_array_contours(filename, array2d, modelgrid,
                                      **kwargs)
                filenames.append(filename)

        if pdfs:
            filename = os.path.join(pdfs_dir, '{}_lay{}.pdf'.format(name, k))
            export_pdf(filename, array2d, nodata=np.nan,
                       text='Layer {} {}'.format(k, name),
                       mfarray_type='array2d')
            filenames.append(filename)


def summarize(model, packages=None, variables=None, output_path=None,
              include_inactive_cells=False,
              verbose=False,
              **kwargs):

    print('summarizing {} input...'.format(model.name))

    if packages is None:
        packages = get_package_list(model)

    if not isinstance(packages, list):
        packages = [packages]

    if variables is not None:
        variables = get_variable_list(variables)

    inactive_cells = get_inactive_cells_mask(model)
    inactive_cells2d = np.all(inactive_cells, axis=0)  # ij locations where all layers are inactive

    summarized = []
    for package in packages:

        if isinstance(package, str):
            package = getattr(model, package)
        if verbose:
            print('\n{} package...'.format(package.name[0]))

        if package.name[0].lower() == 'sfr':
            continue

        package_variables = package.data_list

        for v in package_variables:
            if isinstance(v, DataInterface):
                if v.array is not None:

                    if isinstance(v.name, list):
                        name = v.name[0].strip('_')
                    if isinstance(v.name, str):
                        name = v.name.strip('_')

                    if variables is not None and name.lower() not in variables:
                        return

                    try:
                        summarize_variable(v, package,
                                           inactive_cells, inactive_cells2d,
                                           summarized,
                                           include_inactive_cells=include_inactive_cells,
                                           verbose=verbose,
                                           **kwargs
                                           )
                    except Exception as e:
                        print('skipped, not implemented yet')


    df = pd.DataFrame(summarized)
    if output_path is not None:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        df.to_csv('{}/{}_summary.csv'.format(output_path, model.name), index=False)
    return df


def summarize_variable(variable, package, inactive_cells, inactive_cells2d,
                       summarized,
                       include_inactive_cells=False, verbose=False,
                       **kwargs):

    v = variable
    nlay, nrow, ncol = inactive_cells.shape

    if isinstance(v.name, list):
        name = v.name[0].strip('_')
    if isinstance(v.name, str):
        name = v.name.strip('_')

    if v.data_type == DataType.array2d and len(v.array.shape) == 2 \
            and v.array.shape[1] > 0:
        if verbose:
            print('{}'.format(name))
        array = v.array.copy()
        if not include_inactive_cells:
            array = np.ma.masked_array(array, mask=inactive_cells[0])
        summarized.append({'package': package.name[0],
                           'variable': name,
                           'min': array.min(),
                           'mean': array.mean(),
                           'max': array.max(),
                           })
    elif v.data_type == DataType.array3d:
        if verbose:
            print('{}'.format(name))
        array = v.array.copy()
        if not include_inactive_cells:
            array = np.ma.masked_array(array, mask=inactive_cells)
        for k, array2d in enumerate(array):
            summarized.append({'package': package.name[0],
                               'variable': name,
                               'layer': k,
                               'min': array2d.min(),
                               'mean': array2d.mean(),
                               'max': array2d.max(),
                               })

    elif v.data_type == DataType.transient2d:
        array = v.array[:, 0, :, :].copy()
        if not include_inactive_cells:
            array = np.ma.masked_array(array,
                                       mask=np.broadcast_to(inactive_cells2d,
                                                            array.shape))
        for kper, array2d in enumerate(array):
            summarized.append({'package': package.name[0],
                               'variable': name,
                               'period': kper,
                               'min': array2d.min(),
                               'mean': array2d.mean(),
                               'max': array2d.max(),
                               })

    elif v.data_type == DataType.transient3d:
        pers = [0] + list(np.where(np.diff(v.array.sum(axis=(1, 2, 3))) != 0)[0])
        array = v.array[pers, :, :, :].copy()

        for kper, array3d in enumerate(array):
            if not include_inactive_cells:
                array3d = np.ma.masked_array(array3d,
                                             mask=np.broadcast_to(inactive_cells2d,
                                                                  array3d.shape))
            for k, array2d in enumerate(array3d):
                summarized.append({'package': package.name[0],
                                   'variable': name,
                                   'period': kper,
                                   'layer': k,
                                   'min': array2d.min(),
                                   'mean': array2d.mean(),
                                   'max': array2d.max(),
                                   })

    elif v.data_type == DataType.transientlist:
        df = mftransientlist_to_dataframe(v, squeeze=True)
        tl_variables = get_tl_variables(v)
        if isinstance(package, mf6.ModflowUtlobs):
            obstypes = df.obstype.unique()
            for obstype in obstypes:
                n = len(df.loc[df.obstype == obstype])
                summarized.append({'package': package.name[0],
                                   'variable': obstype,
                                   'count': n
                                   })
            return

        for tlv in tl_variables:
            # todo: need to handle different sfr settings separately in summary
            if verbose:
                print('{}'.format(tlv))
            data_cols = [c for c in df.columns if tlv in c]
            periods = {int(c.strip(tlv)): c for c in data_cols}
            try:
                array = np.zeros((max(list(periods.keys())) + 1,
                                  df['k'].max() + 1,
                                  nrow,
                                  ncol
                                  ))
            except:
                j = 2
            for per, c in periods.items():
                array[per, df['k'], df['i'], df['j']] = df[c]
                array[per, df['k'], df['i'], df['j']] = df[c]
                summary = {'package': package.name[0],
                           'variable': tlv,
                           'period': per,
                           'min': df[c].min(),
                           'mean': df[c].mean(),
                           'max': df[c].max(),
                           }
                summarized.append(summary)
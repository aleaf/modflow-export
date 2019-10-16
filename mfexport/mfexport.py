import os
import numpy as np
import pandas as pd
from flopy.datbase import DataType, DataInterface
from flopy.discretization import StructuredGrid
from .array_export import export_array, export_array_contours
from .list_export import mftransientlist_to_dataframe, get_tl_variables
from .pdf_export import export_pdf
from .shapefile_export import export_shapefile
from .utils import make_output_folders


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
        packages = get_package_list(model)
    else:
        context = 'packages'

    if not isinstance(packages, list):
        packages = [packages]

    if variables is not None:
        context = 'variables'
        variables = get_variable_list(variables)

    if not isinstance(modelgrid, StructuredGrid):
        raise NotImplementedError('Unstructured grids not supported')

    inactive_cells = get_inactive_cells_mask(model)
    inactive_cells2d = np.all(inactive_cells, axis=0)  # ij locations where all layers are inactive

    filenames = []
    for package in packages:

        if isinstance(package, str):
            package = getattr(model, package)
        print('\n{} package...'.format(package.name[0]))

        if package.name[0].lower() == 'sfr':
            export_sfr()
            continue

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
                        for kper, array2d in enumerate(array):

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
                        name = '{}_stress_period_data'.format(packagename)
                        if gis:
                            filename = os.path.join(shps_dir,
                                                    '{}.shp'.format(name)).lower()
                            export_shapefile(filename, v, modelgrid,
                                             squeeze=True, **kwargs)
                            filenames.append(filename)

                        if pdfs:
                            filename = os.path.join(pdfs_dir,
                                                    '{}.pdf'.format(name)).lower()
                            df = mftransientlist_to_dataframe(v, squeeze=True)
                            #df['k'], df['i'], df['j'] = list(zip(*df['cellid']))
                            tl_variables = get_tl_variables(v)

                            for tlv in tl_variables:
                                print('{}:'.format(tlv))
                                data_cols = [c for c in df.columns if tlv in c]
                                array = np.zeros((len(data_cols) + 1,
                                                  df['k'].max() + 1,
                                                  modelgrid.nrow + 1,
                                                  modelgrid.ncol + 1))
                                for c in data_cols:
                                    per = int(c.strip(tlv))
                                    array[per, df['k'], df['i'], df['j']] = df[c]
                                text = '{} package {}'.format(name, tlv)
                                export_pdf(filename, array, text,
                                           mfarray_type='transientlist')
                                filenames.append(filename)
    return filenames


def export_sfr():
    """Not implemented yet"""
    print('skipped, not implemented yet')
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
    nlay, nrow, ncol = inactive_cells.shape

    summarized = []
    for package in packages:

        if isinstance(package, str):
            package = getattr(model, package)
        if verbose:
            print('\n{} package...'.format(package.name[0]))

        if package.name[0] == 'SFR':
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
                        #df['k'], df['i'], df['j'] = list(zip(*df['cellid']))
                        tl_variables = get_tl_variables(v)

                        for tlv in tl_variables:
                            if verbose:
                                print('{}'.format(tlv))
                            data_cols = [c for c in df.columns if tlv in c]
                            array = np.zeros((len(data_cols) + 1,
                                              df['k'].max() + 1,
                                              nrow + 1,
                                              ncol + 1))
                            for c in data_cols:
                                per = int(c.strip(tlv))
                                array[per, df['k'], df['i'], df['j']] = df[c]
                                summarized.append({'package': package.name[0],
                                                   'variable': tlv,
                                                   'period': per,
                                                   'min': df[c].min(),
                                                   'mean': df[c].mean(),
                                                   'max': df[c].max(),
                                                   })
    df = pd.DataFrame(summarized)
    if output_path is not None:
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        df.to_csv('{}/{}_summary.csv'.format(output_path, model.name), index=False)
    return df

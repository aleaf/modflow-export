import os
import numpy as np
from flopy.datbase import DataType, DataInterface
from .array_export import export_array, export_array_contours
from .pdf_export import export_pdf
from .shapefile_export import export_shapefile


def export(model, modelgrid, packages=None, variables=None, output_path='postproc',
           contours=False,
           gis=True, pdfs=True, **kwargs):

    pdfs_dir = os.path.join(output_path, 'pdfs')
    rasters_dir = os.path.join(output_path, 'rasters')
    shps_dir = os.path.join(output_path, 'shps')
    for path in [pdfs_dir, shps_dir, rasters_dir]:
        if not os.path.isdir(path):
            print('creating {}...'.format(path))
            os.makedirs(path)

    if packages is None:
        packages = model.get_package_list()
    if not isinstance(packages, list):
        packages = [packages]

    for package in packages:
        if isinstance(package, str):
            package = getattr(model, package)
        print('\n{} package...'.format(package.name[0]))

        if variables is None:
            variables = package.data_list
        elif isinstance(variables, str):
            variables = [getattr(package, variables)]
        elif not isinstance(variables, list):
            variables = [variables]

        for v in variables:
            if isinstance(v, DataInterface):
                if v.array is not None:

                    if isinstance(v.name, list):
                        name = v.name[0].strip('_')
                    if isinstance(v.name, str):
                        name = v.name.strip('_')

                    if v.data_type == DataType.array2d and len(v.shape) == 2 \
                            and v.array.shape[1] > 0:
                        print('{}:'.format(name))
                        if gis:
                            filename = os.path.join(rasters_dir, '{}.tif'.format(name))
                            export_array(filename, v.array, modelgrid, nodata=-9999,
                                         **kwargs)

                            if contours:
                                filename = os.path.join(shps_dir, '{}_ctr.shp'.format(name))
                                export_array_contours(filename, v.array, modelgrid,
                                                      **kwargs)

                        if pdfs:
                            filename = os.path.join(pdfs_dir, '{}.pdf'.format(name))
                            export_pdf(filename, v.array, nodata=np.nan, text=name,
                                       mfarray_type='array2d')

                    elif v.data_type == DataType.array3d:
                        print('{}:'.format(name))
                        for k, array2d in enumerate(v.array):
                            if gis:
                                filename = os.path.join(rasters_dir, '{}_lay{}.tif'.format(name, k))
                                export_array(filename, array2d, modelgrid, nodata=-9999,
                                             **kwargs)

                                if contours:
                                    filename = os.path.join(shps_dir, '{}_lay{}.tif'.format(name, k))
                                    export_array_contours(filename, array2d, modelgrid,
                                                          **kwargs)

                            if pdfs:
                                filename = os.path.join(pdfs_dir, '{}_lay{}.pdf'.format(name, k))
                                export_pdf(filename, array2d, nodata=np.nan,
                                           text='Layer {} {}'.format(k, name),
                                           mfarray_type='array2d')

                    elif v.data_type == DataType.transient2d:
                        print('{}:'.format(name))
                        for kper, array2d in enumerate(v.array[:, 0, :, :]):

                            if gis:
                                filename = os.path.join(rasters_dir,
                                                        '{}_per{}.tif'.format(name, kper))
                                export_array(filename, array2d, modelgrid, nodata=-9999,
                                             **kwargs)

                            if pdfs:
                                filename = os.path.join(pdfs_dir,
                                                        '{}_per{}.tif'.format(name, kper))
                                export_pdf(filename, array2d, nodata=np.nan,
                                           text='Period {} {}'.format(kper, name),
                                           mfarray_type='array2d')

                    elif v.data_type == DataType.transientlist:
                        print('{}:'.format(name))

                        if gis:
                            filename = os.path.join(shps_dir,
                                                    '{}.shp'.format(name))
                            export_shapefile(filename, v, modelgrid,
                                             squeeze=True, **kwargs)

                        if pdfs:
                            filename = os.path.join(shps_dir,
                                                    '{}.pdf'.format(name))
                            df = v.get_dataframe(squeeze=True)
                            tl_variables = list(v.array.keys())

                            for tlv in tl_variables:
                                data_cols = [c for c in df.columns if tlv in c]
                                array = np.zeros((len(data_cols),
                                                  df['k'].max() + 1,
                                                  modelgrid.nrow + 1,
                                                  modelgrid.ncol + 1))
                                for c in data_cols:
                                    per = int(c.strip(tlv))
                                    array[per, df['k'], df['i'], df['j']] = df[c]
                                text = '{} package {}'.format(name, tlv)
                                export_pdf(filename, array, text,
                                           mfarray_type='transientlist')

import time
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from gisutils import df2shp


def export_array(filename, a, modelgrid, nodata=-9999,
                 fieldname='value', verbose=False,
                 **kwargs):
    """
    Write a numpy array to Arc Ascii grid or shapefile with the model
    reference.

    Parameters
    ----------
    modelgrid : MFsetupGrid instance
    filename : str
        Path of output file. Export format is determined by
        file extention.
        '.asc'  Arc Ascii grid
        '.tif'  GeoTIFF (requries rasterio package)
        '.shp'  Shapefile
    a : 2D numpy.ndarray
        Array to export
    nodata : scalar
        Value to assign to np.nan entries (default -9999)
    fieldname : str
        Attribute field name for array values (shapefile export only).
        (default 'values')
    kwargs:
        keyword arguments to np.savetxt (ascii)
        rasterio.open (GeoTIFF)
        or flopy.export.shapefile_utils.write_grid_shapefile2

    Notes
    -----
    Rotated grids will be either be unrotated prior to export,
    using scipy.ndimage.rotate (Arc Ascii format) or rotation will be
    included in their transform property (GeoTiff format). In either case
    the pixels will be displayed in the (unrotated) projected geographic
    coordinate system, so the pixels will no longer align exactly with the
    model grid (as displayed from a shapefile, for example). A key difference
    between Arc Ascii and GeoTiff (besides disk usage) is that the
    unrotated Arc Ascii will have a different grid size, whereas the GeoTiff
    will have the same number of rows and pixels as the original.

    """
    t0 = time.time()
    if filename.lower().endswith(".tif"):
        if len(np.unique(modelgrid.delr)) != len(np.unique(modelgrid.delc)) != 1 \
                or modelgrid.delr[0] != modelgrid.delc[0]:
            raise ValueError('GeoTIFF export require a uniform grid.')

        trans = modelgrid.transform

        # third dimension is the number of bands
        if len(a.shape) == 2:
            a = np.reshape(a, (1, a.shape[0], a.shape[1]))

        if a.dtype == np.int64:
            a = a.astype(np.int32)
        meta = {'count': a.shape[0],
                'width': a.shape[2],
                'height': a.shape[1],
                'nodata': nodata,
                'dtype': a.dtype,
                'driver': 'GTiff',
                'crs': modelgrid.proj_str,
                'transform': trans,
                'compress': 'lzw'
                }
        meta.update(kwargs)
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(a)
            if isinstance(a, np.ma.masked_array):
                dst.write_mask(~a.mask.transpose(1, 2, 0))
        print('wrote {}'.format(filename))

    elif filename.lower().endswith(".shp"):
        raise NotImplementedError()
    if verbose:
        print("array export took {:.2f}s".format(time.time() - t0))


def export_array_contours(filename, a, modelgrid,
                          fieldname='level',
                          interval=None,
                          levels=None,
                          maxlevels=1000,
                          epsg=None,
                          proj_str=None, verbose=False,
                          **kwargs):
    """
    Contour an array using matplotlib; write shapefile of contours.

    Parameters
    ----------
    filename : str
        Path of output file with '.shp' extention.
    a : 2D numpy array
        Array to contour
    epsg : int
        EPSG code. See https://www.epsg-registry.org/ or spatialreference.org
    prj : str
        Existing projection file to be used with new shapefile.
    **kwargs : keyword arguments to matplotlib.axes.Axes.contour

    """
    t0 = time.time()
    if epsg is None:
        epsg = modelgrid.epsg
    if proj_str is None:
        proj_str = modelgrid.proj_str

    if interval is not None:
        kwargs['levels'] = make_levels(a, interval, maxlevels)

    ax = plt.subplots()[-1]
    contours = ax.contour(modelgrid.xcellcenters,
                          modelgrid.ycellcenters,
                          a, **kwargs)
    plt.close()

    if not isinstance(contours, list):
        contours = [contours]

    if epsg is None:
        epsg = modelgrid.epsg
    if proj_str is None:
        proj_str = modelgrid.proj_str

    geoms = []
    level = []
    for ctr in contours:
        levels = ctr.levels
        for i, c in enumerate(ctr.collections):
            paths = c.get_paths()
            geoms += [LineString(p.vertices) if len(p) > 1 else LineString() for p in paths]
            level += list(np.ones(len(paths)) * levels[i])

    # convert the dictionary to a recarray
    df = pd.DataFrame({'level': level,
                       'geometry': geoms})
    df2shp(df, filename, epsg=epsg, proj_str=proj_str)
    if verbose:
        print("array contour export took {:.2f}s".format(time.time() - t0))
    return


def make_levels(array, interval, maxlevels=1000):
    imin = np.round(np.floor(np.nanmin(array)), 0)
    imax = np.round(np.ceil(np.nanmax(array)), 0)
    levels = np.round(np.arange(imin, imax, interval), 6)
    inrange = (levels >= np.nanmin(array)) & (levels <= np.nanmax(array))
    levels = levels[inrange]
    if len(levels) > maxlevels:
        msg = '{:.0f} levels at interval of {}; setting contours based on maxlevels ({})'.format(
            len(levels),
            interval,
            maxlevels)
        print(msg)
        levels = np.round(np.linspace(imin, imax, maxlevels), 6)
    return levels

import time
import numpy as np
import pandas as pd
import rasterio
from rasterio import Affine
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from .gis import df2shp


def export_array(filename, a, modelgrid, nodata=-9999,
                 fieldname='value',
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

        meta = {'count': a.shape[0],
                'width': a.shape[2],
                'height': a.shape[1],
                'nodata': nodata,
                'dtype': a.dtype,
                'driver': 'GTiff',
                'crs': modelgrid.proj_str,
                'transform': trans
                }
        meta.update(kwargs)
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(a)
        print('wrote {}'.format(filename))

    elif filename.lower().endswith(".shp"):
        from flopy.export.shapefile_utils import write_grid_shapefile2
        epsg = kwargs.get('epsg', None)
        prj = kwargs.get('prj', None)
        if epsg is None and prj is None:
            epsg = modelgrid.epsg
        write_grid_shapefile2(filename, modelgrid, array_dict={fieldname: a},
                              nan_val=nodata,
                              epsg=epsg, prj=prj)
    print("took {:.2f}s".format(time.time() - t0))


def contour_array(modelgrid, ax, a, **kwargs):
    """
    Create a QuadMesh plot of the specified array using pcolormesh

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        ax to add the contours

    a : np.ndarray
        array to contour

    Returns
    -------
    contour_set : ContourSet

    """
    try:
        import matplotlib.tri as tri
    except:
        tri = None
    plot_triplot = False
    if 'plot_triplot' in kwargs:
        plot_triplot = kwargs.pop('plot_triplot')
    if 'extent' in kwargs and tri is not None:
        extent = kwargs.pop('extent')
        idx = (modelgrid.xcellcenters >= extent[0]) & (
                modelgrid.xcellcenters <= extent[1]) & (
                      modelgrid.ycellcenters >= extent[2]) & (
                      modelgrid.ycellcenters <= extent[3])
        a = a[idx].flatten()
        xc = modelgrid.xcellcenters[idx].flatten()
        yc = modelgrid.ycellcenters[idx].flatten()
        triang = tri.Triangulation(xc, yc)
        try:
            amask = a.mask
            mask = [False for i in range(triang.triangles.shape[0])]
            for ipos, (n0, n1, n2) in enumerate(triang.triangles):
                if amask[n0] or amask[n1] or amask[n2]:
                    mask[ipos] = True
            triang.set_mask(mask)
        except:
            mask = None
        contour_set = ax.tricontour(triang, a, **kwargs)
        if plot_triplot:
            ax.triplot(triang, color='black', marker='o', lw=0.75)
    else:
        contour_set = ax.contour(modelgrid.xcellcenters, modelgrid.ycellcenters,
                                 a, **kwargs)
    return contour_set


def export_array_contours(filename, a, modelgrid,
                          fieldname='level',
                          interval=None,
                          levels=None,
                          maxlevels=1000,
                          epsg=None,
                          proj_str=None,
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
    **kwargs : keyword arguments to flopy.export.shapefile_utils.recarray2shp

    """
    t0 = time.time()
    if epsg is None:
        epsg = modelgrid.epsg
    if proj_str is None:
        proj_str = modelgrid.proj_str

    if interval is not None:
        imin = np.nanmin(a)
        imax = np.nanmax(a)
        nlevels = np.round(np.abs(imax - imin) / interval, 2)
        msg = '{:.0f} levels at interval of {} > maxlevels={}'.format(
            nlevels,
            interval,
            maxlevels)
        assert nlevels < maxlevels, msg
        levels = np.arange(imin, imax, interval)
    ax = plt.subplots()[-1]
    contours = contour_array(modelgrid, ax, a, levels=levels)
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
            geoms += [LineString(p.vertices) for p in paths]
            level += list(np.ones(len(paths)) * levels[i])

    # convert the dictionary to a recarray
    df = pd.DataFrame({'level': level,
                       'geometry': geoms})
    df2shp(df, filename, epsg=epsg, proj4=proj_str, **kwargs)
    print("took {:.2f}s".format(time.time() - t0))
    return

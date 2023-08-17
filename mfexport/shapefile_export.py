import time
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from flopy.utils import MfList
from flopy.mf6.data.mfdatalist import MFTransientList
from gisutils import df2shp
from mfexport.list_export import mftransientlist_to_dataframe


def export_shapefile(filename, data, modelgrid, kper=None,
                     squeeze=True,
                     epsg=None, proj_str=None, prj=None,
                     verbose=False):
    t0 = time.time()
    if isinstance(data, MFTransientList) or isinstance(data, MfList):
        df = mftransientlist_to_dataframe(data, squeeze=squeeze)
    elif isinstance(data, np.recarray):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("data needs to be a pandas DataFrame, MFList, or numpy recarray")

    if epsg is None:
        epsg = modelgrid.epsg
    if proj_str is None:
        proj_str = modelgrid.proj4

    if 'cellid' in df.columns and isinstance(df['cellid'].values[0], tuple):
        k, i, j = list(zip(*df['cellid']))
        i = np.array(i)
        j = np.array(j)
    elif 'i' in df.columns and 'j' in df.columns:
        i, j = df['i'].values, df['j'].values
    elif 'geometry' not in df.columns:
        raise ValueError('DataFrame needs cellid, (i, j) or geometry'
                         'information to be exported to shapefile.')

    if kper is not None:
        df = df.loc[df.per == kper]
        verts = np.array(modelgrid._cell_vert_list(i, j))
    elif df is not None:
        verts = modelgrid._cell_vert_list(i, j)
    # use cell geometries from the model grid
    if 'geometry' not in df.columns:
        polys = np.array([Polygon(v) for v in verts])
        df['geometry'] = polys
        # unfortunately, reaches through inactive cells 
        # lose their cellid (k, i, j) location
        # so there is no way to plot these 
        # without geometries from another source (such as the sfrlines)
        # drop such geometries, which are identified by k, i, j == -1
        invalid_geoms = np.any(df[['k', 'i', 'j']] < 0, axis=1)
        df = df.loc[~invalid_geoms].copy()
    if epsg is None:
        epsg = modelgrid.epsg
    if proj_str is None:
        proj_str = modelgrid.proj4
    if prj is None:
        prj = modelgrid.prj
    crs = modelgrid.crs
    df2shp(df, filename, epsg=epsg, crs=crs, prj=prj)
    if verbose:
        print("shapefile export took {:.2f}s".format(time.time() - t0))

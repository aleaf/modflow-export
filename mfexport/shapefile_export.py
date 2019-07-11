import time
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from flopy.utils import MfList
from .gis import df2shp


def export_shapefile(filename, data, modelgrid, kper=None,
                     squeeze=True,
                     epsg=None, proj_str=None, prj=None,
                     verbose=False):
    t0 = time.time()
    if isinstance(data, MfList):
        df = data.get_dataframe(squeeze=squeeze)
    elif isinstance(data, np.recarray):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError("data needs to be a pandas DataFrame, MFList, or numpy recarray")

    if epsg is None:
        epsg = modelgrid.epsg
    if proj_str is None:
        proj_str = modelgrid.proj_str

    if kper is not None:
        df = df.loc[df.per == kper]
        verts = np.array(modelgrid.get_cell_vertices(df.i, df.j))
    elif df is not None:
        verts = modelgrid.get_vertices(df.i.values, df.j.values)
    if 'geometry' not in df.columns:
        polys = np.array([Polygon(v) for v in verts])
        df['geometry'] = polys
    if epsg is None:
        epsg = modelgrid.epsg
    if proj_str is None:
        proj_str = modelgrid.proj_str
    if prj is None:
        prj = modelgrid.prj
    df2shp(df, filename, epsg=epsg, proj4=proj_str, prj=prj)
    if verbose:
        print("shapefile export took {:.2f}s".format(time.time() - t0))

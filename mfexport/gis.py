import os
import warnings
import collections
import time
from functools import partial
import fiona
import rasterio
from rasterio.windows import Window
from rasterio import features
from shapely.ops import transform, unary_union
from shapely.geometry import shape, mapping, Polygon
import pyproj
import numpy as np
import pandas as pd


def df2shp(dataframe, shpname, index=False,
           retain_order=False,
           prj=None, epsg=None, proj4=None, crs=None):
    '''
    Write a DataFrame to a shapefile
    dataframe: dataframe to write to shapefile
    index: If true, write out the dataframe index as a column
    retain_order : boolean
        Retain column order in dataframe, using an OrderedDict. Shapefile will
        take about twice as long to write, since OrderedDict output is not
        supported by the pandas DataFrame object.

    --->there are four ways to specify the projection....choose one
    prj: <file>.prj filename (string)
    epsg: EPSG identifier (integer)
    proj4: pyproj style projection string definition
    crs: crs attribute (dictionary) as read by fiona
    '''
    # first check if output path exists
    if os.path.split(shpname)[0] != '' and not os.path.isdir(os.path.split(shpname)[0]):
        raise IOError("Output folder doesn't exist")

    # check for empty dataframe
    if len(dataframe) == 0:
        raise IndexError("DataFrame is empty!")

    df = dataframe.copy()  # make a copy so the supplied dataframe isn't edited

    # assign none for geometry, to write a dbf file from dataframe
    Type = None
    if 'geometry' not in df.columns:
        df['geometry'] = None
        Type = 'None'
        mapped = [None] * len(df)

    # reset the index to integer index to enforce ordering
    # retain index as attribute field if index=True
    df.reset_index(inplace=True, drop=not index)

    # enforce character limit for names! (otherwise fiona marks it zero)
    # somewhat kludgey, but should work for duplicates up to 99
    df.columns = list(map(str, df.columns))  # convert columns to strings in case some are ints
    overtheline = [(i, '{}{}'.format(c[:8], i)) for i, c in enumerate(df.columns) if len(c) > 10]

    newcolumns = list(df.columns)
    for i, c in overtheline:
        newcolumns[i] = c
    df.columns = newcolumns

    properties = shp_properties(df)
    del properties['geometry']

    # set projection (or use a prj file, which must be copied after shp is written)
    # alternatively, provide a crs in dictionary form as read using fiona
    # from a shapefile like fiona.open(inshpfile).crs

    if epsg is not None:
        from fiona.crs import from_epsg
        crs = from_epsg(int(epsg))
    elif proj4 is not None:
        from fiona.crs import from_string
        crs = from_string(proj4)
    elif crs is not None:
        pass
    else:
        pass

    if Type != 'None':
        for g in df.geometry:
            try:
                Type = g.type
            except:
                continue
        mapped = [mapping(g) for g in df.geometry]

    schema = {'geometry': Type, 'properties': properties}
    length = len(df)

    if not retain_order:
        props = df.drop('geometry', axis=1).astype(object).to_dict(orient='records')
    else:
        props = [collections.OrderedDict(r) for i, r in df.drop('geometry', axis=1).astype(object).iterrows()]
    print('writing {}...'.format(shpname))
    with fiona.collection(shpname, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as output:
        for i in range(length):
            output.write({'properties': props[i],
                          'geometry': mapped[i]})

    if prj is not None:
        """
        if 'epsg' in prj.lower():
            epsg = int(prj.split(':')[1])
            prjstr = getPRJwkt(epsg).replace('\n', '') # get rid of any EOL
            ofp = open("{}.prj".format(shpname[:-4]), 'w')
            ofp.write(prjstr)
            ofp.close()
        """
        try:
            print('copying {} --> {}...'.format(prj, "{}.prj".format(shpname[:-4])))
            shutil.copyfile(prj, "{}.prj".format(shpname[:-4]))
        except IOError:
            print('Warning: could not find specified prj file. shp will not be projected.')


def shp2df(shplist, index=None, index_dtype=None, clipto=[], filter=None,
           true_values=None, false_values=None, layer=None,
           skip_empty_geom=True):
    """Read shapefile/DBF, list of shapefiles/DBFs, or File geodatabase (GDB)
     into pandas DataFrame.

    Parameters
    ----------
    shplist : string or list
        of shapefile/DBF name(s) or FileGDB
    index : string
        Column to use as index for dataframe
    index_dtype : dtype
        Enforces a datatype for the index column (for example, if the index field is supposed to be integer
        but pandas reads it as strings, converts to integer)
    clipto : list
        limit what is brought in to items in index of clipto (requires index)
    filter : tuple (xmin, ymin, xmax, ymax)
        bounding box to filter which records are read from the shapefile.
    true_values : list
        same as argument for pandas read_csv
    false_values : list
        same as argument for pandas read_csv
    layer : str
        Layer name to read (if opening FileGDB)
    skip_empty_geom : True/False, default True
        Drops shapefile entries with null geometries.
        DBF files (which specify null geometries in their schema) will still be read.

    Returns
    -------
    df : DataFrame
        with attribute fields as columns; feature geometries are stored as
    shapely geometry objects in the 'geometry' column.
    """
    if isinstance(shplist, str):
        shplist = [shplist]
    if not isinstance(true_values, list) and true_values is not None:
        true_values = [true_values]
    if not isinstance(false_values, list) and false_values is not None:
        false_values = [false_values]
    if len(clipto) > 0 and index:
        clip = True
    else:
        clip = False

    df = pd.DataFrame()
    for shp in shplist:
        print("\nreading {}...".format(shp))
        if not os.path.exists(shp):
            raise IOError("{} doesn't exist".format(shp))
        shp_obj = fiona.open(shp, 'r', layer=layer)

        if index is not None:
            # handle capitolization issues with index field name
            fields = list(shp_obj.schema['properties'].keys())
            index = [f for f in fields if index.lower() == f.lower()][0]

        attributes = []
        # for reading in shapefiles
        meta = shp_obj.meta
        if meta['schema']['geometry'] != 'None':
            if filter is not None:
                print('filtering on bounding box {}, {}, {}, {}...'.format(*filter))
            if clip:  # limit what is brought in to items in index of clipto
                for line in shp_obj.filter(bbox=filter):
                    props = line['properties']
                    if not props[index] in clipto:
                        continue
                    props['geometry'] = line.get('geometry', None)
                    attributes.append(props)
            else:
                for line in shp_obj.filter(bbox=filter):
                    props = line['properties']
                    props['geometry'] = line.get('geometry', None)
                    attributes.append(props)
            print('--> building dataframe... (may take a while for large shapefiles)')
            shp_df = pd.DataFrame(attributes)
            # reorder fields in the DataFrame to match the input shapefile
            if len(attributes) > 0:
                shp_df = shp_df[list(attributes[0].keys())]

            # handle null geometries
            if len(shp_df) == 0:
                print('Empty dataframe! No features were read.')
                if filter is not None:
                    print('Check filter {} for consistency \
with shapefile coordinate system'.format(filter))
            geoms = shp_df.geometry.tolist()
            if geoms.count(None) == 0:
                shp_df['geometry'] = [shape(g) for g in geoms]
            elif skip_empty_geom:
                null_geoms = [i for i, g in enumerate(geoms) if g is None]
                shp_df.drop(null_geoms, axis=0, inplace=True)
                shp_df['geometry'] = [shape(g) for g in shp_df.geometry.tolist()]
            else:
                shp_df['geometry'] = [shape(g) if g is not None else None
                                      for g in geoms]

        # for reading in DBF files (just like shps, but without geometry)
        else:
            if clip:  # limit what is brought in to items in index of clipto
                for line in shp_obj:
                    props = line['properties']
                    if not props[index] in clipto:
                        continue
                    attributes.append(props)
            else:
                for line in shp_obj:
                    attributes.append(line['properties'])
            print('--> building dataframe... (may take a while for large shapefiles)')
            shp_df = pd.DataFrame(attributes)
            # reorder fields in the DataFrame to match the input shapefile
            if len(attributes) > 0:
                shp_df = shp_df[list(attributes[0].keys())]

        shp_obj.close()
        if len(shp_df) == 0:
            continue
        # set the dataframe index from the index column
        if index is not None:
            if index_dtype is not None:
                shp_df[index] = shp_df[index].astype(index_dtype)
            shp_df.index = shp_df[index].values

        df = df.append(shp_df)

        # convert any t/f columns to numpy boolean data
        if true_values is not None or false_values is not None:
            replace_boolean = {}
            for t in true_values:
                replace_boolean[t] = True
            for f in false_values:
                replace_boolean[f] = False

            # only remap columns that have values to be replaced
            cols = [c for c in df.columns if c != 'geometry']
            for c in cols:
                if len(set(replace_boolean.keys()).intersection(set(df[c]))) > 0:
                    df[c] = df[c].map(replace_boolean)

    return df


def shp_properties(df):

    newdtypes = {'bool': 'str',
                 'object': 'str',
                 'datetime64[ns]': 'str'}

    # fiona/OGR doesn't like numpy ints
    # shapefile doesn't support 64 bit ints,
    # but apparently leaving the ints alone is more reliable
    # than intentionally downcasting them to 32 bit
    # pandas is smart enough to figure it out on .to_dict()?
    for c in df.columns:
        if c != 'geometry':
            df[c] = df[c].astype(newdtypes.get(df.dtypes[c].name,
                                               df.dtypes[c].name))
        if 'int' in df.dtypes[c].name:
            if np.max(np.abs(df[c])) > 2**31 -1:
                df[c] = df[c].astype(str)

    # strip dtypes to just 'float', 'int' or 'str'
    def stripandreplace(s):
        return ''.join([i for i in s
                        if not i.isdigit()]).replace('object', 'str')
    dtypes = [stripandreplace(df[c].dtype.name)
              if c != 'geometry'
              else df[c].dtype.name for c in df.columns]
    properties = collections.OrderedDict(list(zip(df.columns, dtypes)))
    return properties


def get_proj4(prj):
    """Get proj4 string for a projection file

    Parameters
    ----------
    prj : string
        Shapefile or projection file

    Returns
    -------
    proj4 string (http://trac.osgeo.org/proj/)

    """
    '''
    Using fiona (couldn't figure out how to do this with just a prj file)
    from fiona.crs import to_string
    c = fiona.open(shp).crs
    proj4 = to_string(c)
    '''
    # using osgeo
    from osgeo import osr

    prjfile = prj[:-4] + '.prj' # allows shp or prj to be argued
    try:
        with open(prjfile) as src:
            prjtext = src.read()
        srs = osr.SpatialReference()
        srs.ImportFromESRI([prjtext])
        proj4 = srs.ExportToProj4()
        return proj4
    except:
        pass

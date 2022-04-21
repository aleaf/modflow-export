import time
import numpy as np
import pandas as pd
from rasterio import Affine
import pyproj
from shapely.geometry import Polygon
from flopy.discretization import StructuredGrid
from gisutils import df2shp
from mfexport.units import convert_length_units
from mfexport.utils import get_input_arguments, load


class MFexportGrid(StructuredGrid):

    def __init__(self, delc, delr, top=None, botm=None, idomain=None,
                 laycbd=None, lenuni=None,
                 epsg=None, proj_str=None, prj=None, wkt=None, crs=None,
                 xoff=0.0, yoff=0.0, xul=None, yul=None, angrot=0.0):
        super(MFexportGrid, self).__init__(delc=np.array(delc), delr=np.array(delr),
                                          top=top, botm=botm, idomain=idomain,
                                          laycbd=laycbd, lenuni=lenuni,
                                          epsg=epsg, proj4=proj_str, prj=prj,
                                          xoff=xoff, yoff=yoff, angrot=angrot
                                          )

        # properties
        self._crs = None
        # pass all CRS representations through pyproj.CRS.from_user_input
        # to convert to pyproj.CRS instance
        self.crs = get_crs(crs=crs, epsg=epsg, prj=prj, wkt=wkt, proj_str=proj_str)

        # other CRS-related properties are set in the flopy Grid base class
        self._vertices = None
        self._polygons = None
        self._dataframe = None

        # if no epsg, set from proj4 string if possible
        #if epsg is None and proj_str is not None and 'epsg' in proj_str.lower():
        #    self.epsg = int(proj_str.split(':')[1])

        # in case the upper left corner is known but the lower left corner is not
        if xul is not None and yul is not None:
            xll = self._xul_to_xll(xul)
            yll = self._yul_to_yll(yul)
            self.set_coord_info(xoff=xll, yoff=yll, epsg=epsg, proj4=proj_str, angrot=angrot)

    def __eq__(self, other):
        if not isinstance(other, StructuredGrid):
            return False
        if not np.allclose(other.xoffset, self.xoffset):
            return False
        if not np.allclose(other.yoffset, self.yoffset):
            return False
        if not np.allclose(other.angrot, self.angrot):
            return False
        if not other.crs == self.crs:
            return False
        if not np.array_equal(other.delr, self.delr):
            return False
        if not np.array_equal(other.delc, self.delc):
            return False
        return True

    def __repr__(self):
        txt = ''
        if self.nlay is not None:
            txt += f'{self.nlay:d} layer(s), '
        txt += f'{self.nrow:d} row(s), {self.ncol:d} column(s)\n'
        txt += (f'delr: [{self.delr[0]:.2f}...{self.delr[-1]:.2f}]'
                f' {self.units}\n'
                f'delc: [{self.delc[0]:.2f}...{self.delc[-1]:.2f}]'
                f' {self.units}\n'
                )
        txt += f'CRS: {self.crs}\n'
        txt += f'length units: {self.length_units}\n'
        txt += f'xll: {self.xoffset}; yll: {self.yoffset}; rotation: {self.rotation}\n'
        txt += 'Bounds: {}\n'.format(self.extent)
        return txt

    def __str__(self):
        return StructuredGrid.__repr__(self)

    @property
    def xul(self):
        x0 = self.xyedges[0][0]
        y0 = self.xyedges[1][0]
        x0r, y0r = self.get_coords(x0, y0)
        return x0r

    @property
    def yul(self):
        x0 = self.xyedges[0][0]
        y0 = self.xyedges[1][0]
        x0r, y0r = self.get_coords(x0, y0)
        return y0r

    @property
    def bbox(self):
        """Shapely polygon bounding box of the model grid."""
        return get_grid_bounding_box(self)

    @property
    def bounds(self):
        """Grid bounding box in order used by shapely.
        """
        x0, x1, y0, y1 = self.extent
        return x0, y0, x1, y1

    @property
    def size(self):
        if self.nlay is None:
            return self.nrow * self.ncol
        return self.nlay * self.nrow * self.ncol

    @property
    def transform(self):
        """Rasterio Affine object (same as transform attribute of rasters).
        """
        return Affine(self.delr[0], 0., self.xul,
                      0., -self.delc[0], self.yul) * \
               Affine.rotation(-self.angrot)

    @property
    def crs(self):
        """pyproj.crs.CRS instance describing the coordinate reference system
        for the model grid.
        """
        return self._crs

    @crs.setter
    def crs(self, crs):
        """Get a pyproj CRS instance from various inputs
        (epsg, proj string, wkt, etc.).

        crs : obj, optional
            Coordinate reference system for model grid.
            A Python int, dict, str, or pyproj.crs.CRS instance
            passed to the pyproj.crs.from_user_input
            See http://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_user_input.
            Can be any of:
              - PROJ string
              - Dictionary of PROJ parameters
              - PROJ keyword arguments for parameters
              - JSON string with PROJ parameters
              - CRS WKT string
              - An authority string [i.e. 'epsg:4326']
              - An EPSG integer code [i.e. 4326]
              - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
              - An object with a `to_wkt` method.
              - A :class:`pyproj.crs.CRS` class
        """
        crs = get_crs(crs=crs)
        self._crs = crs

    @property
    def epsg(self):
        return self.crs.to_epsg()

    @property
    def proj_str(self):
        return self.crs.to_proj4()

    @property
    def wkt(self):
        return self.crs.to_wkt(pretty=True)

    @property
    def length_units(self):
        return get_crs_length_units(self.crs)

    @property
    def vertices(self):
        """Vertices for grid cell polygons."""
        if self._vertices is None:
            self._set_vertices()
        return self._vertices

    @property
    def polygons(self):
        """Vertices for grid cell polygons."""
        if self._polygons is None:
            self._set_polygons()
        return self._polygons

    @property
    def dataframe(self):
        """Pandas DataFrame of grid cell polygons
        with i, j locations."""
        if self._dataframe is None:
            self._dataframe = self.get_dataframe(layers=True)
        return self._dataframe

    def get_dataframe(self, layers=True):
        """Get a pandas DataFrame of grid cell polygons
        with i, j locations.

        Parameters
        ----------
        layers : bool
            If True, return a row for each k, i, j location
            and a 'k' column; if False, only return i, j
            locations with no 'k' column. By default, True

        Returns
        -------
        layers : DataFrame
            Pandas Dataframe with k, i, j and geometry column
            with a shapely polygon representation of each model cell.
        """
        # get dataframe of model grid cells
        i, j = np.indices((self.nrow, self.ncol))
        geoms = self.polygons
        df = gpd.GeoDataFrame({'i': i.ravel(),
                              'j': j.ravel(),
                              'geometry': geoms}, crs=5070)
        if layers and self.nlay is not None:
            # add layer information
            dfs = []
            for k in range(self.nlay):
                layer_df = df.copy()
                layer_df['k'] = k
                dfs.append(layer_df)
            df = pd.concat(dfs)
            df = df[['k', 'i', 'j', 'geometry']].copy()
        return df

    def write_bbox_shapefile(self, filename='grid_bbox.shp'):
        write_bbox_shapefile(self, filename)

    def write_shapefile(self, filename='grid.shp'):
        i, j = np.indices((self.nrow, self.ncol))
        df = pd.DataFrame({'node': list(range(len(self.polygons))),
                           'i': i.ravel(),
                           'j': j.ravel(),
                           'geometry': self.polygons
                           })
        df2shp(df, filename, epsg=self.epsg, proj_str=self.proj_str)

    def _set_polygons(self):
        """
        Create shapely polygon for each grid cell
        """
        print('creating shapely Polygons of grid cells...')
        t0 = time.time()
        self._polygons = [Polygon(verts) for verts in self.vertices]
        print("finished in {:.2f}s\n".format(time.time() - t0))

    # stuff to conform to sr
    @property
    def length_multiplier(self):
        return convert_length_units(self.lenuni,
                                    2)

    @property
    def rotation(self):
        return self.angrot

    def get_vertices(self, i, j):
        """Get vertices for a single cell or sequence if i, j locations."""
        return self._cell_vert_list(i, j)

    def _set_vertices(self):
        """
        Populate vertices for the whole grid
        """
        jj, ii = np.meshgrid(range(self.ncol), range(self.nrow))
        jj, ii = jj.ravel(), ii.ravel()
        self._vertices = self._cell_vert_list(ii, jj)


def get_grid_bounding_box(modelgrid):
    """Get bounding box of potentially rotated modelgrid
    as a shapely Polygon object.

    Parameters
    ----------
    modelgrid : flopy.discretization.StructuredGrid instance
    """
    mg = modelgrid
    #x0 = mg.xedge[0]
    #x1 = mg.xedge[-1]
    #y0 = mg.yedge[0]
    #y1 = mg.yedge[-1]

    x0 = mg.xyedges[0][0]
    x1 = mg.xyedges[0][-1]
    y0 = mg.xyedges[1][0]
    y1 = mg.xyedges[1][-1]

    # upper left point
    #x0r, y0r = mg.transform(x0, y0)
    x0r, y0r = mg.get_coords(x0, y0)

    # upper right point
    #x1r, y1r = mg.transform(x1, y0)
    x1r, y1r = mg.get_coords(x1, y0)

    # lower right point
    #x2r, y2r = mg.transform(x1, y1)
    x2r, y2r = mg.get_coords(x1, y1)

    # lower left point
    #x3r, y3r = mg.transform(x0, y1)
    x3r, y3r = mg.get_coords(x0, y1)

    return Polygon([(x0r, y0r),
                    (x1r, y1r),
                    (x2r, y2r),
                    (x3r, y3r),
                    (x0r, y0r)])


def load_modelgrid(filename):
    """Create a MFsetupGrid instance from model config json file."""
    cfg = load(filename)
    rename = {'xll': 'xoff',
              'yll': 'yoff',
              }
    for k, v in rename.items():
        if k in cfg:
            cfg[v] = cfg.pop(k)
    if np.isscalar(cfg['delr']):
        cfg['delr'] = np.ones(cfg['ncol'])* cfg['delr']
    if np.isscalar(cfg['delc']):
        cfg['delc'] = np.ones(cfg['nrow']) * cfg['delc']
    kwargs = get_input_arguments(cfg, MFexportGrid)
    return MFexportGrid(**kwargs)


def get_kij_from_node3d(node3d, nrow, ncol):
    """For a consecutive cell number in row-major order
    (row, column, layer), get the zero-based row, column position.
    """
    node2d = node3d % (nrow * ncol)
    k = node3d // (nrow * ncol)
    i = node2d // ncol
    j = node2d % ncol
    return k, i, j


def get_crs(crs=None, epsg=None, prj=None, wkt=None, proj_str=None):
    """Get a pyproj CRS instance from various CRS representations.
    """
    if crs is not None:
        crs = pyproj.CRS.from_user_input(crs)
    elif epsg is not None:
        crs = pyproj.CRS.from_epsg(epsg)
    elif prj is not None:
        with open(prj) as src:
            wkt = src.read()
            crs = pyproj.CRS.from_wkt(wkt)
    elif wkt is not None:
        crs = pyproj.CRS.from_wkt(wkt)
    elif proj_str is not None:
        crs = pyproj.CRS.from_string(proj_str)
    else: # crs is None
        return
    # if possible, have pyproj try to find the closest
    # authority name and code matching the crs
    # so that input from epsg codes, proj strings, and prjfiles
    # results in equal pyproj_crs instances
    authority = crs.to_authority()
    if authority is not None:
        crs = pyproj.CRS.from_user_input(crs.to_authority())
    return crs


def get_crs_length_units(crs):
    length_units = crs.axis_info[0].unit_name
    if 'foot' in length_units.lower() or 'feet' in length_units.lower():
        length_units = 'feet'
    elif 'metre' in length_units.lower() or 'meter' in length_units.lower():
        length_units = 'meters'
    return length_units


def write_bbox_shapefile(modelgrid, outshp):
    outline = get_grid_bounding_box(modelgrid)
    df2shp(pd.DataFrame({'desc': ['model bounding box'],
                         'geometry': [outline]}),
           outshp, epsg=modelgrid.epsg)
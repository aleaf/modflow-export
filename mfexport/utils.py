import os
import inspect
from pathlib import Path
import pprint
import yaml
import json
import numpy as np


def get_input_arguments(kwargs, function, warn=True):
    """Return subset of keyword arguments in kwargs dict
    that are valid parameters to a function or method.

    Parameters
    ----------
    kwargs : dict (parameter names, values)
    function : function of class method

    Returns
    -------
    input_kwargs : dict
    """
    np.set_printoptions(threshold=20)
    print('\narguments to {}:'.format(function.__qualname__))
    params = inspect.signature(function)
    input_kwargs = {}
    not_arguments = {}
    for k, v in kwargs.items():
        if k in params.parameters:
            input_kwargs[k] = v
            print_item(k, v)
        else:
            not_arguments[k] = v
    if warn:
        print('\nother arguments:')
        for k, v in not_arguments.items():
            #print('{}: {}'.format(k, v))
            print_item(k, v)
    print('\n')
    return input_kwargs


def load(filename):
    """Load a configuration file."""
    if str(filename).endswith('.yml') or str(filename).endswith('.yaml'):
        return load_yaml(filename)
    elif str(filename).endswith('.json'):
        return load_json(filename)


def load_json(jsonfile):
    """Convenience function to load a json file; replacing
    some escaped characters."""
    with open(jsonfile) as f:
        return json.load(f)


def load_yaml(yml_file):
    """Load yaml file into a dictionary."""
    with open(yml_file) as src:
        cfg = yaml.load(src, Loader=yaml.Loader)
    return cfg


def make_output_folders(output_path='postproc'):
    output_path = Path(output_path)
    pdfs_dir = output_path / 'pdfs'
    rasters_dir = output_path / 'rasters'
    shps_dir = output_path / 'shps'
    for path in [pdfs_dir, shps_dir, rasters_dir]:
        if not path.is_dir():
            print('creating {}...'.format(path))
            path.mkdir(parents=True)
    return pdfs_dir, rasters_dir, shps_dir


def print_item(k, v):
    print('{}: '.format(k), end='')
    if isinstance(v, dict):
        #print(json.dumps(v, indent=4))
        pprint.pprint(v)
    elif isinstance(v, list):
        pprint.pprint(v)
    else:
        print(v)


def get_water_table(heads, nodata, 
                     valid_min=-1e-4, valid_max=3e4):
    """
    Get a 2D array representing
    the water table elevation for each
    stress period in heads array.

    Parameters
    ----------
    heads : 3 or 4-D np.ndarray
        Heads array.
    nodata : real
        HDRY value indicating dry cells.
    valid_min : float (optional)
        The lowest value regarded as valid, regardless of nodata value.
        By default, -1e4
    valid_max : float (optional)
        The highest value regarded as valid, regardless of nodata value.
        By default, 3e4

    Returns
    -------
    wt : 2 or 3-D np.ndarray of water table elevations
        for each stress period.

    """
    heads = np.array(heads, ndmin=4)
    mask = (heads == nodata) | (heads < valid_min) | (heads > valid_max)
    k = (~mask).argmax(axis=1)
    per, i, j = np.indices(k.shape)
    wt = heads[per.ravel(), k.ravel(), i.ravel(), j.ravel()].reshape(k.shape)
    wt = np.squeeze(wt)
    mask = (wt == nodata) | (wt < valid_min) | (wt > valid_max)
    wt = np.ma.masked_array(wt, mask)
    return wt


def get_flopy_package_fname(package):
    if getattr(package, 'filename', None) is not None:
        return getattr(package, 'filename')
    elif getattr(package, 'file_name') is not None:
        file_name = getattr(package, 'file_name', None)
        return file_name[0]
    elif getattr(package, 'fn_path', None) is not None:
        fn_path = getattr(package, 'fn_path', None)
        return Path(fn_path).name
    else:
        raise AttributeError(f"Can't get filename for package:\n{package}")
import os
import inspect
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
    if filename.endswith('.yml') or filename.endswith('.yaml'):
        return load_yaml(filename)
    elif filename.endswith('.json'):
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
    pdfs_dir = os.path.join(output_path, 'pdfs')
    rasters_dir = os.path.join(output_path, 'rasters')
    shps_dir = os.path.join(output_path, 'shps')
    for path in [pdfs_dir, shps_dir, rasters_dir]:
        if not os.path.isdir(path):
            print('creating {}...'.format(path))
            os.makedirs(path)
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


def get_water_table(heads, nodata):
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

    Returns
    -------
    wt : 2 or 3-D np.ndarray of water table elevations
        for each stress period.

    """
    heads = np.array(heads, ndmin=4)

    k = (heads != nodata).argmax(axis=1)
    per, i, j = np.indices(k.shape)
    wt = heads[per.ravel(), k.ravel(), i.ravel(), j.ravel()].reshape(k.shape)
    wt = np.squeeze(wt)
    wt = np.ma.masked_array(wt, wt==nodata)
    return wt
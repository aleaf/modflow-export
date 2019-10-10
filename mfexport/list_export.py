import pandas as pd
import flopy


def mftransientlist_to_dataframe(mftransientlist, squeeze=True):
    """
    Cast a MFTransientList of stress period data
    into single dataframe containing all stress periods. Output data are
    aggregated (summed) to the model cell level, to avoid
    issues with non-unique row indices.

    Parameters
    ----------
    mftransientlist : flopy.mf6.data.mfdatalist.MFTransientList instance
    squeeze : bool
        Reduce number of columns in dataframe to only include
        stress periods where a variable changes.

    Returns
    -------
    df : dataframe
        Dataframe of shape nrow = ncells, ncol = nvar x nper. If
        the squeeze option is chosen, nper is the number of
        stress periods where at least one cell is different,
        otherwise it is equal to the number of keys in MfList.data.
    """

    data = mftransientlist
    names = ['cellid']
    if isinstance(data.package, flopy.mf6.modflow.ModflowGwfmaw):
        names += ['wellid']

    # monkey patch the mf6 version to behave like the mf2005 version
    if isinstance(mftransientlist, flopy.mf6.data.mfdatalist.MFTransientList):
        mftransientlist.data = {per: ra for per, ra in enumerate(mftransientlist.array)}

    # find relevant variable names
    # may have to iterate over the first stress period
    for per in range(data.model.nper):
        if hasattr(data.data[per], 'dtype'):
            varnames = list([n for n in data.data[per].dtype.names
                             if n not in ['k', 'i', 'j', 'cellid']])
            break

    # create list of dataframes for each stress period
    # each with index of k, i, j
    dfs = []
    for per, recs in data.data.items():

        if recs is None or recs is 0:
            # add an empty dataframe if a stress period is
            # set to 0 (e.g. no pumping during a predevelopment
            # period)
            columns = names + list(['{}{}'.format(c, per)
                                    for c in varnames])
            dfi = pd.DataFrame(data=None, columns=columns)
            dfi = dfi.set_index(names)
        else:
            dfi = pd.DataFrame.from_records(recs)
            if {'k', 'i', 'j'}.issubset(dfi.columns):
                dfi['cellid'] = list(zip(dfi.k, dfi.i, dfi.j))
                dfi.drop(['k', 'i', 'j'], axis=1, inplace=True)
            dfi = dfi.set_index(names)

            # aggregate (sum) data to model cells
            # because pd.concat can't handle a non-unique index
            # (and modflow input doesn't have a unique identifier at sub-cell level)
            dfg = dfi.groupby(names)
            dfi = dfg.sum()  # aggregate
            #dfi.columns = names + list(['{}{}'.format(c, per) for c in varnames])
            dfi.columns = ['{}{}'.format(c, per) if c in varnames else c for c in dfi.columns]
        dfs.append(dfi)
    df = pd.concat(dfs, axis=1)
    if squeeze:
        keep = []
        for var in varnames:
            diffcols = list([n for n in df.columns if var in n])
            squeezed = squeeze_columns(df[diffcols])
            keep.append(squeezed)
        df = pd.concat(keep, axis=1)
    data_cols = df.columns.tolist()
    df['cellid'] = df.index.tolist()
    idx_cols = ['cellid']
    if isinstance(df.index.values[0], tuple):
        df['k'], df['i'], df['j'] = list(zip(*df['cellid']))
        idx_cols += ['k', 'i', 'j']
    cols = idx_cols + data_cols
    df = df[cols]
    return df


def squeeze_columns(df, fillna=0.):
    """Drop columns where the forward difference
    (along axis 1, the column axis) is 0 in all rows.
    In other words, only retain columns where the data
    changed in at least one row.

    Parameters
    ----------
    df : DataFrame
        Containing homogenous data to be differenced (e.g.,
        just flux values, no id or other ancillary information)
    fillna : float
        Value for nan values in DataFrame
    Returns
    -------
    squeezed : DataFrame

    """
    df.fillna(fillna, inplace=True)
    diff = df.diff(axis=1)
    diff[diff.columns[0]] = 1  # always return the first stress period
    changed = diff.sum(axis=0) != 0
    squeezed = df.loc[:, changed.index[changed]]
    return squeezed


def get_tl_variables(mftransientlist):
    """Get variable names in a flopy.utils.MFList or
    flopy.mf6.data.mfdatalist.MFTransientList instance
    """
    # monkey patch the mf6 version to behave like the mf2005 version
    if isinstance(mftransientlist, flopy.mf6.data.mfdatalist.MFTransientList):
        mftransientlist.data = {per: ra for per, ra in enumerate(mftransientlist.array)}

    for per, recarray in mftransientlist.data.items():
        if recarray is not None:
            return [c for c in recarray.dtype.names
                    if c not in ['k', 'i', 'j', 'cellid']]

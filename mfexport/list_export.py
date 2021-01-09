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
    #if isinstance(mftransientlist, flopy.mf6.data.mfdatalist.MFTransientList):
    #    mftransientlist.data = {per: ra for per, ra in enumerate(mftransientlist.array)}

    # find relevant variable names
    # may have to iterate over the first stress period
    varnames = []
    for per in range(data.model.nper):
        if hasattr(data.data.get(per), 'dtype'):
            varnames = list([n for n in data.data[per].dtype.names
                             if n not in ['k', 'i', 'j', 'cellid',
                                          'rno', 'sfrsetting']])
            break

    # create list of dataframes for each stress period
    # each with index of k, i, j
    dfs = []
    reconvert_str_index = False
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
            # convert layer, row, column to cellid
            index_col = 'cellid'  # default index
            if {'k', 'i', 'j'}.issubset(dfi.columns):
                dfi['cellid'] = list(zip(dfi.k, dfi.i, dfi.j))
                dfi.drop(['k', 'i', 'j'], axis=1, inplace=True)
            # cell-by-cell connections; id is the cellid (id2 cellid of connected cell)
            elif 'id' in dfi.columns and 'cellid' not in dfi.columns:
                index_col = 'id'
            # map the cellid to the reach number (SFR package data)
            elif 'rno' in dfi.columns and 'cellid' not in dfi.columns:
                packagedata = data.package.packagedata
                cellid = dict(zip(packagedata.array['rno'], packagedata.array['cellid']))
                dfi['cellid'] = [cellid[rno] for rno in dfi['rno']]
                cols = ['rno', 'cellid']
                # rearrange the column order to start with rno, cellid
                cols = cols + [c for c in dfi.columns if c not in cols]
                dfi = dfi[cols]
                # index on reach number, allowing for multiple instances of a cellid
                # (multiple reaches per cell)
                index_col = 'rno'
            # cast tuple cellids to strings
            # so that pd.concat works in pandas >=1.2
            if 'cellid' in dfi.columns:
                dfi['cellid'] = dfi['cellid'].astype(str)
                # flag to convert string index back to tuples
                reconvert_str_index = True

            dfi.set_index(index_col, drop=False, inplace=True)

            # aggregate (sum) data to model cells
            # because pd.concat can't handle a non-unique index
            # (and modflow input doesn't have a unique identifier at sub-cell level)
            if dfi.index.name != 'rno':
                try:
                    dfg = dfi.reset_index(drop=True).groupby(index_col)
                except:
                    j=2
                dfi = dfg.sum()  # aggregate
            dfi.columns = ['{}{}'.format(c, per) if c in varnames else c for c in dfi.columns]
        dfs.append(dfi)
    df = pd.concat(dfs, axis=1)
    # squeeze the dataframe down to the minimum number of columns (stress periods)
    # to describe changes in stress
    # keep only columns where the stress changes
    # (assuming that missing columns represent the same stress as the previous column)
    # squeeze only the columns with data values
    if squeeze and len(varnames) > 0:
        keep = []
        for var in varnames:
            diffcols = list([n for n in df.columns if var in n])
            if len(diffcols) > 0:
                to_squeeze = df[diffcols].T.astype(float).T
                squeezed = squeeze_columns(to_squeeze)
                keep.append(squeezed)
        squeezed = pd.concat(keep, axis=1)
        squeezed.index = df.index.tolist()
        # join the squeezed data back to other columns
        other_cols = []
        for c in df.columns:
            name = ''.join((char for char in c if not char.isdigit()))
            if name not in varnames:
                other_cols.append(name)

        if len(other_cols) > 0:
            try:
                df = df[other_cols].join(squeezed)
            except:
                j=2
        else:
            df = squeezed
    # add columns for k, i, j
    if reconvert_str_index:
        df.index = [eval(s) for s in df.index]
    for id in ['cellid', 'id']:
        if id not in df.columns and isinstance(df.index.values[0], tuple):
            df['cellid'] = df.index
        if id in df.columns and isinstance(df[id].values[0], tuple):
            cols = df.columns.tolist()
            # get the order right
            pos = [i for i, c in enumerate(cols) if c == id][0]
            for c in reversed(['k', 'i', 'j']):
                cols.insert(pos + 1, c)
            df['k'], df['i'], df['j'] = list(zip(*df[id]))
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
    #if isinstance(mftransientlist, flopy.mf6.data.mfdatalist.MFTransientList):
    #    mftransientlist.data = {per: ra for per, ra in enumerate(mftransientlist.array)}
    non_data_columns = {'k', 'i', 'j', 'cellid', 'rno', 'sfrsetting'}

    for per, recarray in mftransientlist.data.items():
        if recarray is not None:
            return [c for c in recarray.dtype.names
                    if c not in non_data_columns]

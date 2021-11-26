import io
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from flopy.discretization import StructuredGrid
from flopy.utils.sfroutputfile import SfrFile
import flopy.utils.binaryfile as bf
from flopy.mf6.utils.binarygrid_util import MfGrdFile
from mfexport.grid import get_kij_from_node3d


def aggregate_sfr_flow_ja_face(df):
    """SFR streamflow in/out components are saved in MODFLOW 6
    as FLOW-JA-FACE, where all inflows (positive values) and outflows
    (negative values) are listed for each stream reach (node). This
    method aggregates those results and returns a dataframe
    with one row per reach, with Qin, Qout, Qnet and Qmean_cfd
    and Qmean_cfs columns.

    Parameters
    ----------
    df : DataFrame
        Dataframe produced by read_mf6_binary_stress_output.
        Must have columns:
        node : reach number (rno)
        kstpkper : (timestep, stress period) tuples
        time : total time in model units
        q : FLOW-JA-FACE values (+ in, - out)

    Returns
    -------
    agg : DataFrame
        DataFrame with flow values aggregated by node, with columns:
        kstpkper : (timestep, stress period) tuples
        time : total time in model units
        rno : reach number (1-based)
        Qin : total inflows from other reaches
        Qout : total outflows to other reaches
        Qnet : net gain/loss in flow
        Qmean : average of inflows and outflows
    """
    print('aggregating FLOW-JA-FACE results...')
    ta = time.time()

    # Note: the loop/dictionary approach below
    # is ~20 times faster than a groupby approach
    # specifying the aggregation operations by column
    Qin = {}
    Qout = {}
    timedict = {}
    node = df.node.tolist()
    kstpkper = df.kstpkper.tolist()
    times = df.time.tolist()
    q = df.q.tolist()

    tal = time.time()
    for i in range(len(node)):
        k = (kstpkper[i], node[i])
        qi = q[i]

        qin = Qin.get(k, 0)
        qout = Qout.get(k, 0)
        if qi > 0:
            qin += qi
        elif qi < 0:
            qout += qi

        Qin[k] = qin
        Qout[k] = qout
        timedict[k] = times[i]
    print("loop took {:.2f}s".format(time.time() - tal))

    tal = time.time()
    # casting dicts to list first and then constructing dataframe
    # is about 30x faster than pd.DataFrame.from_dict
    keys = list(Qin.keys())
    Qinl = [Qin[k] for k in keys]
    Qoutl = [Qout[k] for k in keys]
    timel = [timedict[k] for k in keys]
    agg = pd.DataFrame({'Qin': Qinl, 'Qout': Qoutl, 'time': timel}, index=keys)
    agg['kstpkper'] = [i[0] for i in agg.index.values]
    agg['rno'] = [i[1] for i in agg.index.values]
    agg.index = range(len(agg))
    agg = agg[['time', 'kstpkper', 'rno', 'Qin', 'Qout']]
    agg['Qnet'] = agg[['Qin', 'Qout']].sum(axis=1)
    agg['Qmean'] = agg[['Qin', 'Qout']].abs().sum(axis=1) / 2
    print("DataFrame construction took {:.2f}s".format(time.time() - tal))

    print("finished in {:.2f}s\n".format(time.time() - ta))
    return agg.sort_values(by=['time', 'rno'])


def aggregate_mf6_stress_budget(mf6_stress_budget_output,
                                text=None,
                                kstpkper=None):
    """Read MODFLOW 6 budget output for a stress package into
    a DataFrame of one node/stress period per row, with fluxes
    listed by column.

    Parameters
    ----------
    mf6_stress_budget_output : file path
        Binary output file
    text : str or list of strings
        Text identifying flow term(s) (e.g. 'FLOW-JA-FACE', 'GWF', etc.).
        If None, all available variables are returned. (default None)

    Returns
    -------
    df : DataFrame
        DataFrame with flow values aggregated by node and stress period.
        Columns derived from FLOW-JA-FACE results are:
        kstpkper : (timestep, stress period) tuples
        time : total time in model units
        node : SFR reach number (zero-based)
        Qin : total inflows from other reaches
        Qout : total outflows to other reaches
        Qnet : net gain/loss in flow
        Qmean : average of inflows and outflows
        Any additional fluxes are listed in columns of the same name.
    """
    print('Getting data from {}...'.format(mf6_stress_budget_output))
    ta = time.time()
    if isinstance(text, str):
        text = [text]
    elif text is None:
        text = get_stress_budget_textlist(mf6_stress_budget_output)
    print('for variables: ' + ' '.join(text))
    dfs = {}
    agg = {}
    for k in text:
        dfs[k] = read_mf6_stress_budget_output(mf6_stress_budget_output, text=k, kstpkper=kstpkper)

        # aggregate FLOW-JA-FACE values to one Qin, Qout value per reach
        if k == 'FLOW-JA-FACE':
            agg[k] = aggregate_sfr_flow_ja_face(dfs['FLOW-JA-FACE'])
            index = pd.MultiIndex.from_tuples(list(zip(agg[k].kstpkper,
                                                       agg[k].rno)),
                                              names=['kstpkper', 'node'])
            agg[k].index = index
        else:
            # get fluxes by reach
            # dfs[k] has one row for each connection
            # reduce to sum of flow terms for each node (i.e. each SFR reach)
            agg[k] = dfs[k].groupby(['kstpkper', 'node']).sum()

    # merge variables into single dataframe
    t = 'FLOW-JA-FACE' if 'FLOW-JA-FACE' in text else text[0]
    text.remove(t)
    df = agg[t]
    for t in text:
        agg[t][t] = agg[t].q  # create column with flux name
        df = df.join(agg[t][t], how='outer') # join on multiindex of kstpkper, node

    # with outer join, ensure that all kstpkper and rno
    # columns fully populated
    df['kstpkper'] = df.index.get_level_values(0)
    df['node'] = df.index.get_level_values(1)
    df.reset_index(inplace=True, drop=True)
    df.sort_values(by=['time', 'node'], inplace=True)
    assert not np.any(df.kstpkper.isna().values)
    #assert not np.any(df.rno.isna().values)  # nan rnos can be unconnected reaches
    df.to_csv('junk.csv')
    #df.dropna(axis=0, subset=['rno'], inplace=True)
    if 'rno' in df.columns:
        #assert np.array_equal(df.node.values, df.rno.values)
        assert np.array_equal(df.dropna(axis=0, subset=['rno']).node.values,
                              df.dropna(axis=0, subset=['rno']).rno.values)

        df.drop('rno', axis=1, inplace=True)
    # convert to zero-based
    # (rnos in flopy package input are zero-based)
    if df['node'].min() == 1:
        df['node'] -= 1
    df.index = range(len(df))
    print("finished in {:.2f}s\n".format(time.time() - ta))
    return df


def get_flowja_face(cell_budget_file, binary_grid_file, kstpkper=(0, 0), idx=0,
                    precision='double'):
    """Get FLOW-JA-FACE (cell by cell flows) from MODFLOW 6 budget
    output and binary grid file.
    TODO: need test for extracted flowja fluxes
    """
    if isinstance(cell_budget_file, str):
        cbb = bf.CellBudgetFile(cell_budget_file)
        if binary_grid_file is None:
            binary_grid_file = cell_budget_file[::-4] + '.dis.grb'
            if not os.path.exists(binary_grid_file):
                binary_grid_file = None
    else:
        cbb = cell_budget_file
    if binary_grid_file is None:
        print("Couldn't get FLOW-JA-FACE, need binary grid file for connection information.")
        return
    bgf = MfGrdFile(binary_grid_file)
    # IA array maps cell number to connection number
    # (one-based index number of first connection at each cell)?
    # taking the forward difference then yields nconnections per cell
    ia = bgf._datadict['IA'] - 1
    # Connections in the JA array correspond directly with the
    # FLOW-JA-FACE record that is written to the budget file.
    ja = bgf._datadict['JA'] - 1  # cell connections
    flowja = cbb.get_data(text='FLOW-JA-FACE')[0][0, 0, :]
    df = get_intercell_connections(ia, ja, flowja)
    cols = ['n', 'm', 'q']

    # get the k, i, j locations for plotting the connections
    if isinstance(bgf.modelgrid, StructuredGrid):
        nlay, nrow, ncol = bgf.modelgrid.nlay, bgf.modelgrid.nrow, bgf.modelgrid.ncol
        k, i, j = get_kij_from_node3d(df['n'].values, nrow, ncol)
        df['kn'], df['in'], df['jn'] = k, i, j
        k, i, j = get_kij_from_node3d(df['m'].values, nrow, ncol)
        df['km'], df['im'], df['jm'] = k, i, j
        df.reset_index()
        cols += ['kn', 'in', 'jn', 'km', 'im', 'jm']
    return df[cols].copy()


def get_intercell_connections(ia, ja, flowja):
    print('Making DataFrame of intercell connections...')
    ta = time.time()
    all_n = []
    m = []
    q = []
    for n in range(len(ia)-1):
        for ipos in range(ia[n] + 1, ia[n+1]):
            all_n.append(n)
            m.append(ja[ipos])  # m is the cell that n connects to
            q.append(flowja[ipos])  # flow across the connection
    df = pd.DataFrame({'n': all_n, 'm': m, 'q': q})
    et = time.time() - ta
    print("finished in {:.2f}s\n".format(et))
    return df


def get_bc_flux(cbbobj, txt, kstpkper=None, idx=None):
    """Read a flow component from MODFLOW binary cell budget output;

    Parameters
    ----------
    cbbobj : open file handle (instance of flopy.utils.binaryfile.CellBudgetFile
    txt : cell budget record to read (e.g. 'STREAM LEAKAGE')
    kstpkper : tuple
        (timestep, stress period) to read
    idx : index of list returned by cbbobj (usually 0)

    Returns
    -------
    arr : ndarray
    """
    nrow, ncol, nlay = cbbobj.nrow, cbbobj.ncol, cbbobj.nlay
    results = cbbobj.get_data(text=txt, kstpkper=kstpkper, idx=idx)
    # this logic needs some cleanup
    if len(results) > 0:
        results = results[0]
    else:
        print('no data found at {} for {}'.format(kstpkper, txt))
        return
    if isinstance(results, list) and txt == 'RECHARGE':
        results = results[1]
    if results.size == 0:
        print('no data found at {} for {}'.format(kstpkper, txt))
        return
    if results.shape == (nlay, nrow, ncol):
        return results
    elif results.shape == (1, nrow, ncol):
        return results[0]
    elif results.shape == (nrow, ncol):
        return results
    elif len(results.shape) == 1 and \
            len({'node', 'q'}.difference(set(results.dtype.names))) == 0:
        arr = np.zeros(nlay * nrow * ncol, dtype=float)
        arr[results.node - 1] = results.q
        arr = np.reshape(arr, (nlay, nrow, ncol))
        arr = arr.sum(axis=0)
        return arr


def get_stress_budget_textlist(mf6_stress_budget_output):
    """Get list of available variable names in a binary budget output file.
    """
    cbobj = bf.CellBudgetFile(mf6_stress_budget_output,
                              precision='double'
                              )
    textlist = [t.strip().decode() for t in cbobj.textlist]
    return textlist


def read_mf6_dependent_variable_output(mf6_dependent_variable_output,
                                       text='head',
                                       kstpkper=None, hdry=-1e30):
    """Reads dependent variable output; for example, heads from the
    groundwater flow solution or output from STAGE FILEOUT in the options
    block of the .sfr6 file). Returns a DataFrame of output values.

    Parameters
    ----------
    mf6_dependent_variable_output : file path
        Binary output file
    text : str
        Text identifying variable (e.g. 'head', 'stage', etc.)
    kstpkper : list of tuples
        List of zero-based (timestep, stress period) tuples. If
        None, all available data are returned. (default None)
    hdry : float
        Value indicating dry cells.

    Returns
    -------
    df : DataFrame
        Table with results. Columns:
        node : Zero-based ID: model cell number for heads, reach number for SFR, etc.
        <text> : results for variable <text>
        kstpkper : (timestep, stress period)
        time : total time in model units
    """
    print('reading {} from\n{}...'.format(text, mf6_dependent_variable_output))
    ta = time.time()
    hdsobj = bf.HeadFile(mf6_dependent_variable_output,
                         text=text) # need to specify text otherwise file may not open
    times = hdsobj.get_times()
    if kstpkper is None:
        kstpkper = hdsobj.get_kstpkper()
    else:
        if not isinstance(kstpkper, list):
            kstpkper = [kstpkper]

    records = []
    for ksp in kstpkper:
        results = hdsobj.get_data(kstpkper=ksp)
        results = np.squeeze(results).ravel().tolist()
        records.append(results)
    nnodes = len(np.squeeze(records[0]).ravel())

    # create dataframe with one result in each row
    # sorted by timestep, then node
    values = []
    kstpkper_values = []
    time_values = []
    for i, rec in enumerate(records):
        values += records[i]
        kstpkper_values += [kstpkper[i]] * nnodes
        time_values += [times[i]] * nnodes
    df = pd.DataFrame({'node': list(range(nnodes)) * len(records),
                       text: values,
                       'kstpkper': kstpkper_values,
                       'time': time_values})
    df.loc[df[text] == hdry, text] = np.nan
    print("finished in {:.2f}s\n".format(time.time() - ta))
    return df[['time', 'kstpkper', 'node', text]]


def read_mf6_stress_budget_output(mf6_stress_budget_output,
                                  text='FLOW-JA-FACE',
                                  kstpkper=None):
    """Reads budget output from any package that follows the imeth=6
    structure (e.g. LAK, MAW, SFR, and UZF package(s); for example,
    output from BUDGET FILEOUT in the options block of the .sfr6 file).

    Parameters
    ----------
    mf6_stress_budget_output : file path
        Binary output file
    text : str
        Text identifying flow term (e.g. 'FLOW-JA-FACE')

    Returns
    -------
    df : DataFrame
        Table with flow results. Columns:
        node : node number (e.g. stream reach; 1-based)
        node2 : connecting node (e.g. up or downstream reach; 1-based)
        q : flow values
        FLOW-AREA : area of JA-FACE?
        kstpkper : (timestep, stress period)
        time : total time in model units
    """
    print('reading {} from\n{}...'.format(text, mf6_stress_budget_output))
    ta = time.time()
    cbobj = bf.CellBudgetFile(mf6_stress_budget_output,
                              precision='double'
                              )
    times = cbobj.get_times()
    dfs = []
    if kstpkper is None:
        kstpkper = cbobj.get_kstpkper()
        # returns a list of recarrays (length: nnodes);
        # one for each timestep, stress period
        records = cbobj.get_data(text=text)
    else:
        # otherwise, just get the results
        # for specified timestep, stress periods
        if not isinstance(kstpkper, list):
            kstpkper = [kstpkper]
        records = []
        for ksp in kstpkper:
            records += cbobj.get_data(text=text, kstpkper=ksp)

    for i, rec in enumerate(records):
        df = pd.DataFrame(rec)
        df['kstpkper'] = [kstpkper[i]] * len(df)
        df['time'] = [times[i]] * len(df)
        dfs.append(df.copy())
    df = pd.concat(dfs)
    print("finished in {:.2f}s\n".format(time.time() - ta))
    return df.sort_values(by=['time', 'node'])


def read_sfr_output(mf2005_sfr_outputfile=None,
                    mf2005_SfrFile_instance=None,
                    mf6_sfr_stage_file=None,
                    mf6_sfr_budget_file=None,
                    mf6_package_data=None,
                    model=None, grid_type='structured'):
    """Read MF-2005 or MF-6 style SFR output; return as DataFrame.
    """
    model_version = None
    packagedata = None
    if model is not None:
        model_version = model.version
        if model_version == 'mf6':
            packagedata = pd.DataFrame(model.sfr.packagedata.array.copy())
    elif mf6_package_data is not None:
        model_version = 'mf6'
        if isinstance(mf6_package_data, str) or isinstance(mf6_package_data, Path):
            
            skiprows = 0
            names = None
            with open(mf6_package_data) as src:
                for line in src:
                    if line.strip().startswith('#'):
                        names = line.strip().split()
                        skiprows += 1
                    else:
                        ncol = len(line.strip().split())
                        break
                        
            if names is None:
                if grid_type == 'structured':
                    names = ['rno', 'k', 'i', 'j', 'rlen', 'rwid', 'rgrd', 'rtp', 'rbth', 'rhk',
                            'man', 'ncon', 'ustrf', 'ndv']
                else:
                    names = ['rno', 'cellid', 'rlen', 'rwid', 'rgrd', 'rtp', 'rbth', 'rhk',
                            'man', 'ncon', 'ustrf', 'ndv']
                for i, _ in enumerate(range(len(names), ncol)):
                    names.append(f'aux_col{i+1}')
            else:
                names[0] = names[0].strip('#')
            
            # read the packagedata as a string to handle "none" values
            with open(mf6_package_data) as src:
                raw_pd = src.read()
                raw_pd = raw_pd.lower().replace('none', '0 0 0')
            packagedata = pd.read_csv(io.StringIO(raw_pd), names=names, 
                                      skiprows=skiprows, delim_whitespace=True)
            for col in ['rno', 'k', 'i', 'j']:
                if col in packagedata:
                    packagedata[col] -= 1
            if 'cellid' in packagedata.columns:
                if not isinstance(packagedata['cellid'][0], int):
                    packagedata['cellid'] = [(c[0]-1, c[1] -1, c[2] -1) for c in packagedata['cellid']]
                else:
                    packagedata['cellid'] -=1
        else:
            # make the dataframe on the .array attribute for flopy objects
            # or mf6_package_data is assumed to be array-like
            packagedata = pd.DataFrame(getattr(mf6_package_data, 'array', mf6_package_data))

    if model_version == 'mf6':

        # get the budget output
        df = aggregate_mf6_stress_budget(mf6_sfr_budget_file)

        # get the stage data
        if mf6_sfr_stage_file is not None:
            stg = read_mf6_dependent_variable_output(mf6_sfr_stage_file,
                                                     text='stage')
            df.sort_values(by=['kstpkper', 'node'], inplace=True)
            stg.sort_values(by=['kstpkper', 'node'], inplace=True)
            df.set_index(['kstpkper', 'node'], inplace=True)
            stg.set_index(['kstpkper', 'node'], inplace=True)
            na_reaches = np.isnan(df.time.values)
            #df.loc[~na_reaches].to_csv('df.csv')
            #stg.loc[~na_reaches].to_csv('stg.csv')
            #assert np.allclose(df.time.values, stg.time.values)
            assert np.allclose(df.loc[~na_reaches].time.values,
                               stg.loc[~na_reaches].time.values)
            assert np.array_equal(df.index, stg.index)
            df['stage'] = stg['stage']
            df.reset_index(inplace=True)

        # get the row, column location of SFR cells;
        # compute stream depths
        if packagedata is not None:
            rd = packagedata
            # convert reach number to zero-based
            if rd.rno.min() == 1:     
                rd['rno'] -= 1
            assert rd.rno.min() == 0
            assert df.node.min() == 0
                
            rno_strtop = dict(zip(rd.rno, rd.rtp))
            df['strtop'] = pd.to_numeric([rno_strtop[rno] for rno in df.node.values], errors='coerce')
            # fill nan stages with their streambed tops
            isna = df['stage'].isna()
            df.loc[isna, 'stage'] = df.loc[isna, 'strtop']
            df['depth'] = df['stage'] - df['strtop']
                
            if 'cellid' not in rd.columns:
                rd['cellid'] = list(zip(rd['k'], rd['i'], rd['j']))
                
            rno_cellid = dict(zip(rd.rno, rd.cellid))
            for i, dim in enumerate(['k', 'i', 'j']):
                df[dim] = pd.to_numeric([rno_cellid[rno][i] for rno in df.node.values], errors='coerce')
            df.dropna(subset=['k', 'i', 'j'], axis=0, inplace=True)
            # can't convert to integers if nans are present
            for dim in ['k', 'i', 'j']:
                df[dim] = df[dim].astype(int)
                assert 'int' in df[dim].dtype.name


    else:
        # SFR output
        if mf2005_sfr_outputfile is not None:
            sfrobj = SfrFile(mf2005_sfr_outputfile)
        elif mf2005_SfrFile_instance is not None:
            sfrobj = mf2005_SfrFile_instance
        else:
            print('Need path to SFR tabular budget output or FloPy SfrFile instance.')

        df = sfrobj.df.copy()
        df.sort_values(by=['segment', 'reach'], inplace=True)

    return df
import numpy as np
from mfexport.list_export import mftransientlist_to_dataframe


def get_period_sums(mftransientlist):
    # monkey patch the mf6 version to behave like the mf2005 version
    #if isinstance(mftransientlist,
    #              flopy.mf6.data.mfdatalist.MFTransientList):
    #    mftransientlist.data = {per: ra for per, ra in enumerate(mftransientlist.array)}

    sums = []
    for per in range(mftransientlist.model.nper):
        recarray = mftransientlist.data.get(per)
        if recarray is None:
            sums.append(0.)
        else:
            varname = [c for c in recarray.dtype.names
                        if c not in ['k', 'i', 'j', 'cellid']][0]
            sums.append(recarray[varname].sum())
    return np.array(sums)


def test_transient_list_to_dataframe(model):
    m, grid, output_path = model
    period_sums = get_period_sums(m.wel.stress_period_data)
    df = mftransientlist_to_dataframe(m.wel.stress_period_data)
    df.drop(['cellid', 'k', 'i', 'j'], axis=1, inplace=True)

    def get_per(column_name):
        return int(''.join(filter(lambda x: x.isdigit(), column_name)))

    # sum the dataframe columns, filling in any missing periods
    data = {get_per(c): df[c].sum() for c in df.columns}
    for i in range(m.nper):
        if i > 0 and i not in data:
            data[i] = data[i-1].sum()
        elif i == 0 and i not in data:
            data[i] = 0.
    period_sums2 = np.array([data[i] for i in range(m.nper)])

    assert np.allclose(period_sums, period_sums2)
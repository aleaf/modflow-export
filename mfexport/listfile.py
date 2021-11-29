"""
Functions for exporting results from the MODFLOW listing file
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import flopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mfexport.units import (get_figure_label_unit_text, parse_flux_units,
                            convert_volume_units, convert_time_units)
from mfexport.utils import make_output_folders


# for each MODFLOW-6 listfile budget term
# get the MF-2005 equivalent
mf2005_terms = {'STO-SS': 'STORAGE',
                'STO-SY': 'STORAGE',
                'WEL': 'WELLS',
                'RCH': 'RECHARGE',
                'SFR': 'STREAM_LEAKAGE',
                'LAK': 'LAKES',
                'CHD': 'CONSTANT_HEAD',
                'GHB': 'HEAD DEP BOUNDS'
                }
plotted = set()

def get_listfile_model_version(listfile):
    with open(listfile) as src:
        firstline = src.readline()
        if 'MODFLOW 6' in firstline:
            return 'mf6'
        elif 'MODFLOW-NWT' in firstline:
            return 'mfnwt'
        elif 'MODFLOW-2005' in firstline:
            return 'mf2005'


def get_budget_keys(listfile, key_suffix='BUDGET FOR ENTIRE MODEL'):
    keys = set()
    with open(listfile) as src:
        for line in src:
            if key_suffix in line:
                keys.add(line.split(' AT ')[0].strip())
    return keys


def get_listfile_data(listfile, model_start_datetime=None,
                      budgetkey=None):

    cls = flopy.utils.MfListBudget
    model_version = get_listfile_model_version(listfile)
    if model_version == 'mf6':
        cls = flopy.utils.Mf6ListBudget

    if budgetkey is not None:
        keys = get_budget_keys(listfile)
        if budgetkey not in keys:
            #budget_package = budgetkey.replace('BUDGET FOR ENTIRE MODEL', '').strip().split('_')[0]
            budget_package = budgetkey.strip().split(' ')[0].split('_')[0]
            budgetkey = [k for k in keys if budget_package in k]
            if len(budgetkey) > 0:
                budgetkey = budgetkey[0]
            else:
                return

        class PackageBudget(cls):
            """Export the a Package Budget from the listing file.
            """
            def set_budget_key(self):
                self.budgetkey = budgetkey
                return
        cls = PackageBudget

    mfl = cls(listfile)
    budget = mfl.get_dataframes(start_datetime=model_start_datetime)
    if budget is not None:
        df_flux, df_vol = budget
        kstp, kper = zip(*mfl.get_kstpkper())
        df_flux['kstp'] = kstp
        df_flux['kper'] = kper
        return df_flux


def plot_list_budget(listfile, model_name=None,
                     model_start_datetime=None,
                     output_path='postproc',
                     model_length_units=None,
                     model_time_units=None,
                     secondary_axis_units=None,
                     xtick_stride=6, plot_start_date=None, plot_end_date=None,
                     plot_pcts=False,
                     datetime_xaxis=True):

    pdfs_dir, _, _ = make_output_folders(output_path)
    if model_name is None:
        model_name = Path(listfile).stem
    if model_start_datetime is None:
        datetime_xaxis=False

    df_flux = get_listfile_data(listfile, model_start_datetime=model_start_datetime)

    df_flux_lake = get_listfile_data(listfile, model_start_datetime=model_start_datetime,
                                     budgetkey='LAK BUDGET FOR ENTIRE MODEL')

    df_flux_sfr = get_listfile_data(listfile, model_start_datetime=model_start_datetime,
                                    budgetkey='SFR BUDGET FOR ENTIRE MODEL')

    out_pdf = pdfs_dir / 'listfile_budget_summary.pdf'
    with PdfPages(out_pdf) as pdf:
        # plot summary with only net values for each term
        ax = plot_budget_summary(df_flux, title_prefix=model_name,
                            term_nets=True,
                            model_length_units=model_length_units,
                            model_time_units=model_time_units,
                            secondary_axis_units=secondary_axis_units,
                            xtick_stride=xtick_stride,
                            plot_start_date=plot_start_date, 
                            plot_end_date=plot_end_date,
                            plot_pcts=plot_pcts)
        if ax is not None:
            pdf.savefig()
            plt.close()
        # plot summary showing in and out values for all terms
        ax = plot_budget_summary(df_flux, title_prefix=model_name,
                            model_length_units=model_length_units,
                            model_time_units=model_time_units,
                            secondary_axis_units=secondary_axis_units,
                            xtick_stride=xtick_stride,
                            plot_start_date=plot_start_date, 
                            plot_end_date=plot_end_date,
                            plot_pcts=plot_pcts)
        if ax is not None:
            pdf.savefig()
            plt.close()    
        # plot summary of annual net means
        ax = plot_budget_summary(df_flux, title_prefix=model_name,
                            term_nets=True,
                            model_length_units=model_length_units,
                            model_time_units=model_time_units,
                            secondary_axis_units=secondary_axis_units,
                            xtick_stride=xtick_stride,
                            plot_start_date=plot_start_date, 
                            plot_end_date=plot_end_date,
                            annual_sums=True,
                            plot_pcts=plot_pcts)
        if ax is not None:
            pdf.savefig()
            plt.close()
    print(f'wrote {out_pdf}')

    pdf_outfile = pdfs_dir / 'listfile_budget_by_term.pdf'
    with PdfPages(pdf_outfile) as pdf:
        plotted = set()
        terms = [c for c in df_flux.columns if c not in {'kstp', 'kper'}]
        for term in terms:
            if term not in plotted:
                plot_budget_term(df_flux, term, title_prefix=model_name, #plotted=plotted,
                                 model_length_units=model_length_units,
                                 model_time_units=model_time_units,
                                 secondary_axis_units=secondary_axis_units,
                                 xtick_stride=xtick_stride, 
                                 plot_start_date=plot_start_date, plot_end_date=plot_end_date,
                                 datetime_xaxis=datetime_xaxis)
                pdf.savefig()
                plt.close()
        if df_flux_lake is not None and len(df_flux_lake) > 0:
            plotted = set()
            terms = [c for c in df_flux_lake.columns if c not in {'kstp', 'kper'}]
            for term in terms:
                if term not in plotted:
                    title_prefix = '{} Lake Package'.format(model_name)
                    plot_budget_term(df_flux_lake, term, title_prefix=title_prefix, #plotted=plotted,
                                 model_length_units=model_length_units,
                                 model_time_units=model_time_units,
                                 secondary_axis_units=secondary_axis_units, 
                                 plot_start_date=plot_start_date, plot_end_date=plot_end_date,
                                 datetime_xaxis=datetime_xaxis)
                    pdf.savefig()
                    plt.close()
        if df_flux_sfr is not None and len(df_flux_sfr) > 0:
            plotted = set()
            terms = [c for c in df_flux_sfr.columns if c not in {'kstp', 'kper'}]
            for term in terms:
                if term not in plotted:
                    title_prefix = '{} SFR Package'.format(model_name)
                    plot_budget_term(df_flux_sfr, term, title_prefix=title_prefix, #plotted=plotted,
                                 model_length_units=model_length_units,
                                 model_time_units=model_time_units,
                                 secondary_axis_units=secondary_axis_units, 
                                 plot_start_date=plot_start_date, plot_end_date=plot_end_date,
                                 datetime_xaxis=datetime_xaxis)
                    pdf.savefig()
                    plt.close()
    print(f'wrote {pdf_outfile}')


def plot_budget_summary(df, title_prefix='', title_suffix='', date_index_fmt='%Y-%m',
                        term_nets=False,
                        model_length_units=None,
                        model_time_units=None,
                        secondary_axis_units=None, xtick_stride=6,
                        plot_start_date=None, plot_end_date=None,
                        plot_pcts=False,
                        annual_sums=False
                        ):
    """Plot a stacked bar chart summary of a MODFLOW listing file budget dataframe.

    Parameters
    ----------
    df : DataFrame
        Table of listing file budget results produced by flopy; typically the flux 
        (not volume) terms (see example below).
    title_prefix : str, optional
        Prefix to insert at the begining of the title, for example the model name.
        by default ''
    title_suffix : str, optional
        Suffix to insert at the end of the title, by default ''
    date_index_fmt : str, optional
        Date format for the plot x-axis, by default '%Y-%m'
    term_nets : bool, optional
        Option to only plot net quantities for each stress period.
        For example if the inflows and outflows for the WEL package 
        were +10 and -10, a bar of zero height would be plotted.
        by default False
    model_length_units : str, optional
        Length units of the model, for labeling and conversion 
        to secondary_axis_units, 
        by default None
    model_time_units : str, optional
        Time units of the model, for labeling and conversion 
        to secondary_axis_units, 
        by default None
    secondary_axis_units : str, optional
        Option to include a secondary y-axis on the right with
        another unit, for example 'mgal/day' for million gallons per day.
        Requires `model_length_units` and `model_time_units`.
        by default None
    xtick_stride : int, optional
        Spacing between x-ticks. May be useful for models with many stress periods.
        by default 6
    plot_start_date : str, optional
        Minimum date to plot on the x-axis, in a string format 
        recognizable by pandas (if `df` has a datetime index) or
        a numeric value (if `df` has a numeric index).
        by default None (plot all dates)
    plot_end_date : str, optional
        Maximum date to plot on the x-axis, in a string format 
        recognizable by pandas (if `df` has a datetime index) or
        a numeric value (if `df` has a numeric index).
        by default None (plot all dates)
    plot_pcts : bool
        Option to include the percentage of each flux.
        by default, False
    annual_sums : bool
        Option to summarize budget by year (e.g. using :py:meth:`pandas.DataFrame.groupby`).
        Requires that ``df`` have a valid datetime index.
        by default, False

    Returns
    -------
    ax : matplotlib axes subplot instance
        
    Examples
    --------
    .. code-block:: python

        from mfexport.listfile import get_listfile_data, plot_budget_summary
        df = get_listfile_data(listfile='model.list', model_start_datetime='2000-01-01')
        plot_budget_summary(df)

    """    

    # slice the dataframe to the specified time range (if any)
    df = df.copy()
    df = df.loc[slice(plot_start_date, plot_end_date)]
    if annual_sums:
        if isinstance(df.index, pd.DatetimeIndex):
            dfa = df.groupby(df.index.year).mean()
            dfa['kper'] = df.groupby(df.index.year).last()['kper']
            dfa['kstp'] = df.groupby(df.index.year).last()['kstp']
            df = dfa
        else:
            print('Skipping, annual_sums requires a datetime index.')
            return
    if len(df) < xtick_stride * 2:
        xtick_stride = 1
        
    fig, ax = plt.subplots(figsize=(11, 8.5))
    in_cols = [c for c in df.columns if '_IN' in c and 'TOTAL' not in c]
    out_cols = [c for c in df.columns if '_OUT' in c and 'TOTAL' not in c]
    if not term_nets:
        ax = df[in_cols].plot.bar(stacked=True, ax=ax,# width=20
                                  )
        df[out_cols] *= -1
        ax = (df[out_cols]).plot.bar(stacked=True, ax=ax,# width=20
                                      )
        df_pcts = df.copy()
        df_pcts[in_cols] = df[in_cols].div(df['TOTAL_IN'], axis=0)
        df_pcts[out_cols] = df[out_cols].div(df['TOTAL_OUT'], axis=0)
    else:
        pairs = list(zip(in_cols, out_cols))
        net_cols = []
        for in_col, out_col in pairs:
            net_col = f"{in_col.split('_')[0]} (net)"
            df[net_col] = df[in_col] - df[out_col]
            net_cols.append(net_col)
        ax = df[net_cols].plot.bar(stacked=True, ax=ax,# width=20
        )
        net_sums = df[net_cols][df[net_cols] > 0].sum(axis=1)
        df_pcts = df[net_cols].div(net_sums, axis=0)

    if isinstance(df.index, pd.DatetimeIndex):
        ax.set_xticklabels(df.index.strftime(date_index_fmt))
    elif annual_sums:
        ax.set_xlabel('Calendar year')
    else:
        ax.set_xlabel('Time since the start of the simulation, in model units')

    ax.axhline(0, zorder=100, lw=0.5, c='k')

    # create ylabel with model units, if provided
    if model_length_units is not None and model_time_units is not None:
        units_text = get_figure_label_unit_text(model_length_units, model_time_units,
                                                length_unit_exp=3)
    else:
        units_text = '$L^3/T$'
    ax.set_ylabel(f'Flow rate, in model units of {units_text}')

    # add commas to y axis values
    formatter = mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.get_yaxis().set_major_formatter(formatter)

    # optional 2nd y-axis with other units
    if secondary_axis_units is not None:
        length_units2, time_units2 = parse_flux_units(secondary_axis_units)
        vol_conversion = convert_volume_units(model_length_units, length_units2)
        time_conversion = convert_time_units(model_time_units, time_units2)

        def fconvert(x):
            return x * vol_conversion * time_conversion

        def rconvert(x):
            return x / (vol_conversion * time_conversion)

        secondary_axis_unit_text = get_figure_label_unit_text(length_units2, time_units2,
                                                              length_unit_exp=3)
        secax = ax.secondary_yaxis('right', functions=(fconvert, rconvert))
        secax.set_ylabel(f'Flow rate, in {secondary_axis_unit_text}')
        secax.get_yaxis().set_major_formatter(formatter)

    # add stress period info
    ymin, ymax = ax.get_ylim()
    xlocs = np.arange(len(df))
    yloc = np.ones(len(df)) * (ymin + 0.03 * (ymax - ymin))
    if xtick_stride is None:
        xtick_stride = int(np.round(len(df) / 10, 0))
        xtick_stride = 1 if xtick_stride < 1 else xtick_stride
    kpers = set()
    for x, y in zip(xlocs[::xtick_stride], yloc[::xtick_stride]):
        kper = int(df.iloc[x]['kper'])
        # only make one line for each stress period
        if kper not in kpers:
            ax.axvline(x, lw=0.5, c='k', zorder=-2)
            ax.text(x, y, f" {kper}", transform=ax.transData, ha='left', va='top')
            kpers.add(kper)
    ax.text(0, # min(kpers), # (the x loc of the first bar is always 0)
            y + abs(0.06*y), 
            ' model stress period:', 
            transform=ax.transData, ha='left', va='top')
    title_text = 'budget summary' 
    if term_nets:
        title_text += ' (net fluxes)'
    title_text = ' '.join((title_prefix, title_text, title_suffix)).strip()
    ax.set_title(title_text)
    
    ax.legend(ncol=2)

    # reduce x-tick density
    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    ax.xaxis.set_ticks(ticks[::xtick_stride])
    ax.xaxis.set_ticklabels(ticklabels[::xtick_stride])
    
    # add percentages for smaller datasets
    if plot_pcts:
        ymin, ymax = ax.get_ylim()
        # max bar height for printing %
        height_cutoff_frac = 0.01
        height_cutoff = ymax * height_cutoff_frac
        for p in ax.patches:
            height = p.get_height()
            if abs(height) > height_cutoff:
                width = p.get_width()
                xloc, yloc = p.get_xy()
                x_value = int(xloc + 0.5 * width)
                
                # get the percentage
                if term_nets:
                    loc = df[net_cols].iloc[x_value] == height
                else:
                    loc = df.iloc[x_value] == height
                pct = df_pcts.iloc[x_value][loc].values[0]
                y_center = yloc + 0.5 * height
                
                # plot the text
                ax.text(x_value, y_center,
                        f'{abs(pct):.1%}', fontsize=8, color='black', 
                        transform=ax.transData, 
                        ha='center', va='center')
    return ax


def plot_budget_term(df, term, title_prefix='', title_suffix='',
                     model_length_units=None, model_time_units=None,
                     secondary_axis_units=None, xtick_stride=None,
                     plot_start_date=None, plot_end_date=None,
                     datetime_xaxis=True):
    """Make a timeseries plot of an individual MODFLOW listing file 
    budget term.

    Parameters
    ----------
    df : DataFrame
        Table of listing file budget results produced by flopy; typically the flux 
        (not volume) terms (see example below).
    title_prefix : str, optional
        Prefix to insert at the begining of the title, for example the model name.
        by default ''
    title_suffix : str, optional
        Suffix to insert at the end of the title, by default ''
    model_length_units : str, optional
        Length units of the model, for labeling and conversion 
        to secondary_axis_units, 
        by default None
    model_time_units : str, optional
        Time units of the model, for labeling and conversion 
        to secondary_axis_units, 
        by default None
    secondary_axis_units : str, optional
        Option to include a secondary y-axis on the right with
        another unit, for example 'mgal/day' for million gallons per day.
        Requires `model_length_units` and `model_time_units`.
        by default None
    xtick_stride : int, optional
        Spacing between x-ticks. May be useful for models with many stress periods.
        by default 6
    plot_start_date : str, optional
        Minimum date to plot on the x-axis, in a string format 
        recognizable by pandas (if `df` has a datetime index) or
        a numeric value (if `df` has a numeric index). May be
        useful if the model has long spinup period(s) that
        would obscure later periods of interest when datetime_xaxis=True.
        by default None (plot all dates)
    plot_end_date : str, optional
        Maximum date to plot on the x-axis, in a string format 
        recognizable by pandas (if `df` has a datetime index) or
        a numeric value (if `df` has a numeric index).
        by default None (plot all dates)
    datetime_xaxis : bool
        Plot budget values as a function of time. If False,
        plot as a function of stress period.
        by default, True

    Returns
    -------
    ax : matplotlib axes subplot instance
        
    Examples
    --------
    .. code-block:: python

        from mfexport.listfile import get_listfile_data, plot_budget_summary
        df = get_listfile_data(listfile='model.list', model_start_datetime='2000-01-01')
        plot_budget_term(df, 'WELLS')
    """    
    # slice the dataframe to the specified time range (if any)
    df = df.copy()
    df = df.loc[slice(plot_start_date, plot_end_date)]
    
    if not datetime_xaxis and 'datetime' in df.index.dtype.name:
        df['datetime'] = df.index
        df.index = df['kper']
    if term not in {'IN-OUT', 'PERCENT_DISCREPANCY'}:

        # get the absolute quantity for the term
        if isinstance(term, list):
            series = df[term].sum(axis=1)
            out_term = [s.replace('_IN', '_OUT') for s in term]
            out_series = df[out_term].sum(axis=1)
        else:
            term = term.replace('_IN', '').replace('_OUT', '')
            in_term = f'{term}_IN'
            out_term = f'{term}_OUT'
            series = df[in_term]
            out_series = df[out_term]

        # get the net
        net_series = series - out_series

        # get the percent relative to the total budget
        pct_series = series/df['TOTAL_IN']
        pct_out_series = out_series/df['TOTAL_OUT']
        pct_net_series = pct_series - pct_out_series

    else:
        series = df[term]
        out_term = None
        out_series = None

    if model_length_units is not None and model_time_units is not None:
        units_text = get_figure_label_unit_text(model_length_units, model_time_units,
                                                length_unit_exp=3)
    else:
        units_text = '$L^3/T$'

    if out_series is not None:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 8.5))
        axes = axes.flat
        ax = axes[0]
        series.plot(ax=ax, c='C0')
        ax.axhline(0, zorder=-1, lw=0.5, c='k')
        (-out_series).plot(ax=ax, c='C1')
        net_series.plot(ax=ax, c='0.5', zorder=-1)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, ['In', 'Out', 'Net'])
        ax.set_ylabel(f'Flow rate, in model units of {units_text}')

        # add commas to y axis values
        formatter = mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
        ax.get_yaxis().set_major_formatter(formatter)

        # optional 2nd y-axis with other units
        if secondary_axis_units is not None:
            length_units2, time_units2 = parse_flux_units(secondary_axis_units)
            vol_conversion = convert_volume_units(model_length_units, length_units2)
            time_conversion = convert_time_units(model_time_units, time_units2)

            def fconvert(x):
                return x * vol_conversion * time_conversion

            def rconvert(x):
                return x / (vol_conversion * time_conversion)

            secondary_axis_unit_text = get_figure_label_unit_text(length_units2, time_units2,
                                                                  length_unit_exp=3)
            secax = ax.secondary_yaxis('right', functions=(fconvert, rconvert))
            secax.set_ylabel(f'Flow rate, in {secondary_axis_unit_text}')
            secax.get_yaxis().set_major_formatter(formatter)

        # plot the percentage of total budget on second axis
        ax2 = axes[1]
        pct_series.plot(ax=axes[1], c='C0')
        ax2.axhline(0, zorder=-1, lw=0.5, c='k')
        (-pct_out_series).plot(ax=ax2, c='C1')
        pct_net_series.plot(ax=ax2, c='0.5', zorder=-1)
        ax2.set_ylabel('Fraction of budget')
        single_subplot = False
        # add stress period info
        #ymin, ymax = ax.get_ylim()
        #yloc = np.ones(len(df)) * (ymin - 0.02 * (ymax - ymin))
        #if xtick_stride is None:
        #    xtick_stride = int(np.round(len(df) / 10, 0))
        #    xtick_stride = 1 if xtick_stride < 1 else xtick_stride
        #kpers = set()
        #for x, y in zip(df.index.values[::xtick_stride], yloc[::xtick_stride]):
        #    kper = int(df.loc[x, 'kper'])
        #    # only make one line for each stress period
        #    if kper not in kpers:
        #        ax.axvline(x, lw=0.5, c='k', zorder=-2)
        #        ax2.axvline(x, lw=0.5, c='k', zorder=-2)
        #        ax2.text(x, y, kper, transform=ax.transData, ha='center', va='top')
        #        kpers.add(kper)

        #ax2.text(0.5, -0.07, 'Model Stress Period', ha='center', va='top', transform=ax.transAxes)

    else:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(11, 8.5))
        series.plot(ax=ax, c='C0')
        ax.axhline(0, zorder=-1, lw=0.5, c='k')
        ax2 = ax
        single_subplot = True

    # add stress period info
    ymin, ymax = ax.get_ylim()
    pad = 0.02
    if single_subplot:
        pad = 0.1
    yloc = np.ones(len(df)) * (ymin - pad * (ymax - ymin))
    if xtick_stride is None:
        xtick_stride = int(np.round(len(df) / 10, 0))
        xtick_stride = 1 if xtick_stride < 1 else xtick_stride
    kpers = set()
    for x, y in zip(df.index.values[::xtick_stride], yloc[::xtick_stride]):
        kper = int(df.loc[x, 'kper'])
        # only make one line for each stress period
        if kper not in kpers:
            ax.axvline(x, lw=0.5, c='k', zorder=-2)
            ax2.axvline(x, lw=0.5, c='k', zorder=-2)
            ax2.text(x, y, kper, transform=ax.transData, ha='center', va='top')
            kpers.add(kper)

        ax2.text(0.5, -0.07, 'Model Stress Period', ha='center', va='top', transform=ax.transAxes)


    title_text = ' '.join((title_prefix, term.split('_')[0], title_suffix)).strip()
    ax.set_title(title_text)

    # set x axis limits
    xmin, xmax = series.index.min(), series.index.max()
        
    if plot_start_date is not None:
        if not datetime_xaxis:
            loc = df.datetime >= plot_start_date
            xmin = df.datetime.loc[loc].index[0]
        else: 
            xmin = pd.Timestamp(plot_start_date)
    if plot_end_date is not None:
        if not datetime_xaxis:
            loc = df.datetime <= plot_end_date
            xmax = df.datetime.loc[loc].index[-1]
        else:
            xmax = pd.Timestamp(plot_end_date)
    ax.set_xlim(xmin, xmax)

    if not datetime_xaxis:
        if 'datetime' in df.columns:
            xticks = ax2.get_xticks()
            datetime_labels = []
            for i in xticks:
                if i in df.index:
                    dt_label = df['datetime'].loc[int(i)].strftime('%Y-%m-%d')
                else:
                    dt_label = ''
                datetime_labels.append(dt_label)
            ax2.set_xticklabels(datetime_labels, rotation=90)
            ax2.set_xlabel(None)
            
    if not isinstance(df.index, pd.DatetimeIndex):
        ax2.set_xlabel('Time since the start of the simulation, in model units')

    plotted.update({term, out_term})
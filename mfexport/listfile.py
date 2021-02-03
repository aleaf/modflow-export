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
            budget_package = budgetkey.replace('BUDGET FOR ENTIRE MODEL', '').strip().split('_')[0]
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
                     secondary_axis_units=None):

    pdfs_dir, _, _ = make_output_folders(output_path)
    if model_name is None:
        model_name = Path(listfile).stem

    df_flux = get_listfile_data(listfile, model_start_datetime=model_start_datetime)

    df_flux_lake = get_listfile_data(listfile, model_start_datetime=model_start_datetime,
                                     budgetkey='LAK BUDGET FOR ENTIRE MODEL')

    df_flux_sfr = get_listfile_data(listfile, model_start_datetime=model_start_datetime,
                                    budgetkey='SFR BUDGET FOR ENTIRE MODEL')

    # plot summary showing in and out values for all terms
    plot_budget_summary(df_flux, title_prefix=model_name,
                        model_length_units=model_length_units,
                        model_time_units=model_time_units,
                        secondary_axis_units=secondary_axis_units)

    # plot summary with only net values for each term
    plot_budget_summary(df_flux, title_prefix=model_name,
                        term_nets=True,
                        model_length_units=model_length_units,
                        model_time_units=model_time_units,
                        secondary_axis_units=secondary_axis_units)

    out_pdf = os.path.join(pdfs_dir, 'listfile_budget_summary.pdf')
    plt.savefig(out_pdf)
    plt.close()
    print(f'wrote {out_pdf}')

    pdf_outfile = os.path.join(pdfs_dir, 'listfile_budget_by_term.pdf')
    with PdfPages(pdf_outfile) as pdf:
        plotted = set()
        terms = [c for c in df_flux.columns if c not in {'kstp', 'kper'}]
        for term in terms:
            if term not in plotted:
                plot_budget_term(df_flux, term, title_prefix=model_name, plotted=plotted,
                                 model_length_units=model_length_units,
                                 model_time_units=model_time_units,
                                 secondary_axis_units=secondary_axis_units)
                pdf.savefig()
                plt.close()
        if df_flux_lake is not None and len(df_flux_lake) > 0:
            plotted = set()
            terms = [c for c in df_flux_lake.columns if c not in {'kstp', 'kper'}]
            for term in terms:
                if term not in plotted:
                    title_prefix = '{} Lake Package'.format(model_name)
                    plot_budget_term(df_flux_lake, term, title_prefix=title_prefix, plotted=plotted,
                                 model_length_units=model_length_units,
                                 model_time_units=model_time_units,
                                 secondary_axis_units=secondary_axis_units)
                    pdf.savefig()
                    plt.close()
        if df_flux_sfr is not None and len(df_flux_sfr) > 0:
            plotted = set()
            terms = [c for c in df_flux_sfr.columns if c not in {'kstp', 'kper'}]
            for term in terms:
                if term not in plotted:
                    title_prefix = '{} SFR Package'.format(model_name)
                    plot_budget_term(df_flux_sfr, term, title_prefix=title_prefix, plotted=plotted,
                                 model_length_units=model_length_units,
                                 model_time_units=model_time_units,
                                 secondary_axis_units=secondary_axis_units)
                    pdf.savefig()
                    plt.close()
    print(f'wrote {pdf_outfile}')


def plot_budget_summary(df, title_prefix='', title_suffix='', date_index_fmt='%Y-%m',
                        term_nets=False,
                        model_length_units=None,
                        model_time_units=None,
                        secondary_axis_units=None):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    if not term_nets:
        in_cols = [c for c in df.columns if '_IN' in c and 'TOTAL' not in c]
        out_cols = [c for c in df.columns if '_OUT' in c and 'TOTAL' not in c]
        ax = df[in_cols].plot.bar(stacked=True, ax=ax)
        ax = (-df[out_cols]).plot.bar(stacked=True, ax=ax)

    if isinstance(df.index, pd.DatetimeIndex):
        ax.set_xticklabels(df.index.strftime(date_index_fmt))
    else:
        ax.set_xlabel('Time since the start of the simulation, in model units')

    ax.axhline(0, zorder=-1, lw=0.5, c='k')

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
    stride = int(np.round(len(df) / 10, 0))
    stride = 1 if stride < 1 else stride
    kpers = set()
    for x, y in zip(xlocs[::stride], yloc[::stride]):
        kper = int(df.iloc[x]['kper'])
        # only make one line for each stress period
        if kper not in kpers:
            ax.axvline(x, lw=0.5, c='k', zorder=-2)
            ax.text(x, y, f" {kper}", transform=ax.transData, ha='left', va='top')
            kpers.add(kper)
    ax.text(min(kpers), y + abs(0.06*y), ' model stress period:', transform=ax.transData, ha='left', va='top')
    title_text = ' '.join((title_prefix, 'budget summary', title_suffix)).strip()
    ax.set_title(title_text)

    # reduce x-tick density
    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    ax.xaxis.set_ticks(ticks[::stride])
    ax.xaxis.set_ticklabels(ticklabels[::stride])
    return ax


def plot_budget_term(df, term, title_prefix='', title_suffix='', plotted=set(),
                     model_length_units=None, model_time_units=None,
                     secondary_axis_units=None):

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
        ax2.set_ylabel('Fraction of model budget')

        # add stress period info
        ymin, ymax = ax.get_ylim()
        yloc = np.ones(len(df)) * (ymin - 0.02 * (ymax - ymin))
        stride = int(np.round(len(df) / 10, 0))
        stride = 1 if stride < 1 else stride
        kpers = set()
        for x, y in zip(df.index.values[::stride], yloc[::stride]):
            kper = df.loc[x, 'kper']
            # only make one line for each stress period
            if kper not in kpers:
                ax.axvline(x, lw=0.5, c='k', zorder=-2)
                ax2.axvline(x, lw=0.5, c='k', zorder=-2)
                ax2.text(x, y, kper, transform=ax.transData, ha='center', va='top')
                kpers.add(kper)

        ax2.text(0.5, -0.07, 'Model Stress Period', ha='center', va='top', transform=ax.transAxes)

    else:
        fig, ax = plt.subplots(1, 1, sharex=True, figsize=(11, 8.5))
        series.plot(ax=ax, c='C0')
        ax.axhline(0, zorder=-1, lw=0.5, c='k')
        ax2 = ax

    title_text = ' '.join((title_prefix, term.split('_')[0], title_suffix)).strip()
    ax.set_title(title_text)
    xmin, xmax = series.index.min(), series.index.max()
    ax.set_xlim(xmin, xmax)

    if not isinstance(df.index, pd.DatetimeIndex):
        ax2.set_xlabel('Time since the start of the simulation, in model units')

    plotted.update({term, out_term})
"""
Functions for exporting results from the MODFLOW listing file
"""
import os
import numpy as np
import pandas as pd
import flopy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mfexport.utils import make_output_folders


class MF6LakListBudget(flopy.utils.Mf6ListBudget):
    """Export the Lake Package Budget from the listing file.
    """

    def set_budget_key(self):
        self.budgetkey = 'LAK BUDGET FOR ENTIRE MODEL'
        return


class MF6SfrListBudget(flopy.utils.Mf6ListBudget):
    """Export the SFR Package Budget from the listing file.
    """

    def set_budget_key(self):
        self.budgetkey = 'SFR BUDGET FOR ENTIRE MODEL'
        return


class MFLakListBudget(flopy.utils.MfListBudget):
    """Export the Lake Package Budget from the listing file.
    """

    def set_budget_key(self):
        self.budgetkey = 'LAK BUDGET FOR ENTIRE MODEL'
        return


class MFSfrListBudget(flopy.utils.MfListBudget):
    """Export the SFR Package Budget from the listing file.
    """

    def set_budget_key(self):
        self.budgetkey = 'SFR BUDGET FOR ENTIRE MODEL'
        return


def get_dataframe(listfile, model_start_datetime=None,
                  model_version=None,
                  budgetkey=None):

    cls = flopy.utils.MfListBudget
    if model_version == 'mf6':
        cls = flopy.utils.Mf6ListBudget

    if budgetkey is not None:
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


def export_mf6_list_budget(model=None, sim=None, model_name=None,
                           model_start_datetime=None,
                           output_path='postproc'):

    if model is None:
        if sim is None or model_name is None:
            raise ValueError('If model=None, must provide simulation and model name!')
        model = sim.get_model(model_name=model_name.lower())
    else:
        model_name = model.name

    pdfs_dir, _, _ = make_output_folders(output_path)

    listfile = os.path.join(model.model_ws, model_name + '.list')

    df_flux = get_dataframe(listfile, model_start_datetime=model_start_datetime,
                            model_version=model.version)

    df_flux_lake = get_dataframe(listfile, model_start_datetime=model_start_datetime,
                                 model_version=model.version,
                                 budgetkey='LAK BUDGET FOR ENTIRE MODEL')

    df_flux_sfr = get_dataframe(listfile, model_start_datetime=model_start_datetime,
                                 model_version=model.version,
                                 budgetkey='SFR BUDGET FOR ENTIRE MODEL')

    pdf_outfile = os.path.join(pdfs_dir, 'listfile_budget.pdf')
    with PdfPages(pdf_outfile) as pdf:
        plotted = set()
        terms = [c for c in df_flux.columns if c not in {'kstp', 'kper'}]
        for term in terms:
            if term not in plotted:
                plot_budget_term(df_flux, term, title_prefix=model_name, plotted=plotted)
                pdf.savefig()
                plt.close()
        if df_flux_lake is not None and len(df_flux_lake) > 0:
            plotted = set()
            terms = [c for c in df_flux_lake.columns if c not in {'kstp', 'kper'}]
            for term in terms:
                if term not in plotted:
                    title_prefix = '{} Lake Package'.format(model_name)
                    plot_budget_term(df_flux_lake, term, title_prefix=title_prefix, plotted=plotted)
                    pdf.savefig()
                    plt.close()
        if df_flux_sfr is not None and len(df_flux_sfr) > 0:
            plotted = set()
            terms = [c for c in df_flux_sfr.columns if c not in {'kstp', 'kper'}]
            for term in terms:
                if term not in plotted:
                    title_prefix = '{} SFR Package'.format(model_name)
                    plot_budget_term(df_flux_sfr, term, title_prefix=title_prefix, plotted=plotted)
                    pdf.savefig()
                    plt.close()


def plot_budget_term(df, term, title_prefix='', title_suffix='', plotted=set()):

    if term not in {'IN-OUT', 'PERCENT_DISCREPANCY'}:

        # get the absolute quantity for the term
        if isinstance(term, list):
            series = df[term].sum(axis=1)
            out_term = [s.replace('_IN', '_OUT') for s in term]
            out_series = df[out_term].sum(axis=1)
        else:
            out_term = term.replace('_IN', '_OUT')
            series = df[term]
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

    if out_series is not None:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 8.5))
        axes = axes.flat
        ax = axes[0]
        ax.axhline(0, zorder=-1, lw=0.5, c='k')
        series.plot(ax=ax, c='C0')
        (-out_series).plot(ax=ax, c='C1')
        net_series.plot(ax=ax, c='0.5', zorder=-1)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h, ['In', 'Out', 'Net'])
        ax.set_ylabel('Flow rate, in model units of $L^3/T$')

        # plot the percentage of total budget on second axis
        ax2 = axes[1]
        ax2.axhline(0, zorder=-1, lw=0.5, c='k')
        pct_series.plot(ax=ax2, c='C0')
        (-pct_out_series).plot(ax=ax2, c='C1')
        pct_net_series.plot(ax=ax2, c='0.5', zorder=-1)
        ax2.set_ylabel('Fraction of model budget')

        # add stress period info
        ymin, ymax = ax.get_ylim()
        yloc = np.ones(len(df)) * (ymin - 0.02 * (ymax - ymin))
        stride = 2
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
        ax.axhline(0, zorder=-1, lw=0.5, c='k')
        series.plot(ax=ax, c='C0')
        ax2 = ax

    title_text = ' '.join((title_prefix, term.split('_')[0], title_suffix)).strip()
    ax.set_title(title_text)

    if not isinstance(df.index, pd.DatetimeIndex):
        ax2.set_xlabel('Time since the start of the simulation, in model units')

    plotted.update({term, out_term})
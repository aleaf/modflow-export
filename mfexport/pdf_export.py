import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from flopy.utils import MfList


def export_pdf(filename, array, text,
               nodata=None, mfarray_type='array2d',
               float_fmt='{:.2f}', verbose=False):
    t0 = time.time()
    if isinstance(array, np.ma.masked_array) and array.mask.all():
        print(f"{filename}: no data to export!")
        return
    elif np.all(np.isnan(array)):
        print(f"{filename}: no data to export!")
        return
    
    if array.min() < 0.01:
        float_fmt = '{:.6e}'
    elif 'int' in array.dtype.name:
        float_fmt = '{:.0f}'

    if len(array.shape) > 2:
        multipage_pdf = PdfPages(filename)

    if mfarray_type == 'array2d':
        multipage_pdf = False
        array = [np.reshape(array, (1, array.shape[0],
                                   array.shape[1]))]
    elif mfarray_type == 'array3d':
        array = [array]
    elif mfarray_type == 'transient2d' or mfarray_type == 'transientlist':
        pass

    for per, array3d in enumerate(array):
        for k, array2d in enumerate(array3d):
            fig, ax = plt.subplots()
            arr = array2d.astype(float)

            if nodata is not None:
                arr[arr == nodata] = np.nan

            mn = np.nanmin(arr)
            mx = np.nanmax(arr)
            mean = np.nanmean(arr)

            im = ax.imshow(array2d)
            titletxt = '{0}'.format(text)
            if mfarray_type == 'array3d':
                titletxt += ', layer {}'.format(k)
            elif mfarray_type == 'transientlist':
                titletxt += ', period {}, layer {}'.format(per, k)
            titletxt += '\nmean: {0}, min: {0}, max: {0}'.format(float_fmt)
            ax.set_title(titletxt.format(mean, mn, mx))
            plt.colorbar(im, shrink=0.8)
            if multipage_pdf:
                multipage_pdf.savefig()
            else:
                plt.savefig(filename)
            plt.close()
    if multipage_pdf:
        multipage_pdf.close()
    if verbose:
        print("pdf export took {:.2f}s".format(time.time() - t0))


def export_pdf_bar_summary(filename, array, title=None, xlabel='Stress Period',
                           method='mean'):
    period_sums = getattr(np.ma, method)(array.data, axis=tuple(range(1, array.ndim))).data
    fig, ax = plt.subplots()
    periods = np.arange(len(period_sums), dtype=int)
    ax.bar(periods, period_sums)
    stride = int(np.round(len(period_sums) / 10, 0))
    stride = 1 if stride < 1 else stride
    ax.set_xticks(periods[::stride])
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(f'{method.capitalize()}, in model units')
    plt.savefig(filename)
    plt.close()


def sfr_baseflow_pdf(outfile, df, pointsize=0.5, verbose=False):
    """make a scatter plot of base flow
    (with point size proportional to Q)
    """
    t0 = time.time()
    fig, ax = plt.subplots()
    wet = df.loc[df.Qmean != 0]
    dry = df.loc[df.Qmean == 0]
    ax.scatter(dry.j, dry.i, s=pointsize, color='0.5')
    Qpointsizes = np.log10(wet.Qmean)
    Qpointsizes[Qpointsizes < 0] = 0.1
    ax.scatter(wet.j, wet.i, s=Qpointsizes, alpha=0.5)
    ax.invert_yaxis()
    ax.set_title('Simulated base flow')
    plt.savefig(outfile)
    print('wrote {}'.format(outfile))
    plt.close()
    if verbose:
        print("pdf export took {:.2f}s".format(time.time() - t0))


def sfr_qaquifer_pdf(outfile, df, pointsize=0.5, verbose=False):
    """make a scatter plot of Qaquifer
    (with color proportional to flux, scaled to largest gaining flow)
    """
    t0 = time.time()
    fig, ax = plt.subplots()
    gaining = df.loc[df.Qaquifer < 0]
    losing = df.loc[df.Qaquifer > 0]
    dry = df.loc[df.Qmean == 0]
    ax.scatter(dry.j, dry.i, pointsize, color='0.5')

    if len(losing) > 0:
        Qpointcolors_l = np.abs(losing.Qaquifer)
        vmax = None
        if len(gaining) > 0:
            vmax = np.percentile(np.abs(gaining.Qaquifer), 95)
        ax.scatter(losing.j, losing.i,
                   s=pointsize, c=Qpointcolors_l,
                   vmax=vmax,
                   cmap='Reds')
    if len(gaining) > 0:
        Qpointcolors_g = np.abs(gaining.Qaquifer)
        vmax = np.percentile(Qpointcolors_g, 95)
        ax.scatter(gaining.j, gaining.i,
                   s=pointsize, c=Qpointcolors_g,
                   vmax=vmax,
                   cmap='Blues')
    ax.invert_yaxis()
    ax.set_title('Simulated stream-aquifer interactions')

    plt.savefig(outfile)
    print('wrote {}'.format(outfile))
    plt.close()
    if verbose:
        print("pdf export took {:.2f}s".format(time.time() - t0))

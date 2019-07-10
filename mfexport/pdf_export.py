import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from flopy.utils import MfList


def export_pdf(filename, array, text,
               nodata=None, mfarray_type='array2d',
               float_fmt='{:.2f}'):
    t0 = time.time()
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
    print("took {:.2f}s".format(time.time() - t0))

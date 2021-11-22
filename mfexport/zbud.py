"""Functions for working with zone budget.
"""
import os
from pathlib import Path
import numpy as np
from flopy.mf6.utils.binarygrid_util import MfGrdFile


def write_zonebudget6_input(zones, budgetfile,
                            binary_grid_file=None,
                            zone_arrays_subfolder='external',
                            outname=None):

    # make sure zones is the right shape
    bgf = MfGrdFile(binary_grid_file)
    nlay, nrow, ncol = bgf.modelgrid.nlay, bgf.modelgrid.nrow, bgf.modelgrid.ncol
    layered = False
    if len(zones.shape) == 3:
        layered = True
        assert zones.shape == (nlay, nrow, ncol)
    elif len(zones.shape) == 2:
        assert zones.shape == (nrow, ncol)
        # if zones is 2d and multiple layers
        # broadcast zones to all layers
        if nlay > 1:
            layered = True
            zones = np.broadcast_to(zones, (nlay, nrow, ncol))
    else:
        assert len(zones) == nlay * nrow * ncol
    assert zones.shape[-2:] == (nrow, ncol)

    budgetfile = Path(budgetfile)
    if outname is None:
        outpath = budgetfile.parent
        outname = budgetfile.stem
    else:
        outpath = Path(outname).parent
        outname = Path(outname).stem

    output_namefile = outpath / (outname + '.zbud.nam')
    output_zonefile = outpath / (outname + '.zbud.zon')
    output_zone_array_format = outpath

    # relative path to budget file from zone budget input
    #budgetfile = budgetfile.relative_to(outpath.absolute())
    # use os.path.relpath because pathlib relative_to() 
    # only works with strings and therefore doesn't handle two paths that branch
    budgetfile = os.path.relpath(budgetfile, outpath.absolute())
    # relative path to grb file
    if binary_grid_file is not None:
        binary_grid_file = os.path.relpath(binary_grid_file, outpath.absolute())

    # write the name file
    with open(output_namefile, 'w') as dest:
        dest.write('begin zonebudget\n')
        dest.write(f'  bud {budgetfile}\n')
        dest.write(f'  zon {output_zonefile.name}\n')
        dest.write(f'  grb {binary_grid_file}\n')
        dest.write('end zonebudget\n')
    print(f'wrote {output_namefile}')

    # write the zone arrays
    zone_arrays_path = outpath / zone_arrays_subfolder
    zone_arrays_path.mkdir(exist_ok=True)
    zone_array_format = 'budget-zones_{:03d}.dat'
    open_close_text = ''
    if not layered:
        array_fname = zone_arrays_path / zone_array_format.format(0)
        np.savetxt(array_fname, zones, fmt='%d')
        array_fname = array_fname.relative_to(outpath)
        print(f'wrote {array_fname}')
        open_close_text += f'    open/close {array_fname}\n'
    else:
        for k, layer_zones in enumerate(zones):
            array_fname = zone_arrays_path / zone_array_format.format(k)
            np.savetxt(array_fname, layer_zones, fmt='%d')
            array_fname = array_fname.relative_to(outpath)
            open_close_text += f'    open/close {array_fname}\n'
            print(f'wrote {array_fname}')

    # write the zone file
    with open(output_zonefile, 'w') as dest:
        dest.write('begin dimensions\n')
        dest.write(f'  ncells {zones.size}\n')
        dest.write('end dimensions\n')
        dest.write('\n')
        dest.write('begin griddata\n')
        izone_text = 'izone'
        if layered:
            izone_text += ' layered'
        dest.write(f'  {izone_text}\n')
        dest.write(open_close_text)
        dest.write('end griddata\n')
    print(f'wrote {output_zonefile}')

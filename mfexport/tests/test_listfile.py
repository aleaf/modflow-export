"""
Tests for listfile module
"""
import os
from pathlib import Path
import pytest
from mfexport import plot_list_budget
from mfexport.listfile import get_listfile_model_version


@pytest.mark.parametrize('listfile,expected', (('mfexport/tests/data/shellmound/shellmound.list', 'mf6'),
                                               ('Examples/data/lpr/lpr_inset.list', 'mfnwt')
                                               ))
def test_get_listfile_model_version(listfile, expected):
    version = get_listfile_model_version(listfile)
    assert version == expected


@pytest.mark.parametrize('listfile,start_datetime',
                         (('mfexport/tests/data/shellmound/shellmound.list', '1998-04-01'),
                          ('Examples/data/lpr/lpr_inset.list', '2011-01-01'),
                          ('Examples/data/lpr/lpr_inset.list', None)
                                      ))
def test_export_list_budget(listfile, test_output_folder,
                                start_datetime, request):
    model_name = Path(listfile).stem
    output_path = test_output_folder / f'{model_name}_{start_datetime}'
    plot_list_budget(listfile=listfile, output_path=output_path,
                     model_start_datetime=start_datetime)
    expected_outfiles = [Path(output_path, f'pdfs/listfile_budget_summary.pdf'),
                         Path(output_path, f'pdfs/listfile_budget_by_term.pdf')
                         ]
    for outfile in expected_outfiles:
        assert outfile.exists()
        assert expected_outfiles[0].stat().st_size > 1e4
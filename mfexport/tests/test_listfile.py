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


@pytest.mark.parametrize('listfile', (('mfexport/tests/data/shellmound/shellmound.list'),
                                      ('Examples/data/lpr/lpr_inset.list')
                                      ))
def test_export_mf6_list_budget(listfile, test_output_folder):
    start_datetime = None
    model_name, _ = os.path.splitext(listfile)
    if model_name == 'shellmound':
        start_datetime = '1998-04-01'
    output_path = test_output_folder / model_name
    plot_list_budget(listfile=listfile, output_path=output_path,
                     model_start_datetime=start_datetime)
    expected_outfiles = [Path(output_path, 'pdfs/listfile_budget_summary.pdf'),
                         Path(output_path, 'pdfs/listfile_budget_by_term.pdf')
                         ]
    for outfile in expected_outfiles:
        assert outfile.exists()
        assert expected_outfiles[0].stat().st_size > 1e4
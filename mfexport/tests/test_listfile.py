"""
Tests for listfile module
"""
from pathlib import Path
from mfexport import plot_list_budget


def test_export_mf6_list_budget(model):
    model, grid, output_path = model
    start_datetime = None
    if model.name == 'shellmound':
        start_datetime = '1998-04-01'
    plot_list_budget(model=model, output_path=output_path,
                     model_start_datetime=start_datetime)
    expected_outfiles = [Path(output_path, 'pdfs/listfile_budget_summary.pdf'),
                         Path(output_path, 'pdfs/listfile_budget_by_term.pdf')
                         ]
    for outfile in expected_outfiles:
        assert outfile.exists()
        assert expected_outfiles[0].stat().st_size > 1e4
"""
Tests for listfile module
"""
from mfexport.listfile import export_mf6_list_budget


def test_export_mf6_list_budget(model):
    model, grid, output_path = model
    start_datetime = None
    if model.name == 'shellmound':
        start_datetime = '1998-04-01'
    export_mf6_list_budget(model=model, output_path=output_path,
                           model_start_datetime=start_datetime)
    j=2
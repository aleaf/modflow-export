"""
Tests for listfile module
"""
from pathlib import Path
import pytest
from mfexport import plot_list_budget
from mfexport.listfile import get_listfile_data, get_listfile_model_version


@pytest.mark.parametrize('listfile,expected', (('mfexport/tests/data/shellmound/shellmound.list', 'mf6'),
                                               ('Examples/data/lpr/lpr_inset.list', 'mfnwt')
                                               ))
def test_get_listfile_model_version(listfile, expected):
    version = get_listfile_model_version(listfile)
    assert version == expected


@pytest.mark.parametrize('listfile,model_start_date,budgetkey,expected_len',
                         (('Examples/data/lpr/lpr_inset.list', '2011-01-01', 'SFR BUDGET', 0),
                          ('Examples/data/lpr/lpr_inset.list', '2011-01-01', None, 12),
                          ('mfexport/tests/data/shellmound/shellmound.list', '1998-04-01', 'SFR BUDGET', 19),
                          )
                         )
def test_get_listfile_data(listfile, model_start_date, budgetkey, expected_len):
    df = get_listfile_data(listfile=listfile, model_start_datetime=model_start_date,
                           budgetkey=budgetkey)
    if expected_len == 0:
        pass
    else:
        assert len(df) == expected_len


@pytest.mark.parametrize('listfile,start_datetime,plot_start_date,plot_end_date,datetime_xaxis',
                         (('mfexport/tests/data/shellmound/shellmound.list', 
                           '1998-04-01', '2010-01-01', '2015-01-01', False),
                          ('mfexport/tests/data/shellmound/shellmound.list', 
                           '1998-04-01', None, None, True),
                          ('Examples/data/lpr/lpr_inset.list', '2011-01-01', None, None, True),
                          ('Examples/data/lpr/lpr_inset.list', None, None, None, True)
                                      ))
def test_export_list_budget(listfile, test_output_folder,
                            start_datetime, 
                            plot_start_date, plot_end_date,
                            datetime_xaxis):
    model_name = Path(listfile).stem
    output_path = test_output_folder / f'{model_name}_{start_datetime}'
    plot_list_budget(listfile=listfile, output_path=output_path,
                     model_start_datetime=start_datetime,
                     model_length_units='meters',
                     model_time_units='days',
                     secondary_axis_units='mgal/day',
                     plot_start_date=plot_start_date, 
                     plot_end_date=plot_end_date,
                     datetime_xaxis=datetime_xaxis
                     )
    expected_outfiles = [Path(output_path, f'pdfs/listfile_budget_summary.pdf'),
                         Path(output_path, f'pdfs/listfile_budget_by_term.pdf')
                         ]
    for outfile in expected_outfiles:
        assert outfile.exists()
        assert expected_outfiles[0].stat().st_size > 1e4
        
        
@pytest.mark.parametrize('listfile,start_datetime',
                         (('mfexport/tests/data/shellmound/shellmound.list', 
                           '1998-04-01'),
                          ('Examples/data/lpr/lpr_inset.list', '2010-12-31'),
                          ('Examples/data/lpr/lpr_inset.list', '2011-01-01')
                                      ))
def test_export_list_budget_annual_means(listfile, test_output_folder,
                            start_datetime):
    model_name = Path(listfile).stem
    output_path = test_output_folder / f'{model_name}_{start_datetime}'
    plot_list_budget(listfile=listfile, output_path=output_path,
                     model_start_datetime=start_datetime,
                     model_length_units='meters',
                     model_time_units='days',
                     secondary_axis_units='mgal/day',
                     plot_pcts=True
                     )
    expected_outfiles = [Path(output_path, f'pdfs/listfile_budget_summary.pdf'),
                         Path(output_path, f'pdfs/listfile_budget_by_term.pdf')
                         ]
    for outfile in expected_outfiles:
        assert outfile.exists()
        assert expected_outfiles[0].stat().st_size > 1e4

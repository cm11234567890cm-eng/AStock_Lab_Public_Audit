"""No slice-derived anchors or pre-listing/future event contamination."""
import pandas as pd
import pytest
from src.data.adjustment_engine import adjust_prices_asof, annotate_history_boundary, build_adjustment_state


def action(date,cash):
    return dict(symbol='TEST',ex_date=date,ex_reference_cash_dividend_per_share=cash,
        ex_reference_bonus_transfer_ratio=0,ex_reference_rights_ratio=0,rights_price=0)


def bars():
    return pd.DataFrame({'date':['2024-01-02','2024-01-03'],
        **{c:[10.,11.] for c in ['open','high','low','close']}})


def test_explicit_boundary_excludes_prelisting_poison_not_valid_past():
    actions=pd.DataFrame([action('2020-01-01',float('nan')),action('2023-06-01',1.)])
    raw=bars();before=raw.copy(deep=True)
    result=adjust_prices_asof('TEST',raw,'2024-12-31','BACK_PIT',actions,
        listing_date='2022-01-01',first_raw_bar_date='2022-01-04')
    assert result.close.tolist()==[11.,12.]
    pd.testing.assert_frame_equal(raw,before)


def test_both_listing_and_actual_first_bar_enforced():
    acts=pd.DataFrame([action('2022-01-01',1),action('2022-01-03',1),action('2022-01-04',1)])
    a=annotate_history_boundary(acts,listing_date='2022-01-02',first_raw_bar_date='2022-01-04')
    assert a.pre_listing_event_flag.tolist()==[True,True,False]
    assert 'pre_listing_event_flag' not in acts


def test_partial_history_metadata_rejected():
    with pytest.raises(ValueError,match='both'):
        adjust_prices_asof('TEST',bars(),'2024-12-31','BACK_PIT',pd.DataFrame([action('2023-01-01',1)]),listing_date='2022-01-01')


def test_empty_history_date_is_not_silently_accepted():
    with pytest.raises(ValueError):
        annotate_history_boundary(pd.DataFrame([action('2023-01-01',1)]),listing_date='',first_raw_bar_date='2022-01-01')


@pytest.mark.parametrize('method',['FRONT_PIT','BACK_PIT'])
def test_future_event_remains_invisible_with_boundary(method):
    acts=pd.DataFrame([action('2025-01-01',float('nan'))])
    got=adjust_prices_asof('TEST',bars(),'2024-12-31',method,acts,
        listing_date='2022-01-01',first_raw_bar_date='2022-01-04')
    pd.testing.assert_frame_equal(got,adjust_prices_asof('TEST',bars(),'2024-12-31','NONE'))


def test_state_and_prices_share_history_boundary():
    a=pd.DataFrame([action('2020-01-01',100),action('2023-01-01',1)])
    state=build_adjustment_state('TEST','2024-12-31',a,
        listing_date='2022-01-01',first_raw_bar_date='2022-01-04')
    assert state['effective_event_count']==1 and state['B']==-1

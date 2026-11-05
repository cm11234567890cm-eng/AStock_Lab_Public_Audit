import pandas as pd
import pytest
from scripts.phase2_readiness import REQUIRED_GATES, readiness, complete_label_dates


def test_current_qualification_is_not_historical_coverage():
    assert len(readiness({'PRICE_SOURCE_QUALIFIED':True,'active_count':5572})) == len(REQUIRED_GATES)


@pytest.mark.parametrize('unknown',[None,False,'YES',1])
def test_unknown_or_truthy_evidence_never_auto_qualifies(unknown):
    assert readiness({k:unknown for k in REQUIRED_GATES}) == list(REQUIRED_GATES)


def test_all_independent_gates_are_required():
    gates={k:True for k in REQUIRED_GATES}
    assert readiness(gates)==[]
    for key in REQUIRED_GATES:
        assert readiness(gates | {key:False}) == [key]


def test_source_endpoint_and_last_five_sessions_are_censored():
    # Synthetic sessions, not a claimed real exchange calendar.
    sessions=pd.date_range('2025-01-01',periods=20,freq='D')
    valid=complete_label_dates(sessions,'2025-01-15',5)
    assert valid.equals(sessions[:10])


def test_short_history_does_not_create_negative_labels():
    assert len(complete_label_dates(['2025-01-01','2025-01-02'],'2025-01-02'))==0


@pytest.mark.parametrize('sessions',[['2025-01-01','2025-01-01'],['2025-01-02','2025-01-01'],['2025-01-01',None]])
def test_duplicate_or_reversed_sessions_rejected(sessions):
    with pytest.raises(ValueError):complete_label_dates(sessions,'2025-01-02')


def test_nonpositive_horizon_rejected():
    with pytest.raises(ValueError):complete_label_dates(['2025-01-01'],'2025-01-01',0)

"""Two invented bars and an invented cash action; no market data or credentials."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.data.adjustment_engine import adjust_prices_asof

bars = pd.DataFrame([
    dict(date='2025-01-02', open=10.0, high=10.0, low=10.0, close=10.0),
    dict(date='2025-01-03', open=9.9, high=9.9, low=9.9, close=9.9),
])
actions = pd.DataFrame([dict(
    symbol='SYNTHETIC_AUDIT', ex_date='2025-01-03',
    cash_dividend_per_10=1.0, actual_cash_dividend_per_share=0.1,
    ex_reference_cash_dividend_per_share=0.1, differential_dividend_flag=False,
    bonus_share_per_10=0.0, transfer_share_per_10=0.0,
    rights_share_per_10=0.0, rights_price=0.0,
    event_type='CASH_DIVIDEND', status='SYNTHETIC', raw_reference='invented-example',
)])
original = bars.copy(deep=True)
before = adjust_prices_asof('SYNTHETIC_AUDIT', bars, '2025-01-02', 'FRONT_PIT', actions)
after = adjust_prices_asof('SYNTHETIC_AUDIT', bars, '2025-01-03', 'FRONT_PIT', actions)
pd.testing.assert_frame_equal(bars, original)
assert before.close.tolist() == [10.0, 9.9]
assert abs(after.close.iloc[0] - 9.9) < 1e-12
print('SYNTHETIC_ONLY: future action excluded before ex-date; effective-date adjustment passed; input unchanged.')

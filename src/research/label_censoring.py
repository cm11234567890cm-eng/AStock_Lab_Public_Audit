"""Independent ex-post label truth and announcement-free OHLCV feature state.

Event presence never qualifies economic parameters. The label truth argument
is deliberately absent from numeric feature/signal interfaces. Historical
price-basis discontinuities have a separate, past-only dependency mask.
"""
from dataclasses import dataclass
from src.universe_policy import predictive_universe
import numpy as np
import pandas as pd
from src.research.target_discovery import features,signals
from src.research.episodes import additional_features,new_signals
from src.research.coverage_recovery import FEATURE_WINDOWS,CORE_SHORT_FEATURE_SET,bounded_streak,interval_flags,eligibility

@dataclass(frozen=True)
class EventTruth:
    symbol: str
    exchange: str
    ex_date: str
    event_type: str
    event_exists: bool = True
    event_presence_qualified: bool = True
    parameter_basis_qualified: bool = False

def price_volume_values(bars,nominal_limit=.10,small_up_max=.015):
    """No event catalog, publication field, A/B, Amount or outcome input."""
    result=pd.concat([features(bars,(),nominal_limit),additional_features(bars,(),small_up_max)],axis=1)
    result['consecutive_small_up_days_10']=bounded_streak(bars,(),small_up_max)
    return result

def intrinsic_validity(bars,values):
    out={};cached={}
    for name,(old,new) in FEATURE_WINDOWS.items():
        if (old,new) not in cached:
            cached[(old,new)]=bars.close.shift(new).rolling(old-new+1,min_periods=old-new+1).count().eq(old-new+1)
        out[name]=cached[(old,new)]&np.isfinite(values[name].astype(float))
    return pd.DataFrame(out,index=bars.index)

def feature_continuity(calendar,intrinsic,historical_price_breaks,unobserved_price_basis_intervals=()):
    """Only effective historical boundaries intersect a feature's past inputs.

    No future label horizon, publication metadata or economic values enter.
    Adding/removing a boundary after t cannot change t's eligibility. This
    conservative mask does not claim a historical live feed of the ledger.
    """
    out={};break_flags={};unknown={};cache={}
    for name,(old,new) in FEATURE_WINDOWS.items():
        if (old,new) not in cache:
            cache[(old,new)]=interval_flags(calendar,historical_price_breaks,unobserved_price_basis_intervals,old,new)
        crossed,unproven=cache[(old,new)]
        out[name]=intrinsic[name].to_numpy()&~crossed&~unproven
        break_flags[name]=crossed;unknown[name]=unproven
    return tuple(pd.DataFrame(x,index=calendar) for x in (out,break_flags,unknown))

def label_state(calendar,event_truth,future_complete,unverified_presence_intervals=(),horizon=5):
    """Exclude every actual CA in (t,t+5], even if A/B was qualified.

    Neither announcement nor publication is part of this interface. A failed
    presence query remains an explicit uncovered interval; it is never NO_CA.
    """
    events=[e.ex_date for e in event_truth if e.event_exists and e.event_presence_qualified]
    crossed,unknown=interval_flags(calendar,events,unverified_presence_intervals,forward=horizon)
    complete=np.asarray(future_complete,bool)
    return pd.DataFrame(dict(LABEL_CROSSES_CA=crossed,LABEL_CA_PRESENCE_UNVERIFIED=unknown,
        label_valid=complete&~crossed&~unknown),index=calendar)

def fixed_signals(values,old_config,new_config):
    if 'exchange' in values:
        values=predictive_universe(values)
    f=values.copy()
    # The frozen J threshold is 3: min(run,10)>=3 iff run>=3. Read the
    # bounded input explicitly so CORE's decision path never needs a 60d run.
    assert new_config['J']['small_up_days_min']<=10
    f['consecutive_small_up_days']=f['consecutive_small_up_days_10']
    for col in ('ma5_above_ma10','ma10_above_ma20'):f[col]=f[col].fillna(False).astype(bool)
    result=pd.concat([signals(f,old_config)[list('ABCDE')],new_signals(f,new_config)],axis=1)
    return result

def strategy_state(values,validity,old_config,new_config,version):
    """Feature-based eligibility and signals: no label or future truth input."""
    if 'exchange' in values:
        values=predictive_universe(values)
        validity=validity.loc[values.index]
    permitted=eligibility(validity,version)
    signal=fixed_signals(values,old_config,new_config)&permitted
    return permitted,signal

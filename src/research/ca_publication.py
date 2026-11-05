"""Research-only corporate-action information sets, separate from outcome truth."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from src.research.target_discovery import features
from src.research.episodes import additional_features
from src.research.coverage_recovery import bounded_streak, FEATURE_WINDOWS

ZONE='Asia/Shanghai'

@dataclass(frozen=True)
class CorporateAction:
    event_id: str
    symbol: str
    exchange: str
    ex_date: str
    announcement_time: str | None = None
    announcement_date: str | None = None
    publication_time: str | None = None
    publication_date: str | None = None
    publication_evidence: str | None = None
    official_source_url: str | None = None
    A: float | None = None
    B: float | None = None
    qualified: bool = False

    def record(self):return asdict(self)

def local_time(value):
    t=pd.Timestamp(value)
    return t.tz_localize(ZONE) if t.tz is None else t.tz_convert(ZONE)

def cutoff(date):
    """Existing daily close research, 15:00 Asia/Shanghai; no intraday claim."""
    return local_time(date).normalize()+pd.Timedelta(hours=15)

def availability_bound(event):
    """Date-only disclosure is available no earlier than the NEXT calendar day.

    This is a conservative bound, never a fabricated publication timestamp.
    A document signature date or URL path cannot substitute for publication.
    """
    if not event.publication_evidence:return None
    if event.publication_time:return local_time(event.publication_time)
    if event.publication_date:return local_time(event.publication_date).normalize()+pd.Timedelta(days=1)
    return None

def event_known_by_t(event,date):
    available=availability_bound(event)
    return available is not None and available<=cutoff(date)

def feature_information_set(events,date):
    return tuple(e for e in events if event_known_by_t(e,date))

def qualified_tuples(events):
    return [(e.ex_date,float(e.A),float(e.B)) for e in events if e.qualified and e.A is not None and e.B is not None]

def causal_features(bars,events,nominal_limit=.1,small_up_max=.015):
    """Each output row sees only officially evidenced disclosures available at t.

    Rows are grouped by identical visible event IDs for efficiency. Label
    validity, future bars, and final event inventory do not enter the values.
    """
    groups={}
    for pos,date in enumerate(bars.index):
        known=tuple(i for i,e in enumerate(events) if e.qualified and event_known_by_t(e,date))
        groups.setdefault(known,[]).append(pos)
    chunks=[]
    for ids,positions in groups.items():
        good=qualified_tuples([events[i] for i in ids])
        f=pd.concat([features(bars,good,nominal_limit),additional_features(bars,good,small_up_max)],axis=1)
        f['consecutive_small_up_days_10']=bounded_streak(bars,good,small_up_max)
        chunks.append(f.iloc[positions])
    return pd.concat(chunks).sort_index()

def economic_publication_mask(calendar,events):
    """Ex-post research QA, NOT an observable feature or a deployed signal.

    Reject a price-basis dependency crossing a qualified action whose parameters
    were not yet evidenced as public at that row. This mask never determines
    numeric features; changing future publication cannot change past values.
    """
    dates=pd.DatetimeIndex(calendar);pos=np.arange(len(dates));out={}
    pending=[(e,np.array([not event_known_by_t(e,t) for t in dates]))
             for e in events if e.qualified]
    for name,(old,new) in FEATURE_WINDOWS.items():
        # prior_runup uses only bars through t-1, but its implementation puts
        # both extrema on t's price basis. A cash translation at t changes the
        # ratio even when all input bars precede ex_date. Shifted compression
        # and volume features instead retain their previous-row basis.
        basis_new=0 if name=='prior_runup_20' else new
        starts=dates[np.maximum(pos-old,0)];ends=dates[np.maximum(pos-basis_new,0)]
        blocked=np.zeros(len(dates),bool)
        for e,not_known in pending:
            crossing=(starts<pd.Timestamp(e.ex_date))&(ends>=pd.Timestamp(e.ex_date))
            blocked|=crossing&not_known
        out[name]=~blocked
    return pd.DataFrame(out,index=dates)

def label_cleaning_truth(calendar,actual_events,horizon=5):
    """Final existence only: publication date NEVER changes retrospective labels."""
    dates=pd.DatetimeIndex(calendar);pos=np.arange(len(dates))
    ends=dates[np.minimum(pos+horizon,len(dates)-1)]
    bad=np.zeros(len(dates),bool)
    for e in actual_events:
        if not e.qualified:bad|=(dates<pd.Timestamp(e.ex_date))&(ends>=pd.Timestamp(e.ex_date))
    return ~bad

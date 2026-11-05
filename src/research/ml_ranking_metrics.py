"""Pure, read-only metrics for out-of-fold ranking research.

The input to :func:`evaluate_oof` is deliberately already ranked.  This
module never fits, re-ranks, fills a missing rank, or changes the input frame.
Labels are used only for evaluation after the rank has been supplied.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.research.episodes import consecutive_groups
from src.universe_policy import report_audit


METHODS = ("M1", "M2", "R3_ALL", "R3_GATE")
RANK_COLUMNS = {method: f"rank_{method}" for method in METHODS}
KS = (1, 3, 5, 10, 20, 50)
THRESHOLDS = (10, 20, 30)
HORIZONS = ("2_5d", "1_5d")
SIGNALS = tuple(f"full_signal_{letter}" for letter in "ABCDEFGHIJ")
KEYS = ("exchange", "symbol", "date")

_REQUIRED = set(KEYS) | {
    "market_pos",
    "block",
    "label_valid",
    "outcome_max_return_1_5d",
    "tie_key",
    *SIGNALS,
    *RANK_COLUMNS.values(),
    *{
        f"outcome_hit{threshold}_{horizon}"
        for threshold in THRESHOLDS
        for horizon in HORIZONS
    },
}


def _is_true(values: pd.Series) -> pd.Series:
    """Return a strict boolean mask without treating missing values as False labels."""

    return values.eq(True).fillna(False).astype(bool)


def _valid_labels(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ``(known, hit, unknown)`` for one outcome column."""

    valid = _is_true(frame["label_valid"])
    known = valid & frame[column].notna()
    hit = known & _is_true(frame[column])
    return known, hit, ~known


def _scope_values(frame: pd.DataFrame) -> list[object]:
    """Stable block values, followed by the synthetic ALL scope."""

    values = list(pd.unique(frame["block"]))
    return values + ["ALL"]


def _scope_mask(frame: pd.DataFrame, block: object) -> pd.Series:
    if block == "ALL":
        return pd.Series(True, index=frame.index)
    return frame["block"].eq(block)


def _pct(numerator: int | float, denominator: int | float) -> float:
    return float(100.0 * numerator / denominator) if denominator else np.nan


def _metric_rows(frame: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    daily_rows: list[dict] = []
    for method in METHODS:
        rank = frame[RANK_COLUMNS[method]]
        for block in _scope_values(frame):
            scope = _scope_mask(frame, block)
            dates = pd.Index(pd.unique(frame.loc[scope, "date"])).sort_values()
            for k in KS:
                selected = scope & rank.notna() & rank.le(k)
                selected_count = int(selected.sum())
                for horizon in HORIZONS:
                    outcome = f"outcome_hit{10}_{horizon}"
                    # The threshold loop below replaces this temporary object;
                    # keeping the construction here explicit prevents labels
                    # from affecting which rows are selected.
                    del outcome
                    for threshold in THRESHOLDS:
                        outcome = f"outcome_hit{threshold}_{horizon}"
                        known, hits, unknown = _valid_labels(frame, outcome)
                        known_selected = selected & known
                        hits_selected = selected & hits
                        unknown_selected = selected & unknown

                        # Only selected slots can contribute to daily counts.
                        # Avoid materialising a full-universe indicator frame;
                        # OOF tables can contain millions of non-top-k rows.
                        selected_rows = frame.loc[selected, "date"]
                        day = pd.DataFrame(
                            {
                                "date": selected_rows,
                                "selected": 1,
                                "evaluable": known_selected.loc[selected].astype(int).to_numpy(),
                                "hits": hits_selected.loc[selected].astype(int).to_numpy(),
                                "unknown": unknown_selected.loc[selected].astype(int).to_numpy(),
                            }
                        ).groupby("date", sort=False).sum()
                        day = day.reindex(dates, fill_value=0)
                        success = day["hits"].gt(0)
                        unresolved = (~success) & day["unknown"].gt(0)

                        rows.append(
                            {
                                "method": method,
                                "block": block,
                                "horizon": horizon,
                                "k": k,
                                "threshold": threshold,
                                "selected": selected_count,
                                "evaluable": int(known_selected.sum()),
                                "hits": int(hits_selected.sum()),
                                "hit_rate_pct": _pct(
                                    int(hits_selected.sum()), int(known_selected.sum())
                                ),
                                "label_coverage_pct": _pct(
                                    int(known_selected.sum()), selected_count
                                ),
                                "days": int(len(dates)),
                                "successful_days": int(success.sum()),
                                "unresolved_days": int(unresolved.sum()),
                                "daily_success_lower_pct": _pct(
                                    int(success.sum()), int(len(dates))
                                ),
                                "daily_success_upper_pct": _pct(
                                    int((success | unresolved).sum()), int(len(dates))
                                ),
                            }
                        )
                        # ``daily`` is the single pooled OOF view consumed by
                        # paired day-level comparisons.  Per-block and second
                        # horizon rows remain available in ``metrics``; adding
                        # them here would duplicate a date in that view and
                        # make a date join many-to-many.
                        if block == "ALL" and horizon == "2_5d":
                            for date, values in day.iterrows():
                                known_count = int(values["evaluable"])
                                hit_count = int(values["hits"])
                                daily_rows.append(
                                    {
                                        "method": method,
                                        "block": block,
                                        "horizon": horizon,
                                        "k": k,
                                        "threshold": threshold,
                                        "date": date,
                                        "selected": int(values["selected"]),
                                        "evaluable": known_count,
                                        "hits": hit_count,
                                        "unknown": int(values["unknown"]),
                                        "daily_hit_rate_pct": _pct(hit_count, known_count),
                                        "successful": bool(hit_count > 0),
                                        "unresolved": bool(
                                            hit_count == 0 and values["unknown"] > 0
                                        ),
                                    }
                                )
    return rows, daily_rows


def _episode_membership(frame: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Build the one common 10% episode universe on frozen 1–5d labels."""

    valid = _is_true(frame["label_valid"])
    hit10 = valid & _is_true(frame["outcome_hit10_1_5d"])
    hit20 = valid & _is_true(frame["outcome_hit20_1_5d"])
    hit30 = valid & _is_true(frame["outcome_hit30_1_5d"])

    # consecutive_groups owns the frozen market_pos gap and security identity
    # rules.  A reset index also makes assignment safe for caller frames with a
    # non-unique index.
    ids = consecutive_groups(frame, hit10)
    membership = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    if len(ids):
        membership.loc[ids.index] = ids.astype("int64").to_numpy()

    members = frame.loc[membership.notna(), ["exchange", "symbol", "date", "market_pos", "block"]].copy()
    members["episode_id"] = membership.loc[members.index].astype("int64").to_numpy()
    members["hit20"] = hit20.loc[members.index].to_numpy()
    members["hit30"] = hit30.loc[members.index].to_numpy()
    return membership, members


def _episode_rows(frame: pd.DataFrame) -> list[dict]:
    membership, members = _episode_membership(frame)
    all_ids = set(members["episode_id"].astype(int))
    threshold_ids = {
        10: set(all_ids),
        20: set(members.loc[members["hit20"], "episode_id"].astype(int)),
        30: set(members.loc[members["hit30"], "episode_id"].astype(int)),
    }
    scopes = _scope_values(frame)
    rows: list[dict] = []
    selections: dict[str, pd.Series] = {
        method: frame[RANK_COLUMNS[method]].notna() for method in METHODS
    }
    for method in METHODS:
        rank = frame[RANK_COLUMNS[method]]
        for k in (3, 5, 10, 20, 50):
            selections[method] = rank.notna() & rank.le(k)
            for threshold in THRESHOLDS:
                hit_column = f"outcome_hit{threshold}_1_5d"
                _, hit, _ = _valid_labels(frame, hit_column)
                selected_hit = selections[method] & hit & membership.notna()
                for block in scopes:
                    scope = _scope_mask(frame, block)
                    involved = set(
                        members.loc[members.index[scope.loc[members.index]], "episode_id"].astype(int)
                    )
                    total_ids = sorted(involved & threshold_ids[threshold])
                    captured = set(
                        membership.loc[selected_hit & scope]
                        .dropna()
                        .astype(int)
                        .tolist()
                    ) & set(total_ids)
                    rows.append(
                        {
                            "method": method,
                            "block": block,
                            "k": k,
                            "threshold": threshold,
                            "horizon": "1_5d",
                            "total_episodes": len(total_ids),
                            "captured_episodes": len(captured),
                            "capture_pct": _pct(len(captured), len(total_ids)),
                            "episode_ids": tuple(total_ids),
                            "captured_episode_ids": tuple(sorted(captured)),
                        }
                    )

    # AJ_ALL is the union of A–J strategy dates and intentionally has no K.
    aj_selected = frame.loc[:, list(SIGNALS)].fillna(False).any(axis=1)
    for threshold in THRESHOLDS:
        hit_column = f"outcome_hit{threshold}_1_5d"
        _, hit, _ = _valid_labels(frame, hit_column)
        selected_hit = aj_selected & hit & membership.notna()
        for block in scopes:
            scope = _scope_mask(frame, block)
            involved = set(
                members.loc[members.index[scope.loc[members.index]], "episode_id"].astype(int)
            )
            total_ids = sorted(involved & threshold_ids[threshold])
            captured = set(
                membership.loc[selected_hit & scope].dropna().astype(int).tolist()
            ) & set(total_ids)
            rows.append(
                {
                    "method": "AJ_ALL",
                    "block": block,
                    "k": "ALL",
                    "threshold": threshold,
                    "horizon": "1_5d",
                    "total_episodes": len(total_ids),
                    "captured_episodes": len(captured),
                    "capture_pct": _pct(len(captured), len(total_ids)),
                    "episode_ids": tuple(total_ids),
                    "captured_episode_ids": tuple(sorted(captured)),
                }
            )
    return rows


def _opportunity(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = _is_true(frame["label_valid"]) & frame["outcome_max_return_1_5d"].notna()
    legal = frame.loc[valid].copy()
    if legal.empty:
        winners = pd.DataFrame(
            columns=[*KEYS, "tie_key", "winner_return", "tie_count", "is_primary", *RANK_COLUMNS.values()]
        )
        return pd.DataFrame(), winners

    peak = legal.groupby("date", sort=False)["outcome_max_return_1_5d"].transform("max")
    winners = legal.loc[legal["outcome_max_return_1_5d"].eq(peak)].copy()
    winners = winners.rename(columns={"outcome_max_return_1_5d": "winner_return"})
    winners["tie_count"] = winners.groupby("date")["date"].transform("size")
    winners = winners.sort_values(["date", "tie_key"], kind="stable")
    winners["is_primary"] = ~winners["date"].duplicated()

    summary: list[dict] = []
    for block in _scope_values(frame):
        if block == "ALL":
            block_dates = pd.Index(pd.unique(frame["date"]))
        else:
            block_dates = pd.Index(pd.unique(frame.loc[_scope_mask(frame, block), "date"]))
        scoped = winners[winners["date"].isin(block_dates)]
        primary = scoped[scoped["is_primary"]]
        days = int(legal.loc[legal["date"].isin(block_dates), "date"].nunique())
        for method in METHODS:
            rank_column = RANK_COLUMNS[method]
            primary_effective = primary[rank_column].astype(float).fillna(np.inf).to_numpy()
            co_effective = scoped[rank_column].astype(float).fillna(np.inf).to_numpy()
            q = (
                [
                    float(np.quantile(primary_effective, p, method="inverted_cdf"))
                    for p in (0.25, 0.5, 0.75)
                ]
                if len(primary_effective)
                else [np.nan, np.nan, np.nan]
            )
            summary_row = {
                "method": method,
                "block": block,
                "days": days,
                "winner_rank_p25": q[0],
                "winner_rank_median": q[1],
                "winner_rank_p75": q[2],
                "unranked_days": int(np.isinf(primary_effective).sum()),
                "multiple_max_winner_days": int((scoped.groupby("date").size() > 1).sum()),
            }
            for k in (3, 5, 10, 20, 50, 100):
                summary_row[f"winner_in_top{k}_pct"] = _pct(
                    int((primary_effective <= k).sum()), days
                )
                by_date = scoped.assign(_in=co_effective <= k).groupby("date")['_in'].any()
                summary_row[f"any_co_winner_in_top{k}_pct"] = _pct(int(by_date.sum()), days)
            summary.append(summary_row)

    for method in METHODS:
        rank_column = RANK_COLUMNS[method]
        winners[f"{rank_column}_effective"] = winners[rank_column].astype(float).fillna(np.inf)
    keep = [
        *KEYS,
        "tie_key",
        "winner_return",
        "tie_count",
        "is_primary",
        *RANK_COLUMNS.values(),
        *[f"{c}_effective" for c in RANK_COLUMNS.values()],
    ]
    return pd.DataFrame(summary), winners[keep].reset_index(drop=True)


def evaluate_oof(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Evaluate supplied OOF ranks and return five tabular research outputs.

    The function is pure with respect to ``frame``.  ``metrics`` and ``daily``
    use both supplied label horizons; ``episodes`` use the frozen 1–5 day
    common-10% universe.  Missing labels remain unknown throughout.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("FRAME_REQUIRED")
    audit = report_audit(frame)
    missing = sorted(_REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"MISSING_COLUMNS: {missing}")
    work = frame.copy(deep=True).reset_index(drop=True)
    metric_rows, daily_rows = _metric_rows(work)
    episodes = pd.DataFrame(_episode_rows(work))
    opportunity, winners = _opportunity(work)
    result = {
        "metrics": pd.DataFrame(metric_rows),
        "daily": pd.DataFrame(daily_rows),
        "opportunity": opportunity,
        "winners": winners,
        "episodes": episodes,
    }
    for table in result.values():
        table['bse_candidate_count'] = audit['bse_candidate_count']
    return result


__all__ = ["evaluate_oof"]

"""A-share execution constraints for future *simulation only*.

No function connects to a broker or submits an order. Backtests must ask this
module whether a hypothetical trade is executable at an observed bar price.
"""
from __future__ import annotations

from dataclasses import dataclass
from src.universe_policy import exclusion_reason


@dataclass(frozen=True)
class TradeDecision:
    allowed: bool
    reason: str


def price_limit_ratio(*, board: str, st_flag: bool, days_since_listing: int | None) -> float | None:
    """Return the daily price-limit ratio, or None for a new-listing unrestricted phase."""
    if days_since_listing is not None and days_since_listing < 5:
        return None
    if st_flag:
        return 0.05
    if board.upper() in {"CHINEXT", "STAR", "BSE"}:
        return 0.20 if board.upper() != "BSE" else 0.30
    return 0.10


def can_hypothetically_buy(*, quantity: int, suspended: bool, one_word_limit_up: bool, at_limit_up: bool, exchange: str | None = None, symbol: str | None = None) -> TradeDecision:
    reason = exclusion_reason(exchange, symbol)
    if reason:
        return TradeDecision(False, reason)
    if quantity <= 0 or quantity % 100:
        return TradeDecision(False, "A股买入必须为100股整数手")
    if suspended:
        return TradeDecision(False, "停牌不可交易")
    if one_word_limit_up or at_limit_up:
        return TradeDecision(False, "涨停价无法假设买入成交")
    return TradeDecision(True, "可模拟买入")


def can_hypothetically_sell(*, quantity: int, shares_bought_today: int, suspended: bool, at_limit_down: bool) -> TradeDecision:
    if quantity <= 0:
        return TradeDecision(False, "卖出数量必须大于零")
    if suspended:
        return TradeDecision(False, "停牌不可交易")
    if at_limit_down:
        return TradeDecision(False, "跌停价无法假设卖出成交")
    if shares_bought_today >= quantity:
        return TradeDecision(False, "T+1：当日买入股份不可当日卖出")
    return TradeDecision(True, "可模拟卖出")

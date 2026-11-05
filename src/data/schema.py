"""Canonical bar schema for every input adapter.

All columns are intentionally present so missing source fields remain explicit as
NA rather than being fabricated.
"""

CANONICAL_BAR_COLUMNS = [
    "symbol", "name", "exchange", "date", "open", "high", "low", "close",
    "volume", "amount", "turnover", "adj_factor", "market_cap", "pe", "pb",
    "roe", "net_profit", "operating_cashflow", "industry", "concept",
    "northbound", "main_money_flow", "limit_up", "limit_down", "st_flag",
    "suspend_flag",
]

REQUIRED_OHLCV_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]

ALIASES = {
    "datetime": "date", "trade_date": "date", "code": "symbol", "ticker": "symbol",
    "vol": "volume", "成交量": "volume", "成交额": "amount", "amount": "amount",
}

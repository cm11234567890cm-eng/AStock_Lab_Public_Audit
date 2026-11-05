from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import ALIASES, CANONICAL_BAR_COLUMNS, REQUIRED_OHLCV_COLUMNS


class DataAdapter:
    """Read CSV, SQLite and Parquet inputs into one explicit A-share schema.

    The adapter only reads external paths. It never refreshes, rewrites or
    normalizes source files in place.
    """

    def read_file(self, path: Path, table: Optional[str] = None) -> pd.DataFrame:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, dtype={"symbol": "string", "code": "string"})
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix in {".sqlite", ".db"}:
            with sqlite3.connect(path) as connection:
                if table is None:
                    found = connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 1"
                    ).fetchone()
                    if not found:
                        raise ValueError(f"SQLite database has no table: {path}")
                    table = found[0]
                return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
        raise ValueError(f"Unsupported input format: {path.suffix}")

    def canonicalize(self, frame: pd.DataFrame, symbol_hint: Optional[str] = None) -> pd.DataFrame:
        frame = frame.rename(columns={key: value for key, value in ALIASES.items() if key in frame.columns}).copy()
        if "symbol" not in frame.columns:
            frame["symbol"] = symbol_hint or pd.NA
        frame["symbol"] = frame["symbol"].astype("string").str.replace(r"^(sh|sz|bj)", "", regex=True).str.zfill(6)
        if "date" not in frame.columns:
            raise ValueError("DATA_NOT_AVAILABLE: input has no date column")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        for column in ("open", "high", "low", "close", "volume", "amount", "turnover", "adj_factor"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "exchange" not in frame.columns:
            frame["exchange"] = frame["symbol"].map(self._exchange_for_symbol)
        for column in CANONICAL_BAR_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        output = frame[CANONICAL_BAR_COLUMNS].dropna(subset=["symbol", "date"]).sort_values(["symbol", "date"])
        return output.drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)

    def load_symbol(self, daily_directory: Path, symbol: str, tail: Optional[int] = None) -> pd.DataFrame:
        symbol = str(symbol).replace("sh", "").replace("sz", "").replace("bj", "").zfill(6)
        candidates = [Path(daily_directory) / f"{symbol}.csv"]
        candidates += list(Path(daily_directory).glob(f"*{symbol}.csv"))
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"DATA_NOT_AVAILABLE: symbol {symbol} is not available")
        result = self.canonicalize(self.read_file(path), symbol_hint=symbol)
        return result.tail(tail).reset_index(drop=True) if tail else result

    @staticmethod
    def _exchange_for_symbol(symbol: object) -> str:
        value = str(symbol)
        if value.startswith(("6", "9")):
            return "SSE"
        if value.startswith(("8", "4")):
            return "BSE"
        return "SZSE"

    @staticmethod
    def validate_required(frame: pd.DataFrame) -> list[str]:
        return [column for column in REQUIRED_OHLCV_COLUMNS if column not in frame.columns]

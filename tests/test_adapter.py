import sqlite3

import pandas as pd

from src.data.adapter import DataAdapter


def test_csv_is_canonicalized(tmp_path):
    path = tmp_path / "000001.csv"
    path.write_text("date,open,high,low,close,volume\n2026-01-02,1,2,0.5,1.5,100\n", encoding="utf-8")
    result = DataAdapter().canonicalize(DataAdapter().read_file(path), symbol_hint="000001")
    assert result.loc[0, "symbol"] == "000001"
    assert result.loc[0, "exchange"] == "SZSE"
    assert result.loc[0, "close"] == 1.5
    assert "adj_factor" in result.columns and pd.isna(result.loc[0, "adj_factor"])


def test_sqlite_is_supported(tmp_path):
    path = tmp_path / "bars.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE bars (date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL, symbol TEXT)")
        connection.execute("INSERT INTO bars VALUES ('2026-01-02',1,2,0.5,1.5,100,'600000')")
    result = DataAdapter().canonicalize(DataAdapter().read_file(path, table="bars"))
    assert result.loc[0, "exchange"] == "SSE"

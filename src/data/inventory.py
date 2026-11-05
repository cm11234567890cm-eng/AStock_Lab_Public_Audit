from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


SUPPORTED_SUFFIXES = {".csv", ".parquet", ".pq", ".sqlite", ".db"}
CORE_FIELDS = ("date", "open", "high", "low", "close", "volume")


@dataclass
class InventorySummary:
    source: str
    directory: str
    source_status: str
    files: int
    stocks: int
    start_date: str | None
    end_date: str | None
    fields: list[str]
    adjustment: str
    latest_file_update: str | None
    missing_rate: float | None
    duplicate_rate: float | None
    notes: list[str]


def _iso_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:10]).date().isoformat()
    except ValueError:
        return None


class DataInventory:
    """Evidence-first inventory; external sources are only opened for reading."""

    def __init__(self, config_path: Path, project_root: Path):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.project_root = Path(project_root)

    def scan(self) -> list[InventorySummary]:
        summaries: list[InventorySummary] = []
        for root_text in self.config["read_only_roots"]:
            root = Path(root_text)
            if not root.exists():
                summaries.append(InventorySummary(
                    source=root.name, directory=str(root), source_status="DATA_NOT_AVAILABLE", files=0, stocks=0,
                    start_date=None, end_date=None, fields=[], adjustment="DATA_NOT_AVAILABLE",
                    latest_file_update=None, missing_rate=None, duplicate_rate=None,
                    notes=["Configured scan root does not exist; no fallback data was invented."],
                ))
            else:
                summaries.append(self._scan_root(root))
        primary = Path(self.config["primary_daily_source"])
        summaries.append(self._scan_daily_csv(primary, "primary_daily_csv"))
        for candidate_text in self.config.get("additional_daily_candidates", []):
            candidate = Path(candidate_text)
            if candidate != primary:
                summaries.append(self._scan_daily_csv(candidate, f"candidate_daily_csv:{candidate.name}"))
        return summaries

    def _scan_root(self, root: Path) -> InventorySummary:
        """Register each requested scan root without assuming its files are bars."""
        try:
            files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
        except OSError:
            files = []
        label = "_".join(part for part in root.parts if part not in {root.drive, "\\"}).replace(":", "")
        return InventorySummary(
            source=f"scan_root:{label}", directory=str(root), source_status="DISCOVERED",
            files=len(files), stocks=0, start_date=None, end_date=None,
            fields=[], adjustment="DATA_NOT_AVAILABLE", latest_file_update=None,
            missing_rate=None, duplicate_rate=None,
            notes=["Discovery-only scan root. It is not assumed to be valid daily-bar data until a source adapter validates dates and schema."],
        )

    def _scan_daily_csv(self, directory: Path, source_name: str) -> InventorySummary:
        if not directory.exists():
            return InventorySummary(source_name, str(directory), "DATA_NOT_AVAILABLE", 0, 0, None, None, [],
                                    "DATA_NOT_AVAILABLE", None, None, None, ["Primary source is absent."])
        files = sorted(directory.glob("*.csv"))
        all_fields: set[str] = set()
        min_date: str | None = None
        max_date: str | None = None
        latest_mtime: float | None = None
        missing = duplicate = rows = 0
        readable = 0
        adjustment_values: set[str] = set()
        for path in files:
            latest_mtime = max(latest_mtime or 0, path.stat().st_mtime)
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    fields = reader.fieldnames or []
                    all_fields.update(fields)
                    seen_dates: set[str] = set()
                    file_rows = 0
                    for row in reader:
                        file_rows += 1
                        rows += 1
                        parsed = _iso_date(row.get("date"))
                        if parsed:
                            min_date = min(min_date, parsed) if min_date else parsed
                            max_date = max(max_date, parsed) if max_date else parsed
                            if parsed in seen_dates:
                                duplicate += 1
                            seen_dates.add(parsed)
                        for field in CORE_FIELDS:
                            if not str(row.get(field, "")).strip():
                                missing += 1
                        flag = str(row.get("adjusted_flag", "")).strip()
                        if flag:
                            adjustment_values.add(flag)
                    readable += int(file_rows > 0 and "date" in fields)
            except (UnicodeDecodeError, OSError, csv.Error):
                continue
        denominator = rows * len(CORE_FIELDS)
        latest_update = datetime.fromtimestamp(latest_mtime).isoformat(timespec="seconds") if latest_mtime else None
        age_days = (date.today() - date.fromisoformat(max_date)).days if max_date else None
        status = "OK" if readable >= 100 and age_days is not None and age_days <= 7 else "STALE_DATA" if readable else "INSUFFICIENT_COVERAGE"
        notes = [
            "Read-only external CSV source; no source file was changed.",
            "Rows without a parseable ISO date are not counted as valid market dates.",
        ]
        if age_days is not None and age_days > 7:
            notes.append(f"STALE_DATA: latest market date is {max_date}, {age_days} calendar days behind local date {date.today()}.")
        if readable < 100:
            notes.append("INSUFFICIENT_COVERAGE: fewer than 100 readable symbols.")
        return InventorySummary(
            source=source_name, directory=str(directory), source_status=status, files=len(files), stocks=readable,
            start_date=min_date, end_date=max_date, fields=sorted(all_fields),
            adjustment=", ".join(sorted(adjustment_values)) if adjustment_values else "DATA_NOT_AVAILABLE",
            latest_file_update=latest_update, missing_rate=round(missing / denominator, 8) if denominator else None,
            duplicate_rate=round(duplicate / rows, 8) if rows else None, notes=notes,
        )

    def write_artifacts(self, summaries: Iterable[InventorySummary]) -> dict:
        records = [asdict(item) for item in summaries]
        primary = next((record for record in records if record["source"] == "primary_daily_csv"), {})
        health = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "data_mode": self.config["data_mode"],
            "latest_stock_date": primary.get("end_date"),
            "readable_stocks": primary.get("stocks", 0),
            "status": primary.get("source_status", "DATA_NOT_AVAILABLE"),
            "missing_rate": primary.get("missing_rate"),
            "duplicate_rate": primary.get("duplicate_rate"),
            "adjustment": primary.get("adjustment"),
            "external_source": primary.get("directory"),
        }
        docs = self.project_root / "docs"
        outputs = self.project_root / "outputs"
        docs.mkdir(exist_ok=True)
        outputs.mkdir(exist_ok=True)
        lines = ["# 数据盘点 / Data Inventory", "", f"生成时间：{health['generated_at']}", "", "## 结论", ""]
        lines += [f"- 状态：`{health['status']}`", f"- 最新股票数据日期：`{health['latest_stock_date'] or 'DATA_NOT_AVAILABLE'}`", f"- 可读取股票数：`{health['readable_stocks']}`", ""]
        for record in records:
            lines += [f"## {record['source']}", "", f"- 目录：`{record['directory']}`", f"- 状态：`{record['source_status']}`", f"- 文件数：{record['files']}；股票数：{record['stocks']}", f"- 日期范围：{record['start_date'] or 'DATA_NOT_AVAILABLE'} 至 {record['end_date'] or 'DATA_NOT_AVAILABLE'}", f"- 字段：{', '.join(record['fields']) or 'DATA_NOT_AVAILABLE'}", f"- 复权情况：{record['adjustment']}", f"- 文件最近更新时间：{record['latest_file_update'] or 'DATA_NOT_AVAILABLE'}", f"- 缺失率：{record['missing_rate'] if record['missing_rate'] is not None else 'DATA_NOT_AVAILABLE'}", f"- 重复率：{record['duplicate_rate'] if record['duplicate_rate'] is not None else 'DATA_NOT_AVAILABLE'}"]
            lines += [f"- 说明：{note}" for note in record["notes"]] + [""]
        (docs / "data_inventory.md").write_text("\n".join(lines), encoding="utf-8")
        (outputs / "data_health.json").write_text(json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8")
        self._write_metadata(records, health)
        return health

    def _write_metadata(self, records: list[dict], health: dict) -> None:
        path = self.project_root / "data" / "metadata.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS source_registry (source TEXT PRIMARY KEY, payload TEXT NOT NULL, scanned_at TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS data_health (generated_at TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            now = health["generated_at"]
            for record in records:
                connection.execute("INSERT OR REPLACE INTO source_registry VALUES (?, ?, ?)", (record["source"], json.dumps(record, ensure_ascii=False), now))
            connection.execute("INSERT OR REPLACE INTO data_health VALUES (?, ?)", (now, json.dumps(health, ensure_ascii=False)))

import struct

from src.data.tdx_discovery import RECORD, classify_symbol, parse_day_file


def test_day_record_is_verified_before_parsing(tmp_path):
    path = tmp_path / "sh600000.day"
    path.write_bytes(RECORD.pack(20260929, 1000, 1100, 900, 1050, 1234.5, 10000, 0) + RECORD.pack(20260930, 1050, 1150, 1000, 1100, 2345.0, 12000, 0))
    parsed = parse_day_file(path)
    assert parsed["status"] == "PARSED"
    assert parsed["latest_date"] == "2026-09-30"
    assert parsed["board"] == "SSE_MAIN"


def test_non_multiple_record_length_is_never_hard_parsed(tmp_path):
    path = tmp_path / "sz300001.day"
    path.write_bytes(struct.pack("<I", 20260930))
    assert parse_day_file(path)["status"] == "FORMAT_SIZE_MISMATCH"


def test_board_rules_do_not_rely_on_directory():
    assert classify_symbol("688001") == ("SSE", "STAR")
    assert classify_symbol("300750") == ("SZSE", "CHINEXT")
    assert classify_symbol("430047", "bj") == ("BSE", "BSE")
    assert classify_symbol("920001", "bj") == ("BSE", "BSE")
    assert classify_symbol("880001", "sh") == ("UNKNOWN", "OTHER_OR_UNCLASSIFIED")

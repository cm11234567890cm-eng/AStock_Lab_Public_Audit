from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from src.data.adapter import DataAdapter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((PROJECT_ROOT / "config" / "data_sources.json").read_text(encoding="utf-8"))
OUTPUTS = PROJECT_ROOT / "outputs"
adapter = DataAdapter()
app = FastAPI(title="AStock Lab", docs_url=None, redoc_url=None)


def _read_json(path: Path, fallback: dict) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AStock Lab</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>body{font-family:system-ui,sans-serif;margin:auto;max-width:1100px;padding:16px;background:#f5f7fb;color:#172033}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.card,table{background:#fff;border-radius:10px;padding:14px;box-shadow:0 1px 4px #dce1eb}table{width:100%;border-collapse:collapse;margin-top:12px}th,td{text-align:left;padding:9px;border-bottom:1px solid #edf0f5}.bad{color:#b42318;font-weight:700}.note{color:#5b6475}@media(max-width:600px){body{padding:10px}th,td{padding:7px;font-size:13px}}</style></head><body><h1>AStock Lab</h1><p class="bad" id="gate">STALE_DATA — 未通过数据验收，禁止显示今日候选或 READY。</p><p class="note">仅本地只读研究与模拟观察；不是投资建议，不连接券商，不会下单。</p><div class="grid" id="health"></div><h2>历史观察样本（非预测排名）</h2><p class="note">仅展示历史本地日线；综合评分、3个月潜力均为 DATA_NOT_AVAILABLE。</p><table><thead><tr><th>代码</th><th>交易所</th><th>日期</th><th>收盘</th><th>状态</th></tr></thead><tbody id="rows"></tbody></table><h2>个股图</h2><select id="symbol"></select><button onclick="chart()">查看</button><div id="chart"></div><script>async function boot(){let h=await (await fetch('/api/health')).json(),shown=h.dashboard_display_status||h.status;document.getElementById('gate').textContent=shown+' — Expected: '+(h.expected_latest_trade_date||'DATA_NOT_AVAILABLE')+' / Actual: '+(h.actual_latest_trade_date||h.latest_stock_date||'DATA_NOT_AVAILABLE')+'；未通过数据验收。';document.getElementById('health').innerHTML=[['验收显示',shown],['Expected',h.expected_latest_trade_date||'DATA_NOT_AVAILABLE'],['Actual',h.actual_latest_trade_date||h.latest_stock_date||'DATA_NOT_AVAILABLE'],['源状态',h.status],['复权',h.adjustment_type||h.adjustment||'DATA_NOT_AVAILABLE']].map(x=>`<div class="card"><b>${x[0]}</b><br><span class="${x[1]!=='PASS'?'bad':''}">${x[1]}</span></div>`).join('');let s=await (await fetch('/api/samples')).json();document.getElementById('rows').innerHTML=s.samples.map(x=>`<tr><td>${x.symbol}</td><td>${x.exchange}</td><td>${x.date}</td><td>${x.close}</td><td>STALE_DATA</td></tr>`).join('');document.getElementById('symbol').innerHTML=s.samples.map(x=>`<option>${x.symbol}</option>`).join('');}async function chart(){let symbol=document.getElementById('symbol').value;let x=await (await fetch('/api/symbol/'+symbol)).json();Plotly.newPlot('chart',[{x:x.bars.map(b=>b.date),open:x.bars.map(b=>b.open),high:x.bars.map(b=>b.high),low:x.bars.map(b=>b.low),close:x.bars.map(b=>b.close),type:'candlestick'}],{margin:{t:25},height:420});}boot();</script></body></html>'''


@app.get("/api/health")
def health() -> dict:
    return _read_json(OUTPUTS / "data_health.json", {"status": "DATA_NOT_AVAILABLE", "latest_stock_date": None, "readable_stocks": 0, "adjustment": "DATA_NOT_AVAILABLE"})


@app.get("/api/samples")
def samples() -> dict:
    return _read_json(OUTPUTS / "dashboard_snapshot.json", {"samples": [], "status": "DATA_NOT_AVAILABLE"})


@app.get("/api/symbol/{symbol}")
def symbol_bars(symbol: str) -> dict:
    if not symbol.isdigit() or len(symbol) != 6:
        raise HTTPException(400, "symbol must be a six-digit A-share code")
    try:
        bars = adapter.load_symbol(Path(CONFIG["primary_daily_source"]), symbol, tail=180)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"symbol": symbol, "bars": json.loads(bars[["date", "open", "high", "low", "close", "volume"]].to_json(orient="records", date_format="iso"))}

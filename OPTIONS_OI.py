
import os
import smtplib
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

# Imports handled by standard library and common data packages
try:
    from fyers_apiv3 import fyersModel
except ImportError:
    try:
        from fyersapi import fyersModel
    except ImportError:
        fyersModel = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# CONFIG
DAILY_LOOKBACK_DAYS = 90
INTRADAY_LOOKBACK_DAYS = 20
IVP_LOOKBACK_DAYS = 252
MIN_OPTION_LTP = 10.0
ITERATIONS_TO_KEEP = 75
OPTION_EMAIL_COLS = [
    "Underlying", "Option Type", "Option Symbol", "Strike", "LTP", "% Change",
    "OI", "Prev OI", "OI Delta",
    "Volume", "Prev Volume", "Volume Delta",
    "OBV", "Prev OBV", "OBV Delta",
    "OI+Volume+OBV Score", "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal",
    "Bull_Signal", "Bear_Signal", "Overall_Signal", "Price_Lead_Status",
    "IVP", "Volatility State", "Last Iteration Time",
]

fyers = None

def init_fyers():
    global fyers
    cid = os.environ.get("CLIENT_ID")
    at = os.environ.get("ACCESS_TOKEN")
    if not cid or not at: return None
    fyers = fyersModel.FyersModel(client_id=cid, is_async=False, token=at, log_path="")
    return fyers

def get_history(symbol: str, resolution: str, days_back: int, include_oi: bool = False) -> pd.DataFrame:
    if fyers is None: return pd.DataFrame()
    now = datetime.now()
    start = now - timedelta(days=days_back)
    payload = {"symbol": symbol, "resolution": resolution, "date_format": "1", "range_from": start.strftime("%Y-%m-%d"), "range_to": now.strftime("%Y-%m-%d"), "cont_flag": "1"}
    if include_oi: payload["oi_flag"] = "1"
    try: res = fyers.history(data=payload)
    except: return pd.DataFrame()
    candles = (res or {}).get("candles", [])
    if not candles: return pd.DataFrame()
    width = max(len(c) for c in candles)
    cols = ["timestamp", "open", "high", "low", "close", "volume", "oi"] if width >= 7 else ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame([c[:len(cols)] for c in candles], columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

def _compute_obv_on_window(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2: return np.nan
    close, vol = pd.to_numeric(df["close"], errors="coerce"), pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    obv = 0.0
    for i in range(1, len(df)):
        if pd.isna(close.iloc[i]) or pd.isna(close.iloc[i-1]): continue
        if close.iloc[i] > close.iloc[i-1]: obv += vol.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]: obv -= vol.iloc[i]
    return round(obv, 2)

def _extract_session_metrics(df: pd.DataFrame, session_date, end_time_value) -> Dict[str, float]:
    d = df.copy().sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d[d["timestamp"].dt.date == session_date].copy()
    if d.empty: return {"oi": np.nan, "volume": np.nan, "obv": np.nan}
    d = d[(d["timestamp"] >= pd.Timestamp.combine(session_date, time(9, 15))) & (d["timestamp"] <= pd.Timestamp.combine(session_date, end_time_value))].copy()
    if d.empty: return {"oi": np.nan, "volume": np.nan, "obv": np.nan}
    v, o = round(pd.to_numeric(d["volume"], errors="coerce").fillna(0.0).sum(), 2), np.nan
    for oc in ["oi", "open_interest", "OI"]:
        if oc in d.columns:
            s = pd.to_numeric(d[oc], errors="coerce").dropna()
            if not s.empty: o = round(float(s.iloc[-1]), 2); break
    return {"oi": o, "volume": v, "obv": _compute_obv_on_window(d)}

def compare_today_vs_previous_metrics(df: pd.DataFrame) -> Dict[str, float]:
    blank = {"OI": np.nan, "Prev OI": np.nan, "OI Delta": np.nan, "Volume": np.nan, "Prev Volume": np.nan, "Volume Delta": np.nan, "OBV": np.nan, "Prev OBV": np.nan, "OBV Delta": np.nan}
    if df is None or df.empty: return blank
    d = df.sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    dates = sorted(d["timestamp"].dt.date.unique())
    if len(dates) < 1: return blank
    cur_d, prev_d = dates[-1], ([x for x in dates if x < dates[-1]] or [None])[-1]
    cur_day = d[d["timestamp"].dt.date == cur_d]
    cur_day = cur_day[cur_day["timestamp"] >= pd.Timestamp.combine(cur_d, time(9, 15))]
    if cur_day.empty: return blank
    end_t = cur_day["timestamp"].iloc[-1].time()
    cur = _extract_session_metrics(d, cur_d, end_t)
    prev = _extract_session_metrics(d, prev_d, end_t) if prev_d else {"oi": np.nan, "volume": np.nan, "obv": np.nan}
    return {
        "OI": cur["oi"], "Prev OI": prev["oi"], "OI Delta": round(cur["oi"]-prev["oi"], 2) if pd.notna(cur["oi"]) and pd.notna(prev["oi"]) else np.nan,
        "Volume": cur["volume"], "Prev Volume": prev["volume"], "Volume Delta": round(cur["volume"]-prev["volume"], 2) if pd.notna(cur["volume"]) and pd.notna(prev["volume"]) else np.nan,
        "OBV": cur["obv"], "Prev OBV": prev["obv"], "OBV Delta": round(cur["obv"]-prev["obv"], 2) if pd.notna(cur["obv"]) and pd.notna(prev["obv"]) else np.nan,
    }

def format_cell(col, val):
    if pd.isna(val): return ""
    return f"{float(val):.2f}%" if col == "% Change" else f"{float(val):.2f}" if isinstance(val, (int, float, np.integer, np.floating)) else str(val)

def _cell_bg(col: str, value: str) -> str:
    v = str(value).strip().upper()
    if col in {"% Change", "OI Delta", "Volume Delta", "OBV Delta", "OBV"}:
        try: num = float(str(value).replace("%", "").replace(",", "").strip())
        except: return '#2d3651'
        return '#2e7d32' if num > 0 else '#c62828' if num < 0 else '#546e7a'
    if col in {"5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal", "Bull_Signal", "Bear_Signal", "Overall_Signal"}:
        if "LONG++" in v or "BUY++" in v: return '#2e7d32'
        if "LONG+" in v or "BUY+" in v: return '#388e3c'
        if v in {"LONG", "BUY"}: return '#43a047'
        if "SHORT++" in v or "SELL++" in v: return '#c62828'
        if "SHORT+" in v or "SELL+" in v: return '#d32f2f'
        if v in {"SHORT", "SELL"}: return '#e53935'
        return '#6b7280'
    return '#2d3651'

def colored_table_html(df: pd.DataFrame, columns: List[str], title: str) -> str:
    html = [f"<div style='margin:10px 0 4px 0; font-family:Arial,Helvetica,sans-serif; font-size:13px;'><b>{title}</b></div>"]
    if df is None or df.empty: return "".join(html) + "<div style='font-size:12px;'>No candidates.</div>"
    view = df[[c for c in columns if c in df.columns]]
    html.append("<table cellpadding='0' cellspacing='1' style='border-spacing:1px; background:#fff; font-family:Arial,Helvetica,sans-serif; font-size:10px;'>")
    html.append("<tr>" + "".join([f"<th style='background:#2f3b59; color:#fff; padding:5px 6px;'>{c}</th>" for c in view.columns]) + "</tr>")
    for _, row in view.iterrows():
        html.append("<tr>" + "".join([f"<td style='background:{_cell_bg(c, format_cell(c, row[c]))}; color:#fff; text-align:center; padding:4px 6px;'>{format_cell(c, row[c])}</td>" for c in view.columns]) + "</tr>")
    return "".join(html) + "</table>"

# Placeholder for remaining business logic (summarize_intraday, build_option_candidates, etc.) 
# Note: For full function, ensure these are included.
if __name__ == "__main__":
    init_fyers()
    # Main logic loop

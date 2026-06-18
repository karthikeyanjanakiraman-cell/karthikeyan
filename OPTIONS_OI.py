#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py

Optimized Index & Options Dual-Verification Scanner via Fyers API.
- CORE INDICES: Tracks Nifty 50, Bank Nifty, Sensex (Multi-timeframe peaks).
- DYNAMIC 21-STRIKE ROUTING: Generates a 21-strike net for deep verification.
- HOLY GRAIL DIVERGENCE: Cross-verifies Spot Sweeps against Option Premium behavior.
- CSV AUDIT: Saves full strike lists to CSV for backtesting.
"""

import os
import sys
import logging
import warnings
import calendar
from datetime import datetime, timedelta, time
import time as time_module
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


class UTF8Formatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        record.msg = msg.encode("ascii", "ignore").decode("ascii")
        return super().format(record)


# Initialize Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = UTF8Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)

warnings.filterwarnings("ignore")

# Configuration
DAILY_LOOKBACK_DAYS = 200  
INTRADAY_LOOKBACK_DAYS = 30
HISTORY_API_MAX_SPAN_DAYS = 99
FYERS_RATE_LIMIT_SLEEP = 0.31
FYERS_RETRY_SLEEP = 2.0
FYERS_MAX_RETRIES = 2

fyers: Optional[fyersModel.FyersModel] = None

INDEX_SYMBOLS = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]

EMAIL_DISPLAY_COLS = [
    "Symbol", "LTP", "% Change", "Signal_Type", "Timeframe",         
    "Top_Band", "Bottom_Band", "Climax_Date", "ATM_Strike",        
    "Option_Contracts", "MTF_15m", "MTF_30m", "MTF_60m", "Last Iteration Time"
]

EMAIL_OPT_COLS = [
    "Symbol", "LTP", "% Change", "Signal_Type", "Timeframe",
    "Top_Band", "Climax_Date", "Breach_Days"
]

# ==========================================
# GLOBAL HELPERS
# ==========================================

def format_value(col: str, val):
    if pd.isna(val) or val == float("inf") or val == float("-inf"): return ""
    if col in ["Timeframe", "Signal_Type", "Option_Contracts", "Breach_Days"]: return str(val)
    if col == "% Change": return f"{float(val):.2f}%"
    if col in ["Top_Band", "Bottom_Band", "ATM_Strike"]: return f"{float(val):.2f}"
    if col == "Climax_Date": return str(val)
    if isinstance(val, (int, float, np.integer, np.floating)): return f"{float(val):.4f}"
    return str(val)

def get_index_meta(symbol: str) -> Tuple[str, str, int]:
    if "NIFTY50" in symbol: return "NSE", "NIFTY", 50
    elif "NIFTYBANK" in symbol: return "NSE", "BANKNIFTY", 100
    elif "SENSEX" in symbol: return "BSE", "SENSEX", 100
    return "NSE", "NIFTY", 50

def get_underlying_spot(opt_symbol: str) -> str:
    if "BANKNIFTY" in opt_symbol: return "NSE:NIFTYBANK-INDEX"
    elif "NIFTY" in opt_symbol: return "NSE:NIFTY50-INDEX"
    elif "SENSEX" in opt_symbol: return "BSE:SENSEX-INDEX"
    return ""

def get_last_weekday(year: int, month: int, target_weekday: int) -> datetime.date:
    last_day = calendar.monthrange(year, month)[1]
    last_date = datetime(year, month, last_day).date()
    offset = (last_date.weekday() - target_weekday) % 7
    return last_date - timedelta(days=offset)

def get_expiry_details(symbol: str) -> Tuple[bool, datetime.date]:
    today = datetime.now().date()
    if "NIFTYBANK" in symbol:
        expiry = get_last_weekday(today.year, today.month, 3) 
        if today > expiry:
            nm, ny = (today.month + 1, today.year) if today.month < 12 else (1, today.year + 1)
            expiry = get_last_weekday(ny, nm, 3)
        return True, expiry
    elif "SENSEX" in symbol:
        days_ahead = 4 - today.weekday()
        if days_ahead < 0: days_ahead += 7
        expiry = today + timedelta(days=days_ahead)
        return (expiry == get_last_weekday(expiry.year, expiry.month, 4)), expiry
    else:
        days_ahead = 3 - today.weekday()
        if days_ahead < 0: days_ahead += 7
        expiry = today + timedelta(days=days_ahead)
        return (expiry == get_last_weekday(expiry.year, expiry.month, 3)), expiry

def get_options_data(symbol: str, ltp: float, side: str) -> Tuple[float, str, List[str]]:
    exch, base_name, interval = get_index_meta(symbol)
    atm_strike = int(round(ltp / interval) * interval)
    is_monthly, nearest_expiry = get_expiry_details(symbol)
    yy = nearest_expiry.strftime("%y")
    expiry_str = nearest_expiry.strftime("%b").upper() if is_monthly else f"{['1','2','3','4','5','6','7','8','9','O','N','D'][nearest_expiry.month-1]}{nearest_expiry.strftime('%d')}"
    opt_type = "CE" if side == "long" else "PE"
    strikes = [atm_strike + (i * interval) for i in range(-10, 11)]
    target_symbols = [f"{exch}:{base_name}{yy}{expiry_str}{s}{opt_type}" for s in strikes]
    display_str = f"21 {opt_type} Contracts (Strikes: {strikes[0]:.2f} to {strikes[-1]:.2f})"
    return float(atm_strike), display_str, target_symbols

def get_options_list_from_df(df: pd.DataFrame) -> List[str]:
    symbols = []
    if df.empty or "Target_Options" not in df.columns: return symbols
    for _, row in df.iterrows():
        opt_list = row.get("Target_Options")
        if isinstance(opt_list, list): symbols.extend(opt_list)
    return list(set(symbols))

# ==========================================
# MATH & API EXECUTION
# ==========================================

def price_stats_from_series(prices: pd.Series) -> dict:
    p = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if len(p) < 3: return {"Directional": np.nan, "Turning": np.nan, "Stability": np.nan, "Balanced": np.nan, "CumsumPlus": np.nan}
    x = np.arange(len(p), dtype=float)
    slope = float(np.polyfit(x, p.values, 1)[0])
    directional = slope + float(p.iloc[-1] - p.iloc[0])
    turning = float(np.mean(np.abs(np.diff(p.values, n=2))))
    return {"Directional": directional, "Turning": turning, "Stability": float(np.std(p.values)), "Balanced": directional - turning + float(np.std(p.values)), "CumsumPlus": float(np.sum(np.clip(np.diff(p.values), 0, None)))}

def build_signals_from_raw_directional(detail_df) -> dict:
    nan = float("nan")
    out = {k: nan for k in ("5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal", "Bull_Signal", "Bear_Signal", "Overall_Signal")}
    if detail_df is None or detail_df.empty: return out
    df = detail_df.copy()
    if "Iteration No" in df.columns: df = df.sort_values("Iteration No")
    vals = pd.to_numeric(df["Directional"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0: return out
    last = vals.size - 1
    def raw_at(offset: int) -> float: return float(vals[max(0, last - offset)])
    out["5m_Signal"], out["15m_Signal"], out["30m_Signal"], out["60m_Signal"] = round(raw_at(0), 4), round(raw_at(3) if last >= 3 else raw_at(0), 4), round(raw_at(6) if last >= 6 else raw_at(0), 4), round(raw_at(12) if last >= 12 else raw_at(0), 4)
    out["Bull_Signal"] = round(float(vals[vals > 0].max()) if (vals > 0).any() else 0.0, 4)
    out["Bear_Signal"] = round(abs(float(vals[vals < 0].min())) if (vals < 0).any() else 0.0, 4)
    out["Overall_Signal"] = round(raw_at(0), 4)
    return out

def classify_mtf_from_window(win, eps: float = 1e-9) -> float:
    s = pd.Series(win).dropna().astype(float)
    if len(s) < 3: return float("nan")
    diff1 = s.diff().dropna()
    if len(diff1) < 2: return float("nan")
    x = np.arange(len(s), dtype=float)
    slope = float(np.polyfit(x, s.values, 1)[0])
    net_move = float(s.iloc[-1] - s.iloc[0])
    turning = float(np.mean(np.abs(np.diff(s.values, n=2)))) if len(s) >= 3 else 0.0
    stability = float(np.std(s.values))
    score = (slope + net_move) + (0.25 * float(np.clip(diff1, 0, None).sum())) - (0.50 * turning) - (0.10 * stability)
    return 1.0 if score > eps else -1.0 if score < -eps else 0.0

def build_mtf_alignment(detail_df: pd.DataFrame) -> Dict[str, object]:
    out = {"MTF_5m": float("nan"), "MTF_15m": float("nan"), "MTF_30m": float("nan"), "MTF_60m": float("nan"), "MTF_SCORE": float("nan"), "MTF_ALIGN": "NA"}
    if detail_df is None or detail_df.empty or "Iteration Change" not in detail_df.columns: return out
    df = detail_df.copy().sort_values("Iteration No").reset_index(drop=True)
    series = pd.to_numeric(df["Iteration Change"], errors="coerce").dropna().astype(float)
    if len(series) < 3: return out
    def classify_from_tail(s: pd.Series, bars: int) -> float: return float("nan") if len(s) < bars else classify_mtf_from_window(s.tail(bars))
    mtf_15, mtf_30, mtf_60 = classify_from_tail(series, 3), classify_from_tail(series.iloc[1::2].reset_index(drop=True), 3), classify_from_tail(series.iloc[3::4].reset_index(drop=True), 3)   
    available = [v for v in [mtf_15, mtf_30, mtf_60] if pd.notna(v)]
    if not available: return out
    score = float(np.nansum([mtf_15, mtf_30, mtf_60]))
    align = "LONG" if all(v == 1.0 for v in available) else "SHORT" if all(v == -1.0 for v in available) else "MIXED"
    out.update({"MTF_15m": mtf_15, "MTF_30m": mtf_30, "MTF_60m": mtf_60, "MTF_SCORE": score, "MTF_ALIGN": align})
    return out

def init_fyers():
    global fyers
    try:
        client_id, access_token = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID"), os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
        if not client_id or not access_token: return
        fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
        logger.info("INIT FyersModel initialized successfully.")
    except Exception as e: logger.warning(f"INIT Failed: {e}")

def get_fyers_history(symbol: str, resolution: str, days_back: int) -> Optional[pd.DataFrame]:
    if not fyers: return None
    try:
        now = datetime.now()
        start_date, end_date = (now - timedelta(days=days_back)).date(), now.date()
        all_candles, current_start = [], start_date
        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=HISTORY_API_MAX_SPAN_DAYS), end_date)
            data = {"symbol": symbol, "resolution": resolution, "date_format": "1", "range_from": current_start.strftime("%Y-%m-%d"), "range_to": current_end.strftime("%Y-%m-%d"), "cont_flag": "1"}
            for attempt in range(1, FYERS_MAX_RETRIES + 1):
                res = fyers.history(data=data)
                if res and res.get("s") == "ok":
                    if candles := res.get("candles", []): all_candles.extend(candles)
                    break
                if isinstance(res, dict) and res.get("code") == 429 and attempt < FYERS_MAX_RETRIES: time_module.sleep(FYERS_RETRY_SLEEP * attempt)
                else: break
            time_module.sleep(FYERS_RATE_LIMIT_SLEEP)
            current_start = current_end + timedelta(days=1)
        if all_candles:
            df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            df.sort_values("timestamp", inplace=True)
            df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            return df.reset_index(drop=True)
        return None
    except Exception as e:
        logger.error(f"FYERS Error fetching {resolution} data for {symbol}: {e}")
        return None

def compute_iteration_volume_profile(intra_df: Optional[pd.DataFrame], prev_close: Optional[float]) -> Tuple[Dict, pd.DataFrame]:
    if intra_df is None or intra_df.empty: return {}, pd.DataFrame()
    df = intra_df.copy()
    df["date"], df["time_only"] = pd.to_datetime(df["timestamp"]).dt.date, pd.to_datetime(df["timestamp"]).dt.time
    dates = sorted(df["date"].unique())
    if len(dates) < 2: return {}, pd.DataFrame()
    curr_df = df[df["date"] == dates[-1]].copy().sort_values("time_only")
    if curr_df.empty: return {}, pd.DataFrame()
    curr_df["Iteration Change"] = ((pd.to_numeric(curr_df["close"], errors="coerce") - float(prev_close)) / float(prev_close) * 100.0) if prev_close else 0.0
    rows = []
    for i in range(len(curr_df)):
        ps = price_stats_from_series(curr_df["Iteration Change"].iloc[: i + 1])
        rows.append({"Iteration No": i+1, "Iteration Time": curr_df.iloc[i]["time_only"].strftime("%H:%M"), "LTP": float(curr_df.iloc[i]["close"]), "Iteration Change": float(curr_df["Iteration Change"].iloc[i]), "Directional": ps["Directional"], "Turning": ps["Turning"], "Stability": ps["Stability"], "Balanced": ps["Balanced"], "CumsumPlus": ps.get("CumsumPlus")})
    detail_df = pd.DataFrame(rows)
    final_ps = price_stats_from_series(curr_df["Iteration Change"])
    summary = {"LTP": float(curr_df["close"].iloc[-1]), "Directional": final_ps["Directional"], "Turning": final_ps["Turning"], "Stability": final_ps["Stability"], "Balanced": final_ps["Balanced"], "CumsumPlus": final_ps.get("CumsumPlus"), "Total Iterations": len(curr_df), "Last Iteration Time": curr_df.iloc[-1]["time_only"].strftime("%H:%M")}
    summary.update(build_signals_from_raw_directional(detail_df))
    summary.update(build_mtf_alignment(detail_df))
    return summary, detail_df

def scan_fno_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, iteration_rows = [], []
    total = len(INDEX_SYMBOLS)
    for idx, sym in enumerate(INDEX_SYMBOLS, start=1):
        logger.info(f"CORE [{idx}/{total}] Processing Spot Index: {sym}")
        daily_df, intra_df = get_fyers_history(sym, "D", DAILY_LOOKBACK_DAYS), get_fyers_history(sym, "15", INTRADAY_LOOKBACK_DAYS)
        prev_close, day_low, day_high, bands = None, float("nan"), float("nan"), {}
        if daily_df is not None and len(daily_df) >= 3:
            daily_df["_date_parsed"] = pd.to_datetime(daily_df["timestamp"]).dt.date
            today_date = datetime.now().date()
            if not (hist_daily := daily_df[daily_df["_date_parsed"] < today_date].copy()).empty:
                prev_close = float(hist_daily["close"].iloc[-1])
                for label, tf_days in [("6M", 135), ("3M", 65), ("2M", 44), ("1M", 22), ("2W", 10), ("1W", 5)]:
                    df_slice = hist_daily.tail(tf_days)
                    if df_slice.empty: bands[f"T_{label}"], bands[f"B_{label}"], bands[f"D_{label}"] = float("nan"), float("nan"), "N/A"
                    else:
                        max_idx = df_slice["volume"].idxmax() if "volume" in df_slice.columns and (df_slice["volume"] > 0).any() else (df_slice["high"] - df_slice["low"]).idxmax()
                        c_day = df_slice.loc[max_idx]
                        bands[f"T_{label}"], bands[f"B_{label}"], bands[f"D_{label}"] = float(c_day["high"]), float(c_day["low"]), str(c_day["_date_parsed"])
        if intra_df is not None and not intra_df.empty:
            intra_df["_d"] = pd.to_datetime(intra_df["timestamp"]).dt.date
            if not (curr_intra := intra_df[intra_df["_d"] == datetime.now().date()]).empty: day_low, day_high = float(curr_intra["low"].min()), float(curr_intra["high"].max())
        if prev_close is None and daily_df is not None and not daily_df.empty: prev_close = float(daily_df["close"].iloc[-1])
        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df, prev_close)
        ltp = iter_summary.get("LTP")
        pct_change = ((ltp - prev_close) / prev_close * 100) if (ltp is not None and prev_close and prev_close != 0) else 0.0
        if not iter_detail.empty:
            iter_detail.insert(0, "Symbol", sym); iter_detail.insert(1, "% Change", pct_change); iteration_rows.append(iter_detail)
        streak_data = {}
        if daily_df is not None and not daily_df.empty and ltp is not None:
            today_date = datetime.now().date()
            history_candles = [(row["_date_parsed"], float(row["close"])) for _, row in daily_df[daily_df["_date_parsed"] < today_date].iterrows()] + [(today_date, ltp)]
            for label in ["6M", "3M", "2M", "1M", "2W", "1W"]:
                t_val, b_val = bands.get(f"T_{label}", float("nan")), bands.get(f"B_{label}", float("nan"))
                l_start = next((d for d, c in reversed(history_candles) if pd.notna(t_val) and c > t_val), None)
                s_start = next((d for d, c in reversed(history_candles) if pd.notna(b_val) and c < b_val), None)
                streak_data[f"Days_L_{label}"] = (today_date - l_start).days if l_start else 999
                streak_data[f"Days_S_{label}"] = (today_date - s_start).days if s_start else 999
        res_row = {"Symbol": sym, "LTP": ltp, "% Change": pct_change, "Prev_Close": prev_close, "Day_Low": day_low, "Day_High": day_high, "Signal_Type": "N/A"}
        res_row.update(bands); res_row.update(streak_data); rows.append(res_row)
    return pd.DataFrame(rows), (pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame())

def scan_options_universe(symbols: List[str]) -> pd.DataFrame:
    rows, total = [], len(symbols)
    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"VERIFICATION [{idx}/{total}] Processing Option Premium: {sym}")
        daily_df, intra_df = get_fyers_history(sym, "D", 60), get_fyers_history(sym, "15", 5)
        ltp, prev_close, day_low, pct_change, bands = np.nan, np.nan, np.nan, 0.0, {}
        if daily_df is not None and len(daily_df) >= 2:
            daily_df["_date_parsed"] = pd.to_datetime(daily_df["timestamp"]).dt.date
            today_date = datetime.now().date()
            if not (hist_daily := daily_df[daily_df["_date_parsed"] < today_date].copy()).empty:
                prev_close = float(hist_daily["close"].iloc[-1])
                max_idx = hist_daily["volume"].idxmax() if "volume" in hist_daily.columns and (hist_daily["volume"] > 0).any() else (hist_daily["high"] - hist_daily["low"]).idxmax()
                c_day = hist_daily.loc[max_idx]
                bands["T_LOC"], bands["D_LOC"] = float(c_day["high"]), str(c_day["_date_parsed"])
        if intra_df is not None and not intra_df.empty:
            intra_df["_d"] = pd.to_datetime(intra_df["timestamp"]).dt.date
            if not (curr_intra := intra_df[intra_df["_d"] == datetime.now().date()]).empty: day_low, ltp = float(curr_intra["low"].min()), float(curr_intra["close"].iloc[-1])
        if pd.isna(ltp) and daily_df is not None and not daily_df.empty: ltp = float(daily_df["close"].iloc[-1])
        if pd.notna(ltp) and pd.notna(prev_close) and prev_close != 0: pct_change = ((ltp - prev_close) / prev_close) * 100
        streak_data = {}
        if daily_df is not None and not daily_df.empty and pd.notna(ltp):
            history_candles = [(row["_date_parsed"], float(row["close"])) for _, row in daily_df[daily_df["_date_parsed"] < today_date].iterrows()] + [(today_date, ltp)]
            l_start = next((d for d, c in reversed(history_candles) if pd.notna(bands.get("T_LOC")) and c > bands.get("T_LOC")), None)
            streak_data["Days_L_LOC"] = (today_date - l_start).days if l_start else 999
        res_row = {"Symbol": sym, "LTP": ltp, "% Change": pct_change, "Prev_Close": prev_close, "Day_Low": day_low, "Signal_Type": "N/A"}
        res_row.update(bands); res_row.update(streak_data); rows.append(res_row)
    return pd.DataFrame(rows)

def build_candidate_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty: return pd.DataFrame(columns=EMAIL_DISPLAY_COLS), pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    base = df.copy()
    for c in ["LTP", "Prev_Close", "Day_Low", "Day_High", "% Change"]:
        if c in base.columns: base[c] = pd.to_numeric(base[c], errors="coerce")
    def prep_side_df(dfside: pd.DataFrame, side: str) -> pd.DataFrame:
        if dfside.empty: return dfside
        out, valid_rows = dfside.copy(), []
        for _, row in out.iterrows():
            ltp, pc, d_low, d_high, sym = row.get("LTP"), row.get("Prev_Close"), row.get("Day_Low"), row.get("Day_High"), row.get("Symbol")
            if pd.isna(ltp) or pd.isna(pc) or pd.isna(d_low) or pd.isna(d_high): continue
            for tf in ["6M", "3M", "2M", "1M", "2W", "1W"]:
                t, b, d, bd_days = row.get(f"T_{tf}"), row.get(f"B_{tf}"), row.get(f"D_{tf}"), row.get(f"Days_L_{tf}" if side == "long" else f"Days_S_{tf}")
                if side == "long" and pd.notna(t) and ltp > t:
                    if bd_days is not None and bd_days <= 10:
                        row["Signal_Type"] = "Fresh Sweep" if d_low <= t else "Fresh Breakout" if pc <= t else "Active Trend"
                        row["Timeframe"], row["Top_Band"], row["Bottom_Band"], row["Climax_Date"], row["Breach_Days"] = tf, t, b, d, bd_days
                        strike, opt_str, opt_list = get_options_data(sym, ltp, "long")
                        row["ATM_Strike"], row["Option_Contracts"], row["Target_Options"] = strike, opt_str, opt_list
                        valid_rows.append(row); break
                elif side == "short" and pd.notna(b) and ltp < b:
                    if bd_days is not None and bd_days <= 10:
                        row["Signal_Type"] = "Fresh Sweep" if d_high >= b else "Fresh Breakdown" if pc >= b else "Active Trend"
                        row["Timeframe"], row["Top_Band"], row["Bottom_Band"], row["Climax_Date"], row["Breach_Days"] = tf, t, b, d, bd_days
                        strike, opt_str, opt_list = get_options_data(sym, ltp, "short")
                        row["ATM_Strike"], row["Option_Contracts"], row["Target_Options"] = strike, opt_str, opt_list
                        valid_rows.append(row); break
        res_df = pd.DataFrame(valid_rows)
        if res_df.empty: return res_df
        return res_df.sort_values(by=['Breach_Days', '% Change'], ascending=[True, False if side=="long" else True], na_position='last')
    
    long_df, short_df = prep_side_df(base, "long").drop_duplicates(subset=["Symbol"]), prep_side_df(base, "short").drop_duplicates(subset=["Symbol"])
    if not long_df.empty and not short_df.empty:
        common = set(long_df["Symbol"]).intersection(set(short_df["Symbol"]))
        for sym in common:
            l_idx, s_idx = long_df.index[long_df["Symbol"] == sym][0], short_df.index[short_df["Symbol"] == sym][0]
            if long_df.at[l_idx, "Breach_Days"] < short_df.at[s_idx, "Breach_Days"]: short_df = short_df.drop(s_idx)
            elif short_df.at[s_idx, "Breach_Days"] < long_df.at[l_idx, "Breach_Days"]: long_df = long_df.drop(l_idx)
            elif float(long_df.at[l_idx, "% Change"]) > 0: short_df = short_df.drop(s_idx)
            else: long_df = long_df.drop(l_idx)
            
    cols = [c for c in EMAIL_DISPLAY_COLS if c in long_df.columns or c in short_df.columns]
    return (long_df[cols] if not long_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)), (short_df[cols] if not short_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS))

def build_option_candidate_tables(df: pd.DataFrame, spot_signal_map: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty: return pd.DataFrame(columns=EMAIL_OPT_COLS), pd.DataFrame(columns=EMAIL_OPT_COLS)
    base = df.copy()
    valid_rows = []
    for _, row in base.iterrows():
        ltp, pc, d_low, sym = row.get("LTP"), row.get("Prev_Close"), row.get("Day_Low"), row.get("Symbol")
        if pd.isna(ltp) or pd.isna(pc) or pd.isna(d_low): continue
        t, d, bd_days = row.get("T_LOC"), row.get("D_LOC"), row.get("Days_L_LOC")
        if pd.notna(t) and ltp > t and bd_days is not None and bd_days <= 10:
            base_signal = "Fresh Sweep" if d_low <= t else "Fresh Breakout" if pc <= t else "Active Trend"
            spot_signal = spot_signal_map.get(get_underlying_spot(sym), "")
            row["Signal_Type"] = "Divergence (Holy Grail)" if spot_signal == "Fresh Sweep" and base_signal == "Active Trend" else "Premium Sweep Divergence" if spot_signal == "Active Trend" and base_signal == "Fresh Sweep" else base_signal
            row["Timeframe"], row["Top_Band"], row["Climax_Date"], row["Breach_Days"] = "LOC", t, d, bd_days
            valid_rows.append(row)
    res_df = pd.DataFrame(valid_rows).sort_values(by=['Breach_Days', '% Change'], ascending=[True, False], na_position='last')
    ce_df, pe_df = res_df[res_df["Symbol"].str.endswith("CE")].head(30), res_df[res_df["Symbol"].str.endswith("PE")].head(30)
    cols = [c for c in EMAIL_OPT_COLS if c in res_df.columns]
    return (ce_df[cols] if not ce_df.empty else pd.DataFrame(columns=EMAIL_OPT_COLS)), (pe_df[cols] if not pe_df.empty else pd.DataFrame(columns=EMAIL_OPT_COLS))

def build_html_table(df: pd.DataFrame, title: str, display_cols: list, max_rows: int = 30) -> str:
    if df is None or df.empty: return f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3><div style="padding:12px;background:#111827;color:#d1d5db;">No candidates.</div>'
    cols = [c for c in display_cols if c in df.columns]
    header_cells = "".join(f'<th style="padding:8px;border:1px solid #4b5563;background:#111827;color:#f9fafb;">{c}</th>' for c in cols)
    body_rows = []
    for _, row in df.head(max_rows).iterrows():
        is_holy_grail, is_sweep = "Holy Grail" in str(row.get("Signal_Type")), "Sweep" in str(row.get("Signal_Type")) and "Holy Grail" not in str(row.get("Signal_Type"))
        row_bg, text_col = ("#581c87", "#e9d5ff") if is_holy_grail else ("#d97706", "#000000") if is_sweep else ("#030712", "#e5e7eb")
        tds = [f'<td style="padding:6px 8px;border:1px solid #4b5563;color:{text_col};background:{row_bg}">{format_value(col, row.get(col, ""))}</td>' for col in cols]
        body_rows.append(f"<tr>{''.join(tds)}</tr>")
    return f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3><table style="border-collapse:collapse;width:100%;background:#030712;"><thead><tr>{header_cells}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'

def save_outputs(summary_df: pd.DataFrame, detail_df: pd.DataFrame, prefix: str = "scan") -> Tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    s_csv, d_csv = f"{prefix}_summary_{ts}.csv", f"{prefix}_detail_{ts}.csv"
    storage_df = summary_df.copy()
    if "Target_Options" in storage_df.columns: storage_df["Target_Options"] = storage_df["Target_Options"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    storage_df.to_csv(s_csv, index=False); detail_df.to_csv(d_csv, index=False)
    return s_csv, d_csv

def send_email_with_tables(long_df, short_df, ce_df, pe_df, csv_filename, detail_csv_filename) -> bool:
    try:
        scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_body = f"""<html><body style="background:#030712;color:#e5e7eb;padding:20px;font-family:Arial,sans-serif;"><h2 style="color:#facc15;">Index Options Strategy Climax Map (Dual Verification)</h2><div style="color:#cbd5e1;font-size:14px;margin-bottom:18px;">Scan completed at {scan_time}</div>{build_html_table(long_df, "Active Index Long Strategy Matrix", EMAIL_DISPLAY_COLS)}<div style="height:28px;"></div>{build_html_table(ce_df, "Call Options (CE) Climax Verification", EMAIL_OPT_COLS)}<div style="height:36px;"></div>{build_html_table(short_df, "Active Index Short Strategy Matrix", EMAIL_DISPLAY_COLS)}<div style="height:28px;"></div>{build_html_table(pe_df, "Put Options (PE) Climax Verification", EMAIL_OPT_COLS)}</body></html>"""
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = sender_email, recipient_email, f"Index Options Climax Blueprint - {scan_time}"
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        for fname in [csv_filename, detail_csv_filename]:
            if fname and os.path.exists(fname):
                with open(fname, "rb") as f:
                    part = MIMEBase("application", "octet-stream"); part.set_payload(f.read()); encoders.encode_base64(part); part.add_header("Content-Disposition", f'attachment; filename={os.path.basename(fname)}'); msg.attach(part)
        with smtplib.SMTP(smtp_host, smtp_port) as server: server.starttls(); server.login(sender_email, sender_password); server.sendmail(sender_email, recipient_email, msg.as_string())
        logger.info("Email sent successfully."); return True
    except Exception as e: logger.error(f"EMAIL Error: {e}"); return False

def main():
    init_fyers()
    if not fyers: logger.error("Fyers not initialized. Exiting."); return
    summary_df, detail_df = scan_fno_universe()
    if summary_df.empty: logger.warning("No summary data produced."); return
    summary_csv, detail_csv = save_outputs(summary_df, detail_df, prefix="scan")
    long_df, short_df = build_candidate_tables(summary_df)
    spot_signal_map = {**{r["Symbol"]: r["Signal_Type"] for _, r in long_df.iterrows()}, **{r["Symbol"]: r["Signal_Type"] for _, r in short_df.iterrows()}}
    all_opt_symbols = list(set(get_options_list_from_df(long_df) + get_options_list_from_df(short_df)))
    ce_opt_df, pe_opt_df = pd.DataFrame(), pd.DataFrame()
    if all_opt_symbols:
        logger.info(f"Dynamically scanning {len(all_opt_symbols)} targeted Option Contracts..."); opt_summary_df = scan_options_universe(all_opt_symbols); ce_opt_df, pe_opt_df = build_option_candidate_tables(opt_summary_df, spot_signal_map)
    else: logger.info("No spot signals breached. Skipping Option Contract scanning.")
    send_email_with_tables(long_df, short_df, ce_opt_df, pe_opt_df, summary_csv, detail_csv)

if __name__ == "__main__": main()

#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py

Optimized Index Options Scanner via Fyers API with email alerts.
- CORE INDICES: Tracks Nifty 50, Bank Nifty, and Sensex.
- 6-LAYER VOLUME CLIMAX: Simultaneously tracks 6M, 3M, 2M, 1M, 2W, and 1W peaks.
- DYNAMIC OPTION CHAIN ROUTING: Pinpoints the exact ATM strike and dynamically builds Fyers valid Weekly/Monthly tokens.
- 10-DAY BREACH AGE FILTER: Price must have broken out/swept the band within <= 10 calendar days.
- BREACH-VELOCITY HIERARCHY: Sorted by days since breach (recency) first, then by % Change (velocity).
"""

import os
import sys
import logging
import warnings
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

# Global Configuration Parameters
DAILY_LOOKBACK_DAYS = 200  
INTRADAY_LOOKBACK_DAYS = 30
HISTORY_API_MAX_SPAN_DAYS = 99
FYERS_RATE_LIMIT_SLEEP = 0.31
FYERS_RETRY_SLEEP = 2.0
FYERS_MAX_RETRIES = 2

fyers: Optional[fyersModel.FyersModel] = None

smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
smtp_port = int(os.environ.get("SMTP_PORT", "587"))
sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
sender_password = os.environ.get("SENDER_PASSWORD", "password")
recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")

INDEX_SYMBOLS = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]

EMAIL_DISPLAY_COLS = [
    "Symbol",
    "LTP",
    "% Change",
    "Signal_Type",       
    "Timeframe",         
    "Top_Band",          
    "Bottom_Band",       
    "Climax_Date",       
    "ATM_Strike",        
    "Option_Contracts",  
    "MTF_15m",
    "MTF_30m",
    "MTF_60m",
    "Last Iteration Time",
]


# ==========================================
# GLOBAL HELPERS & FORMATTERS
# ==========================================

def format_value(col: str, val):
    """Globally scoped cell data visual text formatting engine."""
    if pd.isna(val) or val == float("inf") or val == float("-inf"): return ""
    if col in ["Timeframe", "Signal_Type", "Option_Contracts"]: return str(val)
    if col == "% Change": return f"{float(val):.2f}%"
    if col in ["Top_Band", "Bottom_Band", "ATM_Strike"]: return f"{float(val):.2f}"
    if col == "Climax_Date": return str(val)
    if isinstance(val, (int, float, np.integer, np.floating)): return f"{float(val):.4f}"
    return str(val)


def get_index_meta(symbol: str) -> Tuple[str, str, int]:
    """Maps index symbol to proper Fyers naming parameters and rounding intervals."""
    if "NIFTY50" in symbol:
        return "NSE", "NIFTY", 50
    elif "NIFTYBANK" in symbol:
        return "NSE", "BANKNIFTY", 100
    elif "SENSEX" in symbol:
        return "BSE", "SENSEX", 100
    return "NSE", "NIFTY", 50


def get_expiry_details(symbol: str) -> Tuple[bool, datetime]:
    """Finds the nearest expiry date and determines if it is a Monthly or Weekly contract."""
    today = datetime.now().date()
    if "NIFTYBANK" in symbol:
        target_weekday = 2  # Wednesday Expiry
    elif "SENSEX" in symbol:
        target_weekday = 4  # Friday Expiry
    else:
        target_weekday = 3  # Thursday Expiry (Nifty 50)
        
    days_ahead = target_weekday - today.weekday()
    if days_ahead < 0:
        days_ahead += 7
    nearest_expiry = today + timedelta(days=days_ahead)
    
    # If adding 7 days pushes us into a new month, this is the last expiry of the current month
    next_week = nearest_expiry + timedelta(days=7)
    is_monthly = next_week.month != nearest_expiry.month
    
    return is_monthly, nearest_expiry


def get_options_string(symbol: str, ltp: float, side: str) -> Tuple[float, str]:
    """Generates authentic executable Fyers V3 option chain symbology string maps."""
    exch, base_name, interval = get_index_meta(symbol)
    atm_strike = int(round(ltp / interval) * interval)
    
    is_monthly, nearest_expiry = get_expiry_details(symbol)
    yy = nearest_expiry.strftime("%y")
    
    if is_monthly:
        # Fyers Monthly format: NSE:NIFTY24OCT25000CE
        expiry_str = nearest_expiry.strftime("%b").upper()
    else:
        # Fyers Weekly format: NSE:NIFTY24O1025000CE
        months_map = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "O", 11: "N", 12: "D"}
        m_char = months_map[nearest_expiry.month]
        dd = nearest_expiry.strftime("%d")
        expiry_str = f"{m_char}{dd}"
    
    if side == "long":
        itm = atm_strike - interval
        otm = atm_strike + interval
        opt_type = "CE"
    else:
        itm = atm_strike + interval
        otm = atm_strike - interval
        opt_type = "PE"
        
    itm_sym = f"{exch}:{base_name}{yy}{expiry_str}{itm}{opt_type}"
    atm_sym = f"{exch}:{base_name}{yy}{expiry_str}{atm_strike}{opt_type}"
    otm_sym = f"{exch}:{base_name}{yy}{expiry_str}{otm}{opt_type}"
    
    return float(atm_strike), f"ITM: {itm_sym} | ATM: {atm_sym} | OTM: {otm_sym}"


# ==========================================
# TECHNICAL SIGNAL MATH & COMPUTATION
# ==========================================

def price_stats_from_series(prices: pd.Series) -> dict:
    """Calculates price action statistics (Restored)."""
    p = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if len(p) < 3: return {"Directional": np.nan, "Turning": np.nan, "Stability": np.nan, "Balanced": np.nan, "CumsumPlus": np.nan}
    x = np.arange(len(p), dtype=float)
    slope = float(np.polyfit(x, p.values, 1)[0])
    directional = slope + float(p.iloc[-1] - p.iloc[0])
    turning = float(np.mean(np.abs(np.diff(p.values, n=2))))
    return {
        "Directional": directional, 
        "Turning": turning, 
        "Stability": float(np.std(p.values)), 
        "Balanced": directional - turning + float(np.std(p.values)), 
        "CumsumPlus": float(np.sum(np.clip(np.diff(p.values), 0, None)))
    }


def build_signals_from_raw_directional(detail_df) -> dict:
    nan = float("nan")
    out = {
        k: nan for k in (
            "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal",
            "Bull_Signal", "Bear_Signal", "Overall_Signal"
        )
    }
    if detail_df is None or detail_df.empty: return out

    df = detail_df.copy()
    if "Iteration No" in df.columns: df = df.sort_values("Iteration No")

    vals = pd.to_numeric(df["Directional"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0: return out

    last = vals.size - 1
    def raw_at(offset: int) -> float:
        i = last - offset
        if i < 0: i = 0
        return float(vals[i])

    out["5m_Signal"] = round(raw_at(0), 4)
    out["15m_Signal"] = round(raw_at(3) if last >= 3 else raw_at(0), 4)
    out["30m_Signal"] = round(raw_at(6) if last >= 6 else raw_at(0), 4)
    out["60m_Signal"] = round(raw_at(12) if last >= 12 else raw_at(0), 4)
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
    out = {
        "MTF_5m": float("nan"), "MTF_15m": float("nan"), "MTF_30m": float("nan"),
        "MTF_60m": float("nan"), "MTF_SCORE": float("nan"), "MTF_ALIGN": "NA",
    }
    if detail_df is None or detail_df.empty or "Iteration Change" not in detail_df.columns: return out

    df = detail_df.copy().sort_values("Iteration No").reset_index(drop=True)
    series = pd.to_numeric(df["Iteration Change"], errors="coerce").dropna().astype(float)
    if len(series) < 3: return out

    def classify_from_tail(s: pd.Series, bars: int) -> float:
        if len(s) < bars: return float("nan")
        return classify_mtf_from_window(s.tail(bars))

    mtf_5 = float("nan")                                                       
    mtf_15 = classify_from_tail(series, 3)                                     
    mtf_30 = classify_from_tail(series.iloc[1::2].reset_index(drop=True), 3)   
    mtf_60 = classify_from_tail(series.iloc[3::4].reset_index(drop=True), 3)   

    available = [v for v in [mtf_15, mtf_30, mtf_60] if pd.notna(v)]
    if not available: return out

    score = float(np.nansum([mtf_15, mtf_30, mtf_60]))
    align = "LONG" if all(v == 1.0 for v in available) else "SHORT" if all(v == -1.0 for v in available) else "MIXED"

    out.update({
        "MTF_5m": mtf_5, "MTF_15m": mtf_15, "MTF_30m": mtf_30,
        "MTF_60m": mtf_60, "MTF_SCORE": score, "MTF_ALIGN": align,
    })
    return out


def init_fyers():
    global fyers
    try:
        client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
        if not client_id or not access_token: return
        fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
        logger.info("INIT FyersModel initialized successfully.")
    except Exception as e:
        logger.warning(f"INIT Failed: {e}")


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
                    candles = res.get("candles", [])
                    if candles: all_candles.extend(candles)
                    break
                if isinstance(res, dict) and res.get("code") == 429 and attempt < FYERS_MAX_RETRIES:
                    time_module.sleep(FYERS_RETRY_SLEEP * attempt)
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
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["time_only"] = pd.to_datetime(df["timestamp"]).dt.time
    
    dates = sorted(df["date"].unique())
    if len(dates) < 2: return {}, pd.DataFrame()

    current_date = dates[-1]
    curr_df = df[df["date"] == current_date].copy().sort_values("time_only")
    
    if curr_df.empty: return {}, pd.DataFrame()

    curr_df["Iteration Change"] = ((pd.to_numeric(curr_df["close"], errors="coerce") - float(prev_close)) / float(prev_close) * 100.0) if prev_close else 0.0

    rows = []
    total_iters = 0
    last_iter_time = None

    for i in range(len(curr_df)):
        total_iters += 1
        row = curr_df.iloc[i]
        t = row["time_only"]

        ps = price_stats_from_series(curr_df["Iteration Change"].iloc[: i + 1])

        rows.append({
            "Iteration No": total_iters, "Iteration Time": t.strftime("%H:%M"),
            "LTP": float(row["close"]), "Iteration Change": float(curr_df["Iteration Change"].iloc[i]),
            "Directional": ps["Directional"], "Turning": ps["Turning"], "Stability": ps["Stability"], "Balanced": ps["Balanced"], "CumsumPlus": ps.get("CumsumPlus")
        })
        last_iter_time = t.strftime("%H:%M")

    detail_df = pd.DataFrame(rows)
    ltp = float(curr_df["close"].iloc[-1]) if not curr_df.empty else np.nan

    final_ps = price_stats_from_series(curr_df["Iteration Change"])
    summary = {
        "LTP": ltp, "Directional": final_ps["Directional"], "Turning": final_ps["Turning"], "Stability": final_ps["Stability"],
        "Balanced": final_ps["Balanced"], "CumsumPlus": final_ps.get("CumsumPlus"), 
        "Total Iterations": total_iters, "Last Iteration Time": last_iter_time,
    }

    summary.update(build_signals_from_raw_directional(detail_df))
    summary.update(build_mtf_alignment(detail_df))
    return summary, detail_df


# ==========================================
# PIPELINE EXECUTION & DATA PIPELINES
# ==========================================

def scan_fno_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows, iteration_rows = [], []
    total = len(INDEX_SYMBOLS)

    for idx, sym in enumerate(INDEX_SYMBOLS, start=1):
        logger.info(f"CORE [{idx}/{total}] Processing Spot Index: {sym}")

        daily_df = get_fyers_history(sym, resolution="D", days_back=DAILY_LOOKBACK_DAYS)
        intra_df = get_fyers_history(sym, resolution="15", days_back=INTRADAY_LOOKBACK_DAYS)

        prev_close = None
        bands = {}
        day_low = day_high = float("nan")

        if daily_df is not None and len(daily_df) >= 3:
            daily_df["_date_parsed"] = pd.to_datetime(daily_df["timestamp"]).dt.date
            today_date = datetime.now().date()
            hist_daily = daily_df[daily_df["_date_parsed"] < today_date].copy()
            
            if not hist_daily.empty:
                prev_close = float(hist_daily["close"].iloc[-1])
                
                def extract_bands(df_hist, trading_days, label):
                    df_slice = df_hist.tail(trading_days)
                    if df_slice.empty: 
                        bands[f"T_{label}"] = float("nan")
                        bands[f"B_{label}"] = float("nan")
                        bands[f"D_{label}"] = "N/A"
                        return
                    
                    if "volume" in df_slice.columns and (df_slice["volume"] > 0).any():
                        max_idx = df_slice["volume"].idxmax()
                    else:
                        volatility = df_slice["high"] - df_slice["low"]
                        max_idx = volatility.idxmax()
                        
                    c_day = df_slice.loc[max_idx]
                    bands[f"T_{label}"] = float(c_day["high"])
                    bands[f"B_{label}"] = float(c_day["low"])
                    bands[f"D_{label}"] = str(c_day["_date_parsed"])

                for label, tf_days in [("6M", 135), ("3M", 65), ("2M", 44), ("1M", 22), ("2W", 10), ("1W", 5)]:
                    extract_bands(hist_daily, tf_days, label)
                
        if intra_df is not None and not intra_df.empty:
            intra_df["_d"] = pd.to_datetime(intra_df["timestamp"]).dt.date
            curr_intra = intra_df[intra_df["_d"] == datetime.now().date()]
            if not curr_intra.empty:
                day_low = float(curr_intra["low"].min())
                day_high = float(curr_intra["high"].max())

        if prev_close is None and daily_df is not None and not daily_df.empty:
            prev_close = float(daily_df["close"].iloc[-1])

        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df, prev_close)
        ltp = iter_summary.get("LTP")
        pct_change = ((ltp - prev_close) / prev_close * 100) if (ltp is not None and prev_close and prev_close != 0) else 0.0

        if not iter_detail.empty:
            iter_detail.insert(0, "Symbol", sym)
            iter_detail.insert(1, "% Change", pct_change)
            iteration_rows.append(iter_detail)

        streak_data = {}
        if daily_df is not None and not daily_df.empty and ltp is not None:
            today_date = datetime.now().date()
            history_candles = [(row["_date_parsed"], float(row["close"])) for _, row in daily_df[daily_df["_date_parsed"] < today_date].iterrows()]
            history_candles.append((today_date, ltp))

            for label in ["6M", "3M", "2M", "1M", "2W", "1W"]:
                t_val = bands.get(f"T_{label}", float("nan"))
                b_val = bands.get(f"B_{label}", float("nan"))
                
                l_start = s_start = None
                for d, c in reversed(history_candles):
                    if pd.notna(t_val) and c > t_val: l_start = d
                    else: break
                for d, c in reversed(history_candles):
                    if pd.notna(b_val) and c < b_val: s_start = d
                    else: break
                
                streak_data[f"Days_L_{label}"] = (today_date - l_start).days if l_start else 999
                streak_data[f"Days_S_{label}"] = (today_date - s_start).days if s_start else 999

        res_row = {
            "Symbol": sym, "LTP": ltp, "% Change": pct_change,
            "Prev_Close": prev_close, "Day_Low": day_low, "Day_High": day_high,
            "Signal_Type": "N/A"
        }
        res_row.update(bands)
        res_row.update(streak_data)
        rows.append(res_row)

    return pd.DataFrame(rows), (pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame())


def build_candidate_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty: return pd.DataFrame(columns=EMAIL_DISPLAY_COLS), pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    base = df.copy()
    
    for c in ["LTP", "Prev_Close", "Day_Low", "Day_High", "% Change"]:
        if c in base.columns: base[c] = pd.to_numeric(base[c], errors="coerce")

    def prep_side_df(dfside: pd.DataFrame, side: str) -> pd.DataFrame:
        if dfside.empty: return dfside
        out = dfside.copy()
        valid_rows = []
        
        for _, row in out.iterrows():
            ltp = row.get("LTP")
            pc = row.get("Prev_Close")
            d_low = row.get("Day_Low")
            d_high = row.get("Day_High")
            sym = row.get("Symbol")
            
            if pd.isna(ltp) or pd.isna(pc) or pd.isna(d_low) or pd.isna(d_high): continue

            timeframes = ["6M", "3M", "2M", "1M", "2W", "1W"]
            for tf in timeframes:
                t = row.get(f"T_{tf}")
                b = row.get(f"B_{tf}")
                d = row.get(f"D_{tf}")
                bd_days = row.get(f"Days_L_{tf}" if side == "long" else f"Days_S_{tf}")
                
                if side == "long" and pd.notna(t) and ltp > t:
                    if bd_days is not None and bd_days <= 10:
                        if d_low <= t: row["Signal_Type"] = "Fresh Sweep"
                        elif pc <= t: row["Signal_Type"] = "Fresh Breakout"
                        else: row["Signal_Type"] = "Active Trend"
                        
                        row["Timeframe"], row["Top_Band"], row["Bottom_Band"], row["Climax_Date"] = tf, t, b, d
                        row["Breach_Days"] = bd_days
                        strike, opt_str = get_options_string(sym, ltp, "long")
                        row["ATM_Strike"], row["Option_Contracts"] = strike, opt_str
                        valid_rows.append(row)
                        break
                        
                elif side == "short" and pd.notna(b) and ltp < b:
                    if bd_days is not None and bd_days <= 10:
                        if d_high >= b: row["Signal_Type"] = "Fresh Sweep"
                        elif pc >= b: row["Signal_Type"] = "Fresh Breakdown"
                        else: row["Signal_Type"] = "Active Trend"
                            
                        row["Timeframe"], row["Top_Band"], row["Bottom_Band"], row["Climax_Date"] = tf, t, b, d
                        row["Breach_Days"] = bd_days
                        strike, opt_str = get_options_string(sym, ltp, "short")
                        row["ATM_Strike"], row["Option_Contracts"] = strike, opt_str
                        valid_rows.append(row)
                        break

        res_df = pd.DataFrame(valid_rows)
        if res_df.empty: return res_df
        
        if side == "long":
            return res_df.sort_values(by=['Breach_Days', '% Change'], ascending=[True, False], na_position='last')
        else:
            return res_df.sort_values(by=['Breach_Days', '% Change'], ascending=[True, True], na_position='last')

    long_df = prep_side_df(base, "long").drop_duplicates(subset=["Symbol"]).head(30)
    short_df = prep_side_df(base, "short").drop_duplicates(subset=["Symbol"]).head(30)
    
    cols = [c for c in EMAIL_DISPLAY_COLS if c in long_df.columns or c in short_df.columns]
    return (long_df[cols] if not long_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)), (short_df[cols] if not short_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS))


def build_html_table(df: pd.DataFrame, title: str, max_rows: int = 30) -> str:
    if df is None or df.empty: return f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3><div style="padding:12px;background:#111827;color:#d1d5db;">No indices matched setup limits within the last 10 days.</div>'
    df_slice = df.head(max_rows).copy()
    cols = [c for c in EMAIL_DISPLAY_COLS if c in df_slice.columns]

    header_cells = "".join(f'<th style="padding:8px;border:1px solid #4b5563;background:#111827;color:#f9fafb;">{c}</th>' for c in cols)
    body_rows = []
    
    for _, row in df_slice.iterrows():
        tds = []
        is_sweep = "Sweep" in str(row.get("Signal_Type"))
        row_bg = "#d97706" if is_sweep else "#030712"
        text_col = "#000000" if is_sweep else "#e5e7eb"

        for col in cols:
            bg = row_bg
            if col == "% Change" and not is_sweep:
                try:
                    f_pct = float(row[col])
                    bg = "#14532d" if f_pct > 0 else "#7f1d1d" if f_pct < 0 else "#030712"
                except Exception:
                    pass
            elif col == "Timeframe" and not is_sweep:
                if row[col] in ["6M", "3M"]: bg = "#431407"
                elif row[col] in ["2M", "1M"]: bg = "#1e3a8a"
                else: bg = "#064e3b" 
                
            tds.append(f'<td style="padding:6px 8px;border:1px solid #4b5563;color:{text_col};background:{bg}">{format_value(col, row.get(col, ""))}</td>')
        body_rows.append(f"<tr>{''.join(tds)}</tr>")
    return f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3><table style="border-collapse:collapse;width:100%;background:#030712;"><thead><tr>{header_cells}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def send_email_with_tables(long_df: pd.DataFrame, short_df: pd.DataFrame, csv_filename: str = "", detail_csv_filename: str = "") -> bool:
    try:
        scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_body = f"""
        <html>
        <body style="background:#030712;color:#e5e7eb;padding:20px;font-family:Arial,sans-serif;">
            <h2 style="color:#facc15;">Index Options Strategy Climax Map (Breach Age <= 10 Days)</h2>
            <div style="color:#cbd5e1;font-size:14px;margin-bottom:18px;">Scan completed at {scan_time}</div>
            {build_html_table(long_df, "Active Index Long Strategy Matrix")}
            <div style="height:28px;"></div>
            {build_html_table(short_df, "Active Index Short Strategy Matrix")}
        </body>
        </html>
        """
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = sender_email, recipient_email, f"Index Options Climax Blueprint - {scan_time}"
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        for fname in [csv_filename, detail_csv_filename]:
            if fname and os.path.exists(fname):
                with open(fname, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f'attachment; filename={os.path.basename(fname)}')
                    msg.attach(part)

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        logger.info("Email sent successfully.")
        return True
    except Exception as e:
        logger.error(f"EMAIL Error: {e}")
        return False


def save_outputs(summary_df: pd.DataFrame, detail_df: pd.DataFrame, prefix: str = "scan") -> Tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    s_csv, d_csv = f"{prefix}_summary_{ts}.csv", f"{prefix}_detail_{ts}.csv"
    summary_df.to_csv(s_csv, index=False)
    detail_df.to_csv(d_csv, index=False)
    return s_csv, d_csv


def main():
    init_fyers()
    if not fyers:
        logger.error("Fyers not initialized. Exiting.")
        return

    summary_df, detail_df = scan_fno_universe()
    if summary_df.empty:
        logger.warning("No summary data produced.")
        return

    summary_csv, detail_csv = save_outputs(summary_df, detail_df, prefix="fno")
    long_df, short_df = build_candidate_tables(summary_df)

    send_email_with_tables(
        long_df=long_df, short_df=short_df,
        csv_filename=summary_csv, detail_csv_filename=detail_csv
    )


if __name__ == "__main__":
    main()

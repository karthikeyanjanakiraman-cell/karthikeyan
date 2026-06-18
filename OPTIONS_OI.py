#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py - Complete Master Implementation
Features: Spot/Option Divergence, 21-Strike Net, LOC Climax, CSV Audit
"""

import os
import sys
import logging
import warnings
import calendar
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# ==========================================
# CONFIGURATION
# ==========================================

class Config:
    def __init__(self):
        self.client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        self.access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
        self.sender_password = os.environ.get("SENDER_PASSWORD", "password")
        self.recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")
        self.index_symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]

cfg = Config()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)
warnings.filterwarnings("ignore")

# ==========================================
# GLOBAL HELPERS
# ==========================================

def format_value(col, val):
    if pd.isna(val) or val in [float("inf"), float("-inf")]: return ""
    if col in ["Timeframe", "Signal_Type", "Option_Contracts", "Breach_Days"]: return str(val)
    if col == "% Change": return f"{float(val):.2f}%"
    if col in ["Top_Band", "Bottom_Band", "ATM_Strike"]: return f"{float(val):.2f}"
    if isinstance(val, (int, float, np.integer, np.floating)): return f"{float(val):.4f}"
    return str(val)

def get_index_meta(symbol):
    if "NIFTY50" in symbol: return "NSE", "NIFTY", 50
    if "NIFTYBANK" in symbol: return "NSE", "BANKNIFTY", 100
    return "BSE", "SENSEX", 100

def get_underlying_spot(opt_symbol):
    if "BANKNIFTY" in opt_symbol: return "NSE:NIFTYBANK-INDEX"
    if "NIFTY" in opt_symbol: return "NSE:NIFTY50-INDEX"
    return "BSE:SENSEX-INDEX"

def get_expiry_details(symbol):
    today = datetime.now().date()
    if "NIFTYBANK" in symbol:
        def last_thu(y, m):
            last = calendar.monthrange(y, m)[1]
            d = datetime(y, m, last).date()
            return d - timedelta(days=(d.weekday() - 3) % 7)
        expiry = last_thu(today.year, today.month)
        if today > expiry:
            m = today.month + 1 if today.month < 12 else 1
            y = today.year if today.month < 12 else today.year + 1
            expiry = last_thu(y, m)
        return True, expiry
    days_ahead = (3 if "NIFTY" in symbol else 4) - today.weekday()
    if days_ahead < 0: days_ahead += 7
    return False, today + timedelta(days=days_ahead)

def get_options_data(symbol, ltp, side):
    exch, base_name, interval = get_index_meta(symbol)
    atm_strike = int(round(ltp / interval) * interval)
    is_monthly, expiry = get_expiry_details(symbol)
    yy = expiry.strftime("%y")
    expiry_str = expiry.strftime("%b").upper() if is_monthly else f"{['1','2','3','4','5','6','7','8','9','O','N','D'][expiry.month-1]}{expiry.strftime('%d')}"
    opt_type = "CE" if side == "long" else "PE"
    strikes = [atm_strike + (i * interval) for i in range(-10, 11)]
    return float(atm_strike), f"21 {opt_type} Contracts (Strikes: {strikes[0]:.2f} to {strikes[-1]:.2f})", [f"{exch}:{base_name}{yy}{expiry_str}{s}{opt_type}" for s in strikes]

def get_options_list_from_df(df):
    symbols = []
    if df.empty or "Target_Options" not in df.columns: return symbols
    for _, row in df.iterrows():
        opt_list = row.get("Target_Options")
        if isinstance(opt_list, list): symbols.extend(opt_list)
    return list(set(symbols))

# ==========================================
# MATH & API
# ==========================================

def price_stats(prices):
    p = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if len(p) < 3: return {"Directional": np.nan, "Turning": np.nan, "Stability": np.nan, "Balanced": np.nan, "CumsumPlus": np.nan}
    x = np.arange(len(p), dtype=float)
    slope = float(np.polyfit(x, p.values, 1)[0])
    directional = slope + float(p.iloc[-1] - p.iloc[0])
    turning = float(np.mean(np.abs(np.diff(p.values, n=2))))
    return {"Directional": directional, "Turning": turning, "Stability": float(np.std(p.values)), "Balanced": directional - turning + float(np.std(p.values)), "CumsumPlus": float(np.sum(np.clip(np.diff(p.values), 0, None)))}

def build_signals_from_raw_directional(detail_df):
    nan = float("nan")
    out = {k: nan for k in ("5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal", "Bull_Signal", "Bear_Signal", "Overall_Signal")}
    if detail_df is None or detail_df.empty: return out
    df = detail_df.copy()
    if "Iteration No" in df.columns: df = df.sort_values("Iteration No")
    vals = pd.to_numeric(df["Directional"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0: return out
    last = vals.size - 1
    def raw_at(offset): return float(vals[max(0, last - offset)])
    out["5m_Signal"], out["15m_Signal"], out["30m_Signal"], out["60m_Signal"] = round(raw_at(0), 4), round(raw_at(3) if last >= 3 else raw_at(0), 4), round(raw_at(6) if last >= 6 else raw_at(0), 4), round(raw_at(12) if last >= 12 else raw_at(0), 4)
    out["Bull_Signal"] = round(float(vals[vals > 0].max()) if (vals > 0).any() else 0.0, 4)
    out["Bear_Signal"] = round(abs(float(vals[vals < 0].min())) if (vals < 0).any() else 0.0, 4)
    out["Overall_Signal"] = round(raw_at(0), 4)
    return out

def classify_mtf_from_window(win, eps=1e-9):
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

def build_mtf_alignment(detail_df):
    out = {"MTF_5m": float("nan"), "MTF_15m": float("nan"), "MTF_30m": float("nan"), "MTF_60m": float("nan"), "MTF_SCORE": float("nan"), "MTF_ALIGN": "NA"}
    if detail_df is None or detail_df.empty or "Iteration Change" not in detail_df.columns: return out
    df = detail_df.copy().sort_values("Iteration No").reset_index(drop=True)
    series = pd.to_numeric(df["Iteration Change"], errors="coerce").dropna().astype(float)
    if len(series) < 3: return out
    def classify_from_tail(s, bars): return float("nan") if len(s) < bars else classify_mtf_from_window(s.tail(bars))
    mtf_15, mtf_30, mtf_60 = classify_from_tail(series, 3), classify_from_tail(series.iloc[1::2].reset_index(drop=True), 3), classify_from_tail(series.iloc[3::4].reset_index(drop=True), 3)   
    available = [v for v in [mtf_15, mtf_30, mtf_60] if pd.notna(v)]
    if not available: return out
    score = float(np.nansum([mtf_15, mtf_30, mtf_60]))
    align = "LONG" if all(v == 1.0 for v in available) else "SHORT" if all(v == -1.0 for v in available) else "MIXED"
    out.update({"MTF_15m": mtf_15, "MTF_30m": mtf_30, "MTF_60m": mtf_60, "MTF_SCORE": score, "MTF_ALIGN": align})
    return out

def init_fyers():
    try:
        return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e: 
        logger.warning(f"INIT Failed: {e}"); return None

def get_history(fyers, symbol, res, days):
    try:
        now = datetime.now()
        start, end = (now - timedelta(days=days)).strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")
        res_data = fyers.history(data={"symbol": symbol, "resolution": res, "date_format": "1", "range_from": start, "range_to": end, "cont_flag": "1"})
        if res_data and "candles" in res_data:
            df = pd.DataFrame(res_data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df
        return None
    except Exception as e: return None

def compute_iteration_volume_profile(intra_df, prev_close):
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
        ps = price_stats(curr_df["Iteration Change"].iloc[: i + 1])
        rows.append({"Iteration No": i+1, "Iteration Time": curr_df.iloc[i]["time_only"].strftime("%H:%M"), "LTP": float(curr_df.iloc[i]["close"]), "Iteration Change": float(curr_df["Iteration Change"].iloc[i]), "Directional": ps["Directional"], "Turning": ps["Turning"], "Stability": ps["Stability"], "Balanced": ps["Balanced"], "CumsumPlus": ps.get("CumsumPlus")})
    detail_df = pd.DataFrame(rows)
    final_ps = price_stats(curr_df["Iteration Change"])
    summary = {"LTP": float(curr_df["close"].iloc[-1]), "Directional": final_ps["Directional"], "Turning": final_ps["Turning"], "Stability": final_ps["Stability"], "Balanced": final_ps["Balanced"], "CumsumPlus": final_ps.get("CumsumPlus"), "Total Iterations": len(curr_df), "Last Iteration Time": curr_df.iloc[-1]["time_only"].strftime("%H:%M")}
    summary.update(build_signals_from_raw_directional(detail_df))
    summary.update(build_mtf_alignment(detail_df))
    return summary, detail_df

def scan_fno_universe(fyers):
    rows, iteration_rows = [], []
    for idx, sym in enumerate(cfg.index_symbols, start=1):
        logger.info(f"CORE [{idx}/{len(cfg.index_symbols)}] Processing Spot: {sym}")
        daily, intra = get_history(fyers, sym, "D", 200), get_history(fyers, sym, "15", 30)
        prev_close, bands, streak_data = None, {}, {}
        if daily is not None and len(daily) >= 3:
            today_date = datetime.now().date()
            hist = daily[daily["timestamp"].dt.date < today_date].copy()
            if not hist.empty:
                prev_close = float(hist["close"].iloc[-1])
                for label, tf in [("6M", 135), ("3M", 65), ("2M", 44), ("1M", 22), ("2W", 10), ("1W", 5)]:
                    df_s = hist.tail(tf)
                    idx_val = df_s["volume"].idxmax() if (df_s["volume"]>0).any() else (df_s["high"]-df_s["low"]).idxmax()
                    c = df_s.loc[idx_val]
                    bands.update({f"T_{label}": float(c["high"]), f"B_{label}": float(c["low"]), f"D_{label}": str(c["timestamp"].date())})
                history_candles = list(zip(hist["timestamp"].dt.date, hist["close"])) + [(today_date, float(daily["close"].iloc[-1]))]
                for label in ["6M", "3M", "2M", "1M", "2W", "1W"]:
                    t_val, b_val = bands.get(f"T_{label}"), bands.get(f"B_{label}")
                    streak_data[f"Days_L_{label}"] = (today_date - next((d for d, c in reversed(history_candles) if pd.notna(t_val) and c > t_val), today_date)).days
                    streak_data[f"Days_S_{label}"] = (today_date - next((d for d, c in reversed(history_candles) if pd.notna(b_val) and c < b_val), today_date)).days
        
        iter_s, _ = compute_iteration_volume_profile(intra, prev_close)
        row = {"Symbol": sym, "LTP": iter_s.get("LTP"), "% Change": ((iter_s.get("LTP", 0) - prev_close)/prev_close*100) if prev_close else 0}
        row.update(bands); row.update(streak_data); rows.append(row)
    return pd.DataFrame(rows)

def scan_options_universe(fyers, symbols):
    rows = []
    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"VERIFICATION [{idx}/{len(symbols)}] Option: {sym}")
        daily = get_history(fyers, sym, "D", 60)
        if daily is not None and not daily.empty:
            prev_close = float(daily["close"].iloc[-2]) if len(daily) > 1 else float(daily["close"].iloc[-1])
            max_idx = daily["volume"].idxmax() if (daily["volume"] > 0).any() else (daily["high"] - daily["low"]).idxmax()
            c = daily.loc[max_idx]
            today_date = datetime.now().date()
            history_candles = list(zip(daily["timestamp"].dt.date, daily["close"])) + [(today_date, float(daily["close"].iloc[-1]))]
            l_start = next((d for d, c_val in reversed(history_candles) if c_val > float(c["high"])), today_date)
            rows.append({"Symbol": sym, "LTP": float(daily["close"].iloc[-1]), "T_LOC": float(c["high"]), "D_LOC": str(c["timestamp"].date()), "Days_L_LOC": (today_date - l_start).days, "Prev_Close": prev_close})
    return pd.DataFrame(rows)

def build_candidate_tables(df):
    valid_long, valid_short = [], []
    for _, row in df.iterrows():
        for tf in ["6M", "3M", "2M", "1M", "2W", "1W"]:
            t, b, bd = row.get(f"T_{tf}"), row.get(f"B_{tf}"), row.get(f"Days_L_{tf}")
            if pd.notna(t) and row["LTP"] > t and bd is not None and bd <= 10:
                strike, opt_str, opt_list = get_options_data(row["Symbol"], row["LTP"], "long")
                valid_long.append({**row, "Timeframe": tf, "Top_Band": t, "ATM_Strike": strike, "Option_Contracts": opt_str, "Target_Options": opt_list, "Breach_Days": bd, "Signal_Type": "Active Trend"})
                break
            elif pd.notna(b) and row["LTP"] < b and (sd := row.get(f"Days_S_{tf}")) is not None and sd <= 10:
                strike, opt_str, opt_list = get_options_data(row["Symbol"], row["LTP"], "short")
                valid_short.append({**row, "Timeframe": tf, "Bottom_Band": b, "ATM_Strike": strike, "Option_Contracts": opt_str, "Target_Options": opt_list, "Breach_Days": sd, "Signal_Type": "Active Trend"})
                break
    return pd.DataFrame(valid_long), pd.DataFrame(valid_short)

def build_option_candidate_tables(df, spot_signal_map):
    valid_rows = []
    for _, row in df.iterrows():
        t, d, bd = row.get("T_LOC"), row.get("D_LOC"), row.get("Days_L_LOC")
        if pd.notna(t) and row["LTP"] > t and bd is not None and bd <= 10:
            spot_signal = spot_signal_map.get(get_underlying_spot(row["Symbol"]), "")
            row["Signal_Type"] = "Holy Grail" if spot_signal == "Fresh Sweep" else "Active Trend"
            row["Timeframe"], row["Top_Band"], row["Climax_Date"], row["Breach_Days"] = "LOC", t, d, bd
            valid_rows.append(row)
    res = pd.DataFrame(valid_rows)
    return res[res["Symbol"].str.endswith("CE")], res[res["Symbol"].str.endswith("PE")]

def save_outputs(summary_df, detail_df, prefix="scan"):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    storage = summary_df.copy()
    if "Target_Options" in storage.columns: storage["Target_Options"] = storage["Target_Options"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
    storage.to_csv(f"{prefix}_summary_{ts}.csv", index=False)
    detail_df.to_csv(f"{prefix}_detail_{ts}.csv", index=False)
    return f"{prefix}_summary_{ts}.csv", f"{prefix}_detail_{ts}.csv"

def build_html_table(df, title, cols):
    if df.empty: return f"<h3>{title}</h3><p>No candidates.</p>"
    return f"<h3>{title}</h3><table border='1'><tr>" + "".join([f"<th>{c}</th>" for c in cols]) + "</tr>" + "".join([f"<tr>" + "".join([f"<td>{format_value(c, row.get(c))}</td>" for c in cols]) + "</tr>" for _, row in df.head(30).iterrows()]) + "</table>"

def send_email(long_df, short_df, ce_df, pe_df, csv_file, cfg: Config):
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = cfg.sender_email, cfg.recipient_email, "Index Options Climax Blueprint"
        html = f"<html><body><h2>Index Options Climax</h2>{build_html_table(long_df, 'Long', EMAIL_DISPLAY_COLS)}{build_html_table(ce_df, 'CE Climax', EMAIL_OPT_COLS)}{build_html_table(short_df, 'Short', EMAIL_DISPLAY_COLS)}{build_html_table(pe_df, 'PE Climax', EMAIL_OPT_COLS)}</body></html>"
        msg.attach(MIMEText(html, "html"))
        with open(csv_file, "rb") as f:
            part = MIMEBase("application", "octet-stream"); part.set_payload(f.read()); encoders.encode_base64(part); part.add_header("Content-Disposition", f'attachment; filename={os.path.basename(csv_file)}'); msg.attach(part)
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s: s.starttls(); s.login(cfg.sender_email, cfg.sender_password); s.sendmail(cfg.sender_email, cfg.recipient_email, msg.as_string())
    except Exception as e: logger.error(f"Email Failed: {e}")

def main():
    fyers = init_fyers()
    if not fyers: return
    spot_df, detail_df = scan_fno_universe(fyers)
    long_df, short_df = build_candidate_tables(spot_df)
    spot_map = {**{r["Symbol"]: r["Signal_Type"] for _, r in long_df.iterrows()}, **{r["Symbol"]: r["Signal_Type"] for _, r in short_df.iterrows()}}
    all_opt = list(set(get_options_list_from_df(long_df) + get_options_list_from_df(short_df)))
    ce_df, pe_df = pd.DataFrame(), pd.DataFrame()
    if all_opt:
        opt_df = scan_options_universe(fyers, all_opt)
        ce_df, pe_df = build_option_candidate_tables(opt_df, spot_map)
    csv_f, _ = save_outputs(spot_df, detail_df)
    send_email(long_df, short_df, ce_df, pe_df, csv_f, cfg)
    logger.info("Cycle complete.")

if __name__ == "__main__": main()

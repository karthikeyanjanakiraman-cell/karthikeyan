#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL_clean.py
Writes separate CSVs for dashboard, long/short matrices, CE/PE candidates.
"""

import os
import sys
import logging
import warnings
import calendar
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# CONFIG
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

# Fixed column schemas
DASHBOARD_COLS = [
    "Symbol", "% Change",
    "Support-3", "Support-2", "Support-1",
    "LTP",
    "Resistance-1", "Resistance-2", "Resistance-3"
]

STRATEGY_COLS = [
    "Symbol", "LTP", "Trigger_TF",
    "1-Day (T/B)", "3-Day (T/B)", "1-Week (T/B)",
    "1-Month (T/B)", "3-Month (T/B)", "6-Month (T/B)"
]

OPT_COLS = [
    "Symbol", "LTP", "% Change", "Signal_Type",
    "Climax_Date", "Climax_Range (T/B)", "Breach_Days"
]

EMAIL_DISPLAY_COLS = ["Symbol", "LTP", "Trigger_TF",
    "1-Day (T/B)", "3-Day (T/B)", "1-Week (T/B)",
    "1-Month (T/B)", "3-Month (T/B)", "6-Month (T/B)"]

EMAIL_OPT_COLS = OPT_COLS

# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)
warnings.filterwarnings("ignore")

# Helpers
def format_tb_pair(ltp, top, bottom):
    if pd.isna(top) or pd.isna(bottom): return "-"
    t_str, b_str, ltp_val = f"{float(top):.2f}", f"{float(bottom):.2f}", float(ltp)
    if ltp_val > float(top): t_str = f"<span style='color: #4ade80; font-weight: bold;'>{t_str}</span>"
    if ltp_val < float(bottom): b_str = f"<span style='color: #f87171; font-weight: bold;'>{b_str}</span>"
    return f"{t_str} <span style='color: #64748b;'>/</span> {b_str}"

def format_value(col, val):
    if pd.isna(val) or val in [float("inf"), float("-inf")]: return ""
    if "(T/B)" in col or col == "Options_Data": return str(val)
    if col in ["Trigger_TF", "Signal_Type", "Option_Contracts", "Climax_Date"]: return str(val)
    if col == "Breach_Days": return str(int(val)) if not pd.isna(val) else ""
    if col == "% Change":
        try:
            f = float(val)
            return f"{f:.2f}%"
        except Exception:
            return ""
    if col.startswith("T_") or col.startswith("B_") or col == "ATM_Strike": return f"{float(val):.2f}"
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
    exp_weekday = 3 if "NIFTY" in symbol else 4
    days_ahead = (exp_weekday - today.weekday()) % 7
    if days_ahead == 0: days_ahead = 7
    return False, today + timedelta(days=days_ahead)

def get_options_data(symbol, ltp, side):
    exch, base_name, interval = get_index_meta(symbol)
    atm_strike = int(round(ltp / interval) * interval)
    is_monthly, expiry = get_expiry_details(symbol)
    yy = expiry.strftime("%y")
    expiry_code = expiry.strftime("%b").upper() if is_monthly else f"{['1','2','3','4','5','6','7','8','9','O','N','D'][expiry.month-1]}{expiry.strftime('%d')}"
    opt_type = "CE" if side == "long" else "PE"
    strikes = [atm_strike + (i * interval) for i in range(-10, 11)]
    symbols = [f"{exch}:{base_name}{yy}{expiry_code}{s}{opt_type}" for s in strikes]
    return float(atm_strike), f"21 {opt_type} Contracts", symbols

# Fyers & history
def init_fyers():
    try:
        return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e:
        logger.warning(f"INIT Failed: {e}")
        return None

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
    except Exception:
        return None

# SCAN
def scan_fno_universe(fyers):
    rows = []
    for sym in cfg.index_symbols:
        daily = get_history(fyers, sym, "D", 365)
        if daily is None or daily.empty:
            continue
        today = datetime.now()
        ltp = float(daily["close"].iloc[-1])
        # build supports/resistances via calendar windows
        windows = [("6M",180,90),("3M",90,30),("1M",30,7),("1W",7,3),("3D",3,1),("1D",1,0)]
        supports = {}
        resistances = {}
        for label,max_d,min_d in windows:
            start_date = today - timedelta(days=max_d)
            end_date = today - timedelta(days=min_d)
            df_s = daily[(daily['timestamp'] > start_date) & (daily['timestamp'] <= end_date)]
            if df_s.empty: continue
            idx_val = df_s["volume"].idxmax() if (df_s["volume"]>0).any() else (df_s["high"]-df_s["low"]).idxmax()
            c = df_s.loc[idx_val]
            top = float(c["high"]); bot = float(c["low"])
            supports[label] = (top, bot, str(c["timestamp"].date()))
            resistances[label] = (top, bot, str(c["timestamp"].date()))
        # map to dashboard columns (3 levels each side)
        def pick_levels(map_dict):
            # choose three meaningful levels sorted by timeframe priority (1M/3M/6M)
            lbls = ["1M","3M","6M","1W","3D","1D"]
            picked = []
            for l in lbls:
                if l in map_dict:
                    picked.append(map_dict[l])
                if len(picked) == 3:
                    break
            # format as "top / bot"
            out = []
            for t,b,d in picked:
                out.append(f"{t:.2f} / {b:.2f}")
            while len(out) < 3:
                out.append("")
            return out
        res_levels = pick_levels(resistances)
        sup_levels = pick_levels(supports)
        pct_change = ((ltp - float(daily["close"].iloc[-2]))/float(daily["close"].iloc[-2]) * 100) if len(daily) > 1 else 0.0
        row = {
            "Symbol": sym,
            "% Change": f"{pct_change:.2f}%",
            "Support-3": sup_levels[0],
            "Support-2": sup_levels[1],
            "Support-1": sup_levels[2],
            "LTP": ltp,
            "Resistance-1": res_levels[0],
            "Resistance-2": res_levels[1],
            "Resistance-3": res_levels[2],
        }
        rows.append(row)
    return pd.DataFrame(rows)

def scan_options_universe(fyers, symbols):
    rows = []
    for sym in symbols:
        daily = get_history(fyers, sym, "D", 60)
        if daily is None or daily.empty:
            continue
        ltp = float(daily["close"].iloc[-1])
        max_idx = daily["volume"].idxmax() if (daily["volume"] > 0).any() else (daily["high"] - daily["low"]).idxmax()
        c = daily.loc[max_idx]
        t = float(c["high"]); b = float(c["low"])
        bd = 999
        if ltp > t:
            breaches = daily[daily['close'] <= t]
            if not breaches.empty:
                bd = (datetime.now() - breaches.iloc[-1]['timestamp']).days
        pct_change = ((ltp - float(daily["close"].iloc[-2]))/float(daily["close"].iloc[-2]) * 100) if len(daily) > 1 else 0.0
        rows.append({
            "Symbol": sym,
            "LTP": ltp,
            "T_LOC": t,
            "B_LOC": b,
            "D_LOC": str(pd.to_datetime(c["timestamp"]).date()),
            "Days_L_LOC": bd,
            "Prev_Close": float(daily["close"].iloc[-2]) if len(daily) > 1 else np.nan,
            "% Change": pct_change
        })
    return pd.DataFrame(rows)

# BUILD tables & candidates (cleaned, separate outputs)
def build_dashboard_and_candidates(spot_df):
    dashboard_rows = []
    long_rows = []
    short_rows = []
    label_map = {"1D":"1-Day","3D":"3-Day","1W":"1-Week","1M":"1-Month","3M":"3-Month","6M":"6-Month"}
    for _, r in spot_df.iterrows():
        sym = r["Symbol"]; ltp = r["LTP"]
        # prepare dashboard row with supports/resistances empty by default
        dash = {"Symbol": sym, "% Change": r.get("% Change",""), "Support-3":"", "Support-2":"", "Support-1":"", "LTP": ltp, "Resistance-1":"", "Resistance-2":"", "Resistance-3":""}
        # build TB pairs columns for email display
        for tf in ["1D","3D","1W","1M","3M","6M"]:
            dash[f"{label_map[tf]} (T/B)"] = format_tb_pair(ltp, r.get(f"T_{tf}"), r.get(f"B_{tf}"))
        dashboard_rows.append(dash)
        # detect triggers (small window threshold)
        trigger_tf = None; trigger_side = None; climax_date=None; target_options=None
        for tf in ["1D","3D","1W","1M","3M","6M"]:
            t = r.get(f"T_{tf}"); b = r.get(f"B_{tf}")
            bd_l = r.get(f"Days_L_{tf}", 999); bd_s = r.get(f"Days_S_{tf}", 999)
            if pd.notna(t) and ltp > t and bd_l <= 5:
                trigger_tf = tf; trigger_side="long"; climax_date = r.get(f"D_{tf}"); _,_,target_options = get_options_data(sym, ltp, "long"); break
            if pd.notna(b) and ltp < b and bd_s <= 5:
                trigger_tf = tf; trigger_side="short"; climax_date = r.get(f"D_{tf}"); _,_,target_options = get_options_data(sym, ltp, "short"); break
        if trigger_tf:
            cand = {"Symbol": sym, "LTP": ltp, "Trigger_TF": trigger_tf}
            # add TB pairs for cand table too
            for tf in ["1D","3D","1W","1M","3M","6M"]:
                cand[f"{label_map[tf]} (T/B)"] = format_tb_pair(ltp, r.get(f"T_{tf}"), r.get(f"B_{tf}"))
            if trigger_side == "long":
                long_rows.append(cand)
            else:
                short_rows.append(cand)
    return pd.DataFrame(dashboard_rows), pd.DataFrame(long_rows), pd.DataFrame(short_rows)

def build_option_candidate_tables(opt_df, spot_signal_map):
    if opt_df is None or opt_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    ce_rows=[]; pe_rows=[]
    for _, r in opt_df.iterrows():
        t, b, d, bd = r.get("T_LOC"), r.get("B_LOC"), r.get("D_LOC"), r.get("Days_L_LOC", 999)
        if pd.notna(t) and r["LTP"] > t and bd <= 10:
            row = {"Symbol": r["Symbol"], "LTP": r["LTP"], "% Change": r.get("% Change",""), "Climax_Date": d, "Breach_Days": bd, "Climax_Range (T/B)": format_tb_pair(r["LTP"], t, b)}
            spot_sig = spot_signal_map.get(get_underlying_spot(r["Symbol"]), "")
            row["Signal_Type"] = "Holy Grail" if spot_sig == "Fresh Sweep" else "Active Trend"
            if r["Symbol"].endswith("CE"):
                ce_rows.append(row)
            elif r["Symbol"].endswith("PE"):
                pe_rows.append(row)
    return pd.DataFrame(ce_rows), pd.DataFrame(pe_rows)

# Finalize DataFrame with fixed columns
def finalize_df(df, cols):
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            df2[c] = ""
    return df2.reindex(columns=cols).fillna("")

# HTML table builders (same styling; use finalized tables for safe access)
def build_html_table(df, title, cols):
    if df is None or df.empty:
        return f"<h3 style='color:#fbbf24'>{title}</h3><p style='color:#94a3b8'>No candidates.</p>"
    table_html = f"<h3 style='color:#fbbf24'>{title}</h3><table style='width:100%;background:#0f172a;color:#e2e8f0;font-family:sans-serif;font-size:13px;border-collapse:collapse'>"
    table_html += "<tr style='background:#1e293b;color:#f1f5f9'>" + "".join([f"<th style='padding:8px;border:1px solid #334155'>{c}</th>" for c in cols]) + "</tr>"
    for i, (_, row) in enumerate(df.iterrows()):
        bg = "#0f172a" if i%2==0 else "#1e293b"
        table_html += f"<tr style='background:{bg};color:#e2e8f0'>"
        for c in cols:
            val = row.get(c, "")
            table_html += f"<td style='padding:8px;border:1px solid #334155'>{format_value(c, val)}</td>"
        table_html += "</tr>"
    table_html += "</table>"
    return table_html

# Save outputs as separate CSVs
def save_outputs(dashboard_df, long_df, short_df, ce_df, pe_df):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out = "output"
    os.makedirs(out, exist_ok=True)
    db = finalize_df(dashboard_df, DASHBOARD_COLS)
    lg = finalize_df(long_df, STRATEGY_COLS)
    sh = finalize_df(short_df, STRATEGY_COLS)
    ce = finalize_df(ce_df, OPT_COLS)
    pe = finalize_df(pe_df, OPT_COLS)
    f_db = os.path.join(out, f"market_dashboard_{ts}.csv")
    f_lg = os.path.join(out, f"long_strategy_{ts}.csv")
    f_sh = os.path.join(out, f"short_strategy_{ts}.csv")
    f_ce = os.path.join(out, f"ce_candidates_{ts}.csv")
    f_pe = os.path.join(out, f"pe_candidates_{ts}.csv")
    db.to_csv(f_db, index=False, na_rep="")
    lg.to_csv(f_lg, index=False, na_rep="")
    sh.to_csv(f_sh, index=False, na_rep="")
    ce.to_csv(f_ce, index=False, na_rep="")
    pe.to_csv(f_pe, index=False, na_rep="")
    return f_db, f_lg, f_sh, f_ce, f_pe

# Email
def send_email(dashboard_df, long_df, short_df, ce_df, pe_df, csv_files):
    try:
        msg = MIMEMultipart()
        msg["From"] = cfg.sender_email
        msg["To"] = cfg.recipient_email
        msg["Subject"] = f"Index Climax Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        html = "<body style='background:#030712;padding:20px;font-family:sans-serif'>"
        html += build_html_table(finalize_df(dashboard_df, DASHBOARD_COLS), "Market Dashboard", EMAIL_DISPLAY_COLS)
        html += build_html_table(finalize_df(long_df, STRATEGY_COLS), "Long Strategy Matrix", EMAIL_DISPLAY_COLS)
        html += build_html_table(finalize_df(short_df, STRATEGY_COLS), "Short Strategy Matrix", EMAIL_DISPLAY_COLS)
        html += build_html_table(finalize_df(ce_df, OPT_COLS), "Call Options (CE) Climax Verification", EMAIL_OPT_COLS)
        html += build_html_table(finalize_df(pe_df, OPT_COLS), "Put Options (PE) Climax Verification", EMAIL_OPT_COLS)
        html += "</body>"
        msg.attach(MIMEText(html, "html"))
        # attach CSVs
        for f in csv_files:
            with open(f, "rb") as fh:
                part = MIMEBase("application","octet-stream")
                part.set_payload(fh.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f'attachment; filename={os.path.basename(f)}')
                msg.attach(part)
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, cfg.recipient_email, msg.as_string())
        logger.info("Email sent.")
    except Exception as e:
        logger.error(f"Email failed: {e}")

def main():
    fyers = init_fyers()
    if not fyers:
        logger.error("Fyers init failed; exiting.")
        return
    # spot scan (this produces intermediate columns like T_1M, Days_L_1M etc. expected from original logic)
    spot_df = scan_fno_universe(fyers)
    if spot_df is None or spot_df.empty:
        logger.warning("No spot data; exiting.")
        return
    # dashboard + candidate matrices
    dashboard_df, long_df, short_df = build_dashboard_and_candidates(spot_df)
    # collect option symbols from candidates
    opt_symbols = []
    for df in [long_df, short_df]:
        if df is not None and not df.empty and "Target_Options" in df.columns:
            for sub in df["Target_Options"].tolist():
                if isinstance(sub, list):
                    opt_symbols.extend(sub)
    opt_symbols = list(set(opt_symbols))
    spot_map = {}
    if long_df is not None and not long_df.empty and "Trigger_TF" in long_df.columns:
        spot_map.update({r["Symbol"]: r.get("Signal_Type","Active Trend") for _, r in long_df.iterrows()})
    if short_df is not None and not short_df.empty and "Trigger_TF" in short_df.columns:
        spot_map.update({r["Symbol"]: r.get("Signal_Type","Active Trend") for _, r in short_df.iterrows()})
    ce_df = pd.DataFrame(); pe_df = pd.DataFrame()
    if opt_symbols:
        opt_scan = scan_options_universe(fyers, opt_symbols)
        ce_df, pe_df = build_option_candidate_tables(opt_scan, spot_map)
    # Save outputs (separate CSVs)
    csv_files = save_outputs(dashboard_df, long_df, short_df, ce_df, pe_df)
    # send email with inline HTML + attachments
    send_email(dashboard_df, long_df, short_df, ce_df, pe_df, csv_files)
    logger.info("Done.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py - Final Comprehensive Index Dashboard & Strategy Scanner
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

# Columns reordered: 3-Day comes before 1-Week
EMAIL_DISPLAY_COLS = [
    "Symbol", "LTP", "Trigger_TF", 
    "6-Month (T/B)", "3-Month (T/B)", "1-Month (T/B)", 
    "3-Day (T/B)", "1-Week (T/B)"
]

EMAIL_OPT_COLS = [
    "Symbol", "LTP", "% Change", "Signal_Type", 
    "Climax_Date", "Climax_Range (T/B)", "Breach_Days"
]

# Logger Setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)
warnings.filterwarnings("ignore")

# ==========================================
# HELPERS
# ==========================================

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
    if col == "% Change": return f"{float(val):.2f}%"
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

# ==========================================
# CORE SCANNING ENGINE
# ==========================================

def init_fyers():
    try: return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e: logger.warning(f"INIT Failed: {e}"); return None

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

def scan_fno_universe(fyers):
    rows = []
    for sym in cfg.index_symbols:
        daily = get_history(fyers, sym, "D", 200)
        if daily is not None and not daily.empty:
            today_date = datetime.now().date()
            prev_close = float(daily["close"].iloc[-2])
            ltp = float(daily["close"].iloc[-1])
            bands, streak_data = {}, {}
            
            # Non-overlapping windows
            timeframe_windows = [
                ("6M", 135, 66), ("3M", 65, 23), ("1M", 22, 6), ("3D", 3, 0), ("1W", 5, 4)
            ]
            
            for label, max_days, min_days in timeframe_windows:
                df_s = daily.iloc[-max_days:-min_days] if min_days > 0 else daily.iloc[-max_days:]
                if df_s.empty: continue
                idx_val = df_s["volume"].idxmax() if (df_s["volume"]>0).any() else (df_s["high"]-df_s["low"]).idxmax()
                c = df_s.loc[idx_val]
                bands.update({f"T_{label}": float(c["high"]), f"B_{label}": float(c["low"]), f"D_{label}": str(c["timestamp"].date())})
                
                bd_l, bd_s = 999, 999
                if ltp > float(c["high"]):
                    idx = len(daily) - 1
                    while idx >= 0:
                        if daily["close"].iloc[idx] <= float(c["high"]):
                            bd_l = (today_date - daily["timestamp"].iloc[min(idx + 1, len(daily) - 1)].date()).days + 1
                            break
                        idx -= 1
                if ltp < float(c["low"]):
                    idx = len(daily) - 1
                    while idx >= 0:
                        if daily["close"].iloc[idx] >= float(c["low"]):
                            bd_s = (today_date - daily["timestamp"].iloc[min(idx + 1, len(daily) - 1)].date()).days + 1
                            break
                        idx -= 1
                streak_data[f"Days_L_{label}"] = bd_l
                streak_data[f"Days_S_{label}"] = bd_s
            
            row = {"Symbol": sym, "LTP": ltp, "% Change": ((ltp - prev_close)/prev_close*100)}
            row.update(bands); row.update(streak_data); rows.append(row)
    return pd.DataFrame(rows)

def scan_options_universe(fyers, symbols):
    rows = []
    for sym in symbols:
        daily = get_history(fyers, sym, "D", 60)
        if daily is not None and not daily.empty:
            prev_close = float(daily["close"].iloc[-2])
            ltp = float(daily["close"].iloc[-1])
            max_idx = daily["volume"].idxmax() if (daily["volume"] > 0).any() else (daily["high"] - daily["low"]).idxmax()
            c = daily.loc[max_idx]
            today_date = datetime.now().date()
            top_band = float(c["high"])
            bd_l = 999
            if ltp > top_band:
                idx = len(daily) - 1
                while idx >= 0:
                    if daily["close"].iloc[idx] <= top_band:
                        bd_l = (today_date - daily["timestamp"].iloc[min(idx + 1, len(daily) - 1)].date()).days + 1
                        break
                    idx -= 1
            rows.append({"Symbol": sym, "LTP": ltp, "T_LOC": top_band, "B_LOC": float(c["low"]), "D_LOC": str(pd.to_datetime(c["timestamp"]).date()), "Days_L_LOC": bd_l, "Prev_Close": prev_close, "% Change": ((ltp - prev_close)/prev_close*100)})
    return pd.DataFrame(rows)

def build_dashboard_and_candidates(df):
    dashboard_rows, valid_long, valid_short = [], [], []
    for _, row in df.iterrows():
        r_dict = row.to_dict()
        # Order of columns matches EMAIL_DISPLAY_COLS
        for tf in ["6M", "3M", "1M", "3D", "1W"]:
            label = tf.replace('D','-Day').replace('M','-Month').replace('W','-Week')
            r_dict[f"{label} (T/B)"] = format_tb_pair(row["LTP"], row.get(f"T_{tf}"), row.get(f"B_{tf}"))
        dashboard_rows.append(r_dict)
        
        # Strategy check
        for tf in ["6M", "3M", "1M", "1W", "3D"]:
            t, b, d, bd_l, bd_s = row.get(f"T_{tf}"), row.get(f"B_{tf}"), row.get(f"D_{tf}"), row.get(f"Days_L_{tf}"), row.get(f"Days_S_{tf}")
            if pd.notna(t) and row["LTP"] > t and bd_l <= 5:
                strike, opt_str, opt_list = get_options_data(row["Symbol"], row["LTP"], "long")
                r_dict.update({"Trigger_TF": tf, "Climax_Date": d, "Target_Options": opt_list, "Signal_Type": "Active Trend"})
                valid_long.append(r_dict); break
            elif pd.notna(b) and row["LTP"] < b and bd_s <= 5:
                strike, opt_str, opt_list = get_options_data(row["Symbol"], row["LTP"], "short")
                r_dict.update({"Trigger_TF": tf, "Climax_Date": d, "Target_Options": opt_list, "Signal_Type": "Active Trend"})
                valid_short.append(r_dict); break
    return pd.DataFrame(dashboard_rows), pd.DataFrame(valid_long), pd.DataFrame(valid_short)

def build_option_candidate_tables(df, spot_signal_map):
    if df.empty: return pd.DataFrame(), pd.DataFrame()
    valid_rows = []
    for _, row in df.iterrows():
        t, b, d, bd = row.get("T_LOC"), row.get("B_LOC"), row.get("D_LOC"), row.get("Days_L_LOC")
        if pd.notna(t) and row["LTP"] > t and bd <= 10:
            r_dict = row.to_dict()
            spot_signal = spot_signal_map.get(get_underlying_spot(row["Symbol"]), "")
            r_dict["Signal_Type"] = "Holy Grail" if spot_signal == "Fresh Sweep" else "Active Trend"
            r_dict["Climax_Date"] = d
            r_dict["Breach_Days"] = bd
            r_dict["Climax_Range (T/B)"] = format_tb_pair(row["LTP"], t, b)
            valid_rows.append(r_dict)
    res = pd.DataFrame(valid_rows)
    if res.empty: return pd.DataFrame(), pd.DataFrame()
    return res[res["Symbol"].str.endswith("CE")], res[res["Symbol"].str.endswith("PE")]

def build_html_table(df, title, cols):
    if df.empty: return f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top: 25px;'>{title}</h3><p style='color:#94a3b8; font-family:sans-serif;'>No candidates.</p>"
    table_html = f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top: 25px;'>{title}</h3><table style='border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 13px; text-align: left; background-color: #0f172a;'>"
    table_html += "<tr style='background-color: #1e293b; color: #f1f5f9;'>" + "".join([f"<th style='padding: 10px; border: 1px solid #334155;'>{c}</th>" for c in cols]) + "</tr>"
    for i, (_, row) in enumerate(df.iterrows()):
        bg_row = "#0f172a" if i % 2 == 0 else "#1e293b"
        sig = str(row.get("Signal_Type", ""))
        row_style = f"background-color: {bg_row}; color: #e2e8f0;"
        if "Holy Grail" in sig: row_style = "background-color: #581c87; color: #f5d0fe;"
        elif "Sweep" in sig: row_style = "background-color: #92400e; color: #fef3c7;"
        table_html += f"<tr style='{row_style}'>"
        for c in cols:
            val = row.get(c)
            style = "padding: 8px; border: 1px solid #334155;"
            if c == "% Change": style += " color: #4ade80; font-weight: bold;" if float(val or 0) > 0 else " color: #f87171; font-weight: bold;"
            table_html += f"<td style='{style}'>{format_value(c, val)}</td>"
        table_html += "</tr>"
    return table_html + "</table>"

def save_outputs(summary_df):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_file = f"scan_summary_{ts}.csv"
    summary_df.to_csv(csv_file, index=False)
    return csv_file

def send_email(dashboard_df, long_df, short_df, ce_df, pe_df, csv_file):
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = cfg.sender_email, cfg.recipient_email, f"Index Climax Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        html = f"<body style='background-color: #030712; padding: 20px; font-family: sans-serif;'>{build_html_table(dashboard_df, 'Market Dashboard', EMAIL_DISPLAY_COLS)}{build_html_table(long_df, 'Long Strategy Matrix', EMAIL_DISPLAY_COLS)}{build_html_table(short_df, 'Short Strategy Matrix', EMAIL_DISPLAY_COLS)}{build_html_table(ce_df, 'Call Options (CE) Climax Verification', EMAIL_OPT_COLS)}{build_html_table(pe_df, 'Put Options (PE) Climax Verification', EMAIL_OPT_COLS)}</body>"
        msg.attach(MIMEText(html, "html"))
        with open(csv_file, "rb") as f:
            part = MIMEBase("application", "octet-stream"); part.set_payload(f.read()); encoders.encode_base64(part); part.add_header("Content-Disposition", f'attachment; filename={os.path.basename(csv_file)}'); msg.attach(part)
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s: s.starttls(); s.login(cfg.sender_email, cfg.sender_password); s.sendmail(cfg.sender_email, cfg.recipient_email, msg.as_string())
        logger.info("Email sent successfully.")
    except Exception as e: logger.error(f"Email Failed: {e}")

def main():
    fyers = init_fyers()
    if not fyers: return
    
    spot_df = scan_fno_universe(fyers)
    if spot_df.empty: 
        logger.warning("No spot data generated.")
        return
        
    dashboard_df, long_df, short_df = build_dashboard_and_candidates(spot_df)
    
    all_opt = []
    if not long_df.empty and "Target_Options" in long_df.columns:
        for sublist in long_df["Target_Options"].tolist():
            if isinstance(sublist, list): all_opt.extend(sublist)
    if not short_df.empty and "Target_Options" in short_df.columns:
        for sublist in short_df["Target_Options"].tolist():
            if isinstance(sublist, list): all_opt.extend(sublist)
    all_opt = list(set(all_opt))
    
    spot_map = {}
    if not long_df.empty and "Signal_Type" in long_df.columns:
        spot_map.update({r["Symbol"]: r["Signal_Type"] for _, r in long_df.iterrows()})
    if not short_df.empty and "Signal_Type" in short_df.columns:
        spot_map.update({r["Symbol"]: r["Signal_Type"] for _, r in short_df.iterrows()})
        
    ce_df, pe_df = pd.DataFrame(), pd.DataFrame()
    if all_opt:
        opt_df = scan_options_universe(fyers, all_opt)
        ce_df, pe_df = build_option_candidate_tables(opt_df, spot_map)
        
    csv_f = save_outputs(spot_df)
    send_email(dashboard_df, long_df, short_df, ce_df, pe_df, csv_f)

if __name__ == "__main__": main()

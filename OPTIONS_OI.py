#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py - Comprehensive Calendar-Based Index Dashboard
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

# Top-5 volume climax levels â€” compact: one "Level-N" column per level
# Each cell shows:  DATE  |  Top / Bottom  (vol)
EMAIL_DISPLAY_COLS = [
    "Symbol", "LTP", "% Change",
    "Level-1", "Level-2", "Level-3", "Level-4", "Level-5",
]

# Candidate matrix columns (long / short)
EMAIL_CAND_COLS = [
    "Symbol", "LTP", "% Change",
    "Climax_Date", "Climax_Range (T/B)",
    "Climax_Volume", "Breach_Days", "Signal_Type",
]

EMAIL_OPT_COLS = [
    "Symbol", "LTP", "% Change", "Signal_Type",
    "Climax_Date", "Climax_Range (T/B)", "Breach_Days"
]

# Logger Setup
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)
warnings.filterwarnings("ignore")

# ==========================================
# HELPERS
# ==========================================

def format_tb_pair(ltp, top, bottom):
    if pd.isna(top) or pd.isna(bottom):
        return "-"
    t_str, b_str, ltp_val = f"{float(top):.2f}", f"{float(bottom):.2f}", float(ltp)
    if ltp_val > float(top):
        t_str = f"<span style='color: #4ade80; font-weight: bold;'>{t_str}</span>"
    if ltp_val < float(bottom):
        b_str = f"<span style='color: #f87171; font-weight: bold;'>{b_str}</span>"
    return f"{t_str} <span style='color: #64748b;'>/</span> {b_str}"

def format_value(col, val):
    if pd.isna(val) or val in [float("inf"), float("-inf")]:
        return ""
    if "(T/B)" in col or col == "Options_Data":
        return str(val)
    if col in ["Trigger_TF", "Signal_Type", "Option_Contracts", "Climax_Date"]:
        return str(val)
    if col == "Breach_Days":
        return str(int(val)) if not pd.isna(val) else ""
    if col == "% Change":
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return ""
        return f"{fval:.2f}%"
    if col.startswith("T_") or col.startswith("B_") or col == "ATM_Strike":
        return f"{float(val):.2f}"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.4f}"
    return str(val)

def get_index_meta(symbol):
    if "NIFTY50" in symbol:
        return "NSE", "NIFTY", 50
    if "NIFTYBANK" in symbol:
        return "NSE", "BANKNIFTY", 100
    return "BSE", "SENSEX", 100

def get_underlying_spot(opt_symbol):
    if "BANKNIFTY" in opt_symbol:
        return "NSE:NIFTYBANK-INDEX"
    if "NIFTY" in opt_symbol:
        return "NSE:NIFTY50-INDEX"
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
    # Weekly expiry (NIFTY=Thursday, SENSEX=Friday)
    exp_weekday = 3 if "NIFTY" in symbol else 4
    days_ahead = (exp_weekday - today.weekday()) % 7
    # If today IS the expiry day, use next week's expiry (expired after market close)
    if days_ahead == 0:
        days_ahead = 7
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
    desc = f"21 {opt_type} Contracts (Strikes: {strikes[0]:.2f} to {strikes[-1]:.2f})"
    return float(atm_strike), desc, symbols

# ==========================================
# CORE SCANNING ENGINE
# ==========================================

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
        res_data = fyers.history(data={
            "symbol": symbol,
            "resolution": res,
            "date_format": "1",
            "range_from": start,
            "range_to": end,
            "cont_flag": "1",
        })
        if res_data and "candles" in res_data:
            df = pd.DataFrame(res_data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            return df
        return None
    except Exception:
        return None

def scan_fno_universe(fyers):
    """
    For each index, fetch 6 months of daily data.
    Rank ALL candles by volume descending and keep the top 5.
    Each top-5 candle becomes a 'climax level' (High = Top band, Low = Bottom band).
    Also compute:
      - Days since LTP last closed BELOW the band top  (bd_l)
      - Days since LTP last closed ABOVE the band bot  (bd_s)
    Sorted newestâ†’oldest within the top-5 by date (after volume ranking).
    """
    TOP_N = 5
    rows = []

    for sym in cfg.index_symbols:
        daily = get_history(fyers, sym, "D", 182)   # ~6 months
        if daily is None or daily.empty:
            continue

        today  = datetime.now()
        ltp    = float(daily["close"].iloc[-1])
        pct_ch = ((ltp - float(daily["close"].iloc[-2])) / float(daily["close"].iloc[-2]) * 100)

        # â”€â”€ Rank by volume, pick top N; then re-sort by proximity of band
        #    to LTP (nearest band edge wins) so the most actionable level
        #    always appears as Rank-1 in the email. â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        top_vol = daily.sort_values("volume", ascending=False).head(TOP_N).copy()

        # Proximity = min distance from LTP to either edge of the candle range
        top_vol["_proximity"] = top_vol.apply(
            lambda r: min(abs(ltp - r["high"]), abs(ltp - r["low"])), axis=1
        )
        ranked = top_vol.sort_values("_proximity", ascending=True).reset_index(drop=True)

        row: Dict = {"Symbol": sym, "LTP": ltp, "% Change": pct_ch}

        for rank, (_, c) in enumerate(ranked.iterrows(), start=1):
            top_band = float(c["high"])
            bot_band = float(c["low"])
            vol      = float(c["volume"])
            date_str = str(c["timestamp"].date())

            row[f"T_R{rank}"]    = top_band
            row[f"B_R{rank}"]    = bot_band
            row[f"D_R{rank}"]    = date_str
            row[f"Vol_R{rank}"]  = vol

            bd_l = bd_s = 999
            if ltp > top_band:
                below = daily[daily["close"] <= top_band]
                if not below.empty:
                    bd_l = (today - below.iloc[-1]["timestamp"]).days
            if ltp < bot_band:
                above = daily[daily["close"] >= bot_band]
                if not above.empty:
                    bd_s = (today - above.iloc[-1]["timestamp"]).days

            row[f"BdL_R{rank}"] = bd_l
            row[f"BdS_R{rank}"] = bd_s

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
        top_band = float(c["high"])
        bd_l = 999
        if ltp > top_band:
            breaches = daily[daily["close"] <= top_band]
            if not breaches.empty:
                bd_l = (datetime.now() - breaches.iloc[-1]["timestamp"]).days
        rows.append({
            "Symbol": sym,
            "LTP": ltp,
            "T_LOC": top_band,
            "B_LOC": float(c["low"]),
            "D_LOC": str(pd.to_datetime(c["timestamp"]).date()),
            "Days_L_LOC": bd_l,
            "Prev_Close": float(daily["close"].iloc[-2]),
            "% Change": ((ltp - float(daily["close"].iloc[-2])) / float(daily["close"].iloc[-2]) * 100),
        })
    return pd.DataFrame(rows)

def build_dashboard_and_candidates(df):
    """
    Build:
      1) dashboard_df  â€” one row per symbol, showing all 5 climax levels
      2) long_df       â€” symbols where LTP broke above a climax top (fresh, <=5 days)
      3) short_df      â€” symbols where LTP broke below a climax bottom (fresh, <=5 days)
    Signal fires on the HIGHEST-RANKED (largest volume) qualifying level.
    """
    dashboard_rows, valid_long, valid_short = [], [], []

    for _, row in df.iterrows():
        r_dict = row.to_dict()

        # â”€â”€ Format top-5 as T/B only â€” no date, no volume inside cell â”€â”€â”€â”€â”€â”€â”€â”€â”€
        for rank in range(1, 6):
            t = row.get(f"T_R{rank}")
            b = row.get(f"B_R{rank}")
            r_dict[f"Level-{rank}"] = format_tb_pair(row["LTP"], t, b) if pd.notna(t) and pd.notna(b) else "-"

        # â”€â”€ Signal detection: scan rank-1 â†’ rank-5, first qualifying wins â”€â”€â”€â”€
        active_rank   = None
        trigger_side  = None
        climax_date   = None
        climax_top    = None
        climax_bot    = None
        climax_vol    = None
        target_opts   = None

        for rank in range(1, 6):
            t    = row.get(f"T_R{rank}")
            b    = row.get(f"B_R{rank}")
            d    = row.get(f"D_R{rank}")
            bd_l = row.get(f"BdL_R{rank}", 999)
            bd_s = row.get(f"BdS_R{rank}", 999)
            vol  = row.get(f"Vol_R{rank}")

            if pd.notna(t) and row["LTP"] > t and bd_l <= 5:
                active_rank  = rank
                trigger_side = "long"
                climax_date  = d
                climax_top   = t
                climax_bot   = b
                climax_vol   = vol
                _, _, target_opts = get_options_data(row["Symbol"], row["LTP"], "long")
                break

            if pd.notna(b) and row["LTP"] < b and bd_s <= 5:
                active_rank  = rank
                trigger_side = "short"
                climax_date  = d
                climax_top   = t
                climax_bot   = b
                climax_vol   = vol
                _, _, target_opts = get_options_data(row["Symbol"], row["LTP"], "short")
                break

        r_dict["Signal_Type"]  = f"L{active_rank} Long" if (active_rank and trigger_side=="long") else (f"L{active_rank} Short" if active_rank else "")
        dashboard_rows.append(r_dict.copy())

        if active_rank and target_opts:
            cand = r_dict.copy()
            cand["Climax_Date"]        = climax_date
            cand["Climax_Range (T/B)"] = format_tb_pair(row["LTP"], climax_top, climax_bot)
            cand["Climax_Volume"]      = f"{int(climax_vol):,}" if climax_vol else ""
            cand["Breach_Days"]        = row.get(f"BdL_R{active_rank}" if trigger_side == "long" else f"BdS_R{active_rank}", 999)
            cand["Target_Options"]     = target_opts
            if trigger_side == "long":
                valid_long.append(cand)
            else:
                valid_short.append(cand)

    return pd.DataFrame(dashboard_rows), pd.DataFrame(valid_long), pd.DataFrame(valid_short)


def build_option_candidate_tables(df, spot_signal_map):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
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
    if res.empty:
        return pd.DataFrame(), pd.DataFrame()
    return res[res["Symbol"].str.endswith("CE")], res[res["Symbol"].str.endswith("PE")]

def build_html_table(df, title, cols):
    if df.empty:
        return (
            f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top: 25px;'>{title}</h3>"
            "<p style='color:#94a3b8; font-family:sans-serif;'>No candidates.</p>"
        )
    table_html = (
        f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top: 25px;'>{title}</h3>"
        "<table style='border-collapse: collapse; width: 100%; font-family: sans-serif; "
        "font-size: 13px; text-align: left; background-color: #0f172a;'>"
    )
    table_html += (
        "<tr style='background-color: #1e293b; color: #f1f5f9;'>"
        + "".join([f"<th style='padding: 10px; border: 1px solid #334155;'>{c}</th>" for c in cols])
        + "</tr>"
    )
    for i, (_, row) in enumerate(df.iterrows()):
        bg_row = "#0f172a" if i % 2 == 0 else "#1e293b"
        sig = str(row.get("Signal_Type", ""))
        row_style = f"background-color: {bg_row}; color: #e2e8f0;"
        if "Holy Grail" in sig:
            row_style = "background-color: #581c87; color: #f5d0fe;"
        elif "Sweep" in sig:
            row_style = "background-color: #92400e; color: #fef3c7;"
        table_html += f"<tr style='{row_style}'>"
        for c in cols:
            val = row.get(c)
            style = "padding: 8px; border: 1px solid #334155;"
            if c == "% Change":
                try:
                    fval = float(val)
                    style += " color: #4ade80; font-weight: bold;" if fval > 0 else " color: #f87171; font-weight: bold;"
                except (TypeError, ValueError):
                    pass
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
        msg["From"] = cfg.sender_email
        msg["To"] = cfg.recipient_email
        msg["Subject"] = f"Index Climax Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        html = (
            "<body style='background-color: #030712; padding: 20px; font-family: sans-serif;'>"
            + build_html_table(dashboard_df, "Market Dashboard", EMAIL_DISPLAY_COLS)
            + build_html_table(long_df, "Long Strategy Matrix", EMAIL_CAND_COLS)
            + build_html_table(short_df, "Short Strategy Matrix", EMAIL_CAND_COLS)
            + build_html_table(ce_df, "Call Options (CE) Climax Verification", EMAIL_OPT_COLS)
            + build_html_table(pe_df, "Put Options (PE) Climax Verification", EMAIL_OPT_COLS)
            + "</body>"
        )
        msg.attach(MIMEText(html, "html"))
        with open(csv_file, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_file)}")
            msg.attach(part)
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port) as s:
            s.starttls()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, cfg.recipient_email, msg.as_string())
        logger.info("Email sent successfully.")
    except Exception as e:
        logger.error(f"Email Failed: {e}")

def main():
    fyers = init_fyers()
    if not fyers:
        return

    spot_df = scan_fno_universe(fyers)
    if spot_df.empty:
        logger.warning("No spot data generated.")
        return

    dashboard_df, long_df, short_df = build_dashboard_and_candidates(spot_df)

    all_opt = []
    if not long_df.empty and "Target_Options" in long_df.columns:
        for sublist in long_df["Target_Options"].tolist():
            if isinstance(sublist, list):
                all_opt.extend(sublist)
    if not short_df.empty and "Target_Options" in short_df.columns:
        for sublist in short_df["Target_Options"].tolist():
            if isinstance(sublist, list):
                all_opt.extend(sublist)
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

if __name__ == "__main__":
    main()

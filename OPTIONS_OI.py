#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py - High-Volume Support/Resistance Index Dashboard
"""

import os
import sys
import logging
import warnings
import calendar
from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

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

EMAIL_DISPLAY_COLS = [
    "Symbol", "% Change",
    "Support-3", "Support-2", "Support-1", "LTP",
    "Resistance-1", "Resistance-2", "Resistance-3",
]

EMAIL_CAND_COLS = [
    "Symbol", "% Change",
    "Support-3", "Support-2", "Support-1", "LTP",
    "Resistance-1", "Resistance-2", "Resistance-3",
    "Climax_Date", "Climax_Range (T/B)",
    "Climax_Volume", "Breach_Days", "Signal_Type",
]

EMAIL_OPT_COLS = [
    "Symbol", "LTP", "% Change", "Signal_Type",
    "Climax_Date", "Climax_Range (T/B)", "Breach_Days"
]

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)
warnings.filterwarnings("ignore")

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
    if "(T/B)" in col or col in EMAIL_DISPLAY_COLS:
        return str(val)
    if col in ["Signal_Type", "Option_Contracts", "Climax_Date"]:
        return str(val)
    if col == "Breach_Days":
        return str(int(val)) if not pd.isna(val) else ""
    if col == "% Change":
        try:
            return f"{float(val):.2f}%"
        except (TypeError, ValueError):
            return ""
    if col == "LTP":
        try:
            return f"{float(val):.2f}"
        except (TypeError, ValueError):
            return ""
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
    exp_weekday = 3 if "NIFTY" in symbol else 4
    days_ahead = (exp_weekday - today.weekday()) % 7
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
    base_lookback = 90
    max_lookback = 150
    step = 30
    dedupe_pct = 0.005
    rows = []

    for sym in cfg.index_symbols:
        final_row = None

        for lookback_days in range(base_lookback, max_lookback + 1, step):
            daily = get_history(fyers, sym, "D", lookback_days)
            if daily is None or daily.empty or len(daily) < 5:
                continue

            today = datetime.now()
            ltp = float(daily["close"].iloc[-1])
            prev_close = float(daily["close"].iloc[-2])
            pct_ch = ((ltp - prev_close) / prev_close) * 100

            work = daily.sort_values(["volume", "high"], ascending=[False, False]).reset_index(drop=True)
            candidates = []

            for _, c in work.iterrows():
                top_band = float(c["high"])
                bot_band = float(c["low"])
                mid_band = (top_band + bot_band) / 2.0
                keep = True

                for s in candidates:
                    s_mid = (s["top"] + s["bottom"]) / 2.0
                    mid_close = abs(mid_band - s_mid) / max(abs(ltp), 1.0) <= dedupe_pct
                    overlap = not (top_band < s["bottom"] or bot_band > s["top"])
                    if mid_close or overlap:
                        keep = False
                        break

                if keep:
                    candidates.append({
                        "top": top_band,
                        "bottom": bot_band,
                        "date": str(pd.to_datetime(c["timestamp"]).date()),
                        "volume": float(c["volume"]),
                    })

            if not candidates:
                continue

            bands = pd.DataFrame(candidates)
            supports = bands[bands["top"] < ltp].copy()
            resistances = bands[bands["bottom"] > ltp].copy()
            current_zone = bands[(bands["bottom"] <= ltp) & (bands["top"] >= ltp)].copy()

            supports["sort_key"] = supports["top"]
            supports = supports.sort_values("sort_key", ascending=False).head(3).reset_index(drop=True)

            resistances["sort_key"] = resistances["bottom"]
            resistances = resistances.sort_values("sort_key", ascending=True).head(3).reset_index(drop=True)

            row: Dict = {"Symbol": sym, "LTP": ltp, "% Change": pct_ch, "Lookback_Used": lookback_days}

            for i in range(3):
                if i < len(supports):
                    s = supports.iloc[i]
                    row[f"SUP_T_{i+1}"] = float(s["top"])
                    row[f"SUP_B_{i+1}"] = float(s["bottom"])
                    row[f"SUP_D_{i+1}"] = str(s["date"])
                    row[f"SUP_V_{i+1}"] = float(s["volume"])
                else:
                    row[f"SUP_T_{i+1}"] = np.nan
                    row[f"SUP_B_{i+1}"] = np.nan
                    row[f"SUP_D_{i+1}"] = ""
                    row[f"SUP_V_{i+1}"] = np.nan

                if i < len(resistances):
                    r = resistances.iloc[i]
                    row[f"RES_T_{i+1}"] = float(r["top"])
                    row[f"RES_B_{i+1}"] = float(r["bottom"])
                    row[f"RES_D_{i+1}"] = str(r["date"])
                    row[f"RES_V_{i+1}"] = float(r["volume"])
                else:
                    row[f"RES_T_{i+1}"] = np.nan
                    row[f"RES_B_{i+1}"] = np.nan
                    row[f"RES_D_{i+1}"] = ""
                    row[f"RES_V_{i+1}"] = np.nan

            if len(resistances) > 0:
                top_band = float(resistances.iloc[0]["top"])
                below = daily[daily["close"] <= top_band]
                row["Long_T"] = top_band
                row["Long_B"] = float(resistances.iloc[0]["bottom"])
                row["Long_D"] = str(resistances.iloc[0]["date"])
                row["Long_V"] = float(resistances.iloc[0]["volume"])
                row["Long_Breach_Days"] = (today - below.iloc[-1]["timestamp"]).days if (ltp > top_band and not below.empty) else 999
            else:
                row["Long_T"] = row["Long_B"] = row["Long_V"] = np.nan
                row["Long_D"] = ""
                row["Long_Breach_Days"] = 999

            if len(supports) > 0:
                bot_band = float(supports.iloc[0]["bottom"])
                above = daily[daily["close"] >= bot_band]
                row["Short_T"] = float(supports.iloc[0]["top"])
                row["Short_B"] = bot_band
                row["Short_D"] = str(supports.iloc[0]["date"])
                row["Short_V"] = float(supports.iloc[0]["volume"])
                row["Short_Breach_Days"] = (today - above.iloc[-1]["timestamp"]).days if (ltp < bot_band and not above.empty) else 999
            else:
                row["Short_T"] = row["Short_B"] = row["Short_V"] = np.nan
                row["Short_D"] = ""
                row["Short_Breach_Days"] = 999

            final_row = row
            if len(supports) >= 3 and len(resistances) >= 3:
                break

        if final_row is not None:
            rows.append(final_row)

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
    dashboard_rows, valid_long, valid_short = [], [], []
    for _, row in df.iterrows():
        r_dict = row.to_dict()
        for i in range(1, 4):
            r_dict[f"Support-{i}"] = format_tb_pair(row["LTP"], row.get(f"SUP_T_{i}"), row.get(f"SUP_B_{i}")) if pd.notna(row.get(f"SUP_T_{i}")) else "-"
            r_dict[f"Resistance-{i}"] = format_tb_pair(row["LTP"], row.get(f"RES_T_{i}"), row.get(f"RES_B_{i}")) if pd.notna(row.get(f"RES_T_{i}")) else "-"
        dashboard_rows.append(r_dict.copy())

        # Safely capture the current LTP as a math float
        try:
            ltp_val = float(row["LTP"])
        except (ValueError, TypeError):
            continue

        res_b1 = row.get("RES_B_1")
        res_b2 = row.get("RES_B_2")
        
        try:
            res_b1_val = float(res_b1) if pd.notna(res_b1) and str(res_b1).strip() != "" else np.nan
        except:
            res_b1_val = np.nan
            
        try:
            res_b2_val = float(res_b2) if pd.notna(res_b2) and str(res_b2).strip() != "" else np.nan
        except:
            res_b2_val = np.nan

        if pd.notna(res_b1_val):
            # STRICT LONG LOGIC: LTP must be > Resistance-1. 
            # If Resistance-2 exists, LTP must be < Resistance-2.
            # If Resistance-2 DOES NOT exist, we block the trade to prevent chasing overextended moves.
            is_valid_long = False
            if ltp_val > res_b1_val:
                if pd.notna(res_b2_val) and ltp_val < res_b2_val:
                    is_valid_long = True
                elif pd.isna(res_b2_val): # No second resistance found
                    is_valid_long = False # Block it.

            if is_valid_long:
                cand = {
                    "Symbol": row["Symbol"],
                    "% Change": row["% Change"],
                    "Support-3": r_dict.get("Support-3", "-"),
                    "Support-2": r_dict.get("Support-2", "-"),
                    "Support-1": r_dict.get("Support-1", "-"),
                    "LTP": row["LTP"],
                    "Resistance-1": r_dict.get("Resistance-1", "-"),
                    "Resistance-2": r_dict.get("Resistance-2", "-"),
                    "Resistance-3": r_dict.get("Resistance-3", "-"),
                    "Climax_Date": row.get("Long_D", ""),
                    "Climax_Range (T/B)": format_tb_pair(row["LTP"], row.get("Long_T", res_b1), row.get("Long_B", res_b1)),
                    "Climax_Volume": f"{int(row['Long_V']):,}" if pd.notna(row.get("Long_V")) else "",
                    "Breach_Days": row.get("Long_Breach_Days", 999),
                    "Signal_Type": "Long Resistance Test",
                }
                _, _, cand["Target_Options"] = get_options_data(row["Symbol"], row["LTP"], "long")
                valid_long.append(cand)

        sup_t1 = row.get("SUP_T_1")
        sup_t2 = row.get("SUP_T_2")
        
        try:
            sup_t1_val = float(sup_t1) if pd.notna(sup_t1) and str(sup_t1).strip() != "" else np.nan
        except:
            sup_t1_val = np.nan
            
        try:
            sup_t2_val = float(sup_t2) if pd.notna(sup_t2) and str(sup_t2).strip() != "" else np.nan
        except:
            sup_t2_val = np.nan

        if pd.notna(sup_t1_val):
            # STRICT SHORT LOGIC: LTP must be < Support-1. 
            # If Support-2 exists, LTP must be >= Support-2.
            # If Support-2 DOES NOT exist, we block the trade to prevent chasing.
            is_valid_short = False
            if ltp_val < sup_t1_val:
                if pd.notna(sup_t2_val) and ltp_val >= sup_t2_val:
                    is_valid_short = True
                elif pd.isna(sup_t2_val): # No second support found
                    is_valid_short = False # Block it.

            if is_valid_short:
                cand = {
                    "Symbol": row["Symbol"],
                    "% Change": row["% Change"],
                    "Support-3": r_dict.get("Support-3", "-"),
                    "Support-2": r_dict.get("Support-2", "-"),
                    "Support-1": r_dict.get("Support-1", "-"),
                    "LTP": row["LTP"],
                    "Resistance-1": r_dict.get("Resistance-1", "-"),
                    "Resistance-2": r_dict.get("Resistance-2", "-"),
                    "Resistance-3": r_dict.get("Resistance-3", "-"),
                    "Climax_Date": row.get("Short_D", ""),
                    "Climax_Range (T/B)": format_tb_pair(row["LTP"], row.get("Short_T", sup_t1), row.get("Short_B", sup_t1)),
                    "Climax_Volume": f"{int(row['Short_V']):,}" if pd.notna(row.get("Short_V")) else "",
                    "Breach_Days": row.get("Short_Breach_Days", 999),
                    "Signal_Type": "Short Support Test",
                }
                _, _, cand["Target_Options"] = get_options_data(row["Symbol"], row["LTP"], "short")
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
        return f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top: 25px;'>{title}</h3><p style='color:#94a3b8; font-family:sans-serif;'>No candidates.</p>"
    table_html = f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top: 25px;'>{title}</h3><table style='border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 13px; text-align: left; background-color: #0f172a;'>"
    table_html += "<tr style='background-color: #1e293b; color: #f1f5f9;'>" + "".join([f"<th style='padding: 10px; border: 1px solid #334155;'>{c}</th>" for c in cols]) + "</tr>"
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

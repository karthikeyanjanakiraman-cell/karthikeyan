#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py - High-Volume Support/Resistance F&O Dashboard
Gap-fixed version.
"""

import os
import sys
import time
import logging
import warnings
import calendar
import requests
import re
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


def get_dynamic_fno_universe():
    url = "https://public.fyers.in/sym_details/NSE_FO.csv"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        pattern = re.compile(r"NSE:([A-Z0-9&\-]+)\d{2}[A-Z]{3}")
        matches = set(pattern.findall(response.text))
        indices = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
        fno_stocks = [f"NSE:{t}-EQ" for t in matches if t not in indices]
        return sorted(fno_stocks)
    except Exception:
        return ["NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:INFY-EQ"]


class Config:
    def __init__(self):
        self.client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        self.access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
        self.sender_password = os.environ.get("SENDER_PASSWORD", "password")
        self.recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")
        self.index_symbols = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"] + get_dynamic_fno_universe()
        self.max_symbols = int(os.environ.get("MAX_SYMBOLS", "0"))
        self.scan_sleep = float(os.environ.get("SCAN_SLEEP", "0.05"))
        self.option_scan_sleep = float(os.environ.get("OPTION_SCAN_SLEEP", "0.03"))
        self.breakout_breach_days = int(os.environ.get("BREACH_DAYS_SPOT", "5"))
        self.option_breach_days = int(os.environ.get("BREACH_DAYS_OPTION", "10"))


cfg = Config()
if cfg.max_symbols > 0:
    cfg.index_symbols = cfg.index_symbols[:cfg.max_symbols]

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
    if "(T/B)" in col or col.startswith("Support-") or col.startswith("Resistance-"):
        return str(val)
    if col in ["Signal_Type", "Option_Contracts", "Climax_Date"]:
        return str(val)
    if col == "Breach_Days":
        try:
            return str(int(float(val)))
        except Exception:
            return ""
    if col == "% Change":
        try:
            return f"{float(val):.2f}%"
        except Exception:
            return ""
    if col == "LTP":
        try:
            return f"{float(val):,.2f}"
        except Exception:
            return ""
    if col == "Climax_Volume":
        try:
            return f"{int(float(val)):,}"
        except Exception:
            return str(val)
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):,.2f}"
    return str(val)


def get_symbol_meta(symbol, ltp):
    exch = symbol.split(":")[0]
    base_name = symbol.split(":")[1].split("-")[0]
    if ltp < 150:
        interval = 1
    elif ltp < 500:
        interval = 5
    elif ltp < 2000:
        interval = 10
    elif ltp < 5000:
        interval = 50
    else:
        interval = 100
    if "NIFTY50" in symbol:
        interval = 50
    if "NIFTYBANK" in symbol:
        interval = 100
    if "SENSEX" in symbol:
        interval = 100
    return exch, base_name, interval


def get_underlying_spot(opt_symbol):
    if "BANKNIFTY" in opt_symbol:
        return "NSE:NIFTYBANK-INDEX"
    if "NIFTY" in opt_symbol:
        return "NSE:NIFTY50-INDEX"
    if "SENSEX" in opt_symbol:
        return "BSE:SENSEX-INDEX"
    match = re.match(r"^([A-Z0-9&\-]+)\d+", opt_symbol.split(":")[1])
    return f"NSE:{match.group(1)}-EQ" if match else opt_symbol


def get_expiry_details(symbol):
    today = datetime.now().date()

    def last_thu(y, m):
        last = calendar.monthrange(y, m)[1]
        d = datetime(y, m, last).date()
        return d - timedelta(days=(d.weekday() - 3) % 7)

    if "-EQ" in symbol or "NIFTYBANK" in symbol:
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
    exch, base_name, interval = get_symbol_meta(symbol, ltp)
    atm_strike = int(round(ltp / interval) * interval)
    is_monthly, expiry = get_expiry_details(symbol)
    yy = expiry.strftime("%y")
    expiry_code = expiry.strftime("%b").upper() if is_monthly else f"{['1','2','3','4','5','6','7','8','9','O','N','D'][expiry.month-1]}{expiry.strftime('%d')}"
    opt_type = "CE" if side == "long" else "PE"
    strikes = [atm_strike + (i * interval) for i in range(-5, 6)]
    symbols = [f"{exch}:{base_name}{yy}{expiry_code}{s}{opt_type}" for s in strikes]
    desc = f"11 {opt_type} Contracts"
    return float(atm_strike), desc, symbols


def init_fyers():
    try:
        return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e:
        logger.warning(f"INIT Failed: {e}")
        return None


def get_history(fyers, symbol, res, days, retries=2):
    for attempt in range(retries + 1):
        try:
            now = datetime.now()
            start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
            end = now.strftime("%Y-%m-%d")
            res_data = fyers.history(data={
                "symbol": symbol,
                "resolution": res,
                "date_format": "1",
                "range_from": start,
                "range_to": end,
                "cont_flag": "1"
            })
            if res_data and "candles" in res_data and res_data["candles"]:
                df = pd.DataFrame(res_data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
                return df
            return None
        except Exception:
            if attempt < retries:
                time.sleep(0.3 * (attempt + 1))
            else:
                return None


def scan_fno_universe(fyers):
    base_lookback, max_lookback, step, dedupe_pct = 90, 150, 30, 0.005
    rows = []
    for sym in cfg.index_symbols:
        time.sleep(cfg.scan_sleep)
        final_row = None
        for lookback_days in range(base_lookback, max_lookback + 1, step):
            daily = get_history(fyers, sym, "D", lookback_days)
            if daily is None or daily.empty or len(daily) < 5:
                continue
            today = datetime.now()
            ltp = float(daily["close"].iloc[-1])
            prev_close = float(daily["close"].iloc[-2])
            pct_ch = ((ltp - prev_close) / prev_close) * 100 if prev_close != 0 else np.nan
            work = daily.sort_values(["volume", "high"], ascending=[False, False]).reset_index(drop=True)
            candidates = []
            for _, c in work.iterrows():
                top_band, bot_band = float(c["high"]), float(c["low"])
                mid_band = (top_band + bot_band) / 2.0
                if not any(abs(mid_band - ((s["top"] + s["bottom"]) / 2)) / max(abs(ltp), 1) <= dedupe_pct for s in candidates):
                    candidates.append({
                        "top": top_band,
                        "bottom": bot_band,
                        "date": str(pd.to_datetime(c["timestamp"]).date()),
                        "volume": float(c["volume"])
                    })
            if not candidates:
                continue
            bands = pd.DataFrame(candidates)
            supports = bands[bands["top"] < ltp].sort_values("top", ascending=False).head(3).reset_index(drop=True)
            resistances = bands[bands["bottom"] > ltp].sort_values("bottom", ascending=True).head(3).reset_index(drop=True)
            row: Dict = {"Symbol": sym, "LTP": ltp, "% Change": pct_ch, "Lookback_Used": lookback_days}
            for i in range(3):
                row[f"SUP_T_{i+1}"] = supports.iloc[i]["top"] if i < len(supports) else np.nan
                row[f"SUP_B_{i+1}"] = supports.iloc[i]["bottom"] if i < len(supports) else np.nan
                row[f"SUP_D_{i+1}"] = supports.iloc[i]["date"] if i < len(supports) else ""
                row[f"SUP_V_{i+1}"] = supports.iloc[i]["volume"] if i < len(supports) else np.nan
                row[f"RES_T_{i+1}"] = resistances.iloc[i]["top"] if i < len(resistances) else np.nan
                row[f"RES_B_{i+1}"] = resistances.iloc[i]["bottom"] if i < len(resistances) else np.nan
                row[f"RES_D_{i+1}"] = resistances.iloc[i]["date"] if i < len(resistances) else ""
                row[f"RES_V_{i+1}"] = resistances.iloc[i]["volume"] if i < len(resistances) else np.nan

            row["Long_T"] = resistances.iloc[0]["top"] if len(resistances) > 0 else np.nan
            row["Long_B"] = resistances.iloc[0]["bottom"] if len(resistances) > 0 else np.nan
            row["Long_D"] = resistances.iloc[0]["date"] if len(resistances) > 0 else ""
            row["Long_V"] = resistances.iloc[0]["volume"] if len(resistances) > 0 else np.nan
            if pd.notna(row["Long_T"]) and ltp > row["Long_T"]:
                below = daily[daily["close"] <= row["Long_T"]]
                row["Long_Breach_Days"] = (today - below.iloc[-1]["timestamp"]).days if not below.empty else 999
            else:
                row["Long_Breach_Days"] = 999

            row["Short_T"] = supports.iloc[0]["top"] if len(supports) > 0 else np.nan
            row["Short_B"] = supports.iloc[0]["bottom"] if len(supports) > 0 else np.nan
            row["Short_D"] = supports.iloc[0]["date"] if len(supports) > 0 else ""
            row["Short_V"] = supports.iloc[0]["volume"] if len(supports) > 0 else np.nan
            if pd.notna(row["Short_B"]) and ltp < row["Short_B"]:
                above = daily[daily["close"] >= row["Short_B"]]
                row["Short_Breach_Days"] = (today - above.iloc[-1]["timestamp"]).days if not above.empty else 999
            else:
                row["Short_Breach_Days"] = 999

            final_row = row
            if len(supports) >= 3 and len(resistances) >= 3:
                break
        if final_row:
            rows.append(final_row)
    return pd.DataFrame(rows)


def scan_options_universe(fyers, symbols):
    rows = []
    for sym in list(set(symbols)):
        time.sleep(cfg.option_scan_sleep)
        daily = get_history(fyers, sym, "D", 60)
        if daily is None or daily.empty or len(daily) < 2:
            continue
        ltp = float(daily["close"].iloc[-1])
        max_idx = daily["volume"].idxmax() if (daily["volume"] > 0).any() else (daily["high"] - daily["low"]).idxmax()
        c = daily.loc[max_idx]
        top_band = float(c["high"])
        if ltp > top_band:
            breaches = daily[daily["close"] <= top_band]
            bd_l = (datetime.now() - breaches.iloc[-1]["timestamp"]).days if not breaches.empty else 999
        else:
            bd_l = 999
        prev_close = float(daily["close"].iloc[-2])
        pct_change = ((ltp - prev_close) / prev_close * 100) if prev_close != 0 else np.nan
        rows.append({
            "Symbol": sym,
            "LTP": ltp,
            "T_LOC": top_band,
            "B_LOC": float(c["low"]),
            "D_LOC": str(pd.to_datetime(c["timestamp"]).date()),
            "Days_L_LOC": bd_l,
            "% Change": pct_change
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

        if pd.notna(row.get("Long_T")) and row["LTP"] > row["Long_T"] and row.get("Long_Breach_Days", 999) <= cfg.breakout_breach_days:
            cand = r_dict.copy()
            cand.update({
                "Climax_Date": row.get("Long_D", ""),
                "Climax_Range (T/B)": format_tb_pair(row["LTP"], row["Long_T"], row["Long_B"]),
                "Climax_Volume": row.get("Long_V", np.nan),
                "Breach_Days": row.get("Long_Breach_Days", 999),
                "Signal_Type": "Long Breakout"
            })
            _, _, cand["Target_Options"] = get_options_data(row["Symbol"], row["LTP"], "long")
            valid_long.append(cand)

        if pd.notna(row.get("Short_B")) and row["LTP"] < row["Short_B"] and row.get("Short_Breach_Days", 999) <= cfg.breakout_breach_days:
            cand = r_dict.copy()
            cand.update({
                "Climax_Date": row.get("Short_D", ""),
                "Climax_Range (T/B)": format_tb_pair(row["LTP"], row["Short_T"], row["Short_B"]),
                "Climax_Volume": row.get("Short_V", np.nan),
                "Breach_Days": row.get("Short_Breach_Days", 999),
                "Signal_Type": "Short Breakdown"
            })
            _, _, cand["Target_Options"] = get_options_data(row["Symbol"], row["LTP"], "short")
            valid_short.append(cand)

    long_df = pd.DataFrame(valid_long)
    short_df = pd.DataFrame(valid_short)
    if not long_df.empty:
        long_df = long_df.sort_values(["Breach_Days", "% Change"], ascending=[True, False]).reset_index(drop=True)
    if not short_df.empty:
        short_df = short_df.sort_values(["Breach_Days", "% Change"], ascending=[True, True]).reset_index(drop=True)
    return pd.DataFrame(dashboard_rows), long_df, short_df


def build_option_candidate_tables(df):
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    valid_rows = []
    for _, row in df.iterrows():
        t, b, d, bd = row.get("T_LOC"), row.get("B_LOC"), row.get("D_LOC"), row.get("Days_L_LOC")
        if pd.notna(t) and row["LTP"] > t and bd <= cfg.option_breach_days:
            r_dict = row.to_dict()
            r_dict["Signal_Type"] = "Active Trend"
            r_dict["Climax_Date"] = d
            r_dict["Breach_Days"] = bd
            r_dict["Climax_Range (T/B)"] = format_tb_pair(row["LTP"], t, b)
            valid_rows.append(r_dict)
    res = pd.DataFrame(valid_rows)
    if res.empty:
        return pd.DataFrame(), pd.DataFrame()
    ce_df = res[res["Symbol"].str.endswith("CE")].sort_values(["Breach_Days", "% Change"], ascending=[True, False]).reset_index(drop=True)
    pe_df = res[res["Symbol"].str.endswith("PE")].sort_values(["Breach_Days", "% Change"], ascending=[True, False]).reset_index(drop=True)
    return ce_df, pe_df


def build_html_table(df, title, cols):
    if df.empty:
        return f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top:25px;'>{title}</h3><p style='color:#94a3b8; font-family:sans-serif;'>No candidates.</p>"
    html = f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top:25px;'>{title}</h3><table style='border-collapse:collapse; width:100%; font-family:sans-serif; font-size:13px; text-align:left; background:#0f172a;'>"
    html += "<tr style='background:#1e293b; color:#f1f5f9;'>" + "".join([f"<th style='padding:8px; border:1px solid #334155;'>{c}</th>" for c in cols]) + "</tr>"
    for _, row in df.iterrows():
        sig = str(row.get("Signal_Type", ""))
        row_style = "background:#0f172a; color:#e2e8f0;"
        if "Long" in sig:
            row_style = "background:#052e16; color:#dcfce7;"
        elif "Short" in sig:
            row_style = "background:#450a0a; color:#fee2e2;"
        html += f"<tr style='{row_style}'>"
        for c in cols:
            style = "padding:8px; border:1px solid #334155;"
            if c == "% Change":
                try:
                    style += " color:#4ade80; font-weight:bold;" if float(row.get(c)) > 0 else " color:#f87171; font-weight:bold;"
                except Exception:
                    pass
            html += f"<td style='{style}'>{format_value(c, row.get(c))}</td>"
        html += "</tr>"
    return html + "</table>"


def save_outputs(summary_df):
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    csv_file = f"scan_summary_{ts}.csv"
    summary_df.to_csv(csv_file, index=False)
    return csv_file


def send_email(dashboard_df, long_df, short_df, ce_df, pe_df, csv_file):
    msg = MIMEMultipart()
    msg["From"] = cfg.sender_email
    msg["To"] = cfg.recipient_email
    msg["Subject"] = f"F&O Market Dashboard {datetime.now().strftime('%Y-%m-%d')}"
    html = (
        f"<html><body style='background:#030712; color:#fff; padding:20px;'>"
        f"{build_html_table(dashboard_df, 'Market Dashboard', EMAIL_DISPLAY_COLS)}"
        f"{build_html_table(long_df, 'Long Strategy Matrix', EMAIL_CAND_COLS)}"
        f"{build_html_table(short_df, 'Short Strategy Matrix', EMAIL_CAND_COLS)}"
        f"{build_html_table(ce_df, 'Call Options (CE) Climax Verification', EMAIL_OPT_COLS)}"
        f"{build_html_table(pe_df, 'Put Options (PE) Climax Verification', EMAIL_OPT_COLS)}"
        f"</body></html>"
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

    ce_df, pe_df = pd.DataFrame(), pd.DataFrame()
    if all_opt:
        opt_df = scan_options_universe(fyers, all_opt)
        ce_df, pe_df = build_option_candidate_tables(opt_df)

    csv_f = save_outputs(spot_df)
    send_email(dashboard_df, long_df, short_df, ce_df, pe_df, csv_f)


if __name__ == "__main__":
    main()

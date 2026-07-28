#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL_two_tables.py
Simplified FYERS F&O dashboard email with only two tables:
1) Long Strategy Matrix  -> Long Breakout, Support Sweep
2) Short Strategy Matrix -> Short Breakdown, Resistance Sweep
"""

import os
import sys
import time
import logging
import warnings
import requests
from io import StringIO
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

FYERS_FO_CSV_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
FYERS_TIMEOUT = 20
DAILY_HISTORY_CHUNK_DAYS = 360
MAX_DAILY_LOOKBACK = 730
SCAN_SLEEP_SECONDS = 0.03
ACTIVE_WINDOW_DAYS = 10
FALLBACK_SPOTS = ["NSE:RELIANCE-EQ", "NSE:HDFCBANK-EQ", "NSE:INFY-EQ"]
INDEX_SYMBOLS = ["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX", "BSE:SENSEX-INDEX"]
IGNORED_UNDERLYINGS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50", "SENSEX"}

EMAIL_CAND_COLS = [
    "Symbol", "% Change",
    "Support-3", "Support-2", "Support-1", "LTP",
    "Resistance-1", "Resistance-2", "Resistance-3",
    "Climax_Date", "Climax_Range (T/B)",
    "Climax_Volume", "Breach_Days", "Signal_Type",
]

logger = logging.getLogger("fno_dashboard")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(handler)
warnings.filterwarnings("ignore")

OPTION_PATTERN = re.compile(r"^(?P<exchange>NSE|BSE):(?P<body>[A-Z0-9\-\&]+?)(?P<strike>\d+)(?P<otype>CE|PE)$")
DATE_TOKEN_PATTERN = re.compile(r"^(?P<under>[A-Z0-9\-\&]+?)(?P<token>\d{2}[A-Z0-9]{3,})(?P<strike>\d+)(?P<otype>CE|PE)$")
MONTHLY_TOKEN_PATTERN = re.compile(r"^(?P<yy>\d{2})(?P<mon>[A-Z]{3})$")
WEEKLY_TOKEN_PATTERN = re.compile(r"^(?P<yy>\d{2})(?P<mcode>[1-9OND])(?P<dd>\d{2})$")
MONTH_MAP = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
M_CODE_MAP = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "O": 10, "N": 11, "D": 12}


class Config:
    def __init__(self):
        self.client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        self.access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
        self.smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.environ.get("SMTP_PORT", "587"))
        self.sender_email = os.environ.get("SENDER_EMAIL", "")
        self.sender_password = os.environ.get("SENDER_PASSWORD", "")
        self.recipient_email = os.environ.get("RECIPIENT_EMAIL", "")
        self.enable_email = os.environ.get("ENABLE_EMAIL", "1").strip().lower() in {"1", "true", "yes", "y"}
        self.output_dir = os.environ.get("OUTPUT_DIR", "output")
        self.index_symbols = []

    def validate(self) -> bool:
        ok = True
        if not self.client_id:
            logger.error("Missing CLIENT_ID / CLIENTID env variable")
            ok = False
        if not self.access_token:
            logger.error("Missing ACCESS_TOKEN / ACCESSTOKEN env variable")
            ok = False
        return ok


cfg = Config()


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def read_symbol_master() -> pd.DataFrame:
    logger.info("Downloading FYERS symbol master: %s", FYERS_FO_CSV_URL)
    resp = requests.get(FYERS_FO_CSV_URL, timeout=FYERS_TIMEOUT)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), header=None, dtype=str, low_memory=False)
    df = df.fillna("")
    df.columns = [f"c{i}" for i in range(df.shape[1])]
    return df


def build_symbol_master_views(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = raw_df.copy()
    symbol_like_cols = []
    for c in df.columns:
        s = df[c].astype(str)
        ratio = s.str.contains(r"^(NSE|BSE):", regex=True, na=False).mean()
        if ratio > 0.05:
            symbol_like_cols.append(c)
    if not symbol_like_cols:
        raise RuntimeError("Could not identify symbol columns in NSE_FO.csv")

    long_df = df.melt(value_vars=symbol_like_cols, value_name="symbol").drop(columns=["variable"])
    long_df["symbol"] = long_df["symbol"].astype(str).str.strip()
    long_df = long_df[long_df["symbol"].str.match(r"^(NSE|BSE):", na=False)].drop_duplicates().reset_index(drop=True)

    deriv = long_df[long_df["symbol"].str.contains(r"(CE|PE|FUT)$", regex=True, na=False)].copy()
    eq = long_df[long_df["symbol"].str.endswith("-EQ")].copy()
    idx = long_df[long_df["symbol"].str.endswith("-INDEX")].copy()
    return eq, idx, deriv


def get_dynamic_fno_universe(eq_symbols: pd.DataFrame) -> List[str]:
    if eq_symbols.empty:
        return FALLBACK_SPOTS[:]
    eq = eq_symbols["symbol"].astype(str).str.strip().unique().tolist()
    filtered = []
    for s in eq:
        base = s.split(":", 1)[1].replace("-EQ", "")
        if base not in IGNORED_UNDERLYINGS:
            filtered.append(s)
    return sorted(set(filtered)) or FALLBACK_SPOTS[:]


def format_tb_pair(ltp, top, bottom):
    if pd.isna(top) or pd.isna(bottom):
        return "-"
    upper = max(float(top), float(bottom))
    lower = min(float(top), float(bottom))
    t_str, b_str, ltp_val = f"{upper:.2f}", f"{lower:.2f}", float(ltp)
    if ltp_val > upper:
        t_str = f"<span style='color: #4ade80; font-weight: bold;'>{t_str}</span>"
    if ltp_val < lower:
        b_str = f"<span style='color: #f87171; font-weight: bold;'>{b_str}</span>"
    return f"{t_str} <span style='color: #64748b;'>/</span> {b_str}"


def format_value(col, val):
    if pd.isna(val) or val in [float("inf"), float("-inf")]:
        return ""
    if col == "% Change":
        try:
            return f"{float(val):.2f}%"
        except Exception:
            return ""
    if col == "LTP":
        try:
            return f"{float(val):.2f}"
        except Exception:
            return ""
    if "(T/B)" in col or "Support" in col or "Resistance" in col:
        return str(val)
    if col in ["Signal_Type", "Climax_Date", "Symbol"]:
        return str(val)
    if col == "Breach_Days":
        try:
            return str(int(val))
        except Exception:
            return ""
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.4f}"
    return str(val)


def fetch_history_chunk(fyers, symbol: str, resolution: str, start_dt: date, end_dt: date) -> Optional[pd.DataFrame]:
    payload = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start_dt.strftime("%Y-%m-%d"),
        "range_to": end_dt.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }
    try:
        res_data = fyers.history(data=payload)
        if not isinstance(res_data, dict):
            logger.warning("%s: non-dict response: %r", symbol, res_data)
            return None
        if "candles" not in res_data:
            logger.warning("%s: history error response: %s", symbol, res_data)
            return None
        candles = res_data.get("candles") or []
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
        return df
    except Exception as e:
        logger.warning("%s: history request failed: %s", symbol, e)
        return None


def get_history(fyers, symbol: str, resolution: str = "1D", days: int = 365) -> Optional[pd.DataFrame]:
    end_dt = datetime.now().date()
    start_dt = end_dt - timedelta(days=days)

    if resolution == "1D" and days > DAILY_HISTORY_CHUNK_DAYS:
        dfs = []
        cur = start_dt
        while cur <= end_dt:
            chunk_end = min(cur + timedelta(days=DAILY_HISTORY_CHUNK_DAYS - 1), end_dt)
            df = fetch_history_chunk(fyers, symbol, resolution, cur, chunk_end)
            if df is not None and not df.empty:
                dfs.append(df)
            cur = chunk_end + timedelta(days=1)
            time.sleep(SCAN_SLEEP_SECONDS)
        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return fetch_history_chunk(fyers, symbol, resolution, start_dt, end_dt)


def build_flat_ladder(side_df: pd.DataFrame) -> List[Tuple[float, float]]:
    if side_df is None or side_df.empty:
        return []
    values = []
    for _, r in side_df.iterrows():
        top = safe_float(r.get("top"))
        bottom = safe_float(r.get("bottom"))
        if pd.notna(top):
            values.append(float(top))
        if pd.notna(bottom):
            values.append(float(bottom))
    values = sorted(values)
    bands = []
    for i in range(0, min(len(values), 6), 2):
        if i + 1 < len(values):
            a, b = values[i], values[i + 1]
            bands.append((max(a, b), min(a, b)))
    return bands[:3]


def scan_fno_universe(fyers):
    base_lookback = 90
    max_lookback = MAX_DAILY_LOOKBACK
    step = 30
    rows = []

    logger.info("Scanning %d charts...", len(cfg.index_symbols))

    for idx, sym in enumerate(cfg.index_symbols, start=1):
        if idx % 25 == 0:
            logger.info("Processed %d/%d symbols", idx, len(cfg.index_symbols))

        time.sleep(SCAN_SLEEP_SECONDS)
        final_row = None
        full_history = get_history(fyers, sym, "1D", max_lookback)
        if full_history is None or full_history.empty or len(full_history) < 5:
            continue

        now_ts = pd.Timestamp.now()

        for lookback_days in range(base_lookback, max_lookback + 1, step):
            cutoff_date = now_ts - pd.Timedelta(days=lookback_days)
            daily = full_history[full_history["timestamp"] >= cutoff_date].copy()
            if daily.empty or len(daily) < 5:
                continue

            ltp = float(daily["close"].iloc[-1])
            prev_close = float(daily["close"].iloc[-2])
            pct_ch = ((ltp - prev_close) / prev_close) * 100 if prev_close else np.nan

            work = daily.sort_values(["volume", "high"], ascending=[False, False]).reset_index(drop=True)
            candidates = [
                {
                    "top": float(c["high"]),
                    "bottom": float(c["low"]),
                    "date": str(pd.to_datetime(c["timestamp"]).date()),
                    "volume": float(c["volume"]),
                }
                for _, c in work.iterrows()
            ]
            if not candidates:
                continue

            bands = pd.DataFrame(candidates)
            raw_supports = bands[bands["top"] < ltp].copy().sort_values("top", ascending=False).head(3).reset_index(drop=True)
            raw_resistances = bands[bands["bottom"] > ltp].copy().sort_values("bottom", ascending=True).head(3).reset_index(drop=True)

            support_bands = build_flat_ladder(raw_supports)
            resistance_bands = build_flat_ladder(raw_resistances)

            row: Dict = {"Symbol": sym, "LTP": ltp, "% Change": pct_ch, "Lookback_Used": lookback_days}

            for i in range(3):
                if i < len(support_bands):
                    upper, lower = support_bands[i]
                    src = raw_supports.iloc[min(i, len(raw_supports)-1)] if len(raw_supports) > 0 else None
                    row[f"SUP_T_{i+1}"] = float(upper)
                    row[f"SUP_B_{i+1}"] = float(lower)
                    row[f"SUP_D_{i+1}"] = str(src["date"]) if src is not None else ""
                    row[f"SUP_V_{i+1}"] = float(src["volume"]) if src is not None else np.nan
                else:
                    row[f"SUP_T_{i+1}"] = np.nan
                    row[f"SUP_B_{i+1}"] = np.nan
                    row[f"SUP_D_{i+1}"] = ""
                    row[f"SUP_V_{i+1}"] = np.nan

                if i < len(resistance_bands):
                    upper, lower = resistance_bands[i]
                    src = raw_resistances.iloc[min(i, len(raw_resistances)-1)] if len(raw_resistances) > 0 else None
                    row[f"RES_T_{i+1}"] = float(upper)
                    row[f"RES_B_{i+1}"] = float(lower)
                    row[f"RES_D_{i+1}"] = str(src["date"]) if src is not None else ""
                    row[f"RES_V_{i+1}"] = float(src["volume"]) if src is not None else np.nan
                else:
                    row[f"RES_T_{i+1}"] = np.nan
                    row[f"RES_B_{i+1}"] = np.nan
                    row[f"RES_D_{i+1}"] = ""
                    row[f"RES_V_{i+1}"] = np.nan

            if len(support_bands) > 0:
                sup_t1 = float(row.get("SUP_T_1"))
                sup_b1 = float(row.get("SUP_B_1"))
                was_below = daily[daily["close"] <= sup_t1]
                swept_below = daily[daily["low"] < sup_b1]
                row["Long_Breach_Days"] = int((now_ts.normalize() - was_below.iloc[-1]["timestamp"].normalize()).days) if not was_below.empty else 999
                row["Sup_Sweep_Days"] = int((now_ts.normalize() - swept_below.iloc[-1]["timestamp"].normalize()).days) if not swept_below.empty else 999
            else:
                row["Long_Breach_Days"] = 999
                row["Sup_Sweep_Days"] = 999

            if len(resistance_bands) > 0:
                res_t1 = float(row.get("RES_T_1"))
                res_b1 = float(row.get("RES_B_1"))
                was_above = daily[daily["close"] >= res_b1]
                swept_above = daily[daily["high"] > res_t1]
                row["Short_Breach_Days"] = int((now_ts.normalize() - was_above.iloc[-1]["timestamp"].normalize()).days) if not was_above.empty else 999
                row["Res_Sweep_Days"] = int((now_ts.normalize() - swept_above.iloc[-1]["timestamp"].normalize()).days) if not swept_above.empty else 999
            else:
                row["Short_Breach_Days"] = 999
                row["Res_Sweep_Days"] = 999

            final_row = row
            if len(support_bands) >= 3 and len(resistance_bands) >= 3:
                break

        if final_row is not None:
            rows.append(final_row)

    return pd.DataFrame(rows)


def get_days_ago(date_str):
    if not date_str or pd.isna(date_str):
        return 999
    try:
        return (datetime.now().date() - pd.to_datetime(date_str).date()).days
    except Exception:
        return 999


def build_strategy_tables(df):
    valid_long, valid_short = [], []
    for _, row in df.iterrows():
        r_dict = row.to_dict()
        for i in range(1, 4):
            r_dict[f"Support-{i}"] = format_tb_pair(row["LTP"], row.get(f"SUP_T_{i}"), row.get(f"SUP_B_{i}")) if pd.notna(row.get(f"SUP_T_{i}")) else "-"
            r_dict[f"Resistance-{i}"] = format_tb_pair(row["LTP"], row.get(f"RES_T_{i}"), row.get(f"RES_B_{i}")) if pd.notna(row.get(f"RES_T_{i}")) else "-"

        sup_t1 = row.get("SUP_T_1")
        sup_b1 = row.get("SUP_B_1")
        long_breach = row.get("Long_Breach_Days", 999)
        sup_sweep = row.get("Sup_Sweep_Days", 999)
        long_climax_age = get_days_ago(row.get("SUP_D_1"))

        is_breakout = pd.notna(sup_t1) and (long_breach <= ACTIVE_WINDOW_DAYS or long_climax_age <= ACTIVE_WINDOW_DAYS)
        is_sup_sweep = pd.notna(sup_b1) and sup_sweep <= ACTIVE_WINDOW_DAYS

        if is_breakout or is_sup_sweep:
            sig_type = "Long Breakout" if is_breakout else "Support Sweep"
            valid_long.append({
                "Symbol": row["Symbol"],
                "% Change": row["% Change"],
                "Support-3": r_dict.get("Support-3", "-"),
                "Support-2": r_dict.get("Support-2", "-"),
                "Support-1": r_dict.get("Support-1", "-"),
                "LTP": row["LTP"],
                "Resistance-1": r_dict.get("Resistance-1", "-"),
                "Resistance-2": r_dict.get("Resistance-2", "-"),
                "Resistance-3": r_dict.get("Resistance-3", "-"),
                "Climax_Date": row.get("SUP_D_1", ""),
                "Climax_Range (T/B)": format_tb_pair(row["LTP"], sup_t1, sup_b1),
                "Climax_Volume": f"{int(row['SUP_V_1']):,}" if pd.notna(row.get("SUP_V_1")) else "",
                "Breach_Days": long_breach if is_breakout else sup_sweep,
                "Signal_Type": sig_type,
            })

        res_t1 = row.get("RES_T_1")
        res_b1 = row.get("RES_B_1")
        short_breach = row.get("Short_Breach_Days", 999)
        res_sweep = row.get("Res_Sweep_Days", 999)
        short_climax_age = get_days_ago(row.get("RES_D_1"))

        is_breakdown = pd.notna(res_b1) and (short_breach <= ACTIVE_WINDOW_DAYS or short_climax_age <= ACTIVE_WINDOW_DAYS)
        is_res_sweep = pd.notna(res_t1) and res_sweep <= ACTIVE_WINDOW_DAYS

        if is_breakdown or is_res_sweep:
            sig_type = "Short Breakdown" if is_breakdown else "Resistance Sweep"
            valid_short.append({
                "Symbol": row["Symbol"],
                "% Change": row["% Change"],
                "Support-3": r_dict.get("Support-3", "-"),
                "Support-2": r_dict.get("Support-2", "-"),
                "Support-1": r_dict.get("Support-1", "-"),
                "LTP": row["LTP"],
                "Resistance-1": r_dict.get("Resistance-1", "-"),
                "Resistance-2": r_dict.get("Resistance-2", "-"),
                "Resistance-3": r_dict.get("Resistance-3", "-"),
                "Climax_Date": row.get("RES_D_1", ""),
                "Climax_Range (T/B)": format_tb_pair(row["LTP"], res_t1, res_b1),
                "Climax_Volume": f"{int(row['RES_V_1']):,}" if pd.notna(row.get("RES_V_1")) else "",
                "Breach_Days": short_breach if is_breakdown else res_sweep,
                "Signal_Type": sig_type,
            })

    return pd.DataFrame(valid_long), pd.DataFrame(valid_short)


def build_html_table(df, title, cols):
    if df.empty:
        return f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top:25px;'>{title}</h3><p style='color:#94a3b8; font-family:sans-serif;'>No candidates within {ACTIVE_WINDOW_DAYS} days.</p>"
    table_html = f"<h3 style='color:#fbbf24; font-family:sans-serif; margin-top:25px;'>{title}</h3><table style='border-collapse:collapse; width:100%; font-family:sans-serif; font-size:13px; text-align:left; background-color:#0f172a;'>"
    table_html += "<tr style='background-color:#1e293b; color:#f1f5f9;'>" + "".join([f"<th style='padding:10px; border:1px solid #334155;'>{c}</th>" for c in cols]) + "</tr>"
    for i, (_, row) in enumerate(df.iterrows()):
        bg_row = "#0f172a" if i % 2 == 0 else "#1e293b"
        sig = str(row.get("Signal_Type", ""))
        row_style = f"background-color:{bg_row}; color:#e2e8f0;"
        if "Sweep" in sig:
            row_style = "background-color:#92400e; color:#fef3c7;"
        table_html += f"<tr style='{row_style}'>"
        for c in cols:
            val = row.get(c)
            style = "padding:8px; border:1px solid #334155;"
            if c == "% Change":
                try:
                    fval = float(str(val).replace('%', ''))
                    style += " color:#4ade80; font-weight:bold;" if fval > 0 else " color:#f87171; font-weight:bold;"
                except Exception:
                    pass
            table_html += f"<td style='{style}'>{format_value(c, val)}</td>"
        table_html += "</tr>"
    return table_html + "</table>"


def save_outputs(spot_df, long_df, short_df):
    os.makedirs(cfg.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    paths = {
        "summary_csv": os.path.join(cfg.output_dir, f"scan_summary_{ts}.csv"),
        "long_csv": os.path.join(cfg.output_dir, f"long_strategy_{ts}.csv"),
        "short_csv": os.path.join(cfg.output_dir, f"short_strategy_{ts}.csv"),
    }
    spot_df.to_csv(paths["summary_csv"], index=False)
    long_df.to_csv(paths["long_csv"], index=False)
    short_df.to_csv(paths["short_csv"], index=False)
    return paths


def send_email(long_df, short_df, attachment_file):
    if not cfg.enable_email:
        logger.info("Email disabled via ENABLE_EMAIL")
        return
    if not cfg.sender_email or not cfg.sender_password or not cfg.recipient_email:
        logger.warning("Email skipped due to missing SMTP credentials/recipient")
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = cfg.sender_email
        msg["To"] = cfg.recipient_email
        msg["Subject"] = f"F&O Strategy Matrix - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        html = (
            "<body style='background-color:#030712; padding:20px; font-family:sans-serif;'>"
            + build_html_table(long_df, "Long Strategy Matrix", EMAIL_CAND_COLS)
            + build_html_table(short_df, "Short Strategy Matrix", EMAIL_CAND_COLS)
            + "</body>"
        )
        msg.attach(MIMEText(html, "html"))

        with open(attachment_file, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_file)}")
            msg.attach(part)

        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(cfg.sender_email, cfg.sender_password)
            s.sendmail(cfg.sender_email, [cfg.recipient_email], msg.as_string())
        logger.info("Email sent successfully to %s", cfg.recipient_email)
    except smtplib.SMTPAuthenticationError:
        logger.error("Email failed: SMTP authentication error. For Gmail, use an App Password instead of your normal password.")
    except Exception as e:
        logger.exception("Email failed: %s", e)


def init_fyers():
    try:
        return fyersModel.FyersModel(client_id=cfg.client_id, is_async=False, token=cfg.access_token, log_path="")
    except Exception as e:
        logger.exception("FYERS init failed: %s", e)
        return None


def main():
    start_time = time.time()
    if not cfg.validate():
        return

    try:
        raw_master = read_symbol_master()
        eq_df, idx_df, deriv_df = build_symbol_master_views(raw_master)
        spot_universe = get_dynamic_fno_universe(eq_df)
        cfg.index_symbols = INDEX_SYMBOLS + spot_universe
        logger.info("Resolved %d spot symbols from symbol master", len(spot_universe))
    except Exception as e:
        logger.exception("Symbol master processing failed: %s", e)
        cfg.index_symbols = INDEX_SYMBOLS + FALLBACK_SPOTS

    fyers = init_fyers()
    if not fyers:
        return

    spot_df = scan_fno_universe(fyers)
    if spot_df.empty:
        logger.warning("No spot data generated.")
        return

    long_df, short_df = build_strategy_tables(spot_df)
    paths = save_outputs(spot_df, long_df, short_df)
    send_email(long_df, short_df, paths["summary_csv"])

    elapsed = time.time() - start_time
    logger.info("Done in %.2fs | spot=%d long=%d short=%d", elapsed, len(spot_df), len(long_df), len(short_df))
    logger.info("Saved outputs to %s", cfg.output_dir)


if __name__ == "__main__":
    main()

"""
FO_FNO_FYERS_VOL_REL_EMAIL.py

Uses Fyers historical data to compute Daily Volatility Expansion and Cumulative Intraday Volume Expansion.
- Volume is calculated CUMULATIVELY (from 9:15 AM to current time) and compared to 
  the average cumulative volume for the exact same time window over the last 10/20 days.
"""

import os
import sys
import math
import logging
import configparser
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


# =========================
# Logging
# =========================

class UTF8Formatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        msg = msg.replace("âŒ", "[ERROR]").replace("âœ…", "[OK]")
        msg = msg.replace("ðŸŸ¢", "[GREEN]").replace("ðŸŸ¡", "[YELLOW]").replace("ðŸ”´", "[RED]")
        msg = msg.replace("âš ï¸", "[WARN]").replace("ðŸ“Š", "[DATA]").replace("ðŸŽ¯", "[TARGET]")
        record.msg = msg
        return super().format(record)

log_format = "%(asctime)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(UTF8Formatter(log_format))
logger.addHandler(console_handler)

file_handler = logging.FileHandler("fo_fyers_vol_rel_email.log", encoding="utf-8")
file_handler.setFormatter(UTF8Formatter(log_format))
logger.addHandler(file_handler)

logger.info("[OK] FO Fyers Volatility + Cumulative RVOL Scanner initialized")


# =========================
# Config & Fyers init 
# =========================

config = configparser.ConfigParser()
config.read("config.ini")

def get_cfg(section, key, env_name=None, default=None, is_int=False):
    if env_name:
        val = os.getenv(env_name)
        if val is not None and val.strip() != "":
            return int(val) if is_int else val
    if section and key and config.has_option(section, key):
        val = config.get(section, key)
        return int(val) if is_int else val
    return default

try:
    client_id = get_cfg("fyers_credentials", "client_id", env_name="CLIENT_ID")
    token = (
        get_cfg("fyers_credentials", "access_token", env_name="ACCESS_TOKEN")
        or get_cfg("fyers_credentials", "token", env_name="TOKEN")
    )
    if not client_id or not token:
        raise ValueError("Missing CLIENT_ID or ACCESS_TOKEN")
    fyers = fyersModel.FyersModel(client_id=client_id, token=token)
    logger.info("[OK] Fyers API connected successfully")
except Exception as e:
    logger.error(f"[ERROR] Fyers init failed: {e}")
    fyers = None


# =========================
# Load F&O symbols 
# =========================

def load_fno_symbols_from_sectors(root_dir: str = "sectors") -> List[str]:
    symbols = set()
    if not os.path.isdir(root_dir):
        logger.warning(f"[FNO] Sectors folder '{root_dir}' not found; returning empty list.")
        return []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                df = pd.read_csv(fpath)
                col = None
                for c in df.columns:
                    if c.lower() in ("symbol", "symbols", "ticker"):
                        col = c
                        break
                if col is None:
                    continue
                for s in df[col].dropna().astype(str):
                    s = s.strip()
                    if s:
                        symbols.add(s)
            except Exception as e:
                logger.warning(f"[FNO] Error reading {fpath}: {e}")

    symbols_list = sorted(symbols)
    logger.info(f"[FNO] Loaded {len(symbols_list)} unique F&O symbols.")
    return symbols_list

def format_fyers_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if s.startswith("NSE:") and ("-EQ" in s or "-INDEX" in s):
        return s
    s = s.replace("NSE:", "").replace("-EQ", "")
    if s in ["NIFTY", "NIFTY50"]: return "NSE:NIFTY50-INDEX"
    if s == "BANKNIFTY": return "NSE:NIFTYBANK-INDEX"
    if s == "FINNIFTY": return "NSE:FINNIFTY-INDEX"
    if s == "MIDCPNIFTY": return "NSE:MIDCPNIFTY-INDEX"
    return f"NSE:{s}-EQ"


# =========================
# Fyers history helpers
# =========================

def get_fyers_history(symbol: str, resolution: str, days_back: int) -> Optional[pd.DataFrame]:
    if fyers is None or days_back <= 0:
        return None

    per_call_limit = 366 if resolution == "D" else 100
    all_chunks = []
    end_date = datetime.now()
    remaining_days = days_back

    while remaining_days > 0:
        chunk_days = min(per_call_limit, remaining_days)
        start_date = end_date - timedelta(days=chunk_days - 1)

        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": start_date.strftime("%Y-%m-%d"),
            "range_to": end_date.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }

        try:
            resp = fyers.history(data=data)
            if resp.get("s") != "ok" or "candles" not in resp:
                break
            candles = resp["candles"]
            if not candles:
                break
            df_chunk = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df_chunk["timestamp"] = pd.to_datetime(df_chunk["timestamp"], unit="s")
            all_chunks.append(df_chunk)
        except Exception:
            break

        remaining_days -= chunk_days
        end_date = start_date - timedelta(days=1)

    if not all_chunks:
        return None

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# =========================
# Volatility calculations (Daily)
# =========================

def compute_annualized_volatility_from_daily(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or len(df) < 10:
        return None
    closes = df["close"].astype(float)
    returns = np.log(closes / closes.shift(1)).dropna()
    if len(returns) < 5:
        return None
    vol = returns.std() * math.sqrt(252) * 100.0
    return float(vol)

def compute_volatility_pair(df_daily: pd.DataFrame, short_days: int = 20, long_days: int = 252) -> Dict[str, Optional[float]]:
    if df_daily is None or df_daily.empty:
        return {"DailyCurrVolPct": None, "DailyAvgVolPct": None, "VolExpansion": None}

    df_sorted = df_daily.sort_values("timestamp").reset_index(drop=True)

    curr_vol_ann = compute_annualized_volatility_from_daily(df_sorted.tail(short_days))
    avg_vol_ann = compute_annualized_volatility_from_daily(df_sorted.tail(long_days))

    sqrt_252 = math.sqrt(252)
    daily_curr = (curr_vol_ann / sqrt_252) if curr_vol_ann else None
    daily_avg = (avg_vol_ann / sqrt_252) if avg_vol_ann else None

    vol_expansion = (daily_curr / daily_avg) if (daily_curr and daily_avg and daily_avg > 0) else None

    return {
        "DailyCurrVolPct": daily_curr, 
        "DailyAvgVolPct": daily_avg,
        "VolExpansion": vol_expansion
    }


# =========================
# CUMULATIVE Relative volume (9:15 to Current Time)
# =========================

def compute_cumulative_relative_volume(
    df: pd.DataFrame, lookback_short: int = 10, lookback_long: int = 20
) -> Dict[str, Optional[float]]:
    if df is None or df.empty or len(df) < 10:
        return {"RelVolume10": None, "RelVolume20": None, "CurrentVolume": None, "LTP": None}

    s = df.copy()

    # Ensure datetime format and India timezone
    s["datetime"] = pd.to_datetime(s["timestamp"])

    # Create date and time columns
    s["date"] = s["datetime"].dt.date
    s["time"] = s["datetime"].dt.time

    # Identify the latest timestamp we have in this dataframe
    latest = s.iloc[-1]
    latest_date = latest["date"]
    latest_time = latest["time"]
    ltp = float(latest["close"])

    # 1. Sum up all volume for TODAY up to the latest time
    today_mask = (s["date"] == latest_date) & (s["time"] <= latest_time)
    current_cum_vol = float(s[today_mask]["volume"].sum())

    def _avg_cum_vol(last_n_days: int) -> Optional[float]:
        # Get start date for historical lookback
        start_date = latest_date - timedelta(days=last_n_days * 2) # buffer for weekends

        # 2. Find all historical days, but ONLY include candles up to today's current time
        mask_prior = (
            (s["date"] < latest_date) & 
            (s["date"] >= start_date) & 
            (s["time"] <= latest_time)
        )
        prior_data = s[mask_prior]

        if prior_data.empty:
            return None

        # 3. Sum the volume for each historical day (up to the cutoff time)
        daily_sums = prior_data.groupby("date")["volume"].sum()

        # We only want exactly the last N trading days
        daily_sums = daily_sums.tail(last_n_days)

        if daily_sums.empty:
            return None

        # 4. Return the average cumulative volume of those past N days
        return float(daily_sums.mean())

    avg10_cum = _avg_cum_vol(lookback_short)
    avg20_cum = _avg_cum_vol(lookback_long)

    rel10 = current_cum_vol / avg10_cum if avg10_cum and avg10_cum > 0 else None
    rel20 = current_cum_vol / avg20_cum if avg20_cum and avg20_cum > 0 else None

    return {
        "RelVolume10": rel10,
        "RelVolume20": rel20,
        "CurrentVolume": current_cum_vol,
        "LTP": ltp,
    }


# =========================
# Core scan
# =========================

def scan_fno_universe() -> pd.DataFrame:
    symbols = load_fno_symbols_from_sectors("sectors")
    if not symbols:
        logger.error("[CORE] No F&O symbols found. Exiting.")
        return pd.DataFrame()

    rows = []
    total = len(symbols)
    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"[CORE] ({idx}/{total}) Processing {sym}")

        fyers_sym = format_fyers_symbol(sym)

        daily_df = get_fyers_history(fyers_sym, resolution="D", days_back=365)
        vol_info = compute_volatility_pair(daily_df, short_days=20, long_days=252)

        # Get previous close for % Change
        if daily_df is not None and len(daily_df) >= 2:
            prev_close = float(daily_df["close"].iloc[-2])
        else:
            prev_close = None

        intra_df = get_fyers_history(fyers_sym, resolution="5", days_back=40) # Need enough days for 20-day average

        if intra_df is not None:
            rvol_info = compute_cumulative_relative_volume(intra_df, lookback_short=10, lookback_long=20)
        else:
            rvol_info = {"RelVolume10": None, "RelVolume20": None, "CurrentVolume": None, "LTP": float(daily_df["close"].iloc[-1]) if daily_df is not None else None}

        # Calculate % Change
        if prev_close and rvol_info["LTP"]:
            pct_change = ((rvol_info["LTP"] - prev_close) / prev_close) * 100
        else:
            pct_change = 0.0

        rows.append({
            "Symbol": sym,
            "LTP": rvol_info["LTP"],
            "% Change": pct_change,
            "Current Daily Volatility": vol_info["DailyCurrVolPct"],
            "Avg Daily Volatility": vol_info["DailyAvgVolPct"],
            "Daily Volatility Expansion": vol_info["VolExpansion"],
            "Current Volume": rvol_info["CurrentVolume"],
            "10 Day Relative Volume": rvol_info["RelVolume10"],
            "20 Day Relative Volume": rvol_info["RelVolume20"],
        })

    df = pd.DataFrame(rows)
    if df.empty: return df

    # Daily Volume Expansion is the max of the 10-Day and 20-Day Cumulative RVOL
    df["Daily Volume Expansion"] = df[["10 Day Relative Volume", "20 Day Relative Volume"]].max(axis=1)

    return df


# =========================
# Email helpers
# =========================

def df_to_html_table(df: pd.DataFrame, max_rows: int = 150) -> str:
    if df is None or df.empty: return "<p>No data available.</p>"

    df_disp = df.copy().head(max_rows)

    for col in df_disp.columns:
        if col == "Symbol":
            continue
        elif col == "Current Volume":
            df_disp[col] = df_disp[col].map(lambda x: f"{x:,.0f}" if isinstance(x, (float, int)) and not pd.isna(x) else "")
        elif col == "% Change":
            df_disp[col] = df_disp[col].map(lambda x: f"{x:+.2f}%" if isinstance(x, (float, int)) and not pd.isna(x) else "")
        else:
            df_disp[col] = df_disp[col].map(lambda x: f"{x:.2f}" if isinstance(x, (float, int)) and not pd.isna(x) else "")

    return df_disp.to_html(index=False, border=1, justify="center", escape=False)

def send_email_with_tables(df_vol: pd.DataFrame, df_volum: pd.DataFrame, csv_filename: str) -> bool:
    sender_email = get_cfg("email", "sender_email", env_name="SENDER_EMAIL")
    sender_password = get_cfg("email", "sender_password", env_name="SENDER_PASSWORD")
    recipient_email = get_cfg("email", "recipient_email", env_name="RECIPIENT_EMAIL")

    smtp_server = get_cfg("email", "smtp_server", env_name="SMTP_SERVER", default="smtp.gmail.com")
    smtp_port = get_cfg("email", "smtp_port", env_name="SMTP_PORT", default="587", is_int=True)

    if not all([sender_email, sender_password, recipient_email]):
        logger.warning("[EMAIL] Missing email credentials.")
        return False

    now = datetime.now()
    subject = f"F&O Expansion Scan (Cumulative RVOL) - {now.strftime('%Y-%m-%d %H:%M IST')}"

    table_vol_html = df_to_html_table(df_vol, max_rows=150)
    table_volum_html = df_to_html_table(df_volum, max_rows=150)

    body_html = f"""
    <html>
      <head>
        <style>
          table {{ border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px; }}
          th, td {{ border: 1px solid #dddddd; text-align: right; padding: 6px; }}
          th {{ background-color: #f2f2f2; text-align: center; }}
          td:first-child {{ text-align: left; font-weight: bold; }}
        </style>
      </head>
      <body>
        <p>Hello,</p>
        <p>Below is the latest F&O scan. Volume is calculated <b>cumulatively from market open to the current time</b>.</p>

        <h3>1) Stocks Sorted by Daily Volatility Expansion (Desc)</h3>
        <p><i>Which stocks are swinging wildly compared to their historical daily average?</i></p>
        {table_vol_html}

        <br><br>

        <h3>2) Stocks Sorted by Daily Volume Expansion (Desc)</h3>
        <p><i>Which stocks are experiencing the largest volume breakout for this specific time of day?</i></p>
        {table_volum_html}

        <p>Attached CSV: {os.path.basename(csv_filename)}</p>
        <p>Generated at: {now.strftime('%Y-%m-%d %H:%M:%S')}</p>
      </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body_html, "html"))

    try:
        with open(csv_filename, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(csv_filename)}"')
        msg.attach(part)
    except Exception as e:
        logger.warning(f"Failed to attach CSV: {e}")

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logger.info("[EMAIL] Email sent successfully.")
        return True
    except Exception as e:
        logger.error(f"[EMAIL] Error sending email: {e}")
        return False


# =========================
# Main
# =========================

def main():
    df_all = scan_fno_universe()
    if df_all is None or df_all.empty:
        logger.error("[MAIN] No data to email. Exiting.")
        return

    # Sort Table 1: Daily Volatility Expansion
    df_vol = df_all.sort_values(
        by="Daily Volatility Expansion", 
        ascending=False, 
        na_position="last"
    ).copy()

    # Sort Table 2: Daily Volume Expansion
    df_volum = df_all.sort_values(
        by="Daily Volume Expansion", 
        ascending=False, 
        na_position="last"
    ).copy()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"fo_fyers_expansion_scan_{ts}.csv"

    cols_to_save = [
        "Symbol", "LTP", "% Change", "Current Daily Volatility", "Avg Daily Volatility", 
        "Daily Volatility Expansion", "Current Volume", 
        "10 Day Relative Volume", "20 Day Relative Volume", "Daily Volume Expansion"
    ]

    # Save CSV
    df_vol[cols_to_save].to_csv(csv_filename, index=False)
    logger.info(f"[MAIN] Saved full scan to {csv_filename}")

    # Send email
    send_email_with_tables(df_vol[cols_to_save], df_volum[cols_to_save], csv_filename)

if __name__ == "__main__":
    main()

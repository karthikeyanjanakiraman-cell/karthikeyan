"""
FO_FNO_FYERS_VOL_REL_EMAIL.py

Uses Fyers historical data to:
1) Compute CURRENT and AVERAGE historical volatility for all F&O underlyings:
   - CurrVolPct  : last ~20 trading days (short-term)
   - AvgVolPct   : up to last ~252 trading days (1-year style, average)
   Both use annualized historical volatility: std(log returns) * sqrt(252) * 100.
2) Compute time-of-day relative volume:
   - RelVolume10 : current 5m bar volume vs avg of same slot over last 10 days
   - RelVolume20 : current 5m bar volume vs avg of same slot over last 20 days

Sends an HTML email with:
- Full F&O table: Symbol, LTP, CurrVolPct, AvgVolPct,
  RelVolume10, RelVolume20, CurrentVolume, AvgSlotVolume10, AvgSlotVolume20.
- Second table: ranked by strongest RelVolume10 / RelVolume20

Credential style and Fyers initialization follow EMAIL-2.py.
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
# Logging (similar style)
# =========================

class UTF8Formatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        msg = msg.replace("❌", "[ERROR]").replace("✅", "[OK]")
        msg = msg.replace("🟢", "[GREEN]").replace("🟡", "[YELLOW]").replace("🔴", "[RED]")
        msg = msg.replace("⚠️", "[WARN]").replace("📊", "[DATA]").replace("🎯", "[TARGET]")
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

logger.info("[OK] FO Fyers Volatility + Relative Volume Scanner initialized")


# =========================
# Config & Fyers init (ENV-first, same pattern as EMAIL-2.py)
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
    logger.info("[OK] Fyers API connected successfully (ENV-FIRST MODE)")
except Exception as e:
    logger.error(f"[ERROR] Fyers init failed: {e}")
    fyers = None


# =========================
# Load F&O symbols from sectors folder
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
    logger.info(f"[FNO] Loaded {len(symbols_list)} unique F&O symbols from '{root_dir}/'.")
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
    if fyers is None:
        return None

    if days_back <= 0:
        return None

    if resolution == "D":
        per_call_limit = 366
    else:
        per_call_limit = 100

    all_chunks = []
    now = datetime.now()
    end_date = now
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

            df_chunk = pd.DataFrame(
                candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
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
# Volatility calculations
# =========================

def compute_annualized_volatility_from_daily(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or len(df) < 10:
        return None
    closes = df["close"].astype(float)
    returns = np.log(closes / closes.shift(1))
    returns = returns.dropna()
    if len(returns) < 5:
        return None
    vol = returns.std() * math.sqrt(252) * 100.0
    return float(vol)


def compute_volatility_pair(df_daily: pd.DataFrame,
                            short_days: int = 20,
                            long_days: int = 252) -> Dict[str, Optional[float]]:
    if df_daily is None or df_daily.empty:
        return {"CurrVolPct": None, "AvgVolPct": None}

    df_sorted = df_daily.sort_values("timestamp").reset_index(drop=True)

    short_df = df_sorted.tail(short_days)
    curr_vol = compute_annualized_volatility_from_daily(short_df)

    long_df = df_sorted.tail(long_days)
    avg_vol = compute_annualized_volatility_from_daily(long_df)

    return {"CurrVolPct": curr_vol, "AvgVolPct": avg_vol}


# =========================
# Relative volume
# =========================

def compute_time_of_day_relative_volume_multi(
    df: pd.DataFrame,
    lookback_short: int = 10,
    lookback_long: int = 20,
) -> Dict[str, Optional[float]]:
    if df is None or df.empty or len(df) < 10:
        return {
            "RelVolume10": None,
            "AvgSlotVolume10": None,
            "RelVolume20": None,
            "AvgSlotVolume20": None,
            "CurrentVolume": None,
            "LTP": None,
        }

    s = df.copy()
    s["datetime"] = s["timestamp"]
    s["date"] = s["datetime"].dt.date
    s["time"] = s["datetime"].dt.time

    latest = s.iloc[-1]
    latest_date = latest["date"]
    latest_time = latest["time"]
    current_vol = float(latest["volume"])
    ltp = float(latest["close"])

    def _avg_slot(last_n_days: int) -> Optional[float]:
        start_date = latest_date - timedelta(days=last_n_days)
        mask_prior = (s["date"] < latest_date) & (s["date"] >= start_date)
        same_time_prior = s[mask_prior & (s["time"] == latest_time)]
        if same_time_prior.empty:
            return None
        return float(same_time_prior["volume"].mean())

    avg10 = _avg_slot(lookback_short)
    avg20 = _avg_slot(lookback_long)

    rel10 = current_vol / avg10 if avg10 and avg10 > 0 else None
    rel20 = current_vol / avg20 if avg20 and avg20 > 0 else None

    return {
        "RelVolume10": rel10,
        "AvgSlotVolume10": avg10,
        "RelVolume20": rel20,
        "AvgSlotVolume20": avg20,
        "CurrentVolume": current_vol,
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

        intra_df = get_fyers_history(fyers_sym, resolution="5", days_back=20)
        if intra_df is not None:
            rvol_info = compute_time_of_day_relative_volume_multi(
                intra_df, lookback_short=10, lookback_long=20
            )
        else:
            rvol_info = {
                "RelVolume10": None,
                "AvgSlotVolume10": None,
                "RelVolume20": None,
                "AvgSlotVolume20": None,
                "CurrentVolume": None,
                "LTP": float(daily_df["close"].iloc[-1]) if daily_df is not None else None,
            }

        rows.append({
            "Symbol": sym,
            "LTP": rvol_info["LTP"],
            "CurrVolPct": vol_info["CurrVolPct"],
            "AvgVolPct": vol_info["AvgVolPct"],
            "RelVolume10": rvol_info["RelVolume10"],
            "AvgSlotVolume10": rvol_info["AvgSlotVolume10"],
            "RelVolume20": rvol_info["RelVolume20"],
            "AvgSlotVolume20": rvol_info["AvgSlotVolume20"],
            "CurrentVolume": rvol_info["CurrentVolume"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(
        by=["AvgVolPct", "RelVolume10"],
        ascending=[False, False],
        na_position="last"
    ).reset_index(drop=True)
    return df


# =========================
# Email helpers
# =========================

def df_to_html_table(df: pd.DataFrame, float_cols=None, max_rows: int = 100) -> str:
    if df is None or df.empty:
        return "<p>No data available.</p>"
    float_cols = float_cols or []

    df_disp = df.copy().head(max_rows)
    for col in float_cols:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].map(
                lambda x: f"{x:.2f}"
                if isinstance(x, (float, int)) and not pd.isna(x)
                else ""
            )

    return df_disp.to_html(index=False, border=1, justify="center", escape=False)


def send_email_with_tables(df_all: pd.DataFrame, df_rvol: pd.DataFrame, csv_filename: str) -> bool:
    sender_email = get_cfg("email", "sender_email", env_name="SENDER_EMAIL")
    sender_password = get_cfg("email", "sender_password", env_name="SENDER_PASSWORD")
    recipient_email = get_cfg("email", "recipient_email", env_name="RECIPIENT_EMAIL")

    smtp_server = get_cfg("email", "smtp_server", env_name="SMTP_SERVER", default="smtp.gmail.com")
    smtp_port = get_cfg("email", "smtp_port", env_name="SMTP_PORT", default="587", is_int=True)

    if not all([sender_email, sender_password, recipient_email]):
        logger.warning("[EMAIL] Missing SENDER_EMAIL / SENDER_PASSWORD / RECIPIENT_EMAIL.")
        return False

    now = datetime.now()
    subject = f"F&O Curr/Avg Volatility + Relative Volume (Fyers) - {now.strftime('%Y-%m-%d %H:%M IST')}"

    float_cols = [
        "LTP",
        "CurrVolPct", "AvgVolPct",
        "RelVolume10", "AvgSlotVolume10",
        "RelVolume20", "AvgSlotVolume20",
        "CurrentVolume",
    ]

    table_all_html = df_to_html_table(
        df_all,
        float_cols=float_cols,
        max_rows=200,
    )
    table_rvol_html = df_to_html_table(
        df_rvol,
        float_cols=float_cols,
        max_rows=200,
    )

    body_html = f"""
    <html>
      <body>
        <p>Hello,</p>
        <p>
          Below is the latest F&O scan (Fyers data) showing
          <b>current vs average volatility</b> and
          <b>10-day / 20-day time-of-day relative volume</b>.
        </p>

        <h3>1) All F&O stocks – Curr/Avg volatility & relative volume</h3>
        {table_all_html}

        <h3>2) Relative volume stocks ranked by strongest RelVolume10 / RelVolume20</h3>
        {table_rvol_html}

        <p>Attached CSV: {os.path.basename(csv_filename)}</p>
        <p>Generated at: {now.strftime('%Y-%m-%d %H:%M:%S')}</p>

        <p>This is an automated email from the FO_FNO_FYERS_VOL_REL_EMAIL script.</p>
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
        logger.warning(f"[EMAIL] Failed to attach CSV {csv_filename}: {e}")

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

    # No secret threshold. Just filter rows that have relative volume data.
    df_rvol = df_all[
        df_all["RelVolume10"].notna() | df_all["RelVolume20"].notna()
    ].copy()

    # Find the maximum of the two relative volume fields for sorting
    df_rvol["RelVolumeMax"] = df_rvol[["RelVolume10", "RelVolume20"]].max(axis=1, skipna=True)

    # Sort descending by the maximum relative volume, then by the individual 10/20 values
    df_rvol = df_rvol.sort_values(
        ["RelVolumeMax", "RelVolume10", "RelVolume20"],
        ascending=[False, False, False],
        na_position="last"
    ).drop(columns=["RelVolumeMax"], errors="ignore")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"fo_fyers_vol_rel_scan_{ts}.csv"
    df_all.to_csv(csv_filename, index=False)
    logger.info(f"[MAIN] Saved full scan to {csv_filename} with {len(df_all)} rows.")

    send_email_with_tables(df_all, df_rvol, csv_filename)


if __name__ == "__main__":
    main()

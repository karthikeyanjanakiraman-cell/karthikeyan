"""
FO_FYERS_VOL_REL_EMAIL.py

Scans all F&O stocks from sectors CSVs, computes:
1) Average annualized volatility (approx. NSE style) from daily returns.
2) Daily Volatility Expansion (Current 5-day vol vs 20-day avg vol).
3) Time-of-day relative volume excluding the current day from the historical average (10-day and 20-day).
4) Volume Expansion (Current RelVol vs Average RelVol of the past N days).

Sends an HTML email with the F&O stocks meeting the REL_VOLUME_MIN threshold.
"""

import os
import sys
import math
import logging
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# =========================
# Logging
# =========================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# =========================
# Utility
# =========================

def get_yf_symbol(symbol: str) -> str:
    s = symbol.replace("NSE:", "").replace("-EQ", "").strip()
    if not s.endswith(".NS"):
        s = s + ".NS"
    return s

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
                col = next((c for c in df.columns if c.lower() in ("symbol", "symbols", "ticker")), None)
                if col is None:
                    continue
                for s in df[col].dropna().astype(str):
                    s = s.strip()
                    if s:
                        symbols.add(s)
            except Exception:
                pass

    return sorted(symbols)

# =========================
# Volatility
# =========================

def get_daily_history(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    try:
        yf_sym = get_yf_symbol(symbol)
        ticker = yf.Ticker(yf_sym)
        hist = ticker.history(period=period)
        if hist is None or hist.empty or len(hist) < 30:
            return None
        return hist
    except Exception:
        return None

def compute_volatility_metrics(hist: pd.DataFrame) -> Dict[str, float]:
    closes = hist["Close"].astype(float)
    returns = np.log(closes / closes.shift(1)).dropna()
    
    if len(returns) < 20:
        return {"current_vol": np.nan, "avg_vol": np.nan, "vol_expansion": np.nan}
    
    # Approx NSE daily volatility % (standard deviation of daily log returns * 100)
    daily_vol_series = returns.rolling(window=20).std() * 100.0
    
    if daily_vol_series.dropna().empty:
         return {"current_vol": np.nan, "avg_vol": np.nan, "vol_expansion": np.nan}
         
    current_vol = daily_vol_series.iloc[-1]
    avg_vol = daily_vol_series.mean()
    vol_expansion = current_vol / avg_vol if avg_vol > 0 else np.nan
    
    return {
        "current_vol": current_vol,
        "avg_vol": avg_vol,
        "vol_expansion": vol_expansion
    }

# =========================
# Intraday Volume Metrics
# =========================

def get_intraday_history(symbol: str, days: int = 30, interval: str = "5m") -> Optional[pd.DataFrame]:
    try:
        yf_sym = get_yf_symbol(symbol)
        ticker = yf.Ticker(yf_sym)
        intra = ticker.history(period=f"{days}d", interval=interval)
        if intra is None or intra.empty:
            return None
        return intra
    except Exception:
        return None

def compute_volume_expansion(intra: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Computes time-of-day relative volume and volume expansion.
    EXCLUDES today's volume from the historical average calculation.
    Computes historical relative volume for each past day to find the average relative volume.
    """
    if intra is None or intra.empty or len(intra) < 20:
        return {}

    if intra.index.tz is not None:
        idx = intra.index.tz_convert("Asia/Kolkata")
    else:
        idx = intra.index

    df = intra.copy()
    df["date"] = idx.date
    df["time"] = idx.time

    # Identify the latest bar
    latest_row = df.iloc[-1]
    latest_date = latest_row["date"]
    latest_time = latest_row["time"]
    current_volume = float(latest_row["Volume"])
    ltp = float(latest_row["Close"])

    # Isolate all rows matching the exact time slot, excluding today
    same_time_all = df[(df["time"] == latest_time) & (df["date"] < latest_date)].copy()
    
    if same_time_all.empty:
        return {"ltp": ltp, "current_volume": current_volume}
        
    same_time_all = same_time_all.sort_values("date")

    # Helper function for N days
    def calc_metrics_for_window(n_days: int):
        if len(same_time_all) < n_days:
            return None, None
            
        # Get the past N days for the average volume baseline
        past_n = same_time_all.tail(n_days)
        avg_vol = float(past_n["Volume"].mean())
        
        # Calculate Current Relative Volume vs past N days average
        current_rel_vol = (current_volume / avg_vol) if avg_vol > 0 else np.nan
        
        # To calculate expansion, we need the historical Relative Volumes of those past N days.
        # Rel Vol of Day T = (Volume on Day T) / (Avg Volume of N days before Day T)
        historical_rel_vols = []
        for i in range(len(same_time_all) - n_days, len(same_time_all)):
            target_day_vol = same_time_all.iloc[i]["Volume"]
            # Get the N days strictly BEFORE this target day
            if i >= n_days:
                baseline_window = same_time_all.iloc[i - n_days : i]
                baseline_avg = float(baseline_window["Volume"].mean())
                if baseline_avg > 0:
                    historical_rel_vols.append(target_day_vol / baseline_avg)
                    
        # Average historical relative volume for the past N days
        if historical_rel_vols:
            avg_historical_rel_vol = sum(historical_rel_vols) / len(historical_rel_vols)
            # Expansion = Current Rel Vol / Avg Historical Rel Vol
            expansion = current_rel_vol / avg_historical_rel_vol if avg_historical_rel_vol > 0 else np.nan
        else:
            expansion = np.nan
            
        return current_rel_vol, expansion

    rel_vol_10, exp_10 = calc_metrics_for_window(10)
    rel_vol_20, exp_20 = calc_metrics_for_window(20)

    # Use 10-day relative volume as the primary "Daily Volume Expansion" column for legacy compatibility, 
    # but we now calculate explicit Expansion metrics
    return {
        "ltp": ltp,
        "current_volume": current_volume,
        "rel_vol_10": rel_vol_10,
        "rel_vol_20": rel_vol_20,
        "exp_10": exp_10,
        "exp_20": exp_20,
        "daily_volume_expansion": rel_vol_10 # Keeping as fallback if needed
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
    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"[CORE] ({idx}/{len(symbols)}) Processing {sym}")

        daily_hist = get_daily_history(sym, period="1y")
        if daily_hist is None:
            continue
            
        vol_metrics = compute_volatility_metrics(daily_hist)
        intra = get_intraday_history(sym, days=30, interval="5m")
        vol_exp = compute_volume_expansion(intra) if intra is not None else {}

        rows.append({
            "Symbol": sym,
            "LTP": vol_exp.get("ltp", float(daily_hist["Close"].iloc[-1])),
            "Current Daily Volatility": vol_metrics["current_vol"],
            "Avg Daily Volatility": vol_metrics["avg_vol"],
            "Daily Volatility Expansion": vol_metrics["vol_expansion"],
            "Current Volume": vol_exp.get("current_volume", np.nan),
            "10 Day Relative Volume": vol_exp.get("rel_vol_10", np.nan),
            "10 Day Vol Expansion": vol_exp.get("exp_10", np.nan),
            "20 Day Relative Volume": vol_exp.get("rel_vol_20", np.nan),
            "20 Day Vol Expansion": vol_exp.get("exp_20", np.nan),
        })

    df = pd.DataFrame(rows)
    return df

# =========================
# Email logic
# =========================

def df_to_html_table(df: pd.DataFrame, float_cols=None, max_rows: int = 150) -> str:
    if df is None or df.empty:
        return "<p>No data available.</p>"
    float_cols = float_cols or []

    df_display = df.copy().head(max_rows)
    for col in float_cols:
        if col in df_display.columns:
            df_display[col] = df_display[col].map(
                lambda x: f"{x:.2f}" if isinstance(x, (float, int)) and not pd.isna(x) else ""
            )

    return df_display.to_html(index=False, border=1, justify="center", escape=False)

def send_email_with_tables(df_all: pd.DataFrame, df_rvol: pd.DataFrame, csv_filename: str) -> bool:
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")
    recipient_email = os.getenv("RECIPIENT_EMAIL")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    if not all([sender_email, sender_password, recipient_email]):
        logger.warning("[EMAIL] Missing email credentials.")
        return False

    now = datetime.now()
    subject = f"F&O Expansion & Relative Volume Scan - {now.strftime('%Y-%m-%d %H:%M IST')}"

    cols_to_format = [
        "LTP", "Current Daily Volatility", "Avg Daily Volatility", 
        "Daily Volatility Expansion", "10 Day Relative Volume", 
        "10 Day Vol Expansion", "20 Day Relative Volume", "20 Day Vol Expansion"
    ]

    table_rvol_html = df_to_html_table(df_rvol, float_cols=cols_to_format, max_rows=150)

    body_html = f"""
    <html>
      <body>
        <p>Hello,</p>
        <p>Below is the latest F&amp;O scan targeting Volatility and Volume Expansion.</p>
        <p><i>Note: The historical volume average strictly excludes the current day's volume.</i></p>

        <h3>High Relative Volume Stocks (Rel Vol >= Threshold)</h3>
        {table_rvol_html}

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
        logger.warning(f"[EMAIL] Failed to attach CSV: {e}")

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

    raw_env_min = os.getenv("REL_VOLUME_MIN")

    if raw_env_min is None or raw_env_min.strip() == "":
        logger.error("[MAIN] REL_VOLUME_MIN is missing or empty. Please set it in GitHub Secrets.")
        raise ValueError("REL_VOLUME_MIN is required and cannot be empty")

    try:
        rel_min = float(raw_env_min.strip())
    except ValueError:
        logger.error(f"[MAIN] Invalid REL_VOLUME_MIN value: '{raw_env_min}'. It must be a number.")
        raise ValueError(f"Invalid REL_VOLUME_MIN: {raw_env_min}")

    # Filter stocks where EITHER 10-day OR 20-day relative volume is above the threshold
    df_rvol = df_all[
        (pd.to_numeric(df_all["10 Day Relative Volume"], errors="coerce") >= rel_min) |
        (pd.to_numeric(df_all["20 Day Relative Volume"], errors="coerce") >= rel_min)
    ].copy()

    # Sort heavily by 10-day volume expansion and 10-day relative volume
    df_rvol = df_rvol.sort_values(
        ["10 Day Vol Expansion", "10 Day Relative Volume"],
        ascending=[False, False],
        na_position="last"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"fo_fyers_expansion_scan_{ts}.csv"
    df_all.to_csv(csv_filename, index=False)
    logger.info(f"[MAIN] Saved full scan to {csv_filename} with {len(df_all)} rows.")

    send_email_with_tables(df_all, df_rvol, csv_filename)

if __name__ == "__main__":
    main()

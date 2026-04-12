import os
import sys
import math
import logging
import configparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

try:
    from fyersapiv3 import fyersModel
except ImportError:
    try:
        from fyers_apiv3 import fyersModel
    except ImportError as e:
        raise ImportError(
            "Fyers package not found. Install with: pip install fyers-apiv3"
        ) from e


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
if not logger.handlers:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(UTF8Formatter(log_format))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("fo_fyers_vol_rel_email.log", encoding="utf-8")
    file_handler.setFormatter(UTF8Formatter(log_format))
    logger.addHandler(file_handler)

logger.info("[OK] FO Fyers Volatility + Relative Volume + Email Scanner initialized")

config = configparser.ConfigParser()
config.read("config.ini")


def get_cfg(section, key, env_name=None, default=None, is_int=False):
    if env_name:
        val = os.getenv(env_name)
        if val is not None and str(val).strip() != "":
            return int(val) if is_int else val
    if section and key and config.has_option(section, key):
        val = config.get(section, key)
        return int(val) if is_int else val
    return default


try:
    client_id = get_cfg("fyers_credentials", "client_id", env_name="CLIENTID") or get_cfg("fyers_credentials", "client_id", env_name="CLIENT_ID")
    token = (
        get_cfg("fyers_credentials", "access_token", env_name="ACCESSTOKEN")
        or get_cfg("fyers_credentials", "access_token", env_name="ACCESS_TOKEN")
        or get_cfg("fyers_credentials", "token", env_name="TOKEN")
    )
    if not client_id or not token:
        raise ValueError("Missing CLIENTID/CLIENT_ID or ACCESSTOKEN/ACCESS_TOKEN")
    fyers = fyersModel.FyersModel(client_id=client_id, token=token)
    logger.info("[OK] Fyers API connected successfully")
except Exception as e:
    logger.error(f"[ERROR] Fyers init failed: {e}")
    fyers = None


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
            except Exception as e:
                logger.warning(f"[FNO] Error reading {fpath}: {e}")
    out = sorted(symbols)
    logger.info(f"[FNO] Loaded {len(out)} unique F&O symbols.")
    return out


def format_fyers_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if s.startswith("NSE:") and ("-EQ" in s or "-INDEX" in s):
        return s
    s = s.replace("NSE:", "").replace("-EQ", "")
    if s in ["NIFTY", "NIFTY50"]:
        return "NSE:NIFTY50-INDEX"
    if s == "BANKNIFTY":
        return "NSE:NIFTYBANK-INDEX"
    if s == "FINNIFTY":
        return "NSE:FINNIFTY-INDEX"
    if s == "MIDCPNIFTY":
        return "NSE:MIDCPNIFTY-INDEX"
    return f"NSE:{s}-EQ"


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
        except Exception as e:
            logger.warning(f"[API] {symbol} {resolution} failed: {e}")
            break
        remaining_days -= chunk_days
        end_date = start_date - timedelta(days=1)

    if not all_chunks:
        return None
    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def compute_annualized_volatility_from_daily(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or len(df) < 10:
        return None
    closes = df["close"].astype(float)
    returns = np.log(closes / closes.shift(1)).dropna()
    if len(returns) < 5:
        return None
    return float(returns.std() * math.sqrt(252) * 100.0)


def compute_volatility_pair(df_daily: pd.DataFrame, short_days: int = 20, long_days: int = 252) -> Dict[str, Optional[float]]:
    if df_daily is None or df_daily.empty:
        return {"Current Daily Volatility": None, "Avg Daily Volatility": None, "Daily Volatility Expansion": None}
    df_sorted = df_daily.sort_values("timestamp").reset_index(drop=True)
    curr_vol_ann = compute_annualized_volatility_from_daily(df_sorted.tail(short_days))
    avg_vol_ann = compute_annualized_volatility_from_daily(df_sorted.tail(long_days))
    sqrt_252 = math.sqrt(252)
    daily_curr = (curr_vol_ann / sqrt_252) if curr_vol_ann else None
    daily_avg = (avg_vol_ann / sqrt_252) if avg_vol_ann else None
    vol_exp = (daily_curr / daily_avg) if (daily_curr and daily_avg and daily_avg > 0) else None
    return {
        "Current Daily Volatility": daily_curr,
        "Avg Daily Volatility": daily_avg,
        "Daily Volatility Expansion": vol_exp,
    }


def compute_cumulative_relative_volume(df: pd.DataFrame, lookback_short: int = 10, lookback_long: int = 20) -> Dict[str, Optional[float]]:
    if df is None or df.empty or len(df) < 10:
        return {
            "Today Volume": None,
            "10 Day Relative Volume": None,
            "20 Day Relative Volume": None,
            "Today Volume / 10 Day Relative Volume": None,
            "Today Volume / 20 Day Relative Volume": None,
            "LTP": None,
        }

    s = df.copy()
    s["datetime"] = pd.to_datetime(s["timestamp"])
    s["date"] = s["datetime"].dt.date
    s["time"] = s["datetime"].dt.time

    latest = s.iloc[-1]
    latest_date = latest["date"]
    latest_time = latest["time"]
    ltp = float(latest["close"])

    today_mask = (s["date"] == latest_date) & (s["time"] <= latest_time)
    current_cum_vol = float(s.loc[today_mask, "volume"].sum())

    def avg_cum_base(last_n_days: int) -> Optional[float]:
        start_date = latest_date - timedelta(days=last_n_days * 3)
        mask_prior = (s["date"] < latest_date) & (s["date"] >= start_date) & (s["time"] <= latest_time)
        prior = s.loc[mask_prior]
        if prior.empty:
            return None
        daily_sums = prior.groupby("date")["volume"].sum().tail(last_n_days)
        if daily_sums.empty:
            return None
        return float(daily_sums.mean())

    avg10 = avg_cum_base(lookback_short)
    avg20 = avg_cum_base(lookback_long)
    rel10 = (current_cum_vol / avg10) if avg10 and avg10 > 0 else None
    rel20 = (current_cum_vol / avg20) if avg20 and avg20 > 0 else None

    return {
        "Today Volume": current_cum_vol,
        "10 Day Relative Volume": rel10,
        "20 Day Relative Volume": rel20,
        "Today Volume / 10 Day Relative Volume": rel10,
        "Today Volume / 20 Day Relative Volume": rel20,
        "LTP": ltp,
    }


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
        daily_df = get_fyers_history(fyers_sym, "D", 365)
        intra_df = get_fyers_history(fyers_sym, "5", 40)

        vol_info = compute_volatility_pair(daily_df)
        rvol_info = compute_cumulative_relative_volume(intra_df)

        prev_close = None
        if daily_df is not None and len(daily_df) >= 2:
            prev_close = float(daily_df["close"].iloc[-2])

        ltp = rvol_info.get("LTP")
        pct_change = ((ltp - prev_close) / prev_close) * 100 if (ltp is not None and prev_close not in [None, 0]) else 0.0

        vol_exp = vol_info.get("Daily Volatility Expansion")
        rel10 = rvol_info.get("Today Volume / 10 Day Relative Volume")
        rel20 = rvol_info.get("Today Volume / 20 Day Relative Volume")
        daily_vol_exp = max([x for x in [rel10, rel20] if pd.notna(x)], default=np.nan)
        momentum_score = pct_change * vol_exp * rel10 if pd.notna(pct_change) and pd.notna(vol_exp) and pd.notna(rel10) else np.nan
        short_momentum_score = (-pct_change) * vol_exp * rel10 if pd.notna(pct_change) and pd.notna(vol_exp) and pd.notna(rel10) else np.nan

        trade_side = "Neutral"
        if pd.notna(pct_change) and pd.notna(vol_exp) and pd.notna(rel10):
            if pct_change > 0 and vol_exp > 1 and rel10 > 1:
                trade_side = "Long"
            elif pct_change < 0 and vol_exp > 1 and rel10 > 1:
                trade_side = "Short"

        rows.append({
            "Symbol": sym,
            "LTP": ltp,
            "% Change": pct_change,
            "Current Daily Volatility": vol_info.get("Current Daily Volatility"),
            "Avg Daily Volatility": vol_info.get("Avg Daily Volatility"),
            "Daily Volatility Expansion": vol_exp,
            "Today Volume": rvol_info.get("Today Volume"),
            "10 Day Relative Volume": rvol_info.get("10 Day Relative Volume"),
            "20 Day Relative Volume": rvol_info.get("20 Day Relative Volume"),
            "Today Volume / 10 Day Relative Volume": rel10,
            "Today Volume / 20 Day Relative Volume": rel20,
            "Daily Volume Expansion": daily_vol_exp,
            "Momentum Score": momentum_score,
            "Short Momentum Score": short_momentum_score,
            "Trade Side": trade_side,
        })

    return pd.DataFrame(rows)


def df_to_html_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "<p>No data available.</p>"
    df_disp = df.head(max_rows).copy()
    for col in df_disp.columns:
        if col == "Symbol" or col == "Trade Side":
            continue
        if col == "Today Volume":
            df_disp[col] = df_disp[col].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        elif col == "% Change":
            df_disp[col] = df_disp[col].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "")
        else:
            df_disp[col] = df_disp[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return df_disp.to_html(index=False, border=1, justify="center", escape=False)


def send_email_with_tables(df_all: pd.DataFrame, csv_filename: str) -> bool:
    sender_email = get_cfg("email", "sender_email", env_name="SENDEREMAIL") or get_cfg("email", "sender_email", env_name="SENDER_EMAIL")
    sender_password = get_cfg("email", "sender_password", env_name="SENDERPASSWORD") or get_cfg("email", "sender_password", env_name="SENDER_PASSWORD")
    recipient_email = get_cfg("email", "recipient_email", env_name="RECIPIENTEMAIL") or get_cfg("email", "recipient_email", env_name="RECIPIENT_EMAIL")
    smtp_server = get_cfg("email", "smtp_server", env_name="SMTPSERVER", default="smtp.gmail.com") or get_cfg("email", "smtp_server", env_name="SMTP_SERVER", default="smtp.gmail.com")
    smtp_port = get_cfg("email", "smtp_port", env_name="SMTPPORT", default=587, is_int=True) or get_cfg("email", "smtp_port", env_name="SMTP_PORT", default=587, is_int=True)

    if not all([sender_email, sender_password, recipient_email]):
        logger.warning("[EMAIL] Missing email credentials.")
        return False

    long_df = df_all[(df_all["% Change"] > 0) & (df_all["Daily Volatility Expansion"] > 1) & (df_all["Today Volume / 10 Day Relative Volume"] > 1)].copy()
    long_df = long_df.sort_values(["Momentum Score", "% Change"], ascending=[False, False])

    short_df = df_all[(df_all["% Change"] < 0) & (df_all["Daily Volatility Expansion"] > 1) & (df_all["Today Volume / 10 Day Relative Volume"] > 1)].copy()
    short_df = short_df.sort_values(["Short Momentum Score", "% Change"], ascending=[False, True])

    vol_df = df_all.sort_values("Daily Volatility Expansion", ascending=False, na_position="last").copy()

    long_display_cols = [
        "Symbol", "LTP", "% Change", "Daily Volatility Expansion",
        "Today Volume / 10 Day Relative Volume", "Today Volume / 20 Day Relative Volume",
        "Daily Volume Expansion", "Momentum Score", "Trade Side"
    ]
    short_display_cols = [
        "Symbol", "LTP", "% Change", "Daily Volatility Expansion",
        "Today Volume / 10 Day Relative Volume", "Today Volume / 20 Day Relative Volume",
        "Daily Volume Expansion", "Short Momentum Score", "Trade Side"
    ]

    long_html = df_to_html_table(long_df[long_display_cols], 15)
    short_html = df_to_html_table(short_df[short_display_cols], 15)
    vol_html = df_to_html_table(vol_df[[
        "Symbol", "LTP", "% Change", "Current Daily Volatility", "Avg Daily Volatility",
        "Daily Volatility Expansion", "Today Volume / 10 Day Relative Volume",
        "Today Volume / 20 Day Relative Volume", "Daily Volume Expansion"
    ]], 15)

    now = datetime.now()
    subject = f"FO Expansion Scan - Long / Short / Volatility - {now.strftime('%Y-%m-%d %H:%M IST')}"
    body_html = f"""
    <html>
    <head>
    <style>
    body {{ font-family: Arial, sans-serif; font-size: 13px; color: #111; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; }}
    th, td {{ border: 1px solid #dcdcdc; padding: 6px; text-align: right; }}
    th {{ background: #f2f2f2; text-align: center; }}
    td:first-child, th:first-child {{ text-align: left; }}
    h2, h3 {{ margin-bottom: 8px; }}
    p {{ margin: 6px 0; }}
    </style>
    </head>
    <body>
    <p>Hello,</p>
    <p>Below is the latest FO scan with the requested long and short tables.</p>
    <p><b>Long filter:</b> % Change &gt; 0, Daily Volatility Expansion &gt; 1, Today Volume / 10 Day Relative Volume &gt; 1</p>
    <p><b>Short filter:</b> % Change &lt; 0, Daily Volatility Expansion &gt; 1, Today Volume / 10 Day Relative Volume &gt; 1</p>

    <h3>1) Long Candidates - Top 15</h3>
    {long_html}

    <h3>2) Short Candidates - Top 15</h3>
    {short_html}

    <h3>3) Top Daily Volatility Expansion</h3>
    {vol_html}

    <p><b>Column guide:</b></p>
    <ul>
      <li>Momentum Score = % Change Ã— Daily Volatility Expansion Ã— Today Volume / 10 Day Relative Volume</li>
      <li>Short Momentum Score = -(% Change) Ã— Daily Volatility Expansion Ã— Today Volume / 10 Day Relative Volume</li>
      <li>Daily Volume Expansion = max(10 Day Relative Volume, 20 Day Relative Volume)</li>
    </ul>

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


def main():
    df_all = scan_fno_universe()
    if df_all is None or df_all.empty:
        logger.error("[MAIN] No data to process. Exiting.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"fo_fyers_expansion_scan_{ts}.csv"

    cols_to_save = [
        "Symbol", "LTP", "% Change", "Current Daily Volatility", "Avg Daily Volatility",
        "Daily Volatility Expansion", "Today Volume", "10 Day Relative Volume", "20 Day Relative Volume",
        "Today Volume / 10 Day Relative Volume", "Today Volume / 20 Day Relative Volume",
        "Daily Volume Expansion", "Trade Side"
    ]

    df_all[cols_to_save].sort_values("Daily Volatility Expansion", ascending=False, na_position="last").to_csv(csv_filename, index=False)
    logger.info(f"[MAIN] Saved scan to {csv_filename}")
    send_email_with_tables(df_all, csv_filename)


if __name__ == "__main__":
    main()

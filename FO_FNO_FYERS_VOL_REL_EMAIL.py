import os
import sys
import math
import logging
import configparser
from datetime import datetime, timedelta, time
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


class UTF8Formatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        msg = msg.replace("âŒ", "[ERROR]").replace("âœ…", "[OK]")
        msg = msg.replace("ðŸŸ¢", "[GREEN]").replace("ðŸŸ¡", "[YELLOW]").replace("ðŸ”´", "[RED]")
        msg = msg.replace("âš ï¸", "[WARN]").replace("ðŸ“Š", "[DATA]").replace("ðŸŽ¯", "[TARGET]")
        record.msg = msg
        return super().format(record)


LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(UTF8Formatter(LOG_FORMAT))
logger.addHandler(console_handler)

file_handler = logging.FileHandler("fo_fyers_iteration_email.log", encoding="utf-8")
file_handler.setFormatter(UTF8Formatter(LOG_FORMAT))
logger.addHandler(file_handler)

logger.info("[OK] FO Fyers Iteration-based Volatility + Volume Scanner initialized")


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


MARKET_OPEN = time(9, 15)
DAILY_VOL_THRESHOLD = 1.0
DAILY_VOLUME_THRESHOLD = 1.0
SHORT_DAYS = 20
LONG_DAYS = 252
RVOL_SHORT = 10
RVOL_LONG = 20
INTRADAY_LOOKBACK_DAYS = 45
DAILY_LOOKBACK_DAYS = 365


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
            df_chunk["timestamp"] = pd.to_datetime(df_chunk["timestamp"], unit="s").dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            all_chunks.append(df_chunk)
        except Exception as e:
            logger.warning(f"[API] {symbol} {resolution} history fetch failed: {e}")
            break
        remaining_days -= chunk_days
        end_date = start_date - timedelta(days=1)

    if not all_chunks:
        return None

    df = pd.concat(all_chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_annualized_volatility_from_daily(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or len(df) < 10:
        return None
    closes = df["close"].astype(float)
    returns = np.log(closes / closes.shift(1)).dropna()
    if len(returns) < 5:
        return None
    vol = returns.std() * math.sqrt(252) * 100.0
    return float(vol)


def compute_volatility_pair(df_daily: pd.DataFrame, short_days: int = SHORT_DAYS, long_days: int = LONG_DAYS) -> Dict[str, Optional[float]]:
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
        "Current Daily Volatility": daily_curr,
        "Avg Daily Volatility": daily_avg,
        "Daily Volatility Expansion": vol_expansion,
    }


def _safe_max(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if pd.notna(v)]
    return max(vals) if vals else None


def compute_iteration_volume_profile(
    df: pd.DataFrame,
    lookback_short: int = RVOL_SHORT,
    lookback_long: int = RVOL_LONG,
    market_open: time = MARKET_OPEN,
) -> Tuple[Dict[str, Optional[float]], pd.DataFrame]:
    empty_summary = {
        "Current Volume": None,
        "10 Day Relative Volume": None,
        "20 Day Relative Volume": None,
        "Daily Volume Expansion": None,
        "Total Iterations": 0,
        "Last Iteration Minutes": None,
        "Last Iteration Time": None,
        "LTP": None,
    }
    if df is None or df.empty or len(df) < 10:
        return empty_summary, pd.DataFrame()

    s = df.copy()
    s["datetime"] = pd.to_datetime(s["timestamp"])
    s = s.sort_values("datetime").reset_index(drop=True)
    s["date"] = s["datetime"].dt.date
    s["time"] = s["datetime"].dt.time
    s = s[s["time"] >= market_open].copy()
    if s.empty:
        return empty_summary, pd.DataFrame()

    s["cum_volume"] = s.groupby("date")["volume"].cumsum()
    latest = s.iloc[-1]
    latest_date = latest["date"]
    today_df = s[s["date"] == latest_date].copy().sort_values("datetime")
    if today_df.empty:
        return empty_summary, pd.DataFrame()

    detail_rows = []
    base_dt = datetime.combine(latest_date, market_open)

    for _, row in today_df.iterrows():
        cutoff_dt = row["datetime"]
        minutes_elapsed = int((cutoff_dt - base_dt).total_seconds() // 60)
        if minutes_elapsed <= 0:
            continue

        cutoff_time = row["time"]
        current_cum_vol = float(row["cum_volume"])
        prior = s[(s["date"] < latest_date) & (s["time"] <= cutoff_time)].copy()
        if prior.empty:
            continue

        prior_cum = prior.groupby("date")["cum_volume"].max().sort_index()
        if prior_cum.empty:
            continue

        avg10 = float(prior_cum.tail(lookback_short).mean()) if len(prior_cum.tail(lookback_short)) > 0 else None
        avg20 = float(prior_cum.tail(lookback_long).mean()) if len(prior_cum.tail(lookback_long)) > 0 else None
        rel10 = (current_cum_vol / avg10) if (avg10 and avg10 > 0) else None
        rel20 = (current_cum_vol / avg20) if (avg20 and avg20 > 0) else None
        volume_exp = _safe_max([rel10, rel20])

        detail_rows.append({
            "Iteration No": len(detail_rows) + 1,
            "Iteration Minutes": minutes_elapsed,
            "Iteration Time": cutoff_time.strftime("%H:%M"),
            "Current Volume": current_cum_vol,
            "10 Day Relative Volume": rel10,
            "20 Day Relative Volume": rel20,
            "Daily Volume Expansion": volume_exp,
            "LTP": float(row["close"]),
        })

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        summary = empty_summary.copy()
        summary["LTP"] = float(today_df.iloc[-1]["close"])
        return summary, detail_df

    latest_detail = detail_df.iloc[-1]
    summary = {
        "Current Volume": float(latest_detail["Current Volume"]),
        "10 Day Relative Volume": float(latest_detail["10 Day Relative Volume"]) if pd.notna(latest_detail["10 Day Relative Volume"]) else None,
        "20 Day Relative Volume": float(latest_detail["20 Day Relative Volume"]) if pd.notna(latest_detail["20 Day Relative Volume"]) else None,
        "Daily Volume Expansion": float(latest_detail["Daily Volume Expansion"]) if pd.notna(latest_detail["Daily Volume Expansion"]) else None,
        "Total Iterations": int(len(detail_df)),
        "Last Iteration Minutes": int(latest_detail["Iteration Minutes"]),
        "Last Iteration Time": latest_detail["Iteration Time"],
        "LTP": float(latest_detail["LTP"]),
    }
    return summary, detail_df


def scan_fno_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = load_fno_symbols_from_sectors("sectors")
    if not symbols:
        logger.error("[CORE] No F&O symbols found. Exiting.")
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    iteration_rows = []
    total = len(symbols)

    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"[CORE] ({idx}/{total}) Processing {sym}")
        fyers_sym = format_fyers_symbol(sym)

        daily_df = get_fyers_history(fyers_sym, resolution="D", days_back=DAILY_LOOKBACK_DAYS)
        intra_df = get_fyers_history(fyers_sym, resolution="5", days_back=INTRADAY_LOOKBACK_DAYS)

        vol_info = compute_volatility_pair(daily_df)
        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df)

        if daily_df is not None and len(daily_df) >= 2:
            prev_close = float(daily_df["close"].iloc[-2])
        else:
            prev_close = None

        ltp = iter_summary.get("LTP")
        pct_change = ((ltp - prev_close) / prev_close) * 100 if (ltp and prev_close and prev_close > 0) else 0.0
        daily_vol_exp = vol_info.get("Daily Volatility Expansion")
        daily_volume_exp = iter_summary.get("Daily Volume Expansion")
        momentum_score = (daily_vol_exp * daily_volume_exp) if (daily_vol_exp is not None and daily_volume_exp is not None) else None

        if not iter_detail.empty:
            if daily_vol_exp is not None and daily_vol_exp > DAILY_VOL_THRESHOLD:
                iter_detail["Above DV and DVol"] = iter_detail["Daily Volume Expansion"].gt(DAILY_VOLUME_THRESHOLD)
            else:
                iter_detail["Above DV and DVol"] = False
            above_count = int(iter_detail["Above DV and DVol"].sum())
            iter_detail.insert(0, "Symbol", sym)
            iter_detail.insert(1, "% Change", pct_change)
            iter_detail.insert(2, "Daily Volatility Expansion", daily_vol_exp)
            iteration_rows.append(iter_detail)
        else:
            above_count = 0

        total_iterations = int(iter_summary.get("Total Iterations") or 0)
        above_ratio = (above_count / total_iterations) if total_iterations > 0 else 0.0

        rows.append({
            "Symbol": sym,
            "LTP": ltp,
            "% Change": pct_change,
            "Current Daily Volatility": vol_info.get("Current Daily Volatility"),
            "Avg Daily Volatility": vol_info.get("Avg Daily Volatility"),
            "Daily Volatility Expansion": daily_vol_exp,
            "Current Volume": iter_summary.get("Current Volume"),
            "10 Day Relative Volume": iter_summary.get("10 Day Relative Volume"),
            "20 Day Relative Volume": iter_summary.get("20 Day Relative Volume"),
            "Daily Volume Expansion": daily_volume_exp,
            "Momentum Score": momentum_score,
            "Total Iterations": total_iterations,
            "Above Threshold Iterations": above_count,
            "Above Threshold Ratio": above_ratio,
            "Last Iteration Minutes": iter_summary.get("Last Iteration Minutes"),
            "Last Iteration Time": iter_summary.get("Last Iteration Time"),
        })

    summary_df = pd.DataFrame(rows)
    iteration_df = pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame()
    return summary_df, iteration_df


DISPLAY_COLS = [
    "Symbol",
    "LTP",
    "% Change",
    "Daily Volatility Expansion",
    "10 Day Relative Volume",
    "20 Day Relative Volume",
    "Daily Volume Expansion",
    "Momentum Score",
    "Total Iterations",
    "Above Threshold Iterations",
    "Above Threshold Ratio",
    "Last Iteration Minutes",
    "Last Iteration Time",
]


def build_candidate_tables(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=DISPLAY_COLS), pd.DataFrame(columns=DISPLAY_COLS)

    base = df_all.copy()
    base = base[(base["Daily Volatility Expansion"] > DAILY_VOL_THRESHOLD) & (base["Daily Volume Expansion"] > DAILY_VOLUME_THRESHOLD)].copy()

    long_df = base[base["% Change"] > 0].copy()
    short_df = base[base["% Change"] < 0].copy()

    long_df = long_df.sort_values(
        by=["Above Threshold Iterations", "Momentum Score", "% Change"],
        ascending=[False, False, False],
        na_position="last",
    ).head(15)

    short_df = short_df.sort_values(
        by=["Above Threshold Iterations", "Momentum Score", "% Change"],
        ascending=[False, False, True],
        na_position="last",
    ).head(15)

    return long_df[DISPLAY_COLS].copy(), short_df[DISPLAY_COLS].copy()


def format_value(col: str, val):
    if pd.isna(val):
        return ""
    if col in ["LTP", "Current Daily Volatility", "Avg Daily Volatility", "Daily Volatility Expansion", "10 Day Relative Volume", "20 Day Relative Volume", "Daily Volume Expansion", "Momentum Score"]:
        return f"{val:.2f}"
    if col == "% Change":
        return f"{val:+.2f}%"
    if col == "Current Volume":
        return f"{val:,.0f}"
    if col == "Above Threshold Ratio":
        return f"{val:.2%}"
    if col in ["Total Iterations", "Above Threshold Iterations", "Last Iteration Minutes"]:
        return f"{int(val)}"
    return str(val)


def df_to_html_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "<p>No data available.</p>"
    df_disp = df.copy().head(max_rows)
    for col in df_disp.columns:
        df_disp[col] = df_disp[col].map(lambda x, c=col: format_value(c, x))
    return df_disp.to_html(index=False, border=1, justify="center", escape=False)


def send_email_with_tables(long_df: pd.DataFrame, short_df: pd.DataFrame, csv_filename: str, detail_csv_filename: str) -> bool:
    sender_email = get_cfg("email", "sender_email", env_name="SENDER_EMAIL")
    sender_password = get_cfg("email", "sender_password", env_name="SENDER_PASSWORD")
    recipient_email = get_cfg("email", "recipient_email", env_name="RECIPIENT_EMAIL")
    smtp_server = get_cfg("email", "smtp_server", env_name="SMTP_SERVER", default="smtp.gmail.com")
    smtp_port = get_cfg("email", "smtp_port", env_name="SMTP_PORT", default="587", is_int=True)

    if not all([sender_email, sender_password, recipient_email]):
        logger.warning("[EMAIL] Missing email credentials.")
        return False

    now = datetime.now()
    subject = f"F&O Iteration Expansion Scan - {now.strftime('%Y-%m-%d %H:%M IST')}"
    long_html = df_to_html_table(long_df, max_rows=15)
    short_html = df_to_html_table(short_df, max_rows=15)

    body_html = f"""
    <html>
    <head>
      <style>
        body {{ font-family: Arial, sans-serif; font-size: 13px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; }}
        th, td {{ border: 1px solid #dddddd; padding: 6px; text-align: right; }}
        th {{ background-color: #f2f2f2; text-align: center; }}
        td:first-child {{ text-align: left; font-weight: bold; }}
        p, li {{ line-height: 1.4; }}
      </style>
    </head>
    <body>
      <p>Hello,</p>
      <p>This scan uses rolling cumulative iterations from market open. Example: 9:15-9:20, 9:15-9:25, 9:15-9:30 and so on, with every iteration compared against the same elapsed window over the previous 10 and 20 trading days.</p>
      <p>Momentum Score = Daily Volatility Expansion × Daily Volume Expansion. % Change is not part of the score.</p>
      <h3>Long Candidates - Top 15</h3>
      {long_html}
      <h3>Short Candidates - Top 15</h3>
      {short_html}
      <p><b>Columns</b></p>
      <ul>
        <li><b>Total Iterations</b>: number of 5-minute cumulative windows completed today.</li>
        <li><b>Above Threshold Iterations</b>: number of iterations where Daily Volatility Expansion &gt; 1 and Daily Volume Expansion &gt; 1.</li>
        <li><b>Above Threshold Ratio</b>: Above Threshold Iterations / Total Iterations.</li>
      </ul>
      <p>Attached files: {os.path.basename(csv_filename)} and {os.path.basename(detail_csv_filename)}</p>
      <p>Generated at: {now.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body_html, "html"))

    for fpath in [csv_filename, detail_csv_filename]:
        try:
            with open(fpath, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(fpath)}"')
            msg.attach(part)
        except Exception as e:
            logger.warning(f"[EMAIL] Failed to attach {fpath}: {e}")

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
    df_all, df_iter = scan_fno_universe()
    if df_all is None or df_all.empty:
        logger.error("[MAIN] No data to email. Exiting.")
        return

    long_df, short_df = build_candidate_tables(df_all)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = f"fo_fyers_iteration_scan_{ts}.csv"
    detail_csv = f"fo_fyers_iteration_detail_{ts}.csv"

    df_all.to_csv(summary_csv, index=False)
    logger.info(f"[MAIN] Saved summary scan to {summary_csv}")

    if df_iter is not None and not df_iter.empty:
        df_iter.to_csv(detail_csv, index=False)
    else:
        pd.DataFrame(columns=["Symbol", "Iteration No", "Iteration Minutes", "Iteration Time", "Current Volume", "10 Day Relative Volume", "20 Day Relative Volume", "Daily Volume Expansion", "LTP", "% Change", "Daily Volatility Expansion", "Above DV and DVol"]).to_csv(detail_csv, index=False)
    logger.info(f"[MAIN] Saved iteration detail scan to {detail_csv}")

    send_email_with_tables(long_df, short_df, summary_csv, detail_csv)


if __name__ == "__main__":
    main()

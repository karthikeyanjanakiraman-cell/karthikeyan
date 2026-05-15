#!/usr/bin/env python3
import os
import json
import time
import smtplib
import logging
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

try:
    from fyers_apiv3 import fyersModel
except Exception:
    try:
        import fyersModel
    except Exception:
        from fyersapi import fyersModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Fyers global
fyers = None

# Col name for timestamps in CSV
c_timestamp = "timestamp"

# === Chain Signal Constants ===
# Entry: 5m CONFIRMED + 30m CONFIRMED + T30 >= T30_MIN
# Exit : 30m BROKEN  OR  T30 < T30_MIN
T30_MIN = int(os.environ.get("T30_MIN", "2"))
DAILY_LOOKBACK_DAYS = 252
INTRADAY_LOOKBACK_DAYS = 20
IVP_LOOKBACK_DAYS = 252
OPTION_PAIRS_TO_KEEP = 5
SIGNAL_WINDOW_MINUTES = 5
ITERATIONS_TO_KEEP = 75
SECTORS_DIR = os.environ.get("SECTORS_DIR", "sectors")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
MIN_OPTION_LTP = 10.0
MIN_ATM_CHAIN_VOLUME = int(os.environ.get("MIN_ATM_CHAIN_VOLUME", "100000"))
EMAIL_MAX_ROWS_LONG = int(os.environ.get("EMAIL_MAX_ROWS_LONG", "25"))
EMAIL_MAX_ROWS_SHORT = int(os.environ.get("EMAIL_MAX_ROWS_SHORT", "25"))
TOP_N_UNDERLYINGS = int(os.environ.get("TOP_N_UNDERLYINGS", "60"))
OBV_BREAKOUT_WINDOW = int(os.environ.get("OBV_BREAKOUT_WINDOW", "5"))

OPTION_EMAIL_COLS = [
    "Underlying", "Option Type", "Option Symbol", "Strike", "LTP", "% Change",
    "OI", "Volume", "OBV", "OI+Volume+OBV Score", "EMAIL_RANK_SCORE",
    "Rank Delta", "Cumulative ADX", "5m_Signal", "15m_Signal", "30m_Signal",
    "60m_Signal", "Bull_Signal", "Bear_Signal", "Overall_Signal",
    "Price_Lead_Status", "IVP", "Volatility State", "Last Iteration Time",
]

OPTION_EMAIL_COL_RENAME = {
    "OI+Volume+OBV Score": "Liq Score",
    "EMAIL_RANK_SCORE": "Rank",
    "% Change": "% Chg",
    "Last Iteration Time": "Time",
    "Price_Lead_Status": "Lead",
    "Volatility State": "Vol State",
    "Option Symbol": "Opt Symbol",
}

# Safe float helpers
def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if frame is not None and col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    if frame is None:
        return pd.Series(dtype=float)
    return pd.Series(default, index=frame.index, dtype=float)


def safe_pd_timestamp(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df[col], errors="coerce").dropna()


# === Fyers helpers ===
def init_fyers():
    global fyers
    client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
    access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
    if not client_id or not access_token:
        logger.error("Missing CLIENT_ID / ACCESS_TOKEN; cannot init Fyers.")
        return None
    try:
        fyers = fyersModel.FyersModel(
            client_id=client_id, token=access_token, is_async=False, log_path=""
        )
    except Exception as e:
        logger.error(f"Failed to init FyersModel: {e}")
        fyers = None
    return fyers


def get_history(symbol: str, resolution: str, days_back: int) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    now = datetime.now()
    start = now - timedelta(days=days_back)
    payload = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start.strftime("%Y-%m-%d"),
        "range_to": now.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }
    try:
        res = fyers.history(data=payload)
    except Exception:
        return pd.DataFrame()
    candles = (res or {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(
        candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )
    return df.sort_values("timestamp").reset_index(drop=True)


# === OBV helper ===
def compute_obv(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    direction = np.sign(close.diff())
    obv = (vol * direction).cumsum()
    return round(safe_float(obv.iloc[-1], np.nan), 2)


def compute_today_obv(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    d = df.copy().sort_values("timestamp")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    today = pd.Timestamp.now(tz=None).date()
    today_df = d[d["timestamp"].dt.date == today].copy()
    if not today_df.empty:
        today_df = today_df[
            today_df["timestamp"] >= pd.Timestamp.combine(today, dtime(9, 15))
        ].copy()
        if len(today_df) >= 2:
            return compute_obv(today_df)
    return compute_obv(d)


# === CHAIN SIGNAL LOGIC (5-start + 8-of-11) ===
def read_iteration_history() -> Optional[pd.DataFrame]:
    """Read fo_iteration_history_*.csv if available."""
    history_dir = Path(OUTPUT_DIR).expanduser()
    csv_files = list(history_dir.glob("fo_iteration_history_*.csv"))
    if not csv_files:
        logger.info("No fo_iteration_history_*.csv found; falling back to old CSVs.")
        return None

    latest_csv = max(csv_files, key=os.path.getmtime)
    logger.info(f"INFO | Using iteration history {latest_csv.name} with window_signal column.")

    df = pd.read_csv(latest_csv, parse_dates=False)

    # Use timestamps col (or standardize)
    if c_timestamp not in df.columns:
        for alt in ["timestamp", "time"]:
            if alt in df.columns:
                df.rename(columns={alt: c_timestamp}, inplace=True)
                break
        else:
            logger.error(f"Column '{c_timestamp}' not found in iteration history CSV.")
            return None

    # Parse with explicit format
    df[c_timestamp] = pd.to_datetime(
        df[c_timestamp],
        format="%Y-%m-%d %H:%M:%S",  # adjust if needed
        errors="coerce",
    )

    # Drop rows where timestamp is NaT
    df = df.dropna(subset=[c_timestamp]).reset_index(drop=True)

    # Keep required cols; if any missing, skip
    keep_cols = [
        c_timestamp,
        "underlying",
        "option_type",
        "strike",
        "window_signal",
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns in iteration history: {missing}; skipping.")
        return None

    return df[keep_cols]


def check_chain(signals, timestamps) -> Tuple[int, Optional[datetime]]:
    """signals, timestamps: Series of 1=BUY, -1=SELL, 0=NEUTRAL."""
    signal_vals = signals.values
    time_vals = timestamps.values

    if len(signal_vals) == 0 or len(time_vals) == 0:
        return 0, None

    # Remove zeros (neutral)
    non_zero_signal = []
    non_zero_time = []
    for s, t in zip(signal_vals, time_vals):
        if s != 0:
            non_zero_signal.append(s)
            non_zero_time.append(t)

    if len(non_zero_signal) < 5:
        return 0, None

    # 1. Check 5 consecutive same direction
    last_5 = non_zero_signal[-5:]
    if not all(x == last_5[0] for x in last_5):
        return 0, None

    # 2. Check 8 of last 11 same direction
    last_11 = non_zero_signal[-11:]
    if len(last_11) < 8:
        return 0, None

    direction = last_11[0]
    same_count = sum(1 for x in last_11 if x == direction)

    if same_count >= 8:
        return direction, non_zero_time[-5]  # BUY / SELL and 5th‑start time
    return 0, None


def evaluate_chain_from_iteration(iter_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    ce_candidates = []
    pe_candidates = []

    for (underlying, opt_type), group in iter_df.groupby(["underlying", "option_type"]):
        signals = group["window_signal"]
        timestamps = group[c_timestamp]

        result, _ = check_chain(signals, timestamps)

        if result == +1:
            ce_candidates.append(underlying)
        elif result == -1:
            pe_candidates.append(underlying)

    return list(set(ce_candidates)), list(set(pe_candidates))


# === STATE PERSISTENCE (cepebuy_state.json) ===
DAILY_STATE_FILE = os.path.join(OUTPUT_DIR, "cepebuy_state.json")

def load_state() -> Dict[str, List[str]]:
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.isfile(DAILY_STATE_FILE):
        return {"date": today, "ce_buy": [], "pe_buy": []}

    try:
        with open(DAILY_STATE_FILE, "r") as f:
            state = json.load(f)
        if state.get("date") != today:
            logger.info("New day detected; resetting state.")
            return {"date": today, "ce_buy": [], "pe_buy": []}
        return state
    except Exception as e:
        logger.error(f"Error reading state {DAILY_STATE_FILE}: {e}")
        return {"date": today, "ce_buy": [], "pe_buy": []}


def save_state(state: Dict[str, List[str]]) -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    state["date"] = today
    with open(DAILY_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"State saved; {len(state['ce_buy'])} CE BUY, {len(state['pe_buy'])} PE BUY")


# === CANDIDATE CSV FALLBACK (safe) ===
def read_candidates() -> pd.DataFrame:
    long_csvs = list(Path(OUTPUT_DIR).glob("fo_long_candidates_*.csv"))
    if long_csvs:
        long_df = pd.read_csv(max(long_csvs, key=os.path.getmtime))
        long_df["candidate_type"] = "long"
    else:
        long_df = pd.DataFrame()

    short_csvs = list(Path(OUTPUT_DIR).glob("fo_short_candidates_*.csv"))
    if short_csvs:
        short_df = pd.read_csv(max(short_csvs, key=os.path.getmtime))
        short_df["candidate_type"] = "short"
    else:
        short_df = pd.DataFrame()

    if long_df.empty and short_df.empty:
        return pd.DataFrame()

    return pd.concat([long_df, short_df], ignore_index=True)


def update_state_from_candidates(state: Dict[str, List[str]], cand_df: pd.DataFrame) -> None:
    if cand_df.empty:
        logger.info("Candidate DataFrame empty; skipping fallback integration.")
        return

    required_cols = ["underlying", "option_type", "strike", "strike_diff"]
    missing = [c for c in required_cols if c not in cand_df.columns]
    if missing:
        logger.warning(
            f"Missing candidate cols {missing}; "
            "skipping fallback candidate integration from old CSVs."
        )
        return

    for (underlying, opt_type), group in cand_df.groupby(
        ["underlying", "option_type"]
    ):
        if group.empty:
            continue
        idx = group["strike_diff"].abs().idxmin()
        atm_strike = group.loc[idx, "strike"]
        # Just add underlying to state
        if opt_type == "CE":
            state["ce_buy"] = sorted(list(set(state["ce_buy"] + [underlying])))
        elif opt_type == "PE":
            state["pe_buy"] = sorted(list(set(state["pe_buy"] + [underlying])))


# === EMAIL ===
def build_html_table(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"<h3>{title}</h3><p>No stocks today.</p>"

    header = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, row in df.iterrows():
        tds = "".join(f"<td>{value}</td>" for value in row)
        rows += f"<tr>{tds}</tr>"
    return f"<h3>{title}</h3><table border='1'><tr>{header}</tr>{rows}</table>"


def build_html_body(ce_today: List[str], pe_today: List[str]) -> str:
    ce_count = len(ce_today)
    pe_count = len(pe_today)

    ce_df = pd.DataFrame({"CE BUY today": ce_today})
    pe_df = pd.DataFrame({"PE BUY today": pe_today})

    ce_table = build_html_table(ce_df, f"CE BUY ({ce_count} stocks today)")
    pe_table = build_html_table(pe_df, f"PE BUY ({pe_count} stocks today)")

    html_body = (
        "<html><body>"
        "<h2>CE / PE Momentum Buy Report</h2>"
        f"<p>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        "<p>Chain rules: 5‑start + 8‑of‑11 same‑direction signals. Mixed chains rejected.</p>"
        f"{ce_table}<br/>{pe_table}"
        "</body></html>"
    )
    return html_body


def send_email(subject: str, html_body: str) -> None:
    user = os.getenv("EMAIL_USER")
    pwd = os.getenv("EMAIL_PASSWORD")
    if not user or not pwd:
        logger.error("Missing EMAIL_USER / EMAIL_PASSWORD; skipping email.")
        return

    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = os.getenv("TO_EMAILS", "admin@example.com")

    part = MIMEText(html_body, "html")
    msg.attach(part)

    try:
        server = smtplib.SMTP(smtp_host, smtp_port)
        server.starttls()
        server.login(user, pwd)
        server.send_message(msg)
        server.quit()
        logger.info(f"Email sent to {msg['To']}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")


# === MAIN LOGIC ===
def main():
    init_fyers()

    # 1. Read iteration history (preferred)
    iter_df = read_iteration_history()
    if iter_df is None:
        cand_df = read_candidates()
        if cand_df.empty:
            logger.info("No candidates files found; exiting.")
            return
        state = load_state()
        update_state_from_candidates(state, cand_df)
        ce_today = state["ce_buy"]
        pe_today = state["pe_buy"]
    else:
        ce_iter, pe_iter = evaluate_chain_from_iteration(iter_df)
        state = load_state()
        # Add this day's iteration signals
        state["ce_buy"] = sorted(list(set(state["ce_buy"] + ce_iter)))
        state["pe_buy"] = sorted(list(set(state["pe_buy"] + pe_iter)))

        cand_df = read_candidates()
        update_state_from_candidates(state, cand_df)
        save_state(state)

        ce_today = ce_iter
        pe_today = pe_iter

    # 2. Build & send email
    html_body = build_html_body(ce_today, pe_today)
    subject = f"CE / PE Momentum Buy Report - {datetime.now().strftime('%d %b %H:%M')}"
    send_email(subject, html_body)


if __name__ == "__main__":
    main()

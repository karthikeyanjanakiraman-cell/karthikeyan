#!/usr/bin/env python3
"""
CEPEBUY.py — CE / PE Momentum Buy Email Sender with 5‑start / 8‑of‑11 chain logic.
Chain rules (applied independently per option symbol):
  1. MOMENTUM START  : 5 consecutive same-direction signals (all BUY or all SELL)
  2. FINAL SIGNAL    : 8 of the last 11 signals are in the same direction
  3. MIXED REJECTION : if chain is mixed (neither rule met), skip it
"""
import os
import sys
import json
import smtplib
import logging
from datetime import datetime, date
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# === CONFIG =========================================================================================

# user config
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
TO_EMAILS = os.getenv("TO_EMAILS", "admin@example.com").split(",")

# file paths
PROJECT_DIR = Path(__file__).parent
HISTORY_DIR = PROJECT_DIR
CEPEBUY_STATE_FILE = PROJECT_DIR / "cepebuy_state.json"
STATE_DATE = datetime.now().strftime("%Y-%m-%d")

# Column name for timestamp in CSV
c_timestamp = "timestamp"   # must match your CSV header

# === READING DATA ================================================================================

def read_iteration_history() -> Optional[pd.DataFrame]:
    """Read fo_iteration_history_*.csv if available."""
    csv_files = list(HISTORY_DIR.glob("fo_iteration_history_*.csv"))
    if not csv_files:
        logging.info("No fo_iteration_history_*.csv found; falling back to old candidate CSVs.")
        return None

    latest_csv = max(csv_files, key=os.path.getmtime)
    logging.info(f"INFO | Using iteration history {latest_csv.name} with window_signal column.")

    df = pd.read_csv(latest_csv, parse_dates=False)

    # Ensure c_timestamp is in the columns
    if c_timestamp not in df.columns:
        if "timestamp" in df.columns:
            # Use the default column name if it exists
            pass
        else:
            logging.error(f"Column {c_timestamp} not found in iteration history.")
            return None

    # Parse as datetime with explicit format to avoid UserWarning
    df[c_timestamp] = pd.to_datetime(
        df[c_timestamp],
        format="%Y-%m-%d %H:%M:%S",     # CHANGE if your CSV uses a different format
        errors="coerce"
    )

    # Drop rows where timestamp is NaT
    df = df.dropna(subset=[c_timestamp]).reset_index(drop=True)

    # Keep only required columns
    keep_cols = [
        c_timestamp,
        "underlying",
        "option_type",   # "CE" / "PE"
        "strike",
        "window_signal", # 1 for BUY, -1 for SELL, 0 for neutral
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        logging.error(f"Missing columns in fo_iteration_history: {missing}")
        return None

    return df[keep_cols]


def read_candidates() -> pd.DataFrame:
    """Read fo_long_candidates_*.csv and fo_short_candidates_*.csv."""
    long_csvs = list(HISTORY_DIR.glob("fo_long_candidates_*.csv"))
    if long_csvs:
        long_df = pd.read_csv(max(long_csvs, key=os.path.getmtime))
        long_df["candidate_type"] = "long"
    else:
        long_df = pd.DataFrame()

    short_csvs = list(HISTORY_DIR.glob("fo_short_candidates_*.csv"))
    if short_csvs:
        short_df = pd.read_csv(max(short_csvs, key=os.path.getmtime))
        short_df["candidate_type"] = "short"
    else:
        short_df = pd.DataFrame()

    if long_df.empty and short_df.empty:
        return pd.DataFrame()

    return pd.concat([long_df, short_df], ignore_index=True)


# === CHAIN LOGIC ===================================================================================

def check_chain(signals, timestamps) -> Tuple[int, Optional[datetime]]:
    """
    signals: Series of 1 (BUY), -1 (SELL), 0 (neutral)
    timestamps: Series of timestamps (parsed)
    Returns:
        (result, five_start_time)
        result: +1 (CE BUY), -1 (PE BUY), 0 (no signal)
    """
    # Convert to numpy arrays to avoid pandas ambiguity
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
        return 0, None   # 5‑start not met

    # 2. Check 8 of last 11 in same direction
    last_11 = non_zero_signal[-11:]
    if len(last_11) < 8:
        return 0, None

    direction = last_11[0]
    same_count = sum(1 for x in last_11 if x == direction)

    if same_count >= 8:
        return direction, non_zero_time[-5]   # BUY/SELL & 5th‑start time
    else:
        return 0, None


def evaluate_chain_from_iteration(iter_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    iter_df: has columns: timestamp, underlying, option_type, strike, window_signal
    Returns lists of underlying symbols that qualify for CE BUY / PE BUY.
    """
    ce_all = []   # underlying that qualify for CE BUY
    pe_all = []   # underlying that qualify for PE BUY

    # Group by (underlying, option_type)
    for (underlying, opt_type), group in iter_df.groupby(["underlying", "option_type"]):
        signals = group["window_signal"]
        timestamps = group[c_timestamp]

        result, five_start_time = check_chain(signals, timestamps)

        if result == +1:
            ce_all.append(underlying)
        elif result == -1:
            pe_all.append(underlying)

    return list(set(ce_all)), list(set(pe_all))


# === STATE (cepebuy_state.json) ====================================================================

def load_state() -> Dict[str, List[str]]:
    """
    Load or create cepebuy_state.json.
    Structure:
        {
            "date": "2026-05-15",
            "ce_buy": ["INFY", "SBIN"],
            "pe_buy": ["HDFC", "RELIANCE"]
        }
    """
    if not CEPEBUY_STATE_FILE.exists():
        return {
            "date": STATE_DATE,
            "ce_buy": [],
            "pe_buy": [],
        }

    try:
        with CEPEBUY_STATE_FILE.open("r") as f:
            state = json.load(f)
        if state.get("date") != STATE_DATE:
            # Reset if new day
            logging.info(f"New day detected: resetting state to {STATE_DATE}")
            return {
                "date": STATE_DATE,
                "ce_buy": [],
                "pe_buy": [],
            }
        else:
            return state
    except Exception as e:
        logging.error(f"Error loading state {CEPEBUY_STATE_FILE}: {e}")
        return {
            "date": STATE_DATE,
            "ce_buy": [],
            "pe_buy": [],
        }


def save_state(state: Dict[str, List[str]]) -> None:
    state["date"] = STATE_DATE
    with CEPEBUY_STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2)
    logging.info(f"State saved: {len(state['ce_buy'])} CE BUY, {len(state['pe_buy'])} PE BUY")


def update_state_from_iteration(
    ce_candidates: List[str], pe_candidates: List[str]
) -> Dict[str, List[str]]:
    """
    Merge today's iteration‑based candidates into state.
    Already qualified symbols stay; new ones are added.
    """
    state = load_state()
    ce_buy = set(state["ce_buy"]) | set(ce_candidates)
    pe_buy = set(state["pe_buy"]) | set(pe_candidates)
    state["ce_buy"] = sorted(ce_buy)
    state["pe_buy"] = sorted(pe_buy)
    save_state(state)
    return state


def update_state_from_candidates(state: Dict[str, List[str]], cand_df: pd.DataFrame) -> None:
    """
    Fill in any missing ATM‑strike candidates from old CSVs into state.
    """
    if cand_df.empty:
        return

    # Group by underlying and option_type, pick near‑ATM strike
    candidates = []
    for (underlying, opt_type), group in cand_df.groupby(["underlying", "option_type"]):
        atm_strike = group.loc[group["strike_diff"].abs().idxmin(), "strike"]
        candidates.append((underlying, opt_type, atm_strike))

    # Add to state if not already present
    ce_buy = set(state["ce_buy"])
    pe_buy = set(state["pe_buy"])

    for underlying, opt_type, _ in candidates:
        if opt_type == "CE":
            ce_buy.add(underlying)
        elif opt_type == "PE":
            pe_buy.add(underlying)

    state["ce_buy"] = sorted(ce_buy)
    state["pe_buy"] = sorted(pe_buy)
    save_state(state)


# === EMAIL BUILDING ===============================================================================

def build_html_table(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"<h3>{title}</h3><p>No stocks today.</p>"

    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    rows = ""
    for _, row in df.iterrows():
        tds = "<tr>"
        for value in row:
            tds += f"<td>{value}</td>"
        tds += "</tr>"
        rows += tds

    return f"<h3>{title}</h3><table border='1'>{header}{rows}</table>"


def build_html_body(
    ce_today: List[str], pe_today: List[str], ce_full: List[str], pe_full: List[str]
) -> str:
    ce_count = len(ce_today)
    pe_count = len(pe_today)

    # Build minimal tables (just symbols for now)
    ce_df = pd.DataFrame({"CE BUY today": ce_today})
    pe_df = pd.DataFrame({"PE BUY today": pe_today})

    ce_table = build_html_table(ce_df, f"CE BUY ({ce_count} stocks today)")
    pe_table = build_html_table(pe_df, f"PE BUY ({pe_count} stocks today)")

    html_body = (
        "<html><body>"
        "<h2>CE / PE Momentum Buy Report</h2>"
        f"<p>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
        "<p>Chain rules: 5‑start followed by 8‑of‑11 same‑direction signals. Mixed chains rejected.</p>"
        f"{ce_table}<br/>{pe_table}"
        "</body></html>"
    )

    return html_body


def send_email(subject: str, html_body: str) -> None:
    if not EMAIL_USER or not EMAIL_PASSWORD:
        logging.error("Missing EMAIL_USER or EMAIL_PASSWORD; skipping email.")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_USER
    msg["To"] = ", ".join(TO_EMAILS)

    part = MIMEText(html_body, "html")
    msg.attach(part)

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        logging.info(f"Email sent to {TO_EMAILS}")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


# === MAIN ENTRY ===================================================================================

def main() -> None:
    # Read iteration history (preferred)
    iter_df = read_iteration_history()
    if iter_df is None:
        # Fallback to old CSVs only
        cand_df = read_candidates()
        if cand_df.empty:
            logging.info("No candidates files found; exiting.")
            return

        # Just load existing state, don't add new logic from chain
        state = load_state()
        update_state_from_candidates(state, cand_df)
        ce_today = state["ce_buy"]
        pe_today = state["pe_buy"]
    else:
        # 1. Evaluate chains from iteration history
        ce_iter, pe_iter = evaluate_chain_from_iteration(iter_df)

        # 2. Update state with today's iteration‑based candidates
        state = update_state_from_iteration(ce_iter, pe_iter)

        # 3. Read candidates as fallback only (if needed)
        cand_df = read_candidates()
        update_state_from_candidates(state, cand_df)

        # 4. Today's qualified symbols (from iteration)
        ce_today = ce_iter
        pe_today = pe_iter

    # Build email body from today's qualified symbols only
    ce_all = state["ce_buy"]
    pe_all = state["pe_buy"]

    html_body = build_html_body(
        ce_today=ce_today,
        pe_today=pe_today,
        ce_full=ce_all,
        pe_full=pe_all,
    )

    subject = f"CE / PE Momentum Buy Report - {datetime.now().strftime('%d %b %H:%M')}"
    send_email(subject, html_body)


if __name__ == "__main__":
    main()

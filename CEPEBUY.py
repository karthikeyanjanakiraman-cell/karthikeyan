#!/usr/bin/env python3
"""
CEPEBUY.py — CE / PE Momentum Buy Email Sender with 5-start / 8-of-11 chain logic.

Chain rules (applied independently per option symbol):
  1. MOMENTUM START  : 5 consecutive same-direction signals (all BUY or all SELL)
  2. FINAL SIGNAL    : 8 of the last 11 signals are in the same direction
  3. MIXED REJECTION : if either CE or PE chain is mixed (neither rule met), skip it

A symbol qualifies for CE BUY when its CE chain meets the above.
A symbol qualifies for PE BUY when its PE chain meets the above.

Once qualified during the day, the symbol stays in all subsequent emails
(persisted in a daily JSON state file).

CSV sources (from OPTIONS_OI.py):
  fo_iteration_history_*.csv  — per-symbol per-timestamp signals
  fo_long_candidates_*.csv    — fallback if iteration history is unavailable
  fo_short_candidates_*.csv   — fallback if iteration history is unavailable
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SENDER_EMAIL    = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")
SMTP_HOST       = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.environ.get("SMTP_PORT", "587"))

# Chain rules
CONSEC_START    = int(os.environ.get("CONSEC_START", "5"))   # 5 consecutive
CONFIRM_OF      = int(os.environ.get("CONFIRM_OF", "8"))     # 8 of last...
CONFIRM_WINDOW  = int(os.environ.get("CONFIRM_WINDOW", "11")) # ...11 signals

# Daily state file (persists across runs within the same calendar day)
STATE_FILE = os.environ.get("CEPE_STATE_FILE", "/tmp/cepebuy_state.json")


# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------
def _search_roots() -> set:
    roots = set()
    roots.add(Path.cwd().resolve())
    roots.add((Path.cwd() / "output").resolve())
    try:
        sd = Path(__file__).parent.resolve()
        roots.add(sd)
        roots.add((sd / "output").resolve())
        for p in sd.parents[:4]:
            roots.add(p.resolve())
            roots.add((p / "output").resolve())
    except Exception:
        pass
    for ev in ("GITHUB_WORKSPACE", "RUNNER_WORKSPACE", "HOME", "CI_WORKSPACE"):
        v = os.environ.get(ev)
        if v:
            roots.add(Path(v).resolve())
            roots.add((Path(v) / "output").resolve())
    for fb in ("/github/workspace", "/home/runner/work", "/workspace", "/mnt/data", "/tmp"):
        if os.path.isdir(fb):
            roots.add(Path(fb).resolve())
    return roots

def _find_latest(pattern: str) -> Optional[str]:
    matches = []
    seen = set()
    for root in _search_roots():
        if not root.exists():
            continue
        for match in root.rglob(pattern):
            ap = str(match.resolve())
            if ap in seen:
                continue
            seen.add(ap)
            try:
                matches.append((os.path.getmtime(ap), ap))
            except OSError:
                pass
    if not matches:
        return None
    matches.sort(reverse=True, key=lambda x: x[0])
    best = matches[0][1]
    logger.info("Selected %s (mtime=%s)",
                best,
                datetime.fromtimestamp(matches[0][0]).strftime("%Y-%m-%d %H:%M:%S"))
    return best

def _load_any(patterns: List[str]) -> pd.DataFrame:
    for pat in patterns:
        path = _find_latest(pat)
        if not path:
            continue
        try:
            df = pd.read_csv(path)
            logger.info("Loaded %s: %d rows x %d cols", os.path.basename(path), len(df), len(df.columns))
            return df
        except Exception as exc:
            logger.exception("Failed to read %s: %s", path, exc)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Daily state persistence
# ---------------------------------------------------------------------------
def _load_state() -> Dict:
    today = str(date.today())
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            if state.get("date") == today:
                logger.info("Loaded daily state: %d CE, %d PE already qualified",
                            len(state.get("ce", {})), len(state.get("pe", {})))
                return state
    except Exception:
        pass
    return {"date": today, "ce": {}, "pe": {}}

def _save_state(state: Dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Could not save state: %s", exc)


# ---------------------------------------------------------------------------
# 5-start / 8-of-11 chain logic
# ---------------------------------------------------------------------------
def _direction(signal_str: str) -> int:
    """Convert a signal string to +1 (bullish) or -1 (bearish) or 0 (neutral)."""
    s = str(signal_str).upper()
    if any(k in s for k in ("BUY", "BULL", "ENTER", "CONFIRM", "LONG")):
        return 1
    if any(k in s for k in ("SELL", "BEAR", "EXIT", "SHORT")):
        return -1
    return 0

def check_chain(signals: List[str]) -> Optional[int]:
    """
    Apply the 5-start / 8-of-11 chain rules to a list of signal strings.

    Returns:
      +1  if CE BUY condition met (bullish momentum confirmed)
      -1  if PE BUY condition met (bearish momentum confirmed)
       0  if mixed / not enough data — REJECT
    """
    if not signals:
        return 0

    dirs = [_direction(s) for s in signals]
    non_zero = [d for d in dirs if d != 0]
    if len(non_zero) < max(CONSEC_START, CONFIRM_WINDOW):
        return 0  # not enough data yet

    # Rule 1: 5 consecutive same direction in the LAST signals
    recent = non_zero[-CONSEC_START:]
    if len(set(recent)) == 1 and recent[0] != 0:
        direction = recent[0]
        # Rule 2: 8 of last 11 in the same direction
        last11 = non_zero[-CONFIRM_WINDOW:]
        count_same = sum(1 for d in last11 if d == direction)
        if count_same >= CONFIRM_OF:
            return direction  # +1 bullish, -1 bearish

    # Mixed chain — reject
    return 0

def evaluate_chain_from_iteration(iter_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Scan iteration history and apply 5-start / 8-of-11 per (Underlying, Option Type).

    Returns:
        ce_qualify  : {Underlying: latest_row_dict}  for confirmed CE BUY
        pe_qualify  : {Underlying: latest_row_dict}  for confirmed PE BUY
    """
    ce_qualify: Dict = {}
    pe_qualify: Dict = {}

    if iter_df.empty:
        return ce_qualify, pe_qualify

    cols = {c.lower().strip(): c for c in iter_df.columns}

    def _col(cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    c_underlying = _col(["underlying", "symbol", "stock"])
    c_opt_type   = _col(["option type", "option_type", "otype", "type"])
    c_signal     = _col(["window_signal", "window signal", "signal", "5m_signal", "chain_signal", "chain signal"])
    c_timestamp  = _col(["timestamp", "time", "datetime", "ts"])

    if any(v is None for v in (c_underlying, c_opt_type, c_signal)):
        logger.warning("Iteration CSV missing required columns. Available: %s", list(iter_df.columns))
        return ce_qualify, pe_qualify

    # Sort by timestamp if available
    if c_timestamp:
        try:
            iter_df = iter_df.copy()
            iter_df[c_timestamp] = pd.to_datetime(iter_df[c_timestamp], errors="coerce")
            iter_df = iter_df.sort_values(c_timestamp)
        except Exception:
            pass

    for (underlying, opt_type), grp in iter_df.groupby([c_underlying, c_opt_type]):
        signals = grp[c_signal].tolist()
        result = check_chain(signals)
        latest_row = grp.iloc[-1].to_dict()

        if result == 1 and str(opt_type).upper() == "CE":
            ce_qualify[underlying] = latest_row
        elif result == -1 and str(opt_type).upper() == "PE":
            pe_qualify[underlying] = latest_row

    logger.info("Chain evaluation: CE qualify=%d | PE qualify=%d", len(ce_qualify), len(pe_qualify))
    return ce_qualify, pe_qualify

def evaluate_chain_from_candidates(long_df: pd.DataFrame, short_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Fallback when iteration history is not available.
    Uses the ENTER + OK HOLD filter on pre-computed candidate CSVs.
    """
    ce_qualify: Dict = {}
    pe_qualify: Dict = {}

    combined = pd.concat([long_df, short_df], ignore_index=True) if (not long_df.empty or not short_df.empty) else pd.DataFrame()
    if combined.empty:
        return ce_qualify, pe_qualify

    cols = {c.lower().strip(): c for c in combined.columns}

    def _col(cands):
        for c in cands:
            if c.lower() in cols:
                return cols[c.lower()]
        return None

    u  = _col(["underlying", "symbol", "stock"])
    t  = _col(["option type", "option_type", "otype", "type"])
    cs = _col(["chain signal", "chain_signal", "signal"])
    es = _col(["exit signal", "exit_signal", "exit"])

    if any(v is None for v in (u, t, cs, es)):
        logger.warning("Candidates CSV missing columns. Available: %s", list(combined.columns))
        return ce_qualify, pe_qualify

    mask = (
        combined[cs].astype(str).str.contains("ENTER", case=False, na=False) &
        combined[es].astype(str).str.contains("OK HOLD", case=False, na=False)
    )
    filt = combined[mask].copy()
    if filt.empty:
        return ce_qualify, pe_qualify

    for _, row in filt.iterrows():
        underlying = row[u]
        opt_type = str(row[t]).upper()
        if opt_type == "CE" and underlying not in ce_qualify:
            ce_qualify[underlying] = row.to_dict()
        elif opt_type == "PE" and underlying not in pe_qualify:
            pe_qualify[underlying] = row.to_dict()

    logger.info("Fallback candidate filter: CE=%d | PE=%d", len(ce_qualify), len(pe_qualify))
    return ce_qualify, pe_qualify


# ---------------------------------------------------------------------------
# Merge with daily state (persist once qualified)
# ---------------------------------------------------------------------------
def merge_with_state(state: Dict, ce_qualify: Dict, pe_qualify: Dict) -> Tuple[Dict, Dict]:
    """
    New qualifications are added to state.
    Existing state qualifications are always kept (once in = stays all day).
    """
    # Add any new ones
    for sym, row in ce_qualify.items():
        state["ce"][sym] = row  # overwrite with latest row data

    for sym, row in pe_qualify.items():
        state["pe"][sym] = row

    logger.info("After state merge: CE=%d | PE=%d (cumulative today)", len(state["ce"]), len(state["pe"]))
    return state["ce"], state["pe"]


# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------
def _build_table(rows_dict: Dict, title: str) -> str:
    if not rows_dict:
        return f"<h3>{title}</h3><p>No signals.</p>"

    df = pd.DataFrame(list(rows_dict.values()))
    html = [
        f"<h3>{title}</h3>",
        '<table border="1" cellspacing="0" cellpadding="4" style="border-collapse:collapse;">',
        "<tr style='background:#f0f0f0;'>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>",
    ]
    for _, r in df.iterrows():
        html.append(f"<tr>" + "".join(f"<td>{r[c]}</td>" for c in df.columns) + "</tr>")
    html.append("</table>")
    return "
".join(html)

def send_email(subject: str, html: str) -> bool:
    if not all((SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL)):
        logger.warning("Email credentials missing; skipping.")
        return False

    recipients = [a.strip() for a in RECIPIENT_EMAIL.replace(";", ",").split(",") if a.strip()]
    if not recipients:
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = ",".join(recipients)
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        logger.info("Connecting %s:%d ...", SMTP_HOST, SMTP_PORT)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.starttls()
            s.login(SENDER_EMAIL, SENDER_PASSWORD)
            s.sendmail(SENDER_EMAIL, recipients, msg.as_string())
        logger.info("Email sent: %s", subject)
        return True
    except Exception as exc:
        logger.exception("SMTP failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("=== CEPEBUY starting (5-start / 8-of-11 chain logic) ===")

    # Load daily state (persists qualified symbols across runs)
    state = _load_state()

    # --- Primary path: use iteration history for full chain evaluation ---
    iter_df = _load_any([
        "fo_iteration_history_*.csv",
        "iteration_history_*.csv",
        "fo_iteration_history.csv",
        "iteration_history.csv",
    ])

    if not iter_df.empty:
        logger.info("Using iteration history for 5-start / 8-of-11 chain evaluation.")
        ce_qualify, pe_qualify = evaluate_chain_from_iteration(iter_df)
    else:
        # --- Fallback: use pre-filtered candidate CSVs ---
        logger.warning("Iteration history not found. Falling back to candidate CSVs (ENTER + OK HOLD filter).")
        long_df = _load_any([
            "fo_long_candidates_*.csv", "long_candidates_*.csv",
            "fo_long_candidates.csv", "long_candidates.csv",
        ])
        short_df = _load_any([
            "fo_short_candidates_*.csv", "short_candidates_*.csv",
            "fo_short_candidates.csv", "short_candidates.csv",
        ])
        if long_df.empty and short_df.empty:
            logger.error("No data source available. Abort.")
            sys.exit(1)
        ce_qualify, pe_qualify = evaluate_chain_from_candidates(long_df, short_df)

    # Merge new qualifications into daily state (once qualified = stays all day)
    ce_today, pe_today = merge_with_state(state, ce_qualify, pe_qualify)

    # Save updated state
    _save_state(state)

    if not ce_today and not pe_today:
        logger.info("No CE/PE BUY signals today. No email sent.")
        return

    html = (
        "<html><body>"
        f"<h2>CE / PE Momentum Buy Report &mdash; {datetime.now().strftime('%d %b %Y %H:%M')}</h2>"
        f"<p>Chain rule: {CONSEC_START} consecutive + {CONFIRM_OF} of {CONFIRM_WINDOW}. Mixed chains rejected.</p>"
        + _build_table(ce_today, f"CE BUY ({len(ce_today)} stocks)")
        + "<br/>"
        + _build_table(pe_today, f"PE BUY ({len(pe_today)} stocks)")
        + "</body></html>"
    )

    subject = f"CE / PE Momentum Buy Report - {datetime.now().strftime('%d %b %H:%M')}"
    send_email(subject, html)


if __name__ == "__main__":
    main()

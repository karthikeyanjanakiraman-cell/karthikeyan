#!/usr/bin/env python3
"""
CEPEBUY.py â€” CE / PE Momentum Buy Email Sender
Companion for OPTIONS_OI.py. Finds the latest timestamped CSVs and sends
the CE BUY / PE BUY momentum email.

Expected CSVs (written by OPTIONS_OI.py):
  fo_long_candidates_YYYYMMDD_HHMM.csv
  fo_short_candidates_YYYYMMDD_HHMM.csv
"""

import os
import sys
import smtplib
import logging
from datetime import datetime
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Tuple, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Email config
# ---------------------------------------------------------------------------
SENDER_EMAIL    = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")
SMTP_HOST       = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.environ.get("SMTP_PORT", "587"))

# ---------------------------------------------------------------------------
# CSV discovery
# ---------------------------------------------------------------------------

def _search_roots() -> set:
    """Return a set of Path roots to search for CSVs."""
    roots = set()

    # CWD + output/
    roots.add(Path.cwd().resolve())
    roots.add((Path.cwd() / "output").resolve())

    # This script's directory + parents + output/
    try:
        sd = Path(__file__).parent.resolve()
        roots.add(sd)
        roots.add((sd / "output").resolve())
        for p in sd.parents[:4]:
            roots.add(p.resolve())
            roots.add((p / "output").resolve())
    except Exception:
        pass

    # CI / GitHub Actions env vars
    for ev in ("GITHUB_WORKSPACE", "RUNNER_WORKSPACE", "HOME", "CI_WORKSPACE"):
        v = os.environ.get(ev)
        if v:
            roots.add(Path(v).resolve())
            roots.add((Path(v) / "output").resolve())

    # Common hardcoded paths
    for fallback in ("/github/workspace", "/home/runner/work", "/workspace", "/mnt/data", "/tmp"):
        if os.path.isdir(fallback):
            roots.add(Path(fallback).resolve())

    return roots


def _find_latest(pattern: str) -> Optional[str]:
    """Recursively search roots for `pattern` and return the most recent match by mtime."""
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
        logger.warning("No files matched pattern: %s", pattern)
        return None

    matches.sort(reverse=True, key=lambda x: x[0])
    best = matches[0][1]
    logger.info(
        "Selected %s (mtime=%s)",
        best,
        datetime.fromtimestamp(matches[0][0]).strftime("%Y-%m-%d %H:%M:%S"),
    )
    return best


def _load_any(patterns) -> pd.DataFrame:
    """Try each pattern in order; return DataFrame from the first hit."""
    for pat in patterns:
        path = _find_latest(pat)
        if not path:
            continue
        try:
            df = pd.read_csv(path)
            logger.info(
                "Loaded %s: %d rows x %d cols",
                os.path.basename(path),
                len(df),
                len(df.columns),
            )
            return df
        except Exception as exc:
            logger.exception("Failed to read %s: %s", path, exc)
            continue
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Signal logic
# ---------------------------------------------------------------------------

def build_ce_pe_buy_rows(long_df: pd.DataFrame, short_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ce_buy, pe_buy) from long+short candidate DataFrames.

    Keeps rows where:
      - Chain Signal contains "ENTER"
      - Exit Signal contains "OK HOLD"
    Collapses to one row per Underlying (one stock = one row).
    """
    if long_df is None:
        long_df = pd.DataFrame()
    if short_df is None:
        short_df = pd.DataFrame()

    combined = pd.concat([long_df, short_df], ignore_index=True) if (not long_df.empty or not short_df.empty) else pd.DataFrame()
    if combined.empty:
        return pd.DataFrame(), pd.DataFrame()

    cols = {c.lower().strip(): c for c in combined.columns}

    def _col(candidates):
        for c in candidates:
            key = c.lower()
            if key in cols:
                return cols[key]
        return None

    u  = _col(["underlying", "symbol", "stock"])
    t  = _col(["option type", "option_type", "otype", "type"])
    cs = _col(["chain signal", "chain_signal", "signal"])
    es = _col(["exit signal", "exit_signal", "exit"])

    if any(v is None for v in (u, t, cs, es)):
        logger.warning("Missing columns. Available: %s", list(combined.columns))
        return pd.DataFrame(), pd.DataFrame()

    mask = (
        combined[cs].astype(str).str.contains("ENTER", case=False, na=False) &
        combined[es].astype(str).str.contains("OK HOLD", case=False, na=False)
    )
    filt = combined[mask].copy()
    if filt.empty:
        return pd.DataFrame(), pd.DataFrame()

    ot = filt[t].astype(str).str.upper()
    ce = filt[ot == "CE"].copy()
    pe = filt[ot == "PE"].copy()

    for df_side in (ce, pe):
        if not df_side.empty:
            df_side.sort_values(by=[u], inplace=True)
            df_side.drop_duplicates(subset=[u], keep="first", inplace=True)

    logger.info("CE BUY: %d | PE BUY: %d", len(ce), len(pe))
    return ce, pe


# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------

def _build_table(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"<h3>{title}</h3><p>No signals.</p>"
    rows = [
        f"<h3>{title}</h3>",
        '<table border="1" cellspacing="0" cellpadding="4">',
        "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>",
    ]
    for _, r in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{r[c]}</td>" for c in df.columns) + "</tr>")
    rows.append("</table>")
    return "\n".join(rows)


def send_email(subject: str, html: str) -> bool:
    if not all((SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL)):
        logger.warning("Email credentials missing; skipping.")
        return False

    recipients = [a.strip() for a in RECIPIENT_EMAIL.replace(";", ",").split(",") if a.strip()]
    if not recipients:
        logger.warning("No recipients parsed; skipping.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = ",".join(recipients)
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        logger.info("Connecting %s:%d â€¦", SMTP_HOST, SMTP_PORT)
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
    logger.info("=== CEPEBUY starting ===")

    long_df = _load_any([
        "fo_long_candidates_*.csv",
        "long_candidates_*.csv",
        "fo_long_candidates.csv",
        "long_candidates.csv",
    ])

    short_df = _load_any([
        "fo_short_candidates_*.csv",
        "short_candidates_*.csv",
        "fo_short_candidates.csv",
        "short_candidates.csv",
    ])

    if long_df.empty and short_df.empty:
        logger.error("Both long and short CSVs missing/empty. Abort.")
        sys.exit(1)

    ce_df, pe_df = build_ce_pe_buy_rows(long_df, short_df)
    if ce_df.empty and pe_df.empty:
        logger.info("No CE/PE BUY signals today.")
        return

    html = (
        "<html><body>"
        f"<h2>CE / PE Momentum Buy Report â€” {datetime.now().strftime('%d %b %Y %H:%M')}</h2>"
        + _build_table(ce_df, "CE BUY Candidates")
        + "<br/>"
        + _build_table(pe_df, "PE BUY Candidates")
        + "</body></html>"
    )

    subject = f"CE / PE Momentum Buy Report - {datetime.now().strftime('%d %b %H:%M')}"
    send_email(subject, html)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CE/PE Momentum Buy Email Sender â€” standalone companion for OPTIONS_OI.py

Usage:
    python OPTIONS_OI_cepe_momentum2.py

Environment variables:
    CSV_DIR          â€” directory where fo_long_candidates.csv / fo_short_candidates.csv live
    SENDER_EMAIL     â€” SMTP login address
    SENDER_PASSWORD  â€” SMTP password / app password
    RECIPIENT_EMAIL  â€” comma-separated recipient list
"""

import os
import sys
import smtplib
import glob
import logging
from datetime import datetime
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from typing import Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Accept overrides from environment; fallback filenames match OPTIONS_OI.py defaults
LONG_CSV = os.environ.get("LONG_CSV", "fo_long_candidates.csv")
SHORT_CSV = os.environ.get("SHORT_CSV", "fo_short_candidates.csv")

SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))


# -----------------------------------------------------------------------------
# CSV Discovery
# -----------------------------------------------------------------------------

def find_csv(name: str) -> str | None:
    """Look for a CSV in multiple places so the user doesn't have to babysit paths."""

    # 1. Exact env override or current working directory
    candidates = [name]

    # 2. If CSV_DIR is set, look there
    csv_dir = os.environ.get("CSV_DIR")
    if csv_dir:
        candidates.append(os.path.join(csv_dir, name))

    # 3. Look in the same directory as THIS script
    script_dir = Path(__file__).parent.resolve()
    candidates.append(str(script_dir / name))

    # 4. Search up to 3 parent directories (common in GitHub Actions / repo layouts)
    for level in range(1, 4):
        candidates.append(str(script_dir.parents[level - 1] / name))

    # 5. Try a loose glob in home / repo area (last resort)
    home = Path.home()
    glob_matches = list(home.rglob(name))
    if not glob_matches:
        glob_matches = list(Path("/workspace").rglob(name)) if os.path.exists("/workspace") else []
    if not glob_matches:
        glob_matches = list(Path("/github/workspace").rglob(name)) if os.path.exists("/github/workspace") else []

    for path in candidates:
        if os.path.isfile(path):
            logger.info("Found %s at: %s", name, os.path.abspath(path))
            return path

    if glob_matches:
        best = str(glob_matches[0])
        logger.info("Found %s via search at: %s", name, best)
        return best

    logger.warning("Could not find %s. Searched:\n  %s", name, "\n  ".join(candidates))
    return None


def load_csv(name: str) -> pd.DataFrame:
    path = find_csv(name)
    if not path:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        logger.info("Loaded %s: %d rows x %d cols", os.path.basename(path), len(df), len(df.columns))
        return df
    except Exception as exc:
        logger.exception("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# CE/PE BUY Logic
# -----------------------------------------------------------------------------

def build_ce_pe_buy_rows(long_df: pd.DataFrame, short_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Derive CE BUY / PE BUY rows from already-filtered long/short candidates.

    Rules:
      - Chain Signal contains "ENTER"
      - Exit Signal contains "OK HOLD"
      - Collapse to one row per Underlying (one stock one row)
    """
    if long_df is None:
        long_df = pd.DataFrame()
    if short_df is None:
        short_df = pd.DataFrame()

    combined = pd.concat([long_df, short_df], ignore_index=True) if (not long_df.empty or not short_df.empty) else pd.DataFrame()
    if combined.empty:
        logger.info("No long/short candidates; CE/PE BUY universe is empty.")
        return pd.DataFrame(), pd.DataFrame()

    # Flexible column lookup
    cols = {c.lower().strip(): c for c in combined.columns}

    def get_col(candidates):
        for cand in candidates:
            key = cand.lower()
            if key in cols:
                return cols[key]
        return None

    col_underlying = get_col(["underlying", "symbol", "stock", "scrip"])
    col_opt_type   = get_col(["option type", "option_type", "otype", "type"])
    col_chain_sig  = get_col(["chain signal", "chain_signal", "signal"])
    col_exit_sig   = get_col(["exit signal", "exit_signal", "exit"])

    missing = [n for n, c in [("Underlying", col_underlying), ("Option Type", col_opt_type),
                               ("Chain Signal", col_chain_sig), ("Exit Signal", col_exit_sig)] if c is None]
    if missing:
        logger.warning("Missing columns in candidates CSVs: %s | Available columns: %s",
                       ", ".join(missing), list(combined.columns))
        return pd.DataFrame(), pd.DataFrame()

    mask_enter = combined[col_chain_sig].astype(str).str.contains("ENTER", case=False, na=False)
    mask_hold  = combined[col_exit_sig].astype(str).str.contains("OK HOLD", case=False, na=False)
    filtered = combined[mask_enter & mask_hold].copy()

    if filtered.empty:
        logger.info("No rows satisfy ENTER + OK HOLD.")
        return pd.DataFrame(), pd.DataFrame()

    opt_type_series = filtered[col_opt_type].astype(str).str.upper()
    ce_df = filtered[opt_type_series == "CE"].copy()
    pe_df = filtered[opt_type_series == "PE"].copy()

    if not ce_df.empty:
        ce_df = ce_df.sort_values(by=[col_underlying])
        ce_df = ce_df.drop_duplicates(subset=[col_underlying], keep="first")
    if not pe_df.empty:
        pe_df = pe_df.sort_values(by=[col_underlying])
        pe_df = pe_df.drop_duplicates(subset=[col_underlying], keep="first")

    logger.info("CE BUY: %d | PE BUY: %d", len(ce_df), len(pe_df))
    return ce_df, pe_df


# -----------------------------------------------------------------------------
# Email Building
# -----------------------------------------------------------------------------

def build_table_html(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"<h3>{title}</h3><p>No signals.</p>"
    html = [f"<h3>{title}</h3>", '<table border="1" cellspacing="0" cellpadding="4">']
    html.append("<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>")
    for _, row in df.iterrows():
        html.append("<tr>" + "".join(f"<td>{row[c]}</td>" for c in df.columns) + "</tr>")
    html.append("</table>")
    return "\n".join(html)


def build_ce_pe_buy_email_html(ce_df: pd.DataFrame, pe_df: pd.DataFrame) -> str:
    parts = [
        "<html><body>",
        f"<h2>CE / PE Momentum Buy Report &mdash; {datetime.now().strftime('%d %b %Y %H:%M')}</h2>",
    ]
    parts.append(build_table_html(ce_df, "CE BUY Candidates"))
    parts.append("<br/>")
    parts.append(build_table_html(pe_df, "PE BUY Candidates"))
    parts.append("</body></html>")
    return "\n".join(parts)


def send_single_email(subject: str, html_body: str) -> bool:
    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
        logger.warning("Email credentials/recipients not configured; skipping CE/PE email.")
        return False

    recipients = [addr.strip() for addr in RECIPIENT_EMAIL.replace(";", ",").split(",") if addr.strip()]
    if not recipients:
        logger.warning("Recipient list empty after parsing; skipping CE/PE email.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = ",".join(recipients)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        logger.info("Connecting SMTP %s:%d as %s", SMTP_HOST, SMTP_PORT, SENDER_EMAIL)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, recipients, msg.as_string())
        logger.info("Email sent: CE / PE Momentum Buy Report - %s", datetime.now().strftime("%d %b %H:%M"))
        return True
    except Exception as exc:
        logger.exception("Failed to send CE/PE email: %s", exc)
        return False


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    logger.info("Starting CE/PE momentum email from CSV candidates.")
    long_df = load_csv(LONG_CSV)
    short_df = load_csv(SHORT_CSV)

    if long_df.empty and short_df.empty:
        logger.error("Both long and short CSVs are missing or empty. Cannot proceed.")
        sys.exit(1)

    ce_df, pe_df = build_ce_pe_buy_rows(long_df, short_df)
    if ce_df.empty and pe_df.empty:
        logger.info("No CE/PE BUY rows; nothing to email.")
        return

    html_body = build_ce_pe_buy_email_html(ce_df, pe_df)
    subject = f"CE / PE Momentum Buy Report - {datetime.now().strftime('%d %b %H:%M')}"
    send_single_email(subject, html_body)


if __name__ == "__main__":
    main()

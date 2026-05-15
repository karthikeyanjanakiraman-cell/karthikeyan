#!/usr/bin/env python3
import os
import smtplib
import logging
from datetime import datetime
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
LONG_CSV = os.environ.get("LONG_CSV", "fo_long_candidates.csv")
SHORT_CSV = os.environ.get("SHORT_CSV", "fo_short_candidates.csv")

# SMTP / email settings – same names you already use in the main script
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning("CSV not found: %s", path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        logger.info("Loaded %s: %d rows", path, len(df))
        return df
    except Exception as exc:
        logger.exception("Failed to read %s: %s", path, exc)
        return pd.DataFrame()


def build_ce_pe_buy_rows(long_df: pd.DataFrame, short_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build CE BUY and PE BUY rows from long/short candidate CSVs.

    This does NOT change your existing pipeline. It simply reads the
    already-filtered long/short candidates and applies a clean CE/PE
    BUY rule:
      - Chain Signal contains "ENTER"
      - Exit Signal contains "OK HOLD" (still in trend)

    It then keeps only one row per Underlying (first row), which matches
    your "one stock one row" requirement based on the current candidates.
    """

    if long_df is None:
        long_df = pd.DataFrame()
    if short_df is None:
        short_df = pd.DataFrame()

    combined = pd.concat([long_df, short_df], ignore_index=True) if (not long_df.empty or not short_df.empty) else pd.DataFrame()
    if combined.empty:
        logger.info("No long/short candidates; CE/PE BUY universe is empty.")
        return pd.DataFrame(), pd.DataFrame()

    # Normalise column names a bit for robustness
    cols = {c.lower().strip(): c for c in combined.columns}

    def get_col(name_candidates):
        for cand in name_candidates:
            key = cand.lower()
            if key in cols:
                return cols[key]
        return None

    col_underlying = get_col(["underlying", "symbol", "stock"])
    col_opt_type = get_col(["option type", "option_type", "otype"])
    col_chain_signal = get_col(["chain signal", "chain_signal"])
    col_exit_signal = get_col(["exit signal", "exit_signal"])

    missing = [
        name
        for name, col in [
            ("Underlying", col_underlying),
            ("Option Type", col_opt_type),
            ("Chain Signal", col_chain_signal),
            ("Exit Signal", col_exit_signal),
        ]
        if col is None
    ]
    if missing:
        logger.warning("Missing required columns in candidates CSVs: %s", ", ".join(missing))
        return pd.DataFrame(), pd.DataFrame()

    # Filter for strong in-trend entries: ENTER + OK HOLD like your chain email
    mask_enter = combined[col_chain_signal].astype(str).str.contains("ENTER", case=False, na=False)
    mask_hold = combined[col_exit_signal].astype(str).str.contains("OK HOLD", case=False, na=False)
    filtered = combined[mask_enter & mask_hold].copy()

    if filtered.empty:
        logger.info("No rows satisfy ENTER + OK HOLD; no CE/PE BUY signals.")
        return pd.DataFrame(), pd.DataFrame()

    # Split CE vs PE
    opt_type_series = filtered[col_opt_type].astype(str).str.upper()
    ce_df = filtered[opt_type_series == "CE"].copy()
    pe_df = filtered[opt_type_series == "PE"].copy()

    # One row per underlying (nearest ATM etc. already handled upstream)
    if not ce_df.empty:
        ce_df = ce_df.sort_values(by=[col_underlying])
        ce_df = ce_df.drop_duplicates(subset=[col_underlying], keep="first")
    if not pe_df.empty:
        pe_df = pe_df.sort_values(by=[col_underlying])
        pe_df = pe_df.drop_duplicates(subset=[col_underlying], keep="first")

    logger.info("CE BUY: %d | PE BUY: %d", len(ce_df), len(pe_df))

    return ce_df, pe_df


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
        f"<h2>CE / PE Momentum Buy Report - {datetime.now().strftime('%d %b %Y %H:%M')}</h2>",
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
        logger.warning("Recipient list is empty after parsing; skipping CE/PE email.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = ",".join(recipients)

    part_html = MIMEText(html_body, "html", "utf-8")
    msg.attach(part_html)

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


def main() -> None:
    logger.info("Starting CE/PE momentum email from CSV candidates.")
    long_df = load_csv(LONG_CSV)
    short_df = load_csv(SHORT_CSV)

    ce_df, pe_df = build_ce_pe_buy_rows(long_df, short_df)
    if ce_df.empty and pe_df.empty:
        logger.info("No CE/PE BUY rows; nothing to email.")
        return

    html_body = build_ce_pe_buy_email_html(ce_df, pe_df)
    subject = f"CE / PE Momentum Buy Report - {datetime.now().strftime('%d %b %H:%M')}"
    send_single_email(subject, html_body)


if __name__ == "__main__":
    main()

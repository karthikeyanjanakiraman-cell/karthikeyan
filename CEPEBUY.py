# Optional FYERS/OPTIONS_OI integration module for CEPEBUY rank-only workflow
import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Local imports only; no missing symbols assumed
try:
    import OPTIONS_OI as opt
except Exception as e:
    opt = None
    logger.warning("OPTIONS_OI unavailable: %s", e)


def _safe_getattr(name, default=None):
    return getattr(opt, name, default) if opt is not None else default


def get_fyers_client():
    initfyers = _safe_getattr("initfyers")
    if callable(initfyers):
        try:
            return initfyers()
        except Exception as e:
            logger.warning("initfyers failed: %s", e)
    return None


def normalize_symbol(raw: str) -> str:
    formateqsymbol = _safe_getattr("formateqsymbol", lambda x: f"NSE:{str(x).strip().upper()}-EQ")
    s = str(raw).strip().upper()
    if s.startswith("NSE:NSE:"):
        s = s.replace("NSE:NSE:", "NSE:", 1)
    if s.endswith("-EQ-EQ"):
        s = s[:-3]
    return s if s.startswith("NSE:") else formateqsymbol(s)


def extract_chain(resp) -> list:
    if not isinstance(resp, dict):
        return []
    data = resp.get("data", resp)
    if isinstance(data, dict):
        for key in ("optionsChain", "optionChain", "chain", "contracts", "data"):
            if key in data:
                data = data[key]
                break
    return data if isinstance(data, list) else []


def fetch_option_chain(underlying: str) -> pd.DataFrame:
    fyers = get_fyers_client()
    if fyers is None:
        return pd.DataFrame()
    try:
        resp = fyers.optionchain({"symbol": normalize_symbol(underlying), "strikecount": 50})
    except Exception as e:
        logger.warning("optionchain failed for %s: %s", underlying, e)
        return pd.DataFrame()
    safefloat = _safe_getattr("safefloat", lambda v, default=np.nan: default if v is None or str(v).strip()=="" else float(v))
    rows = []
    for item in extract_chain(resp):
        if not isinstance(item, dict):
            continue
        typ = str(item.get("optionType") or item.get("type") or "").upper()
        strike = safefloat(item.get("strikePrice") or item.get("strike"), np.nan)
        if pd.isna(strike) or typ not in ("CE", "PE"):
            continue
        rows.append({
            "Underlying": underlying,
            "Option Type": typ,
            "Strike": strike,
            "Option Symbol": str(item.get("symbol") or item.get("tradingSymbol") or ""),
            "OptionLTP": safefloat(item.get("ltp") or item.get("lp"), np.nan),
            "OI": safefloat(item.get("oi") or item.get("openInterest"), np.nan),
            "Volume": safefloat(item.get("volume"), np.nan),
        })
    return pd.DataFrame(rows)


def fetch_iteration_history(option_symbol: str) -> pd.DataFrame:
    gethistory = _safe_getattr("gethistory")
    builditerationhistory = _safe_getattr("builditerationhistory")
    SIGNALWINDOWMINUTES = _safe_getattr("SIGNALWINDOWMINUTES", 5)
    ITERATIONSTOKEEP = _safe_getattr("ITERATIONSTOKEEP", 75)
    INTRADAYLOOKBACKDAYS = _safe_getattr("INTRADAYLOOKBACKDAYS", 20)
    if not callable(gethistory) or not callable(builditerationhistory):
        return pd.DataFrame()
    try:
        hist_symbol = option_symbol if str(option_symbol).startswith("NSE:") else f"NSE:{option_symbol}"
        intradf = gethistory(hist_symbol, "5", INTRADAYLOOKBACKDAYS)
        if intradf is None or intradf.empty:
            return pd.DataFrame()
        return builditerationhistory(intradf, SIGNALWINDOWMINUTES, ITERATIONSTOKEEP)
    except Exception as e:
        logger.warning("history failed for %s: %s", option_symbol, e)
        return pd.DataFrame()


import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.application import MIMEApplication


def build_near_atm_signals(options_df: pd.DataFrame) -> pd.DataFrame:
    if options_df is None or options_df.empty:
        return pd.DataFrame(columns=["Stock", "Side", "Signal", "ATMStrike", "EntryPrice", "LTP", "Entry", "Exit15", "Exit39"])
    df = options_df.copy()
    if "Underlying" not in df.columns:
        return pd.DataFrame(columns=["Stock", "Side", "Signal", "ATMStrike", "EntryPrice", "LTP", "Entry", "Exit15", "Exit39"])
    df["Stock"] = df["Underlying"].astype(str).str.replace(r"^NSE:", "", regex=True).str.replace(r"-EQ$", "", regex=True)
    df["Side"] = df["Option Type"].map({"CE": "CE", "PE": "PE"}).fillna("")
    df["Signal"] = np.where(df["Side"] == "CE", "BUY CE", "BUY PE")
    df["ATMStrike"] = pd.to_numeric(df.get("Strike"), errors="coerce")
    df["EntryPrice"] = pd.to_numeric(df.get("OptionLTP"), errors="coerce")
    df["LTP"] = pd.to_numeric(df.get("OptionLTP"), errors="coerce")
    df["Entry"] = ""
    df["Exit15"] = "-"
    df["Exit39"] = "-"
    cols = ["Stock", "Side", "Signal", "ATMStrike", "EntryPrice", "LTP", "Entry", "Exit15", "Exit39"]
    return df[cols].drop_duplicates(subset=["Stock", "Side", "ATMStrike"]).reset_index(drop=True)


def send_email_with_attachment(subject: str, body_html: str, attachment_path: str | None = None) -> bool:
    email_address = os.environ.get("EMAIL_ADDRESS")
    email_password = os.environ.get("EMAIL_PASSWORD")
    email_to = os.environ.get("EMAIL_TO", email_address)
    if not email_address or not email_password or not email_to:
        logger.warning("Email env vars missing; skipping email")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = email_address
        msg["To"] = email_to
        msg["Subject"] = subject
        msg.attach(MIMEText(body_html, "html", "utf-8"))
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(attachment_path)}")
            msg.attach(part)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(email_address, email_password)
            server.send_message(msg)
        return True
    except Exception as e:
        logger.warning("Email send failed: %s", e)
        return False

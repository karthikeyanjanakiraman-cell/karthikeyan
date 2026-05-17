import os
import math
import logging
import smtplib
from pathlib import Path
from typing import List, Optional, Tuple
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(os.environ.get("DATA_DIR", "."))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASIT_GLOB = os.environ.get("ASIT_GLOB", "asit*.csv")
TOP_N = int(os.environ.get("TOP_N", "20"))
CHAIN_UP = int(os.environ.get("CHAIN_UP", "5"))
CHAIN_DOWN = int(os.environ.get("CHAIN_DOWN", "5"))

ENTRY_THRESHOLD = float(os.environ.get("ENTRY_THRESHOLD", "0"))
EXIT_THRESHOLD = float(os.environ.get("EXIT_THRESHOLD", "0"))
MIN_OPTION_LTP = float(os.environ.get("MIN_OPTION_LTP", "10"))

ATTACHMENT_NAME = os.environ.get("ATTACHMENT_NAME", "near_atm_signals.csv")
EMAIL_ENABLED = os.environ.get("EMAIL_ENABLED", "1") == "1"
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))

SENDER_EMAIL = os.environ.get("SENDER_EMAIL", os.environ.get("EMAIL_ADDRESS", ""))
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", os.environ.get("EMAIL_PASSWORD", ""))
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", os.environ.get("EMAIL_TO", ""))


def pick_latest_file(pattern: str, base_dir: Path = BASE_DIR) -> Optional[Path]:
    files = sorted(base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def clean_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().replace({"NAN": np.nan, "NONE": np.nan, "NULL": np.nan, "": np.nan})


def score_frame(df: pd.DataFrame):
    bull_col = find_col(df, ["BullMultiTFScore", "bull", "bullish", "buy", "long", "BullScore", "bull_score", "5min_BullScore", "15min_BullScore", "1hour_BullScore", "4hour_BullScore", "1day_BullScore"])
    bear_col = find_col(df, ["BearMultiTFScore", "bear", "bearish", "sell", "short", "BearScore", "bear_score", "5min_BearScore", "15min_BearScore", "1hour_BearScore", "4hour_BearScore", "1day_BearScore"])
    rank_col = find_col(df, ["RankScore15Tier", "score", "rank", "Score15Tier", "5min_Score15Tier", "15min_Score15Tier", "1hour_Score15Tier", "4hour_Score15Tier", "1day_Score15Tier"])
    sym_col = find_col(df, ["Symbol", "symbol", "underlying", "ticker", "tradingsymbol", "name"])
    trend_col = find_col(df, ["DominantTrend", "trend", "direction"])
    if sym_col is None:
        raise ValueError("No symbol column found in ASIT CSV")
    for c in [bull_col, bear_col, rank_col]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if trend_col is not None:
        df[trend_col] = clean_text(df[trend_col])
    df[sym_col] = clean_text(df[sym_col])
    return df, sym_col, bull_col, bear_col, rank_col, trend_col


def top_rows(df: pd.DataFrame, sort_col: Optional[str], n: int) -> pd.DataFrame:
    out = df.copy()
    if sort_col is not None:
        out = out.sort_values(sort_col, ascending=False, na_position="last")
    return out.head(n).reset_index(drop=True)


def build_top_lists(asit_path: Path, top_n: int = TOP_N) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_csv(asit_path)
    df, sym_col, bull_col, bear_col, rank_col, trend_col = score_frame(df)
    bull_df = top_rows(df, bull_col or rank_col, top_n).copy()
    bear_df = top_rows(df, bear_col or rank_col, top_n).copy()
    bull_df["Side"] = "BULLISH"
    bear_df["Side"] = "BEARISH"
    bull_df["Underlying"] = bull_df[sym_col]
    bear_df["Underlying"] = bear_df[sym_col]
    return bull_df, bear_df, pd.concat([bull_df, bear_df], ignore_index=True)


def infer_option_type(side: str, trend: str = "") -> str:
    s = str(side).upper()
    t = str(trend).upper()
    if s.startswith("BULL"):
        return "CE"
    if s.startswith("BEAR"):
        return "PE"
    if "UP" in t:
        return "CE"
    if "DOWN" in t:
        return "PE"
    return "CE"


def choose_atm_row(stock_df: pd.DataFrame) -> pd.Series:
    if stock_df.empty:
        return pd.Series(dtype=object)
    stock_df = stock_df.copy()
    stock_df["Strike"] = pd.to_numeric(stock_df["Strike"], errors="coerce")
    stock_df = stock_df.dropna(subset=["Strike"])
    if stock_df.empty:
        return pd.Series(dtype=object)
    if "UnderlyingPrice" in stock_df.columns:
        px = pd.to_numeric(stock_df["UnderlyingPrice"], errors="coerce").iloc[0]
    else:
        px = np.nan
    if pd.isna(px):
        return stock_df.iloc[(stock_df["Strike"] - stock_df["Strike"].median()).abs().argsort().iloc[0]]
    idx = (stock_df["Strike"] - px).abs().idxmin()
    return stock_df.loc[idx]


def build_chain_window(chain_df: pd.DataFrame, atm_strike: float, up: int = CHAIN_UP, down: int = CHAIN_DOWN) -> pd.DataFrame:
    if chain_df.empty or pd.isna(atm_strike):
        return pd.DataFrame()
    chain_df = chain_df.copy()
    chain_df["Strike"] = pd.to_numeric(chain_df["Strike"], errors="coerce")
    chain_df = chain_df.dropna(subset=["Strike"])
    strikes = sorted(chain_df["Strike"].unique())
    if atm_strike not in strikes:
        nearest = min(strikes, key=lambda x: abs(x - atm_strike))
        atm_strike = nearest
    pos = strikes.index(atm_strike)
    lo = max(0, pos - down)
    hi = min(len(strikes) - 1, pos + up)
    window = strikes[lo:hi + 1]
    return chain_df[chain_df["Strike"].isin(window)].sort_values(["Strike", "Option Type"]).reset_index(drop=True)


def analyze_tandem(chain_window: pd.DataFrame) -> str:
    if chain_window.empty:
        return "NO_DATA"
    if "OptionLTP" not in chain_window.columns:
        return "NO_LTP"
    df = chain_window.copy()
    df["OptionLTP"] = pd.to_numeric(df["OptionLTP"], errors="coerce")
    diffs = df.groupby("Option Type")["OptionLTP"].diff().dropna()
    if diffs.empty:
        return "INCONCLUSIVE"
    mean_change = diffs.mean()
    if mean_change > ENTRY_THRESHOLD:
        return "MOVE_UP_TOGETHER"
    if mean_change < -abs(EXIT_THRESHOLD):
        return "MOVE_DOWN_TOGETHER"
    return "MIXED"


def build_report(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df is None or seed_df.empty:
        return pd.DataFrame(columns=["Stock", "Side", "Signal", "ATMStrike", "EntryPrice", "LTP", "Entry", "Exit15", "Exit39"])
    rows = []
    for _, row in seed_df.iterrows():
        stock = str(row.get("Underlying", row.get("Symbol", ""))).strip().upper()
        side = str(row.get("Side", "")).upper()
        signal = f"BUY {'CE' if side.startswith('BULL') else 'PE'}"
        atm = pd.to_numeric(row.get("ATMStrike", np.nan), errors="coerce")
        entry = pd.to_numeric(row.get("EntryPrice", row.get("LTP", np.nan)), errors="coerce")
        ltp = pd.to_numeric(row.get("LTP", np.nan), errors="coerce")
        rows.append({
            "Stock": stock,
            "Side": "CE" if signal.endswith("CE") else "PE",
            "Signal": signal,
            "ATMStrike": atm,
            "EntryPrice": entry,
            "LTP": ltp,
            "Entry": row.get("Entry", ""),
            "Exit15": row.get("Exit15", "-"),
            "Exit39": row.get("Exit39", "-")
        })
    return pd.DataFrame(rows)


def send_email(subject: str, html_body: str, attachment_path: Optional[Path] = None) -> bool:
    if not EMAIL_ENABLED:
        logger.info("Email disabled")
        return False
    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
        logger.warning("Email env vars missing")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECIPIENT_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        if attachment_path and attachment_path.exists():
            with open(attachment_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={attachment_path.name}")
            msg.attach(part)
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        logger.exception("Email send failed: %s", e)
        return False


def main():
    asit_file = pick_latest_file(ASIT_GLOB)
    if asit_file is None:
        raise FileNotFoundError(f"No file found for pattern {ASIT_GLOB} in {BASE_DIR.resolve()}")
    logger.info("Using ASIT file: %s", asit_file.resolve())
    bull_df, bear_df, combined = build_top_lists(asit_file, top_n=TOP_N)
    signals = build_report(combined)
    signals_path = OUTPUT_DIR / ATTACHMENT_NAME
    signals.to_csv(signals_path, index=False)
    bull_df.to_csv(OUTPUT_DIR / "top_20_bullish.csv", index=False)
    bear_df.to_csv(OUTPUT_DIR / "top_20_bearish.csv", index=False)
    combined.to_csv(OUTPUT_DIR / "top_20_bull_bear.csv", index=False)
    html = f"<html><body><p>ASIT file: {asit_file.name}</p><p>Signals: {len(signals)}</p></body></html>"
    send_email(f"CEPEBUY ATM Signals - {asit_file.stem}", html, signals_path)
    print(f"ASIT file: {asit_file.name}")
    print(f"Top bullish: {len(bull_df)}")
    print(f"Top bearish: {len(bear_df)}")
    print(f"Signals: {len(signals)}")


if __name__ == "__main__":
    main()

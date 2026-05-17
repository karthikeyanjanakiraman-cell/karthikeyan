import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from OPTIONS_OI import (
        initfyers,
        gethistory,
        builditerationhistory,
        SIGNALWINDOWMINUTES,
        ITERATIONSTOKEEP,
        INTRADAYLOOKBACKDAYS,
        formateqsymbol,
        safefloat,
        sendsingleemail,
        sendcepebuyemail,
    )
except Exception:
    initfyers = None
    gethistory = None
    builditerationhistory = None
    SIGNALWINDOWMINUTES = 5
    ITERATIONSTOKEEP = 75
    INTRADAYLOOKBACKDAYS = 20
    formateqsymbol = lambda x: f"NSE:{str(x).strip().upper()}-EQ"
    safefloat = lambda v, default=np.nan: default if v is None or str(v).strip() == "" else float(v)
    sendsingleemail = None
    sendcepebuyemail = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(os.environ.get("DATA_DIR", "."))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASIT_GLOB = os.environ.get("ASIT_GLOB", "asit*.csv")
TOP_N = int(os.environ.get("TOP_N", "20"))
MIN_OPTION_LTP = float(os.environ.get("MIN_OPTION_LTP", "10"))
CHAIN_UP = int(os.environ.get("CHAIN_UP", "5"))
CHAIN_DOWN = int(os.environ.get("CHAIN_DOWN", "5"))
ENTRY_THRESHOLD = float(os.environ.get("ENTRY_THRESHOLD", "0"))
EXIT_THRESHOLD = float(os.environ.get("EXIT_THRESHOLD", "0"))
ATTACHMENT_NAME = os.environ.get("ATTACHMENT_NAME", "near_atm_signals.csv")


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


def fetch_option_chain(underlying: str) -> pd.DataFrame:
    if initfyers is None:
        return pd.DataFrame()
    fy = initfyers()
    if fy is None:
        return pd.DataFrame()
    eqsymbol = formateqsymbol(underlying)
    try:
        res = fy.optionchain({"symbol": eqsymbol, "strikecount": max(CHAIN_UP, CHAIN_DOWN) * 2 + 5})
    except Exception as e:
        logger.warning("optionchain failed for %s: %s", underlying, e)
        return pd.DataFrame()
    chain = (res or {}).get("data") or (res or {}).get("optionsChain") or []
    rows = []
    for item in chain:
        strike = safefloat(item.get("strikePrice") or item.get("strike"), np.nan)
        typ = str(item.get("optionType") or item.get("type") or "").upper()
        sym = str(item.get("symbol") or item.get("tradingSymbol") or item.get("optionSymbol") or "").strip()
        if pd.isna(strike) or typ not in ("CE", "PE"):
            continue
        rows.append({
            "Underlying": underlying,
            "Option Type": typ,
            "Strike": strike,
            "Option Symbol": sym,
            "OptionLTP": safefloat(item.get("ltp") or item.get("lp"), np.nan),
            "OI": safefloat(item.get("oi") or item.get("openInterest"), np.nan),
            "Volume": safefloat(item.get("volume"), np.nan),
        })
    return pd.DataFrame(rows)


def infer_atm_from_spot(underlying: str, chain: pd.DataFrame) -> float:
    if chain.empty:
        return np.nan
    fx = initfyers() if initfyers is not None else None
    spot = np.nan
    if fx is not None:
        try:
            q = fx.quotes({"symbols": [formateqsymbol(underlying)]})
            data = (q or {}).get("d") or (q or {}).get("data") or []
            if data:
                spot = safefloat(data[0].get("v", {}).get("lp") or data[0].get("lp"), np.nan)
        except Exception:
            spot = np.nan
    strikes = pd.to_numeric(chain["Strike"], errors="coerce").dropna().unique()
    if len(strikes) == 0:
        return np.nan
    strikes = sorted(map(float, strikes))
    if pd.isna(spot):
        return float(strikes[len(strikes) // 2])
    return float(min(strikes, key=lambda x: abs(x - spot)))


def chain_window(chain: pd.DataFrame, atm: float, up: int = CHAIN_UP, down: int = CHAIN_DOWN) -> pd.DataFrame:
    if chain.empty or pd.isna(atm):
        return pd.DataFrame()
    chain = chain.copy()
    chain["Strike"] = pd.to_numeric(chain["Strike"], errors="coerce")
    chain = chain.dropna(subset=["Strike"])
    strikes = sorted(chain["Strike"].unique())
    if not strikes:
        return pd.DataFrame()
    nearest = min(strikes, key=lambda x: abs(x - atm))
    idx = strikes.index(nearest)
    lo = max(0, idx - down)
    hi = min(len(strikes) - 1, idx + up)
    keep = strikes[lo:hi + 1]
    return chain[chain["Strike"].isin(keep)].sort_values(["Strike", "Option Type"]).reset_index(drop=True)


def tandem_status(win: pd.DataFrame) -> str:
    if win.empty or "OptionLTP" not in win.columns:
        return "NO_DATA"
    df = win.copy()
    df["OptionLTP"] = pd.to_numeric(df["OptionLTP"], errors="coerce")
    df = df.dropna(subset=["OptionLTP"])
    if df.empty:
        return "NO_DATA"
    pivot = df.pivot_table(index="Strike", columns="Option Type", values="OptionLTP", aggfunc="last").sort_index()
    if pivot.empty or pivot.shape[0] < 2:
        return "INCONCLUSIVE"
    diff = pivot.ffill().bfill().diff().dropna(how="all")
    if diff.empty:
        return "INCONCLUSIVE"
    ce = diff.get("CE")
    pe = diff.get("PE")
    if ce is None or pe is None:
        return "INCONCLUSIVE"
    ce_up = (ce > ENTRY_THRESHOLD).sum()
    pe_up = (pe > ENTRY_THRESHOLD).sum()
    ce_dn = (ce < -EXIT_THRESHOLD).sum()
    pe_dn = (pe < -EXIT_THRESHOLD).sum()
    total = len(diff)
    if ce_up >= total * 0.6 and pe_up >= total * 0.6:
        return "MOVE_UP_TOGETHER"
    if ce_dn >= total * 0.6 and pe_dn >= total * 0.6:
        return "MOVE_DOWN_TOGETHER"
    return "MIXED"


def infer_signal(side: str) -> str:
    return "BUY CE" if str(side).upper().startswith("BULL") else "BUY PE"


def build_report(seed_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Stock", "Side", "Signal", "ATMStrike", "EntryPrice", "LTP", "Entry", "Exit15", "Exit39"]
    if seed_df is None or seed_df.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for _, row in seed_df.iterrows():
        underlying = str(row.get("Underlying", row.get("Symbol", ""))).strip().upper()
        side = str(row.get("Side", "")).upper()
        chain = fetch_option_chain(underlying)
        if chain.empty:
            continue
        atm = infer_atm_from_spot(underlying, chain)
        win = chain_window(chain, atm, CHAIN_UP, CHAIN_DOWN)
        status = tandem_status(win)
        opt_type = "CE" if side.startswith("BULL") else "PE"
        leg = win[win["Option Type"] == opt_type].copy()
        if leg.empty:
            leg = win.copy()
        leg = leg.sort_values("Strike")
        ltp = pd.to_numeric(leg["OptionLTP"], errors="coerce").dropna()
        ltp_val = float(ltp.iloc[0]) if not ltp.empty else np.nan
        rows.append({
            "Stock": underlying,
            "Side": opt_type,
            "Signal": infer_signal(side),
            "ATMStrike": atm,
            "EntryPrice": ltp_val,
            "LTP": ltp_val,
            "Entry": "ENTER" if status == "MOVE_UP_TOGETHER" else "WAIT",
            "Exit15": "EXIT" if status == "MOVE_DOWN_TOGETHER" else "HOLD",
            "Exit39": status,
        })
    out = pd.DataFrame(rows, columns=cols)
    if not out.empty:
        out = out.sort_values(["Signal", "Stock"], ascending=[True, True]).reset_index(drop=True)
    return out


def build_html_report(signals: pd.DataFrame, title: str) -> str:
    if signals.empty:
        body = "<p>No signals generated.</p>"
    else:
        body = signals.to_html(index=False, border=0, classes="report")
    return f"""
    <html><body>
    <h3>{title}</h3>
    <p>Top N: {TOP_N} | Chain window: {CHAIN_DOWN} down / {CHAIN_UP} up | Entry threshold: {ENTRY_THRESHOLD} | Exit threshold: {EXIT_THRESHOLD}</p>
    {body}
    </body></html>
    """


def send_email(subject: str, html_body: str, attachments: Optional[List[str]] = None) -> bool:
    if sendcepebuyemail is not None:
        try:
            # keep compatibility with existing OPTIONS_OI mailer when available
            return bool(sendcepebuyemail([], [], attachments or []))
        except Exception:
            pass
    if sendsingleemail is not None:
        try:
            return bool(sendsingleemail(subject, html_body, attachments or []))
        except Exception:
            pass
    logger.warning("No email helper available from OPTIONS_OI")
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
    html = build_html_report(signals, f"CEPEBUY ATM Signals - {asit_file.stem}")
    send_email(f"CEPEBUY ATM Signals - {asit_file.stem}", html, [str(signals_path)])
    print(f"ASIT file: {asit_file.name}")
    print(f"Top bullish: {len(bull_df)}")
    print(f"Top bearish: {len(bear_df)}")
    print(f"Signals: {len(signals)}")


if __name__ == "__main__":
    main()

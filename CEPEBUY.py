import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(os.environ.get("DATA_DIR", "."))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASIT_GLOB = os.environ.get("ASIT_GLOB", "asit*.csv")
OUTPUT_BULL = OUTPUT_DIR / "top_10_bullish.csv"
OUTPUT_BEAR = OUTPUT_DIR / "top_10_bearish.csv"
OUTPUT_COMBINED = OUTPUT_DIR / "top_10_bull_bear.csv"
TOP_N = int(os.environ.get("TOP_N", "10"))

from OPTIONS_OI import gethistory, builditerationhistory, SIGNALWINDOWMINUTES, ITERATIONSTOKEEP, INTRADAYLOOKBACKDAYS, formateqsymbol, safefloat

fyers = None


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
    if sym_col is None:
        raise ValueError("No symbol column found in ASIT CSV")
    for c in [bull_col, bear_col, rank_col]:
        if c is not None:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[sym_col] = clean_text(df[sym_col])
    return df, sym_col, bull_col, bear_col, rank_col


def top_rows(df: pd.DataFrame, sort_col: Optional[str], n: int) -> pd.DataFrame:
    out = df.copy()
    if sort_col is not None:
        out = out.sort_values(sort_col, ascending=False, na_position="last")
    return out.head(n).reset_index(drop=True)


def build_top_lists(asit_path: Path, top_n: int = TOP_N) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_csv(asit_path)
    df, sym_col, bull_col, bear_col, rank_col = score_frame(df)
    bull_df = top_rows(df, bull_col or rank_col, top_n).copy()
    bear_df = top_rows(df, bear_col or rank_col, top_n).copy()
    bull_df["Side"] = "BULLISH"
    bear_df["Side"] = "BEARISH"
    bull_df["Underlying"] = bull_df[sym_col]
    bear_df["Underlying"] = bear_df[sym_col]
    bull_df["Option Symbol"] = bull_df[sym_col]
    bear_df["Option Symbol"] = bear_df[sym_col]
    return bull_df, bear_df, pd.concat([bull_df, bear_df], ignore_index=True)


def init_client_once():
    global fyers
    if fyers is None:
        fyers = initfyers()
    return fyers


def normalize_symbol(raw: str) -> str:
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
    client = init_client_once()
    if client is None:
        logger.warning("Fyers client unavailable; skipping option chain for %s", underlying)
        return pd.DataFrame()
    eqsymbol = normalize_symbol(underlying)
    logger.info("API chain fetch underlying=%s eqsymbol=%s", underlying, eqsymbol)
    try:
        resp = client.optionchain({"symbol": eqsymbol, "strikecount": 50})
    except Exception as e:
        logger.warning("optionchain failed for %s: %s", underlying, e)
        return pd.DataFrame()
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
    if gethistory is None or builditerationhistory is None:
        return pd.DataFrame()
    hist_symbol = option_symbol if str(option_symbol).startswith("NSE:") else f"NSE:{option_symbol}"
    try:
        intradf = gethistory(hist_symbol, "5", INTRADAYLOOKBACKDAYS)
        if intradf is None or intradf.empty:
            return pd.DataFrame()
        return builditerationhistory(intradf, SIGNALWINDOWMINUTES, ITERATIONSTOKEEP)
    except Exception as e:
        logger.warning("history fetch failed for %s: %s", hist_symbol, e)
        return pd.DataFrame()


def build_api_scans(seed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    option_rows, iter_rows = [], []
    if seed_df is None or seed_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    for _, row in seed_df.iterrows():
        underlying = str(row.get("Underlying", "")).strip()
        side = str(row.get("Side", "")).upper()
        req_type = "CE" if side.startswith("BULL") else "PE"
        chain = fetch_option_chain(underlying)
        if chain.empty:
            continue
        chain = chain[chain["Option Type"] == req_type].copy()
        if chain.empty:
            continue
        option_rows.append(chain)
        for _, crow in chain.iterrows():
            hist = fetch_iteration_history(crow.get("Option Symbol", ""))
            if hist is None or hist.empty:
                continue
            hist.insert(0, "Source Underlying", underlying)
            hist.insert(1, "Source Option Symbol", crow.get("Option Symbol", ""))
            hist.insert(2, "Source Option Type", req_type)
            iter_rows.append(hist)
    return (pd.concat(option_rows, ignore_index=True) if option_rows else pd.DataFrame(),
            pd.concat(iter_rows, ignore_index=True) if iter_rows else pd.DataFrame())


def main():
    asit_file = pick_latest_file(ASIT_GLOB)
    if asit_file is None:
        raise FileNotFoundError(f"No file found for pattern {ASIT_GLOB} in {BASE_DIR.resolve()}")
    logger.info("Using ASIT file: %s", asit_file.resolve())
    bull_df, bear_df, combined = build_top_lists(asit_file, top_n=TOP_N)
    bull_df.to_csv(OUTPUT_BULL, index=False)
    bear_df.to_csv(OUTPUT_BEAR, index=False)
    combined.to_csv(OUTPUT_COMBINED, index=False)
    logger.info("Saved %s, %s, %s", OUTPUT_BULL.name, OUTPUT_BEAR.name, OUTPUT_COMBINED.name)
    options_df, iter_df = build_api_scans(combined)
    if not options_df.empty:
        options_df.to_csv(OUTPUT_DIR / "api_options.csv", index=False)
    if not iter_df.empty:
        iter_df.to_csv(OUTPUT_DIR / "api_iteration_history.csv", index=False)
    print(f"ASIT file: {asit_file}")
    print(f"Top bullish: {len(bull_df)}")
    print(f"Top bearish: {len(bear_df)}")
    print(f"API option rows: {len(options_df)}")
    print(f"API iteration rows: {len(iter_df)}")


if __name__ == "__main__":
    main()

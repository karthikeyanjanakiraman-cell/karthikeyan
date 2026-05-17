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
MIN_OPTION_LTP = float(os.environ.get("MIN_OPTION_LTP", "10"))
TOP_N = int(os.environ.get("TOP_N", "10"))

try:
    from OPTIONS_OI import initfyers, gethistory, builditerationhistory, SIGNALWINDOWMINUTES, ITERATIONSTOKEEP, INTRADAYLOOKBACKDAYS, formateqsymbol, safefloat
except Exception:
    initfyers = None
    gethistory = None
    builditerationhistory = None
    SIGNALWINDOWMINUTES = 5
    ITERATIONSTOKEEP = 75
    INTRADAYLOOKBACKDAYS = 20
    formateqsymbol = lambda x: f"NSE:{str(x).strip().upper()}-EQ"
    def safefloat(v, default=np.nan):
        try:
            if v is None or str(v).strip() == "":
                return default
            return float(v)
        except Exception:
            return default

CLIENT_ID = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
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
    bull_df["Option Symbol"] = bull_df[sym_col]
    bear_df["Option Symbol"] = bear_df[sym_col]
    combined = pd.concat([bull_df, bear_df], ignore_index=True)
    return bull_df, bear_df, combined


def init_client_once():
    global fyers
    if fyers is not None:
        return fyers
    if not CLIENT_ID or not ACCESS_TOKEN:
        logger.warning("Missing CLIENT_ID/ACCESS_TOKEN")
        return None
    try:
        try:
            from fyers_apiv3 import fyersModel as fm
        except Exception:
            from fyersapiv3 import fyersModel as fm
        fyers = fm.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False, log_path="")
        logger.info("Fyers client initialized using %s", fm.__module__)
        return fyers
    except Exception as e:
        logger.warning("Fyers client not available: %s", e)
        return None


def fetch_option_chain(underlying: str) -> pd.DataFrame:
    fy = init_client_once()
    if fy is None:
        return pd.DataFrame([{
            "Underlying": underlying,
            "Option Type": "CE",
            "Strike": np.nan,
            "Option Symbol": "",
            "OptionLTP": np.nan,
            "OI": np.nan,
            "Volume": np.nan,
            "Source": "fallback"
        }])
    eqsymbol = formateqsymbol(underlying)
    logger.info("API chain fetch underlying=%s eqsymbol=%s", underlying, eqsymbol)
    try:
        chainres = fy.optionchain({"symbol": eqsymbol, "strikecount": 50})
    except Exception as e:
        logger.warning("optionchain failed for %s: %s", underlying, e)
        return pd.DataFrame()
    if isinstance(chainres, dict):
        chain = chainres.get("data") or chainres.get("optionsChain") or []
        if isinstance(chain, dict):
            chain = chain.get("optionsChain") or chain.get("data") or []
    else:
        chain = []
    rows = []
    for item in chain if isinstance(chain, list) else []:
        if not isinstance(item, dict):
            continue
        strike = safefloat(item.get("strikePrice") or item.get("strike"), np.nan)
        typ = str(item.get("optionType") or item.get("type") or "").upper()
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
    if builditerationhistory is None or gethistory is None:
        return pd.DataFrame()
    hist_symbol = option_symbol if str(option_symbol).startswith("NSE:") else f"NSE:{option_symbol}"
    intradf = gethistory(hist_symbol, "5", INTRADAYLOOKBACKDAYS)
    if intradf is None or intradf.empty:
        return pd.DataFrame()
    return builditerationhistory(intradf, SIGNALWINDOWMINUTES, ITERATIONSTOKEEP)


def build_api_scans(seed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    option_rows = []
    iter_rows = []
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
    options_df = pd.concat(option_rows, ignore_index=True) if option_rows else pd.DataFrame()
    iter_df = pd.concat(iter_rows, ignore_index=True) if iter_rows else pd.DataFrame()
    return options_df, iter_df


def save_outputs(bull_df: pd.DataFrame, bear_df: pd.DataFrame, combined: pd.DataFrame):
    bull_df.to_csv(OUTPUT_BULL, index=False)
    bear_df.to_csv(OUTPUT_BEAR, index=False)
    combined.to_csv(OUTPUT_COMBINED, index=False)
    logger.info("Saved %s, %s, %s", OUTPUT_BULL, OUTPUT_BEAR, OUTPUT_COMBINED)


def main():
    asit_file = pick_latest_file(ASIT_GLOB)
    if asit_file is None:
        raise FileNotFoundError(f"No file found for pattern {ASIT_GLOB} in {BASE_DIR.resolve()}")
    logger.info("Using ASIT file: %s", asit_file.resolve())
    bull_df, bear_df, combined = build_top_lists(asit_file, top_n=TOP_N)
    save_outputs(bull_df, bear_df, combined)
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

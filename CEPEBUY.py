import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# â”€â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE_DIR   = Path(os.environ.get("DATA_DIR",   "."))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "."))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ASIT_GLOB      = os.environ.get("ASIT_GLOB", "asit*.csv")
OUTPUT_BULL    = OUTPUT_DIR / "top_10_bullish.csv"
OUTPUT_BEAR    = OUTPUT_DIR / "top_10_bearish.csv"
OUTPUT_COMBINED= OUTPUT_DIR / "top_10_bull_bear.csv"
MIN_OPTION_LTP = float(os.environ.get("MIN_OPTION_LTP", "10"))
TOP_N          = int(os.environ.get("TOP_N", "10"))
CLIENT_ID      = os.environ.get("CLIENT_ID")  or os.environ.get("CLIENTID")
ACCESS_TOKEN   = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")

# â”€â”€â”€ Import helpers from OPTIONS_OI with graceful fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_initfyers_fn       = None
_gethistory_fn      = None
_builditer_fn       = None
SIGNALWINDOWMINUTES = 5
ITERATIONSTOKEEP    = 75
INTRADAYLOOKBACKDAYS= 20

def _default_formateqsymbol(x: str) -> str:
    sym = str(x).strip().upper()
    if sym.startswith("NSE:"):
        return sym if sym.endswith("-EQ") else sym + "-EQ"
    return f"NSE:{sym}-EQ"

def _default_safefloat(v, default=np.nan):
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default

formateqsymbol = _default_formateqsymbol
safefloat      = _default_safefloat

try:
    from OPTIONS_OI import (
        initfyers          as _initfyers_fn,
        gethistory         as _gethistory_fn,
        builditerationhistory as _builditer_fn,
        SIGNALWINDOWMINUTES,
        ITERATIONSTOKEEP,
        INTRADAYLOOKBACKDAYS,
        formateqsymbol,
        safefloat,
    )
    logger.info("OPTIONS_OI helpers imported successfully")
except Exception as _e:
    logger.warning("Could not import from OPTIONS_OI (%s) â€“ using built-in fallbacks", _e)

# â”€â”€â”€ Fyers client â€“ robust multi-module init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_fyers = None

def init_client_once():
    """
    Returns a live FyersModel instance.
    Priority:
      1. Reuse the global _fyers if already initialised.
      2. Call OPTIONS_OI.initfyers() so the same global fyers object is shared.
      3. Try fyers_apiv3 (official pip name), fyersapiv3 (old name), fyersapi.
    """
    global _fyers
    if _fyers is not None:
        return _fyers

    # â”€â”€ Try OPTIONS_OI.initfyers first â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if _initfyers_fn is not None:
        try:
            fy = _initfyers_fn()
            if fy is not None:
                _fyers = fy
                logger.info("Fyers client ready via OPTIONS_OI.initfyers()")
                return _fyers
        except Exception as e:
            logger.warning("OPTIONS_OI.initfyers() failed: %s", e)

    # â”€â”€ Fallback: build our own client â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not CLIENT_ID or not ACCESS_TOKEN:
        logger.warning("CLIENT_ID / ACCESS_TOKEN not set â€“ Fyers API disabled")
        return None

    for mod_name in ("fyers_apiv3", "fyersapiv3", "fyersapi"):
        try:
            mod = __import__(mod_name, fromlist=["fyersModel"])
            fm  = mod.fyersModel
            _fyers = fm.FyersModel(
                client_id=CLIENT_ID,
                token=ACCESS_TOKEN,
                is_async=False,
                log_path=""
            )
            logger.info("Fyers client ready using module '%s'", mod_name)
            return _fyers
        except Exception as e:
            logger.warning("FyersModel init via '%s' failed: %s", mod_name, e)

    logger.error("Fyers client could not be initialised â€“ all module names exhausted")
    return None

# â”€â”€â”€ Utility helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
    return (
        s.astype(str).str.strip().str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan, "NULL": np.nan, "": np.nan})
    )

# â”€â”€â”€ ASIT CSV scoring / ranking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def score_frame(df: pd.DataFrame):
    bull_col  = find_col(df, ["BullMultiTFScore","bull","bullish","buy","long","BullScore","bull_score",
                               "5min_BullScore","15min_BullScore","1hour_BullScore","4hour_BullScore","1day_BullScore"])
    bear_col  = find_col(df, ["BearMultiTFScore","bear","bearish","sell","short","BearScore","bear_score",
                               "5min_BearScore","15min_BearScore","1hour_BearScore","4hour_BearScore","1day_BearScore"])
    rank_col  = find_col(df, ["RankScore15Tier","score","rank","Score15Tier",
                               "5min_Score15Tier","15min_Score15Tier","1hour_Score15Tier","4hour_Score15Tier","1day_Score15Tier"])
    sym_col   = find_col(df, ["Symbol","symbol","underlying","ticker","tradingsymbol","name"])
    trend_col = find_col(df, ["DominantTrend","trend","direction"])
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

def build_top_lists(
    asit_path: Path, top_n: int = TOP_N
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_csv(asit_path)
    df, sym_col, bull_col, bear_col, rank_col, trend_col = score_frame(df)
    bull_df = top_rows(df, bull_col or rank_col, top_n).copy()
    bear_df = top_rows(df, bear_col or rank_col, top_n).copy()
    bull_df["Side"]         = "BULLISH"
    bear_df["Side"]         = "BEARISH"
    bull_df["Underlying"]   = bull_df[sym_col]
    bear_df["Underlying"]   = bear_df[sym_col]
    bull_df["Option Symbol"]= bull_df[sym_col]
    bear_df["Option Symbol"]= bear_df[sym_col]
    combined = pd.concat([bull_df, bear_df], ignore_index=True)
    return bull_df, bear_df, combined

# â”€â”€â”€ Option chain fetch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def fetch_option_chain(underlying: str) -> pd.DataFrame:
    fy = init_client_once()
    if fy is None:
        return pd.DataFrame()
    eqsymbol = formateqsymbol(underlying)
    logger.info("Fetching option chain: underlying=%s eqsymbol=%s", underlying, eqsymbol)
    try:
        chainres = fy.optionchain({"symbol": eqsymbol, "strikecount": 50})
    except Exception as e:
        logger.warning("optionchain() failed for %s: %s", underlying, e)
        return pd.DataFrame()
    chain = (chainres or {}).get("data") or (chainres or {}).get("optionsChain") or []
    if not chain:
        logger.warning("Empty option chain response for %s â€“ keys: %s",
                       underlying, list((chainres or {}).keys()))
        return pd.DataFrame()
    rows = []
    for item in chain:
        strike = safefloat(item.get("strikePrice") or item.get("strike"), np.nan)
        typ    = str(item.get("optionType") or item.get("type") or "").upper()
        if pd.isna(strike) or typ not in ("CE", "PE"):
            continue
        rows.append({
            "Underlying":    underlying,
            "Option Type":   typ,
            "Strike":        strike,
            "Option Symbol": str(item.get("symbol") or item.get("tradingSymbol") or ""),
            "OptionLTP":     safefloat(item.get("ltp")  or item.get("lp"),            np.nan),
            "OI":            safefloat(item.get("oi")   or item.get("openInterest"),  np.nan),
            "Volume":        safefloat(item.get("volume"),                            np.nan),
        })
    return pd.DataFrame(rows)

# â”€â”€â”€ Iteration history fetch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def fetch_iteration_history(option_symbol: str) -> pd.DataFrame:
    build_hist = _builditer_fn
    get_hist   = _gethistory_fn
    win        = SIGNALWINDOWMINUTES
    keep       = ITERATIONSTOKEEP

    # Extra safety â€“ try importing again if the module-level import failed
    if build_hist is None or get_hist is None:
        try:
            import OPTIONS_OI as _opt
            build_hist = getattr(_opt, "builditerationhistory", None)
            get_hist   = getattr(_opt, "gethistory",            None)
            win        = getattr(_opt, "SIGNALWINDOWMINUTES",   win)
            keep       = getattr(_opt, "ITERATIONSTOKEEP",      keep)
        except Exception as e:
            logger.warning("OPTIONS_OI retry import failed: %s", e)

    if get_hist is None or build_hist is None:
        logger.warning("gethistory / builditerationhistory not available â€“ skipping history for %s", option_symbol)
        return pd.DataFrame()

    hist_symbol = (
        option_symbol if str(option_symbol).startswith("NSE:")
        else f"NSE:{option_symbol}"
    )
    intradf = get_hist(hist_symbol, "5", INTRADAYLOOKBACKDAYS)
    if intradf is None or intradf.empty:
        logger.warning("No intraday data for %s", hist_symbol)
        return pd.DataFrame()
    return build_hist(intradf, win, keep)

# â”€â”€â”€ Main API scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def build_api_scans(
    seed_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    option_rows: List[pd.DataFrame] = []
    iter_rows:   List[pd.DataFrame] = []
    if seed_df is None or seed_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    for _, row in seed_df.iterrows():
        underlying = str(row.get("Underlying", "")).strip()
        if not underlying or underlying.upper() in ("NAN", "NONE", ""):
            continue
        side     = str(row.get("Side", "")).upper()
        req_type = "CE" if side.startswith("BULL") else "PE"

        chain = fetch_option_chain(underlying)
        if chain.empty:
            continue
        chain = chain[chain["Option Type"] == req_type].copy()
        if chain.empty:
            continue

        # Filter by MIN_OPTION_LTP
        if "OptionLTP" in chain.columns:
            chain = chain[pd.to_numeric(chain["OptionLTP"], errors="coerce").fillna(0) >= MIN_OPTION_LTP].copy()
        if chain.empty:
            continue

        option_rows.append(chain)

        for _, crow in chain.iterrows():
            sym = str(crow.get("Option Symbol", "")).strip()
            if not sym:
                continue
            hist = fetch_iteration_history(sym)
            if hist is None or hist.empty:
                continue
            hist = hist.copy()
            hist.insert(0, "Source Underlying",    underlying)
            hist.insert(1, "Source Option Symbol", sym)
            hist.insert(2, "Source Option Type",   req_type)
            iter_rows.append(hist)

    options_df = pd.concat(option_rows, ignore_index=True) if option_rows else pd.DataFrame()
    iter_df    = pd.concat(iter_rows,   ignore_index=True) if iter_rows   else pd.DataFrame()
    return options_df, iter_df

# â”€â”€â”€ Save outputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def save_outputs(
    bull_df: pd.DataFrame, bear_df: pd.DataFrame, combined: pd.DataFrame
) -> None:
    bull_df.to_csv(OUTPUT_BULL,     index=False)
    bear_df.to_csv(OUTPUT_BEAR,     index=False)
    combined.to_csv(OUTPUT_COMBINED, index=False)
    logger.info("Saved %s, %s, %s", OUTPUT_BULL, OUTPUT_BEAR, OUTPUT_COMBINED)

# â”€â”€â”€ Entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main() -> None:
    asit_file = pick_latest_file(ASIT_GLOB)
    if asit_file is None:
        raise FileNotFoundError(
            f"No file found for pattern '{ASIT_GLOB}' in {BASE_DIR.resolve()}"
        )
    logger.info("Using ASIT file: %s", asit_file.resolve())

    bull_df, bear_df, combined = build_top_lists(asit_file, top_n=TOP_N)
    save_outputs(bull_df, bear_df, combined)

    options_df, iter_df = build_api_scans(combined)

    if not options_df.empty:
        out = OUTPUT_DIR / "api_options.csv"
        options_df.to_csv(out, index=False)
        logger.info("Saved api_options.csv (%d rows)", len(options_df))

    if not iter_df.empty:
        out = OUTPUT_DIR / "api_iteration_history.csv"
        iter_df.to_csv(out, index=False)
        logger.info("Saved api_iteration_history.csv (%d rows)", len(iter_df))

    print(f"ASIT file:         {asit_file}")
    print(f"Top bullish:       {len(bull_df)}")
    print(f"Top bearish:       {len(bear_df)}")
    print(f"API option rows:   {len(options_df)}")
    print(f"API iteration rows:{len(iter_df)}")

if __name__ == "__main__":
    main()

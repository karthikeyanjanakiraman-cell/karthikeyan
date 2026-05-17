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

try:
    import CEPEBUY_optional_integration as fy_mod
except Exception:
    fy_mod = None


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


def save_outputs(bull_df: pd.DataFrame, bear_df: pd.DataFrame, combined: pd.DataFrame):
    bull_df.to_csv(OUTPUT_BULL, index=False)
    bear_df.to_csv(OUTPUT_BEAR, index=False)
    combined.to_csv(OUTPUT_COMBINED, index=False)
    logger.info("Saved %s, %s, %s", OUTPUT_BULL.name, OUTPUT_BEAR.name, OUTPUT_COMBINED.name)


def main():
    asit_file = pick_latest_file(ASIT_GLOB)
    if asit_file is None:
        raise FileNotFoundError(f"No file found for pattern {ASIT_GLOB} in {BASE_DIR.resolve()}")
    logger.info("Using ASIT file: %s", asit_file.resolve())
    bull_df, bear_df, combined = build_top_lists(asit_file, top_n=TOP_N)
    save_outputs(bull_df, bear_df, combined)
    if fy_mod is not None:
        try:
            # optional FYERS scan hook; safe to ignore if unavailable
            fyers_options = []
            for sym in combined["Underlying"].head(10).tolist():
                chain = fy_mod.fetch_option_chain(sym)
                if not chain.empty:
                    fyers_options.append(chain)
            if fyers_options:
                pd.concat(fyers_options, ignore_index=True).to_csv(OUTPUT_DIR / "api_options.csv", index=False)
        except Exception as e:
            logger.warning("Optional FYERS scan skipped: %s", e)
    print(f"ASIT file: {asit_file}")
    print(f"Top bullish: {len(bull_df)}")
    print(f"Top bearish: {len(bear_df)}")
    print(f"Combined rows: {len(combined)}")


if __name__ == "__main__":
    main()

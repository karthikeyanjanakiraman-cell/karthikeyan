import os
import glob
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

GREEKS_GLOB = os.environ.get("GREEKS_GLOB", "asit_intraday_greeks_v3_0_*.csv")
ITERATION_GLOB = os.environ.get("ITERATION_GLOB", "fo_iteration_history_*.csv")

OUTPUT_FILE = OUTPUT_DIR / "cepebuy_output.csv"
MISSING_FILE = OUTPUT_DIR / "cepebuy_missing_report.csv"

MIN_OPTION_LTP = float(os.environ.get("MIN_OPTION_LTP", "10"))
MIN_REQUIRED_ITERATIONS = int(os.environ.get("MIN_REQUIRED_ITERATIONS", "1"))


def pick_latest_file(pattern: str, base_dir: Path = BASE_DIR) -> Path:
    files = sorted(base_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern} in {base_dir.resolve()}")
    return files[0]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %s: %s rows", path.name, len(df))
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def standardize_text_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    s = s.replace({"": np.nan, "NAN": np.nan, "NONE": np.nan, "NULL": np.nan})
    return s


def normalize_text(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = standardize_text_series(df[col])
    return df


def to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def find_first_column(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def extract_option_type_from_symbol(series: pd.Series) -> pd.Series:
    return standardize_text_series(series).str.extract(r"(CE|PE)\s*$", expand=False)


def extract_strike_from_symbol(series: pd.Series) -> pd.Series:
    s = standardize_text_series(series).str.replace("NSE:", "", regex=False)
    extracted = s.str.extract(r"\d{1,2}[A-Z]{3}(\d+(?:\.\d+)?)(?:CE|PE)$", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def extract_underlying_from_symbol(series: pd.Series) -> pd.Series:
    s = standardize_text_series(series).str.replace("NSE:", "", regex=False)
    extracted = s.str.extract(r"^([A-Z0-9&_-]+?)(?:\d{1,2}[A-Z]{3}\d+(?:\.\d+)?(?:CE|PE))$", expand=False)
    return extracted


def rename_greeks_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    alias_map = {
        "Underlying": ["underlying", "symbol", "stock", "name", "ticker", "root", "base_symbol"],
        "Option Type": ["option type", "option_type", "type", "cepe", "cp", "right", "optiontype"],
        "Strike": ["strike", "strike price", "strike_price", "strikeprice"],
        "LTP": ["ltp", "close", "price", "last_price", "last traded price", "last"],
        "Option Symbol": ["option symbol", "option_symbol", "tradingsymbol", "trading symbol", "instrument", "instrument_name", "symbol_name", "contract", "contract_symbol"],
    }
    rename_map = {}
    for std_col, aliases in alias_map.items():
        found = find_first_column(df, aliases)
        if found and found != std_col:
            rename_map[found] = std_col
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def normalize_greeks_df(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_greeks_columns(df)
    df = normalize_text(df, [c for c in ["Underlying", "Option Type", "Option Symbol"] if c in df.columns])
    for col in ["Underlying", "Option Type", "Strike", "LTP", "Option Symbol"]:
        if col not in df.columns:
            df[col] = np.nan
    df = to_numeric(df, ["Strike", "LTP"])
    if df["Option Symbol"].notna().any():
        miss_type = df["Option Type"].isna()
        if miss_type.any():
            df.loc[miss_type, "Option Type"] = extract_option_type_from_symbol(df.loc[miss_type, "Option Symbol"])
        miss_underlying = df["Underlying"].isna()
        if miss_underlying.any():
            df.loc[miss_underlying, "Underlying"] = extract_underlying_from_symbol(df.loc[miss_underlying, "Option Symbol"])
        miss_strike = df["Strike"].isna()
        if miss_strike.any():
            df.loc[miss_strike, "Strike"] = extract_strike_from_symbol(df.loc[miss_strike, "Option Symbol"])
    df["Option Type"] = standardize_text_series(df["Option Type"])
    df["Underlying"] = standardize_text_series(df["Underlying"])
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    df["LTP"] = pd.to_numeric(df["LTP"], errors="coerce")
    logger.info("Greeks missing summary | Underlying=%s | Option Type=%s | Strike=%s | LTP=%s | rows=%s", int(df["Underlying"].isna().sum()), int(df["Option Type"].isna().sum()), int(df["Strike"].isna().sum()), int(df["LTP"].isna().sum()), len(df))
    return df


def rename_iteration_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    alias_map = {
        "Option Symbol": ["option symbol", "option_symbol"],
        "Underlying": ["underlying"],
        "Strike": ["strike"],
        "Option Type": ["option type", "option_type"],
        "iteration": ["iteration"],
        "timestamp": ["timestamp"],
        "current_window_score": ["current_window_score"],
        "previous_trading_day_same_time_score": ["previous_trading_day_same_time_score"],
        "window_delta": ["window_delta"],
        "window_signal": ["window_signal"],
        "close": ["close"],
    }
    rename_map = {}
    for std_col, aliases in alias_map.items():
        found = find_first_column(df, aliases)
        if found and found != std_col:
            rename_map[found] = std_col
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def normalize_iteration_df(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_iteration_columns(df)
    df = normalize_text(df, ["Option Symbol", "Underlying", "Option Type", "window_signal"])
    df = to_numeric(df, ["Strike", "iteration", "current_window_score", "previous_trading_day_same_time_score", "window_delta", "close"])
    required = {"Underlying", "Option Type", "Strike", "iteration", "window_signal", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Iteration file missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["Underlying", "Option Type", "Strike", "iteration"])
    df["Option Type"] = standardize_text_series(df["Option Type"])
    df["Underlying"] = standardize_text_series(df["Underlying"])
    df["Strike"] = pd.to_numeric(df["Strike"], errors="coerce")
    df["iteration"] = pd.to_numeric(df["iteration"], errors="coerce")
    df = df.sort_values(["Underlying", "Option Type", "Strike", "iteration"]).reset_index(drop=True)
    logger.info("Iteration summary | rows=%s | underlyings=%s | option_symbols=%s", len(df), df["Underlying"].nunique(), df["Option Symbol"].nunique() if "Option Symbol" in df.columns else 0)
    return df


def build_candidate_rows(greeks_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["Underlying", "Option Type", "Strike", "LTP", "Option Symbol"] if c in greeks_df.columns]
    candidates = greeks_df[cols].copy()
    candidates["Underlying"] = standardize_text_series(candidates["Underlying"])
    candidates["Option Type"] = standardize_text_series(candidates["Option Type"])
    candidates["Strike"] = pd.to_numeric(candidates["Strike"], errors="coerce")
    if "LTP" in candidates.columns:
        candidates["LTP"] = pd.to_numeric(candidates["LTP"], errors="coerce")
    candidates = candidates.dropna(subset=["Underlying", "Option Type", "Strike"])
    if "LTP" in candidates.columns:
        candidates = candidates[candidates["LTP"].fillna(0) >= MIN_OPTION_LTP]
    candidates = candidates.drop_duplicates(subset=["Underlying", "Option Type", "Strike"]).reset_index(drop=True)
    logger.info("Candidate rows after cleanup: %s", len(candidates))
    return candidates


def latest_buy_entry(iter_slice: pd.DataFrame) -> Optional[pd.Series]:
    if iter_slice.empty:
        return None
    buy_df = iter_slice[iter_slice["window_signal"].astype(str).str.contains("BUY", case=False, na=False)].copy()
    if buy_df.empty:
        return None
    buy_df = buy_df.sort_values("iteration")
    return buy_df.iloc[-1]


def process_cepebuy() -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=== CEPEBUY starting ===")
    greeks_path = pick_latest_file(GREEKS_GLOB)
    logger.info("Using greeks file: %s", greeks_path.resolve())
    greeks_df = normalize_greeks_df(load_csv(greeks_path))
    iter_cols = ["Underlying", "Option Type", "Strike", "iteration", "window_signal", "close", "timestamp", "Option Symbol", "current_window_score", "previous_trading_day_same_time_score", "window_delta"]
    try:
        iteration_path = pick_latest_file(ITERATION_GLOB)
        logger.info("Using iteration file: %s", iteration_path.resolve())
        iter_df = normalize_iteration_df(load_csv(iteration_path))
    except FileNotFoundError:
        logger.warning("No iteration history file found; using empty iteration dataframe")
        iter_df = pd.DataFrame(columns=iter_cols)
    candidates = build_candidate_rows(greeks_df)
    results = []
    missing_rows = []
    for _, row in candidates.iterrows():
        underlying = row["Underlying"]
        option_type = row["Option Type"]
        strike = float(row["Strike"])
        greeks_ltp = row["LTP"] if "LTP" in row.index else np.nan
        greeks_option_symbol = row["Option Symbol"] if "Option Symbol" in row.index else np.nan
        if iter_df.empty:
            missing_rows.append({"Underlying": underlying, "Option Type": option_type, "Strike": strike, "Option Symbol": greeks_option_symbol, "Status": "NO_ITERATION_FILE", "Reason": "No iteration history file available"})
            continue
        iter_slice = iter_df[(iter_df["Underlying"] == underlying) & (iter_df["Option Type"] == option_type) & (iter_df["Strike"] == strike)].copy()
        if iter_slice.empty:
            missing_rows.append({"Underlying": underlying, "Option Type": option_type, "Strike": strike, "Option Symbol": greeks_option_symbol, "Status": "NO_ITERATION_ROWS", "Reason": "No rows found in latest iteration file for underlying/type/strike"})
            continue
        if len(iter_slice) < MIN_REQUIRED_ITERATIONS:
            missing_rows.append({"Underlying": underlying, "Option Type": option_type, "Strike": strike, "Option Symbol": greeks_option_symbol, "Status": "LOW_ITERATION_COUNT", "Reason": f"Only {len(iter_slice)} rows in iteration history"})
            continue
        entry = latest_buy_entry(iter_slice)
        if entry is None:
            missing_rows.append({"Underlying": underlying, "Option Type": option_type, "Strike": strike, "Option Symbol": greeks_option_symbol, "Status": "NO_ENTRY_CHAIN", "Reason": "No BUY signal present in iteration rows"})
            continue
        ordered = iter_slice.sort_values("iteration")
        last_row = ordered.iloc[-1]
        results.append({"Underlying": underlying, "Option Type": option_type, "Strike": strike, "Greeks Option Symbol": greeks_option_symbol, "Iter Option Symbol": last_row["Option Symbol"] if "Option Symbol" in last_row.index else np.nan, "Greeks LTP": greeks_ltp, "Entry Iteration": int(entry["iteration"]) if pd.notna(entry["iteration"]) else np.nan, "Entry Timestamp": entry["timestamp"] if "timestamp" in entry.index else "", "Entry Signal": entry["window_signal"], "Entry Close": entry["close"], "Entry Score": entry["current_window_score"] if "current_window_score" in entry.index else np.nan, "Entry Delta": entry["window_delta"] if "window_delta" in entry.index else np.nan, "Last Seen Iteration": int(last_row["iteration"]) if pd.notna(last_row["iteration"]) else np.nan, "Last Seen Signal": last_row["window_signal"], "Last Seen Close": last_row["close"], "Total Iter Rows": len(iter_slice)})
    out_df = pd.DataFrame(results)
    missing_df = pd.DataFrame(missing_rows)
    if not out_df.empty:
        out_df = out_df.sort_values(["Entry Iteration", "Entry Delta", "Total Iter Rows"], ascending=[False, False, False]).reset_index(drop=True)
        out_df.to_csv(OUTPUT_FILE, index=False)
        logger.info("Saved trade output: %s | rows=%s", OUTPUT_FILE, len(out_df))
    else:
        logger.info("No trade candidates found")
    missing_df.to_csv(MISSING_FILE, index=False)
    logger.info("Saved missing report: %s | rows=%s", MISSING_FILE, len(missing_df))
    if not missing_df.empty:
        logger.info("Missing status summary: %s", missing_df["Status"].value_counts().to_dict())
    return out_df, missing_df


def main():
    try:
        process_cepebuy()
    except Exception as e:
        logger.exception("CEPEBUY failed: %s", e)
        raise


if __name__ == "__main__":
    main()

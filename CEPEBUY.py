import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

BASE_DIR = Path(".")
GREEKS_GLOB = "asit_intraday_greeks_v3_0_*.csv"
ITERATION_GLOB = "fo_iteration_history_*.csv"

OUTPUT_FILE = "cepebuy_output.csv"
MISSING_REPORT_FILE = "cepebuy_missing_report.csv"
MIN_REQUIRED_ITERATIONS = 5


def pick_latest_file(pattern: str) -> Path:
    files = sorted(
        BASE_DIR.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not files:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return files[0]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    logging.info("Loaded %s: %s rows", path.name, len(df))
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_text(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            df.loc[df[col].isin(["", "NAN", "NONE"]), col] = np.nan
    return df


def normalize_iteration_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = normalize_text(df, ["Underlying", "Option Type", "window_signal", "Option Symbol"])

    for col in [
        "Strike",
        "iteration",
        "current_window_score",
        "previous_trading_day_same_time_score",
        "window_delta",
        "close",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = {"Underlying", "Option Type", "iteration", "window_signal", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Iteration file missing columns: {sorted(missing)}")

    df = df.dropna(subset=["Underlying", "Option Type", "iteration"])
    df = df.sort_values(["Underlying", "Option Type", "iteration"]).reset_index(drop=True)

    logging.info(
        "Iteration summary | rows=%s | underlyings=%s | option_symbols=%s",
        len(df),
        df["Underlying"].nunique(),
        df["Option Symbol"].nunique() if "Option Symbol" in df.columns else 0
    )
    return df


def extract_option_type_from_symbol(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.upper()
        .str.extract(r"(CE|PE)s*$", expand=False)
    )


def extract_underlying_from_symbol(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.replace("NSE:", "", regex=False)
    extracted = s.str.extract(
        r"^(?:NSE)?([A-Z0-9&_-]+?)(?:d{1,2}[A-Z]{3}d+(?:.d+)?(?:CE|PE))$",
        expand=False
    )
    return extracted


def normalize_greeks_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    rename_map = {}
    for c in df.columns:
        lc = c.lower().strip()

        if lc in {"symbol", "stock", "name", "underlying"}:
            rename_map[c] = "Underlying"
        elif lc in {"option type", "option_type", "type", "cepe", "cp", "right"}:
            rename_map[c] = "Option Type"
        elif lc in {"strike", "strike price", "strike_price"}:
            rename_map[c] = "Strike"
        elif lc in {"ltp", "close", "price", "last_price"}:
            rename_map[c] = "LTP"
        elif lc in {
            "option symbol", "option_symbol", "tradingsymbol", "trading symbol",
            "instrument", "instrument_name", "symbol_name"
        }:
            rename_map[c] = "Option Symbol"

    if rename_map:
        df = df.rename(columns=rename_map)

    df = normalize_text(
        df,
        [c for c in ["Underlying", "Option Type", "Option Symbol"] if c in df.columns]
    )

    if "Option Type" not in df.columns:
        df["Option Type"] = np.nan

    if "Option Type" in df.columns and df["Option Type"].isna().any() and "Option Symbol" in df.columns:
        fill_mask = df["Option Type"].isna()
        df.loc[fill_mask, "Option Type"] = extract_option_type_from_symbol(
            df.loc[fill_mask, "Option Symbol"]
        )

    if "Underlying" not in df.columns:
        df["Underlying"] = np.nan

    if df["Underlying"].isna().any() and "Option Symbol" in df.columns:
        fill_mask = df["Underlying"].isna()
        df.loc[fill_mask, "Underlying"] = extract_underlying_from_symbol(
            df.loc[fill_mask, "Option Symbol"]
        )

    for col in ["Strike", "LTP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Underlying" not in df.columns or df["Underlying"].isna().all():
        raise ValueError("Greeks file must contain Underlying or parsable Option Symbol")

    missing_opt = int(df["Option Type"].isna().sum())
    logging.info("Greeks Option Type missing rows: %s / %s", missing_opt, len(df))

    return df


def latest_buy_entry(iter_slice: pd.DataFrame):
    buy_df = iter_slice[
        iter_slice["window_signal"].astype(str).str.contains("BUY", case=False, na=False)
    ].copy()

    if buy_df.empty:
        return None

    buy_df = buy_df.sort_values("iteration")
    return buy_df.iloc[-1]


def build_candidate_rows(greeks_df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["Underlying", "Option Type", "Strike", "LTP", "Option Symbol"] if c in greeks_df.columns]
    if not keep:
        raise ValueError("Greeks file does not contain usable columns")

    candidates = greeks_df[keep].copy()

    if "Option Type" in candidates.columns:
        candidates = candidates.dropna(subset=["Option Type"])

    candidates = candidates.drop_duplicates().reset_index(drop=True)
    logging.info("Candidate rows after cleanup: %s", len(candidates))
    return candidates


def process_cepebuy():
    logging.info("=== CEPEBUY starting ===")

    greeks_path = pick_latest_file(GREEKS_GLOB)
    iteration_path = pick_latest_file(ITERATION_GLOB)

    logging.info("Using greeks file: %s", greeks_path.resolve())
    logging.info("Using iteration file: %s", iteration_path.resolve())

    greeks_df = normalize_greeks_df(load_csv(greeks_path))
    iter_df = normalize_iteration_df(load_csv(iteration_path))
    candidates = build_candidate_rows(greeks_df)

    results = []
    missing_rows = []

    for _, row in candidates.iterrows():
        underlying = row["Underlying"] if "Underlying" in row.index else np.nan
        option_type = row["Option Type"] if "Option Type" in row.index else np.nan
        greeks_option_symbol = row["Option Symbol"] if "Option Symbol" in row.index else ""

        if pd.isna(underlying):
            missing_rows.append({
                "Underlying": "",
                "Option Type": option_type if pd.notna(option_type) else "",
                "Option Symbol": greeks_option_symbol,
                "Status": "MISSING_UNDERLYING_IN_GREEKS",
                "Reason": "Underlying missing and could not be derived"
            })
            continue

        if pd.isna(option_type):
            missing_rows.append({
                "Underlying": underlying,
                "Option Type": "",
                "Option Symbol": greeks_option_symbol,
                "Status": "MISSING_OPTION_TYPE_IN_GREEKS",
                "Reason": "Option Type missing and could not be derived"
            })
            continue

        iter_slice = iter_df[
            (iter_df["Underlying"] == underlying) &
            (iter_df["Option Type"] == option_type)
        ].copy()

        if iter_slice.empty:
            missing_rows.append({
                "Underlying": underlying,
                "Option Type": option_type,
                "Option Symbol": greeks_option_symbol,
                "Status": "NO_ITERATION_ROWS",
                "Reason": "No rows found in latest iteration file for this underlying/type"
            })
            continue

        if len(iter_slice) < MIN_REQUIRED_ITERATIONS:
            missing_rows.append({
                "Underlying": underlying,
                "Option Type": option_type,
                "Option Symbol": greeks_option_symbol,
                "Status": "LOW_ITERATION_COUNT",
                "Reason": f"Only {len(iter_slice)} iteration rows"
            })
            continue

        entry = latest_buy_entry(iter_slice)
        if entry is None:
            missing_rows.append({
                "Underlying": underlying,
                "Option Type": option_type,
                "Option Symbol": greeks_option_symbol,
                "Status": "NO_ENTRY_CHAIN",
                "Reason": "No BUY signal present in iteration rows"
            })
            continue

        results.append({
            "Underlying": underlying,
            "Option Type": option_type,
            "Greeks Option Symbol": greeks_option_symbol,
            "Entry Strike": entry["Strike"] if "Strike" in entry.index else np.nan,
            "Entry Iteration": int(entry["iteration"]) if pd.notna(entry["iteration"]) else np.nan,
            "Entry Timestamp": entry["timestamp"] if "timestamp" in entry.index else "",
            "Entry Signal": entry["window_signal"],
            "Entry Close": entry["close"],
            "Entry Score": entry["current_window_score"] if "current_window_score" in entry.index else np.nan,
            "Entry Delta": entry["window_delta"] if "window_delta" in entry.index else np.nan,
            "Last Seen Iteration": int(iter_slice["iteration"].max()),
            "Last Seen Signal": iter_slice.iloc[-1]["window_signal"],
            "Last Seen Close": iter_slice.iloc[-1]["close"],
            "Total Iter Rows": len(iter_slice),
            "Greeks Strike": row["Strike"] if "Strike" in row.index else np.nan,
            "Greeks LTP": row["LTP"] if "LTP" in row.index else np.nan,
        })

    out_df = pd.DataFrame(results)
    missing_df = pd.DataFrame(missing_rows)

    if not out_df.empty:
        out_df = out_df.sort_values(
            ["Entry Iteration", "Entry Delta", "Total Iter Rows"],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        out_df.to_csv(OUTPUT_FILE, index=False)
        logging.info("Saved trade output: %s | rows=%s", OUTPUT_FILE, len(out_df))
    else:
        logging.info("No trade candidates found")

    if not missing_df.empty:
        missing_df.to_csv(MISSING_REPORT_FILE, index=False)
        logging.info("Saved missing report: %s | rows=%s", MISSING_REPORT_FILE, len(missing_df))

    status_counts = missing_df["Status"].value_counts().to_dict() if not missing_df.empty else {}
    logging.info("Missing status summary: %s", status_counts)

    return out_df, missing_df


def main():
    try:
        process_cepebuy()
    except Exception as e:
        logging.exception("CEPEBUY failed: %s", e)
        raise


if __name__ == "__main__":
    main()

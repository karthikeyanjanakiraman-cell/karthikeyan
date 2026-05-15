#!/usr/bin/env python3
# cepebuy.py  â€”  Full rewrite with CSV-based Exit15 / Exit39
# Run at 09:35, 10:00, 10:15, etc.  Auto-fills exits from history CSV.
# No threads. No timers. No API calls at minute 15 or 39.

import pandas as pd
import glob
import os
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List

HISTORY_DIR = "."
SIGNALS_JSON = "active_signals.json"
OUTPUT_CSV = "cepebuy_signals.csv"
HISTORY_TIME_COL = "datetime"
HISTORY_STOCK_COL = "Stock"
HISTORY_TYPE_COL = "Type"
HISTORY_STRIKE_COL = "Strike"
HISTORY_LTP_COL = "LTP"
_lock = threading.Lock()
SignalKey = Tuple[str, float, str]

def load_signals(path: str = SIGNALS_JSON) -> Dict[SignalKey, dict]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        parts = k.split("|")
        out[(parts[0], float(parts[1]), parts[2])] = v
    return out

def save_signals(signals: Dict[SignalKey, dict], path: str = SIGNALS_JSON) -> None:
    out = {}
    for k, v in signals.items():
        out[f"{k[0]}|{k[1]}|{k[2]}"] = v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

def get_today_history_path(directory: str = HISTORY_DIR) -> str:
    today = datetime.now().strftime("%Y%m%d")
    pattern = os.path.join(directory, f"fo_iteration_history_{today}_*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No history CSV found for pattern: {pattern}")
    return max(files, key=os.path.getmtime)

def read_history(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=[HISTORY_TIME_COL])
    df[HISTORY_STRIKE_COL] = df[HISTORY_STRIKE_COL].astype(float)
    df[HISTORY_STOCK_COL] = df[HISTORY_STOCK_COL].str.upper().str.strip()
    df[HISTORY_TYPE_COL] = df[HISTORY_TYPE_COL].str.upper().str.strip()
    return df

def get_exit_from_csv(df, entry_time_str, stock, opt_type, strike, exit_minutes):
    entry_dt = pd.to_datetime(entry_time_str)
    target = entry_dt + timedelta(minutes=exit_minutes)
    mask = (
        (df[HISTORY_TIME_COL] >= target) &
        (df[HISTORY_STOCK_COL] == stock.upper().strip()) &
        (df[HISTORY_TYPE_COL] == opt_type.upper().strip()) &
        (df[HISTORY_STRIKE_COL] == float(strike))
    )
    hit = df.loc[mask].head(1)
    if not hit.empty:
        return float(hit[HISTORY_LTP_COL].iloc[0])
    return None

def fill_pending_exits(signals, df, lock):
    filled = 0
    with lock:
        for key, row in signals.items():
            stock, strike, opt_type = key
            entry = row.get("Entry")
            if not entry:
                continue
            val15 = row.get("Exit15")
            if val15 in (None, "", "-", "PENDING", 0.0, "0.0"):
                ltp15 = get_exit_from_csv(df, entry, stock, opt_type, strike, 15)
                if ltp15 is not None:
                    row["Exit15"] = ltp15
                    filled += 1
                else:
                    row["Exit15"] = "PENDING"
            val39 = row.get("Exit39")
            if val39 in (None, "", "-", "PENDING", 0.0, "0.0"):
                ltp39 = get_exit_from_csv(df, entry, stock, opt_type, strike, 39)
                if ltp39 is not None:
                    row["Exit39"] = ltp39
                    filled += 1
                else:
                    row["Exit39"] = "PENDING"
            if isinstance(row.get("Exit15"), (int, float)):
                row["LTP"] = row["Exit15"]
    print(f"[EXIT FILL] Filled {filled} exit values")
    return filled

def scanner_get_new_signals():
    # TODO: paste your real scanner logic here
    # Return list of dicts with keys: Stock, Type, Signal, Strike, EntryPrice, LTP, Entry, Score
    return []

def main():
    signals = load_signals(SIGNALS_JSON)
    print(f"[STATE] Loaded {len(signals)} signals")
    history_path = get_today_history_path(HISTORY_DIR)
    print(f"[HISTORY] {history_path}")
    df_hist = read_history(history_path)
    fill_pending_exits(signals, df_hist, _lock)
    new_raw = scanner_get_new_signals()
    added = 0
    for raw in new_raw:
        stock = str(raw["Stock"]).upper().strip()
        opt_type = str(raw["Type"]).upper().strip()
        strike = float(raw["Strike"])
        key = (stock, strike, opt_type)
        row = {
            "Stock": stock,
            "Type": opt_type,
            "Signal": raw.get("Signal"),
            "Strike": strike,
            "EntryPrice": float(raw.get("EntryPrice", 0)),
            "LTP": float(raw.get("LTP", 0)),
            "Entry": raw.get("Entry"),
            "Score": float(raw.get("Score", 0)),
            "Exit15": "PENDING",
            "Exit39": "PENDING",
        }
        with _lock:
            signals[key] = row
        added += 1
        print(f"[NEW] {key} Entry={row[chr(39)+chr(39)]Entry{chr(39)+chr(39)]} Score={row[chr(39)+chr(39)]Score{chr(39)+chr(39)]}")
    print(f"[SCAN] Added {added}; total={len(signals)}")
    save_signals(signals, SIGNALS_JSON)
    rows = []
    with _lock:
        for key, row in signals.items():
            rows.append(row)
    if rows:
        df_out = pd.DataFrame(rows)
        cols = ["Stock","Type","Signal","Strike","EntryPrice","LTP","Entry","Exit15","Exit39","Score"]
        cols = [c for c in cols if c in df_out.columns]
        df_out = df_out[cols]
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"[OUT] {OUTPUT_CSV} ({len(df_out)} rows)")
    print("\n" + "="*70)
    print(f"{'Stock':<12} {'Type':<4} {'Strike':<8} {'Entry':<20} {'Exit15':<10} {'Score':<6}")
    print("-"*70)
    with _lock:
        for key, row in signals.items():
            print(f"{row[chr(39)+chr(39)]Stock{chr(39)+chr(39)]:<12} {row[chr(39)+chr(39)]Type{chr(39)+chr(39)]:<4} "
                  f"{row[chr(39)+chr(39)]Strike{chr(39)+chr(39)]:<8} {str(row.get(chr(39)+chr(39)]Entry{chr(39)+chr(39),chr(39)+chr(39)):<20} "
                  f"{str(row.get(chr(39)+chr(39)]Exit15{chr(39)+chr(39),chr(39)+chr(39)):<10} {row.get(chr(39)+chr(39)]Score{chr(39)+chr(39),chr(39)+chr(39)):<6}")
    print("="*70)

if __name__ == "__main__":
    main()

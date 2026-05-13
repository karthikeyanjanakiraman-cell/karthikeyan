# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# OPTIONS_OI.py  â€”  Chain-Based Signal System
# Entry : 5m CONFIRMED + 30m CONFIRMED + T30 >= 10
# Exit  : 30m BROKEN  OR  T30 < 10
# Email : LONG (CE) table + SHORT (PE) table  â€” sorted by Entry Time
#         Columns: Underlying | Option Type | Strike | LTP |
#                  5m | T5 | 15m | T15 | 30m | T30 |
#                  Chain Signal | Exit Signal | Entry Time
#         Sticky: once fired ENTER, row retained all day
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import os
import time
import smtplib
import logging
import json
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

try:
    from fyers_apiv3 import fyersModel
except Exception:
    try:
        import fyersModel
    except Exception:
        from fyersapi import fyersModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
INTRADAY_LOOKBACK_DAYS = 20
SIGNAL_WINDOW_MINUTES  = 5
ITERATIONS_TO_KEEP     = 75
SECTORS_DIR            = os.environ.get("SECTORS_DIR", "sectors")
OUTPUT_DIR             = os.environ.get("OUTPUT_DIR", ".")
MIN_OPTION_LTP         = 10.0
MIN_ATM_CHAIN_VOLUME   = int(os.environ.get("MIN_ATM_CHAIN_VOLUME", "100000"))
PER_SYMBOL_SLEEP_SEC   = float(os.environ.get("PER_SYMBOL_SLEEP_SEC", "0.25"))
TOP_N_UNDERLYINGS      = int(os.environ.get("TOP_N_UNDERLYINGS", "60"))
OPTION_PAIRS_TO_KEEP   = 5
T30_MIN                = 10
DAILY_STATE_FILE       = os.path.join(OUTPUT_DIR, "chain_signal_state.json")

fyers = None

# â”€â”€ Fyers init â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def init_fyers() -> Optional[object]:
    global fyers
    client_id    = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
    access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
    if not client_id or not access_token:
        logger.error("Missing CLIENT_ID / ACCESS_TOKEN environment variables.")
        fyers = None
        return None
    fyers = fyersModel.FyersModel(
        client_id=client_id, token=access_token,
        is_async=False, log_path=""
    )
    return fyers

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default

def format_eq_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if symbol.startswith("NSE:"):
        return symbol if symbol.endswith("-EQ") else f"{symbol}-EQ"
    return f"NSE:{symbol}-EQ"

def get_history(symbol: str, resolution: str, days_back: int) -> pd.DataFrame:
    if fyers is None:
        return pd.DataFrame()
    now   = datetime.now()
    start = now - timedelta(days=days_back)
    payload = {
        "symbol":     symbol,
        "resolution": resolution,
        "date_format":"1",
        "range_from": start.strftime("%Y-%m-%d"),
        "range_to":   now.strftime("%Y-%m-%d"),
        "cont_flag":  "1",
    }
    try:
        res = fyers.history(data=payload)
    except Exception:
        return pd.DataFrame()
    candles = (res or {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles,
                      columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert("Asia/Kolkata")
        .dt.tz_localize(None)
    )
    return (df.sort_values("timestamp")
              .drop_duplicates(subset=["timestamp"], keep="last")
              .reset_index(drop=True))

def nearest_step(value: float) -> int:
    v = abs(safe_float(value, 0))
    if v >= 20000: return 100
    if v >= 10000: return 50
    if v >= 2000:  return 20
    if v >= 500:   return 10
    if v >= 100:   return 5
    return 1

# â”€â”€ Sector / symbol loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def discover_sector_csvs(root_dir: str = SECTORS_DIR) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                paths.append(os.path.join(dirpath, fname))
    return sorted(set(paths))

def load_fno_symbols_from_sectors(root_dir: str = SECTORS_DIR) -> List[str]:
    symbols = set()
    for path in discover_sector_csvs(root_dir):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        lowered = {str(c).strip().lower(): c for c in df.columns}
        symbol_col = None
        for key in ["symbol", "symbols", "ticker", "tradingsymbol"]:
            if key in lowered:
                symbol_col = lowered[key]
                break
        if symbol_col is None:
            continue
        for raw in df[symbol_col].dropna().astype(str):
            sym = raw.strip().upper()
            if sym and sym not in {"NAN", "NONE"}:
                symbols.add(sym)
    return sorted(symbols)

# â”€â”€ Option chain fetch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def fetch_option_chain(symbol: str,
                       pair_count: int = OPTION_PAIRS_TO_KEEP
                       ) -> Tuple[pd.DataFrame, bool]:
    """
    Returns (chain_df, atm_vol_ok).
    chain_df  : Strike | Option Type | Option Symbol | OI | Chain Volume | LTP
    atm_vol_ok: False  â†’ skip this symbol entirely (ATM volume too low)
    """
    if fyers is None:
        return pd.DataFrame(), False
    eq_sym = format_eq_symbol(symbol)
    try:
        quote     = fyers.quotes({"symbols": eq_sym})
        ltp       = safe_float(
            quote.get("d", [{}])[0].get("v", {}).get("lp"), np.nan)
        chain_res = fyers.optionchain(data={"symbol": eq_sym, "strikecount": 50})
    except Exception:
        return pd.DataFrame(), False
    chain = ((chain_res or {}).get("data") or {}).get("optionsChain", [])
    if not chain:
        return pd.DataFrame(), False

    rows = []
    for item in chain:
        strike = safe_float(
            item.get("strike_price") or item.get("strike"), np.nan)
        typ = str(item.get("option_type") or item.get("type") or "").upper()
        if pd.isna(strike) or typ not in {"CE", "PE"}:
            continue
        rows.append({
            "Strike":        strike,
            "Option Type":   typ,
            "Option Symbol": str(item.get("symbol", "")),
            "OI":            safe_float(
                item.get("oi") or item.get("open_interest"), 0),
            "Chain Volume":  safe_float(item.get("volume"), 0),
            "LTP":           safe_float(
                item.get("ltp") or item.get("lp"), 0),
        })
    if not rows:
        return pd.DataFrame(), False

    oc   = pd.DataFrame(rows)
    step = nearest_step(ltp if pd.notna(ltp) else oc["Strike"].median())
    atm  = (round(ltp / step) * step
            if pd.notna(ltp) else oc["Strike"].median())

    # ATM volume gate â€” at least one of CE/PE must pass
    atm_rows = oc[oc["Strike"] == atm]
    atm_vol_ok = False
    for opt_type in ["CE", "PE"]:
        sub = atm_rows[atm_rows["Option Type"] == opt_type]
        if not sub.empty:
            vol = safe_float(sub["Chain Volume"].iloc[0], 0)
            if vol >= MIN_ATM_CHAIN_VOLUME:
                atm_vol_ok = True
                break
    if not atm_vol_ok:
        logger.debug("SKIP %s: ATM volume too low", symbol)
        return pd.DataFrame(), False

    strikes = sorted(
        oc["Strike"].dropna().unique(),
        key=lambda x: abs(x - atm)
    )[:pair_count]
    return oc[oc["Strike"].isin(strikes)].reset_index(drop=True), True

# â”€â”€ Per-candle window signals (CORRECT: one signal per candle vs prev day) â”€â”€â”€â”€
def build_iteration_history(intra_df: pd.DataFrame,
                             window_minutes: int = SIGNAL_WINDOW_MINUTES,
                             iterations: int = ITERATIONS_TO_KEEP
                             ) -> pd.DataFrame:
    """
    For EACH 5-min candle on today compute:
      current_score  = % change in last `window_minutes`
      prev_score     = same window on previous trading day same candle
      window_signal  = Buy++/Buy+/Buy/Sell++/Sell+/Sell/Neutral
    Returns DataFrame per candle â€” NOT a single global signal.
    """
    if intra_df is None or intra_df.empty:
        return pd.DataFrame()

    full_df = intra_df.copy().sort_values("timestamp").reset_index(drop=True)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    today    = full_df["timestamp"].dt.date.max()
    today_df = full_df[full_df["timestamp"].dt.date == today].copy().reset_index(drop=True)

    anchor_s = pd.Timestamp.combine(today, dtime(9, 15))
    anchor_e = pd.Timestamp.combine(today, dtime(15, 30))
    today_df = today_df[
        (today_df["timestamp"] >= anchor_s) &
        (today_df["timestamp"] <= anchor_e)
    ].reset_index(drop=True)
    if today_df.empty:
        return pd.DataFrame()

    # Build prev-day lookup  {(hour, minute): score}
    trading_days = sorted(full_df["timestamp"].dt.date.unique())
    prev_days    = [d for d in trading_days if d < today]
    prev_lookup: Dict[Tuple[int, int], float] = {}
    if prev_days:
        prev_df = full_df[full_df["timestamp"].dt.date == prev_days[-1]].copy()
        for i in range(len(prev_df)):
            end_ts   = pd.to_datetime(prev_df.iloc[i]["timestamp"])
            start_ts = end_ts - timedelta(minutes=window_minutes)
            pw = prev_df[
                (prev_df["timestamp"] >= start_ts) &
                (prev_df["timestamp"] <= end_ts)
            ]
            if len(pw) >= 2:
                fc = safe_float(pw["close"].iloc[0])
                lc = safe_float(pw["close"].iloc[-1])
                if pd.notna(fc) and fc != 0:
                    prev_lookup[(end_ts.hour, end_ts.minute)] = round(
                        (lc - fc) / fc * 100, 2)

    rows = []
    for i in range(len(today_df)):
        end_ts   = pd.to_datetime(today_df.iloc[i]["timestamp"])
        start_ts = end_ts - timedelta(minutes=window_minutes)
        cur = today_df[
            (today_df["timestamp"] >= start_ts) &
            (today_df["timestamp"] <= end_ts)
        ]
        if len(cur) < 2:
            continue
        fc = safe_float(cur["close"].iloc[0])
        lc = safe_float(cur["close"].iloc[-1])
        if pd.isna(fc) or fc == 0:
            continue
        cur_score  = round((lc - fc) / fc * 100, 2)
        prev_score = prev_lookup.get((end_ts.hour, end_ts.minute), np.nan)

        if pd.isna(prev_score):
            signal = "Neutral"
        else:
            delta = round(cur_score - prev_score, 2)
            if delta >= 0.50:    signal = "Buy++"
            elif delta >= 0.20:  signal = "Buy+"
            elif delta >= 0.05:  signal = "Buy"
            elif delta <= -0.50: signal = "Sell++"
            elif delta <= -0.20: signal = "Sell+"
            elif delta <= -0.05: signal = "Sell"
            else:                signal = "Neutral"

        rows.append({
            "iteration":     len(rows) + 1,
            "timestamp":     end_ts.strftime("%H:%M"),
            "window_start":  start_ts.strftime("%H:%M"),
            "window_end":    end_ts.strftime("%H:%M"),
            "current_score": cur_score,
            "prev_score":    round(prev_score, 2) if pd.notna(prev_score) else np.nan,
            "window_delta":  round(cur_score - prev_score, 2) if pd.notna(prev_score) else np.nan,
            "window_signal": signal,
            "close":         lc,
        })
        if len(rows) >= iterations:
            break
    return pd.DataFrame(rows)

# â”€â”€ Chain status from a list of per-candle signals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_chain_status(signals: List[str]) -> Tuple[str, int, int, int]:
    """
    signals : list of window_signal strings
    Returns : (status, buy_count, sell_count, total_count)
    status  : CONFIRMED | MIXED | BROKEN
    """
    b = sum(1 for s in signals if str(s).startswith("Buy"))
    s = sum(1 for s in signals if str(s).startswith("Sell"))
    t = b + s
    if t == 0:
        return "MIXED", 0, 0, 0
    ratio = b / t
    if ratio >= 0.65:
        return "CONFIRMED", b, s, t
    elif ratio <= 0.35:
        return "BROKEN", b, s, t
    else:
        return "MIXED", b, s, t

# â”€â”€ Latest chain state for a given block size â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def latest_chain_state(iter_df: pd.DataFrame,
                        block_minutes: int
                        ) -> Tuple[str, int, int, int]:
    """
    Extracts the CURRENT (latest) block's signals from iter_df
    and returns chain status for that block.
    """
    if iter_df is None or iter_df.empty:
        return "MIXED", 0, 0, 0
    iters_per_block = max(1, block_minutes // SIGNAL_WINDOW_MINUTES)
    last_it   = int(iter_df["iteration"].max())
    block_num = ((last_it - 1) // iters_per_block) + 1
    b_start   = (block_num - 1) * iters_per_block + 1
    b_end     = block_num * iters_per_block
    block_rows = iter_df[
        (iter_df["iteration"] >= b_start) &
        (iter_df["iteration"] <= b_end)
    ]
    signals = block_rows["window_signal"].tolist() if not block_rows.empty else []
    return compute_chain_status(signals)

# â”€â”€ Per-option scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def scan_option(option_symbol: str,
                option_type: str,
                strike: float,
                underlying: str) -> Optional[Dict]:
    sym = (option_symbol
           if option_symbol.startswith("NSE:")
           else f"NSE:{option_symbol}")
    intra_df = get_history(sym, "5", INTRADAY_LOOKBACK_DAYS)
    if intra_df is None or intra_df.empty:
        return None

    ltp = safe_float(intra_df["close"].iloc[-1])
    if pd.isna(ltp) or ltp < MIN_OPTION_LTP:
        return None

    # Per-candle iteration history (correct per-candle signals)
    iter_df = build_iteration_history(intra_df)
    if iter_df.empty:
        return None

    # Chain state for 5m / 15m / 30m blocks
    stat5,  b5,  s5,  t5  = latest_chain_state(iter_df, 5)
    stat15, b15, s15, t15 = latest_chain_state(iter_df, 15)
    stat30, b30, s30, t30 = latest_chain_state(iter_df, 30)

    c5  = "CONFIRMED" if stat5  == "CONFIRMED" else ("BROKEN" if stat5  == "BROKEN" else "MIXED")
    c15 = "CONFIRMED" if stat15 == "CONFIRMED" else ("BROKEN" if stat15 == "BROKEN" else "MIXED")
    c30 = "CONFIRMED" if stat30 == "CONFIRMED" else ("BROKEN" if stat30 == "BROKEN" else "MIXED")

    # Entry rule
    entry_ok = (stat5  == "CONFIRMED" and
                stat30 == "CONFIRMED" and
                t30 >= T30_MIN)

    # Exit rule
    exit_ok  = (stat30 == "BROKEN" or t30 < T30_MIN)

    if entry_ok:
        chain_signal = "ENTER"
    elif exit_ok:
        chain_signal = "EXIT"
    else:
        chain_signal = "WAIT"

    exit_signal = "EXIT NOW" if exit_ok else "HOLD"

    return {
        "Underlying":    underlying,
        "Option Type":   option_type,
        "Option Symbol": option_symbol,
        "Strike":        strike,
        "LTP":           round(ltp, 2),
        "5m":            c5,
        "T5":            t5,
        "15m":           c15,
        "T15":           t15,
        "30m":           c30,
        "T30":           t30,
        "Chain Signal":  chain_signal,
        "Exit Signal":   exit_signal,
        "_iter_df":      iter_df,   # internal â€” stripped before JSON save
    }

# â”€â”€ Sticky state  (all-day persistence in JSON) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def load_daily_state() -> Dict:
    today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(DAILY_STATE_FILE):
        try:
            with open(DAILY_STATE_FILE) as f:
                state = json.load(f)
            if state.get("date") == today:
                return state
        except Exception:
            pass
    return {"date": today, "rows": {}}

def save_daily_state(state: Dict):
    try:
        with open(DAILY_STATE_FILE, "w") as f:
            json.dump(state, f, default=str)
    except Exception as e:
        logger.warning("Could not save state: %s", e)

def update_sticky_rows(state: Dict, new_rows: List[Dict]) -> List[Dict]:
    """
    Unified CE + PE state in one dict.
    Key = Underlying|OptionType|Strike

    Logic:
    - New row with Chain Signal = ENTER  â†’ add with Entry Time frozen
    - Existing row                       â†’ update live columns, Entry Time unchanged
    - Returns all rows sorted by Entry Time ascending
    """
    existing = state.get("rows", {})
    live_cols = ["LTP", "5m", "T5", "15m", "T15", "30m", "T30",
                 "Chain Signal", "Exit Signal"]

    for row in new_rows:
        key   = f"{row['Underlying']}|{row['Option Type']}|{row['Strike']}"
        clean = {k: v for k, v in row.items() if not k.startswith("_")}
        if key not in existing:
            if row.get("Chain Signal") == "ENTER":
                clean["Entry Time"] = datetime.now().strftime("%H:%M")
                existing[key] = clean
        else:
            for col in live_cols:
                if col in clean:
                    existing[key][col] = clean[col]

    state["rows"] = existing
    save_daily_state(state)
    return sorted(existing.values(),
                  key=lambda r: r.get("Entry Time", "99:99"))

# â”€â”€ HTML email â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EMAIL_STYLE = """
<style>
  body { font-family: Arial, sans-serif; font-size: 12px; }
  h3   { margin: 14px 0 4px 0; padding: 8px 12px;
         border-radius: 4px; font-size: 13px; }
  table { border-collapse: collapse; width: 100%; margin-bottom: 22px; }
  th  { background: #1a1a2e; color: #e0e0e0; padding: 6px 8px;
        font-size: 11px; text-align: center; border: 1px solid #444; }
  td  { padding: 5px 7px; border: 1px solid #ddd;
        text-align: center; font-size: 11px; }
  tr:nth-child(even) { background: #f7f7f7; }
  .confirmed { color: #155724; font-weight: bold; }
  .broken    { color: #721c24; font-weight: bold; }
  .mixed     { color: #856404; }
  .enter     { background: #d4edda; color: #155724; font-weight: bold; }
  .exit_now  { background: #f8d7da; color: #721c24; font-weight: bold; }
  .hold      { background: #fff3cd; color: #856404; }
  .wait      { color: #555; }
  .long_h    { background: #155724; color: white; }
  .short_h   { background: #721c24; color: white; }
  .info      { font-size: 11px; color: #555; margin-bottom: 10px; }
  .ts        { color: #999; font-size: 10px; }
</style>
"""

DISPLAY_COLS = [
    "Underlying", "Option Type", "Strike", "LTP",
    "5m", "T5", "15m", "T15", "30m", "T30",
    "Chain Signal", "Exit Signal", "Entry Time"
]

def _td(col: str, val) -> str:
    v   = str(val) if val is not None else ""
    cls = ""
    if col in ("5m", "15m", "30m"):
        cls = ("confirmed" if v == "CONFIRMED"
               else "broken" if v == "BROKEN"
               else "mixed")
    elif col == "Chain Signal":
        cls = ("enter"    if v == "ENTER"
               else "exit_now" if v == "EXIT"
               else "wait")
    elif col == "Exit Signal":
        cls = "exit_now" if v == "EXIT NOW" else "hold"
    return f'<td class="{cls}">{v}</td>'

def build_table_html(rows: List[Dict], title: str, css_class: str) -> str:
    header = "".join(f"<th>{c}</th>" for c in DISPLAY_COLS)
    if rows:
        body = "".join(
            "<tr>" + "".join(_td(c, r.get(c, "")) for c in DISPLAY_COLS) + "</tr>"
            for r in rows
        )
    else:
        body = (f'<tr><td colspan="{len(DISPLAY_COLS)}" '
                f'style="color:#999;padding:10px;">â€” No signals yet â€”</td></tr>')
    return (f'<h3 class="{css_class}">{title}</h3>'
            f'<table><thead><tr>{header}</tr></thead>'
            f'<tbody>{body}</tbody></table>')

def build_email_html(long_rows: List[Dict], short_rows: List[Dict]) -> str:
    ts = datetime.now().strftime("%d %b %Y  %H:%M IST")
    long_tbl  = build_table_html(long_rows,  "ðŸ“ˆ LONG CANDIDATES  (CE)", "long_h")
    short_tbl = build_table_html(short_rows, "ðŸ“‰ SHORT CANDIDATES (PE)", "short_h")
    return f"""<html><head>{EMAIL_STYLE}</head>
<body>
<p class="ts">Chain Signal Report â€” {ts}</p>
<p class="info">
  <b>Entry:</b> 5m CONFIRMED + 30m CONFIRMED + T30 &ge; {T30_MIN}
  &nbsp;|&nbsp;
  <b>Exit:</b> 30m BROKEN or T30 &lt; {T30_MIN}
  &nbsp;|&nbsp;
  <b>Sticky:</b> rows retained all day once ENTER fires
</p>
{long_tbl}
{short_tbl}
</body></html>"""

# â”€â”€ Send email â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def send_email(html: str, subject: str = None):
    sender   = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = os.environ.get("EMAIL_RECEIVER")
    if not all([sender, password, receiver]):
        logger.warning("Email env vars not set â€” skipping send.")
        return
    subject = subject or f"Chain Signal {datetime.now().strftime('%H:%M')}"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = receiver
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        logger.info("Email sent: %s", subject)
    except Exception as e:
        logger.error("Email failed: %s", e)

# â”€â”€ Main scan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_scan():
    init_fyers()
    state      = load_daily_state()
    symbols    = load_fno_symbols_from_sectors()[:TOP_N_UNDERLYINGS]
    all_scanned: List[Dict] = []

    logger.info("Scanning %d symbols ...", len(symbols))
    for sym in symbols:
        try:
            chain_df, vol_ok = fetch_option_chain(sym, OPTION_PAIRS_TO_KEEP)
        except Exception as e:
            logger.warning("fetch_option_chain %s: %s", sym, e)
            continue
        if not vol_ok or chain_df.empty:
            continue

        for _, row in chain_df.iterrows():
            opt_sym  = str(row.get("Option Symbol", ""))
            opt_type = str(row.get("Option Type",  "")).upper()
            strike   = safe_float(row.get("Strike"), np.nan)
            if not opt_sym or opt_type not in {"CE", "PE"} or pd.isna(strike):
                continue
            try:
                result = scan_option(opt_sym, opt_type, strike, sym)
            except Exception as e:
                logger.warning("scan_option %s: %s", opt_sym, e)
                result = None
            if result:
                all_scanned.append(result)
            time.sleep(PER_SYMBOL_SLEEP_SEC)

    # Single unified sticky-row merge (CE + PE together)
    all_rows = update_sticky_rows(state, all_scanned)

    # Split for display â€” sort by Entry Time
    long_rows  = sorted(
        [r for r in all_rows if r.get("Option Type") == "CE"],
        key=lambda r: r.get("Entry Time", "99:99")
    )
    short_rows = sorted(
        [r for r in all_rows if r.get("Option Type") == "PE"],
        key=lambda r: r.get("Entry Time", "99:99")
    )

    # Build and send email
    html = build_email_html(long_rows, short_rows)
    send_email(html)

    # Save CSVs
    ts_str = datetime.now().strftime("%Y%m%d_%H%M")
    def _save(rows: List[Dict], tag: str):
        if not rows:
            return
        (pd.DataFrame(rows)
           .drop(columns=["_iter_df"], errors="ignore")
           .to_csv(os.path.join(OUTPUT_DIR, f"chain_{tag}_{ts_str}.csv"),
                   index=False))
    _save(long_rows,  "long")
    _save(short_rows, "short")
    logger.info("Done â€” Long: %d  |  Short: %d", len(long_rows), len(short_rows))


if __name__ == "__main__":
    run_scan()

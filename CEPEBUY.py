#!/usr/bin/env python3
import os
import time
import smtplib
import logging
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

try:
    from fyers_apiv3 import fyersModel
except Exception:
    try:
        import fyersModel
    except Exception:
        try:
            from fyersapi import fyersModel
        except Exception:
            fyersModel = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DAILY_LOOKBACK_DAYS = 252
INTRADAY_LOOKBACK_DAYS = 20
OPTION_PAIRS_TO_KEEP = int(os.environ.get("OPTION_PAIRS_TO_KEEP", "5"))
SIGNAL_WINDOW_MINUTES = int(os.environ.get("SIGNAL_WINDOW_MINUTES", "5"))
ITERATIONS_TO_KEEP = int(os.environ.get("ITERATIONS_TO_KEEP", "75"))
SECTORS_DIR = os.environ.get("SECTORS_DIR", "sectors")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
MIN_OPTION_LTP = float(os.environ.get("MIN_OPTION_LTP", "5"))
MIN_ATM_CHAIN_VOLUME = int(os.environ.get("MIN_ATM_CHAIN_VOLUME", "50000"))
MIN_OPTION_DAY_VOLUME = int(os.environ.get("MIN_OPTION_DAY_VOLUME", "50000"))
PER_SYMBOL_SLEEP_SEC = float(os.environ.get("PER_SYMBOL_SLEEP_SEC", "0.10"))
EMAIL_MAX_ROWS_LONG = int(os.environ.get("EMAIL_MAX_ROWS_LONG", "25"))
EMAIL_MAX_ROWS_SHORT = int(os.environ.get("EMAIL_MAX_ROWS_SHORT", "25"))
TOP_N_UNDERLYINGS = int(os.environ.get("TOP_N_UNDERLYINGS", "20"))
MIN_CHAIN_LEGS = int(os.environ.get("MIN_CHAIN_LEGS", "3"))
REUSE_SHORTLIST = str(os.environ.get("REUSE_SHORTLIST", "1")).strip().lower() in {"0", "false", "no", "n"}
TODAY_TAG = datetime.now().strftime("%Y%m%d")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, f"fo_summary_{TODAY_TAG}.csv")
LONG_SEED_PATH = os.path.join(OUTPUT_DIR, f"fo_long_seed_{TODAY_TAG}.csv")
SHORT_SEED_PATH = os.path.join(OUTPUT_DIR, f"fo_short_seed_{TODAY_TAG}.csv")
LONG_CANDIDATES_PATH = os.path.join(OUTPUT_DIR, f"fo_long_candidates_{TODAY_TAG}.csv")
SHORT_CANDIDATES_PATH = os.path.join(OUTPUT_DIR, f"fo_short_candidates_{TODAY_TAG}.csv")
ITER_PATH = os.path.join(OUTPUT_DIR, f"fo_iteration_history_{TODAY_TAG}.csv")
CHAIN_LONG_PATH = os.path.join(OUTPUT_DIR, f"chain_long_{TODAY_TAG}.csv")
CHAIN_SHORT_PATH = os.path.join(OUTPUT_DIR, f"chain_short_{TODAY_TAG}.csv")
STATE_CSV_PATH = os.path.join(OUTPUT_DIR, f"chain_state_{TODAY_TAG}.csv")

OPTION_EMAIL_COLS = [
    "Underlying",
    "Option Type",
    "Option Symbol",
    "Strike",
    "LTP",
    "OI",
    "Volume",
    "OBV",
    "OI+Volume+OBV Score",
    "EMAIL_RANK_SCORE",
    "Entry Value",
    "Entry Time",
    "Exit Value",
    "Exit Time",
    "Chain Signal",
    "Chain Legs",
]

OPTION_EMAIL_COL_RENAME = {
    "OI+Volume+OBV Score": "Liq Score",
    "EMAIL_RANK_SCORE": "Rank",
    "Option Symbol": "Opt Symbol",
    "Entry Value": "Entry",
    "Exit Value": "Exit",
}

fyers = None


def safe_float(value, default=np.nan) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def initfyers() -> Optional[object]:
    global fyers
    if fyers is not None:
        return fyers
    if fyersModel is None:
        logger.warning("Fyers SDK not available; running in dry mode")
        return None

    clientid = os.environ.get("CLIENTID") or os.environ.get("CLIENT_ID")
    accesstoken = os.environ.get("ACCESSTOKEN") or os.environ.get("ACCESS_TOKEN")
    if not clientid or not accesstoken:
        logger.warning("Missing CLIENTID / ACCESSTOKEN environment variables")
        return None

    try:
        fyers = fyersModel.FyersModel(client_id=clientid, token=accesstoken, is_async=False, log_path="")
        logger.info("Fyers client initialized")
        return fyers
    except Exception as e:
        logger.exception("Failed to initialize Fyers: %s", e)
        fyers = None
        return None



def find_latest_asit_csv(rootdir: str = ".") -> Optional[str]:
    matches = []
    for dirpath, _, filenames in os.walk(rootdir):
        for fname in filenames:
            low = fname.lower()
            if low.startswith("asit") and low.endswith(".csv"):
                path = os.path.join(dirpath, fname)
                try:
                    matches.append((os.path.getmtime(path), path))
                except Exception:
                    pass
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][1]


def normalize_underlying_symbol(value: str) -> str:
    s = str(value or "").strip().upper().replace("NSE:", "")
    if s.endswith("-EQ"):
        s = s[:-3]
    return s.strip()


def load_asit_shortlist(path: str, topn: int = TOP_N_UNDERLYINGS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in df.columns}
    sym_col = cols.get("symbol")
    rank_col = cols.get("rankscore15tier") or cols.get("rank")
    bull_col = cols.get("bullmultitfscore")
    bear_col = cols.get("bearmultitfscore")
    trend_col = cols.get("dominanttrend")
    if not sym_col:
        raise RuntimeError(f"ASIT csv missing Symbol column: {path}")
    work = df.copy()
    work["Symbol"] = work[sym_col].astype(str).map(normalize_underlying_symbol)
    work["__rank"] = pd.to_numeric(work[rank_col], errors="coerce").fillna(0) if rank_col and rank_col in work.columns else 0.0
    work["__bull"] = pd.to_numeric(work[bull_col], errors="coerce").fillna(0) if bull_col and bull_col in work.columns else 0.0
    work["__bear"] = pd.to_numeric(work[bear_col], errors="coerce").fillna(0) if bear_col and bear_col in work.columns else 0.0
    trends = work[trend_col].astype(str).str.upper() if trend_col and trend_col in work.columns else pd.Series("", index=work.index)
    bull_mask = trends.str.contains("BULL") | (work["__bull"] >= work["__bear"])
    bear_mask = trends.str.contains("BEAR") | (work["__bear"] > work["__bull"])
    long_df = work[bull_mask].copy().sort_values(["__rank", "__bull", "__bear"], ascending=[False, False, True]).drop_duplicates(subset=["Symbol"]).head(topn)
    short_df = work[bear_mask].copy().sort_values(["__rank", "__bear", "__bull"], ascending=[False, False, True]).drop_duplicates(subset=["Symbol"]).head(topn)
    return long_df[["Symbol"]].reset_index(drop=True), short_df[["Symbol"]].reset_index(drop=True)


def format_eq_symbol(symbol: str) -> str:
    symbol = str(symbol).strip().upper()
    if symbol.startswith("NSE:") and symbol.endswith("-EQ"):
        return symbol
    if symbol.startswith("NSE:") and not symbol.endswith("-EQ"):
        return f"{symbol}-EQ"
    if symbol.endswith("-EQ"):
        return f"NSE:{symbol}"
    return f"NSE:{symbol}-EQ"


def normalize_option_symbol(symbol: str) -> str:
    s = str(symbol).strip()
    if not s:
        return s
    return s if s.startswith("NSE:") else f"NSE:{s}"


def gethistory(symbol: str, resolution: str, days: int) -> pd.DataFrame:
    client = initfyers()
    if client is None:
        return pd.DataFrame()

    now = datetime.now()
    start = now - timedelta(days=days)
    payload = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start.strftime("%Y-%m-%d"),
        "range_to": now.strftime("%Y-%m-%d"),
        "cont_flag": "1",
    }
    try:
        res = client.history(payload)
        candles = (res or {}).get("candles", [])
    except Exception as e:
        logger.warning("History fetch failed symbol=%s res=%s err=%s", symbol, resolution, e)
        return pd.DataFrame()

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)


def nearest_step(val: float) -> int:
    val = abs(safe_float(val, 0))
    if val >= 20000:
        return 100
    if val >= 10000:
        return 50
    if val >= 2000:
        return 20
    if val >= 500:
        return 10
    if val >= 100:
        return 5
    return 1


def compute_obv(df: pd.DataFrame) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    obv = (vol * np.sign(close.diff().fillna(0))).cumsum()
    return round(safe_float(obv.iloc[-1], np.nan), 2)


def intraday_window_score(df: pd.DataFrame, window_minutes: int = SIGNAL_WINDOW_MINUTES) -> float:
    if df is None or df.empty or len(df) < 2:
        return np.nan
    d = df.copy().sort_values("timestamp")
    endts = pd.to_datetime(d["timestamp"].iloc[-1])
    startts = endts - timedelta(minutes=window_minutes)
    cur = d[(d["timestamp"] >= startts) & (d["timestamp"] <= endts)]
    if len(cur) < 2:
        return np.nan
    first_close = safe_float(cur["close"].iloc[0], np.nan)
    last_close = safe_float(cur["close"].iloc[-1], np.nan)
    if pd.isna(first_close) or first_close == 0:
        return np.nan
    return round(((last_close - first_close) / first_close) * 100.0, 2)


def previous_trading_day_same_time_score(full_df: pd.DataFrame, endts: Optional[pd.Timestamp] = None, window_minutes: int = SIGNAL_WINDOW_MINUTES) -> float:
    if full_df is None or full_df.empty or len(full_df) < 4:
        return np.nan
    d = full_df.copy().sort_values("timestamp").reset_index(drop=True)
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    endts = pd.to_datetime(endts) if endts is not None else pd.to_datetime(d["timestamp"].iloc[-1])
    trading_days = sorted(d["timestamp"].dt.date.unique())
    prev_days = [day for day in trading_days if day < endts.date()]
    if not prev_days:
        return np.nan
    prev_day = prev_days[-1]
    prev_day_df = d[d["timestamp"].dt.date == prev_day].copy()
    if prev_day_df.empty:
        return np.nan
    same_time_rows = prev_day_df[(prev_day_df["timestamp"].dt.hour == endts.hour) & (prev_day_df["timestamp"].dt.minute == endts.minute)]
    prev_end = pd.to_datetime(same_time_rows["timestamp"].iloc[-1] if not same_time_rows.empty else prev_day_df["timestamp"].iloc[-1])
    prev_start = prev_end - timedelta(minutes=window_minutes)
    prev_window = prev_day_df[(prev_day_df["timestamp"] >= prev_start) & (prev_day_df["timestamp"] <= prev_end)]
    if len(prev_window) < 2:
        return np.nan
    first_close = safe_float(prev_window["close"].iloc[0], np.nan)
    last_close = safe_float(prev_window["close"].iloc[-1], np.nan)
    if pd.isna(first_close) or first_close == 0:
        return np.nan
    return round(((last_close - first_close) / first_close) * 100.0, 2)


def compare_window_signal(current_score: float, previous_score: float) -> Tuple[float, str]:
    if pd.isna(current_score) or pd.isna(previous_score):
        return np.nan, "Neutral"
    delta = round(current_score - previous_score, 2)
    if delta >= 0.05:
        return delta, "Buy"
    if delta <= -0.05:
        return delta, "Sell"
    return delta, "Neutral"


def build_iteration_history(intra_df: pd.DataFrame, window_minutes: int = SIGNAL_WINDOW_MINUTES, iterations: int = ITERATIONS_TO_KEEP) -> pd.DataFrame:
    if intra_df is None or intra_df.empty:
        return pd.DataFrame()

    full_df = intra_df.copy().sort_values("timestamp").reset_index(drop=True)
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    last_day = full_df["timestamp"].dt.date.max()
    d = full_df[full_df["timestamp"].dt.date == last_day].copy().reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()

    start_anchor = pd.Timestamp.combine(pd.Timestamp(last_day).date(), dtime(9, 15))
    end_anchor = pd.Timestamp.combine(pd.Timestamp(last_day).date(), dtime(15, 30))
    d = d[(d["timestamp"] >= start_anchor) & (d["timestamp"] <= end_anchor)].copy().reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()

    rows = []
    for i in range(len(d)):
        endts = pd.to_datetime(d.loc[i, "timestamp"])
        startts = endts - timedelta(minutes=window_minutes)
        cur = d[(d["timestamp"] >= startts) & (d["timestamp"] <= endts)]
        if len(cur) < 2:
            continue
        current_score = intraday_window_score(cur, window_minutes)
        previous_score = previous_trading_day_same_time_score(full_df, endts, window_minutes)
        delta, signal = compare_window_signal(current_score, previous_score)
        rows.append({
            "iteration": len(rows) + 1,
            "timestamp": endts.strftime("%H:%M"),
            "windowminutes": window_minutes,
            "windowstart": startts.strftime("%H:%M"),
            "windowend": endts.strftime("%H:%M"),
            "currentwindowscore": current_score,
            "previoustradingdaysametimescore": previous_score,
            "windowdelta": delta,
            "windowsignal": signal,
            "close": safe_float(cur["close"].iloc[-1], np.nan),
        })
        if len(rows) >= iterations:
            break
    return pd.DataFrame(rows)


def summarize_intraday(intra_df: pd.DataFrame, daily_df: pd.DataFrame) -> Dict[str, object]:
    if intra_df is None or intra_df.empty:
        return {}
    close = pd.to_numeric(intra_df["close"], errors="coerce")
    prev_close = safe_float(daily_df["close"].iloc[-2] if daily_df is not None and len(daily_df) >= 2 else np.nan, np.nan)
    ltp = safe_float(close.iloc[-1], np.nan)
    iteration_history = build_iteration_history(intra_df)
    return {
        "LTP": round(ltp, 2),
        "% Change": round(((ltp - prev_close) / prev_close * 100.0), 2) if pd.notna(prev_close) and prev_close != 0 else np.nan,
        "Last Iteration Time": pd.to_datetime(intra_df["timestamp"].iloc[-1]).strftime("%H:%M"),
        "Iteration History": iteration_history,
    }


def choose_top_candidates(summary_df: pd.DataFrame, topn: int = TOP_N_UNDERLYINGS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = summary_df.copy()
    work["% Change"] = pd.to_numeric(work.get("% Change"), errors="coerce").fillna(0)
    long_df = work[work["BullMultiTFScore"] > 0].copy().sort_values(["BullMultiTFScore", "BullMultiTFScore"], ascending=[False, False]).head(topn)
    short_df = work[work["BearMultiTFScore"] < 0].copy().sort_values(["BearMultiTFScore", "BearMultiTFScore"], ascending=[True, False]).head(topn)
    return long_df.reset_index(drop=True), short_df.reset_index(drop=True)


def parse_chain_records(chain_res) -> List[Dict]:
    if isinstance(chain_res, dict):
        data = chain_res.get("data")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ["optionsChain", "chain", "optionChain", "options", "records"]:
                if isinstance(data.get(key), list):
                    return data.get(key)
        for key in ["optionsChain", "chain", "optionChain", "options", "records"]:
            if isinstance(chain_res.get(key), list):
                return chain_res.get(key)
    if isinstance(chain_res, list):
        return chain_res
    return []


def fetch_optionpairs(symbol: str, paircount: int = OPTION_PAIRS_TO_KEEP) -> pd.DataFrame:
    client = initfyers()
    if client is None:
        return pd.DataFrame()

    eqsymbol = format_eq_symbol(symbol)
    ltp = np.nan
    try:
        quote = client.quotes({"symbols": eqsymbol})
        if isinstance(quote, dict):
            data = quote.get("d", [])
            if data:
                ltp = safe_float(data[0].get("v", {}).get("lp"), np.nan)
    except Exception as e:
        logger.warning("Quote fetch failed symbol=%s err=%s", symbol, e)

    chain_attempts = [eqsymbol, symbol if str(symbol).startswith("NSE:") else f"NSE:{str(symbol).strip().upper()}"]
    chain_res = None
    for chain_symbol in chain_attempts:
        try:
            chain_res = client.optionchain({"symbol": chain_symbol, "strikecount": 50})
            break
        except Exception as e:
            logger.warning("optionchain failed symbol=%s err=%s", chain_symbol, e)

    if chain_res is None:
        return pd.DataFrame()

    chain = parse_chain_records(chain_res)
    rows = []
    for item in chain:
        if not isinstance(item, dict):
            continue
        strike = safe_float(item.get("strikePrice") or item.get("strike") or item.get("strike_price"), np.nan)
        opt_type = str(item.get("optionType") or item.get("type") or item.get("option_type") or "").upper()
        opt_symbol = str(item.get("symbol") or item.get("tradingSymbol") or item.get("trading_symbol") or item.get("optionSymbol") or "").strip()
        oi = safe_float(item.get("oi") or item.get("openInterest") or item.get("open_interest"), np.nan)
        vol = safe_float(item.get("volume") or item.get("vol"), np.nan)
        leg_ltp = safe_float(item.get("ltp") or item.get("lp") or item.get("last_price"), np.nan)
        if pd.isna(strike) or opt_type not in ("CE", "PE") or not opt_symbol:
            continue
        rows.append({
            "Underlying": symbol,
            "Strike": strike,
            "Option Type": opt_type,
            "Option Symbol": opt_symbol,
            "OI": oi,
            "Chain Volume": vol,
            "LTP": leg_ltp,
        })

    if not rows:
        return pd.DataFrame()

    oc = pd.DataFrame(rows)
    if pd.isna(ltp):
        ltp = safe_float(oc["Strike"].median(), np.nan)
    step = nearest_step(ltp if pd.notna(ltp) else oc["Strike"].median())
    atm = round(ltp / step) * step if pd.notna(ltp) and step else oc["Strike"].median()
    strikes = sorted(oc["Strike"].dropna().unique(), key=lambda x: abs(x - atm))[:max(paircount, 1)]
    return oc[oc["Strike"].isin(strikes)].copy().sort_values(["Strike", "Option Type"]).reset_index(drop=True)


def scansingleoption(option_symbol: str, option_type: str, strike: float, underlying: str) -> Optional[Dict]:
    hist_symbol = normalize_option_symbol(option_symbol)
    daily_df = gethistory(hist_symbol, "D", DAILY_LOOKBACK_DAYS)
    intra_df = gethistory(hist_symbol, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        return None
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    summary.update({
        "Underlying": underlying,
        "Option Type": option_type,
        "Option Symbol": option_symbol,
        "Strike": strike,
        "OBV": compute_obv(intra_df),
        "OI": np.nan,
        "Volume": safe_float(intra_df["volume"].sum() if "volume" in intra_df.columns else 0, 0),
    })
    return summary


def option_liquidity_score(oi, volume, obv) -> float:
    return round(
        np.log1p(max(safe_float(oi, 0), 0)) * 0.45
        + np.log1p(max(safe_float(volume, 0), 0)) * 0.35
        + np.log1p(max(abs(safe_float(obv, 0)), 0)) * 0.20,
        4,
    )


def rank_option_candidates(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["OI+Volume+OBV Score"] = pd.to_numeric(out.get("OI+Volume+OBV Score"), errors="coerce").fillna(0)
    out["Volume"] = pd.to_numeric(out.get("Volume"), errors="coerce").fillna(0)
    out["OI"] = pd.to_numeric(out.get("OI"), errors="coerce").fillna(0)
    out["EMAIL_RANK_SCORE"] = out["OI+Volume+OBV Score"]
    return out.sort_values(["EMAIL_RANK_SCORE", "Volume", "OI"], ascending=[False, False, False]).reset_index(drop=True)


def buildoptioncandidates(candidates_df: pd.DataFrame, side: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidates_df is None or candidates_df.empty or "Symbol" not in candidates_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    iter_rows = []
    for underlying in candidates_df["Symbol"].dropna().astype(str):
        pair_df = fetch_optionpairs(underlying)
        if pair_df.empty:
            continue

        req_type = "CE" if side == "long" else "PE"
        type_df = pair_df[pair_df["Option Type"] == req_type].copy()
        if type_df.empty:
            continue

        atm_strike = sorted(type_df["Strike"].dropna().unique(), key=lambda x: abs(x - type_df["Strike"].median()))[0]
        atm_rows = type_df[type_df["Strike"] == atm_strike]
        atm_vol = safe_float(atm_rows["Chain Volume"].max() if not atm_rows.empty else 0, 0)
        if atm_vol < MIN_ATM_CHAIN_VOLUME:
            logger.info("SKIP %s %s atm chain volume %.0f < %d", underlying, req_type, atm_vol, MIN_ATM_CHAIN_VOLUME)
            continue

        for _, row in type_df.iterrows():
            strike = safe_float(row.get("Strike"), np.nan)
            opt_type = str(row.get("Option Type", "")).upper()
            sym = str(row.get("Option Symbol", "")).strip()
            scanned = scansingleoption(sym, opt_type, strike, underlying)
            if not scanned:
                continue

            scanned["OI"] = safe_float(row.get("OI"), np.nan)
            scanned["Volume"] = max(safe_float(scanned.get("Volume", 0), 0), safe_float(row.get("Chain Volume", 0), 0))
            scanned["OI+Volume+OBV Score"] = option_liquidity_score(scanned.get("OI", 0), scanned.get("Volume", 0), scanned.get("OBV", 0))
            if safe_float(scanned.get("LTP", 0.0), 0.0) < MIN_OPTION_LTP:
                continue
            if safe_float(scanned.get("Volume", 0), 0) < MIN_OPTION_DAY_VOLUME:
                continue

            rows.append(scanned)
            hist = scanned.get("Iteration History")
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                temp = hist.copy()
                temp.insert(0, "Option Symbol", sym)
                temp.insert(1, "Underlying", underlying)
                temp.insert(2, "Strike", strike)
                temp.insert(3, "Option Type", opt_type)
                iter_rows.append(temp)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    out = pd.DataFrame(rows)
    out = rank_option_candidates(out, side)
    final_out = out[[c for c in OPTION_EMAIL_COLS if c in out.columns]].reset_index(drop=True)

    iter_df = pd.DataFrame()
    if iter_rows and not final_out.empty:
        all_iters = pd.concat(iter_rows, ignore_index=True)
        all_iters = all_iters[all_iters["Option Symbol"].isin(final_out["Option Symbol"])].copy()
        group_cols = [c for c in ["Underlying", "Option Type", "Strike", "Option Symbol"] if c in all_iters.columns]
        if group_cols and not all_iters.empty:
            sort_cols = group_cols + (["timestamp"] if "timestamp" in all_iters.columns else [])
            all_iters = all_iters.sort_values(sort_cols).reset_index(drop=True)
            all_iters["iteration"] = all_iters.groupby(group_cols).cumcount() + 1
            all_iters["iteration"] = pd.to_numeric(all_iters["iteration"], errors="coerce").astype("Int64")
            all_iters = all_iters[all_iters["iteration"].between(1, ITERATIONS_TO_KEEP)]
        iter_df = all_iters.reset_index(drop=True)

    return final_out, iter_df


def load_state_df() -> pd.DataFrame:
    cols = ["State Key", "entered", "down_streak", "Entry Value", "Entry Time", "Exit Value", "Exit Time", "Chain Signal", "last_iteration"]
    if not os.path.isfile(STATE_CSV_PATH):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(STATE_CSV_PATH)
        for col in cols:
            if col not in df.columns:
                df[col] = np.nan
        return df[cols].copy()
    except Exception as e:
        logger.warning("Could not read state csv %s: %s", STATE_CSV_PATH, e)
        return pd.DataFrame(columns=cols)


def save_state_df(state_df: pd.DataFrame):
    cols = ["State Key", "entered", "down_streak", "Entry Value", "Entry Time", "Exit Value", "Exit Time", "Chain Signal", "last_iteration"]
    out = state_df.copy() if state_df is not None else pd.DataFrame(columns=cols)
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    out[cols].to_csv(STATE_CSV_PATH, index=False)


def get_state_row(state_df: pd.DataFrame, state_key: str) -> Dict[str, object]:
    if state_df is None or state_df.empty or "State Key" not in state_df.columns:
        return {}
    hit = state_df[state_df["State Key"].astype(str) == str(state_key)]
    if hit.empty:
        return {}
    return hit.iloc[0].to_dict()


def upsert_state_row(state_df: pd.DataFrame, state_key: str, row: Dict[str, object]) -> pd.DataFrame:
    data = dict(row)
    data["State Key"] = state_key
    if state_df is None or state_df.empty:
        return pd.DataFrame([data])
    mask = state_df["State Key"].astype(str) == str(state_key)
    if mask.any():
        for k, v in data.items():
            state_df.loc[mask, k] = v
        return state_df
    return pd.concat([state_df, pd.DataFrame([data])], ignore_index=True)


def chain_entry_exit_from_iters(iter_df: pd.DataFrame, state_key: Optional[str] = None, state_df: Optional[pd.DataFrame] = None) -> Tuple[Dict[str, object], pd.DataFrame]:
    empty = {
        "Chain Signal": "WAIT",
        "Entry Value": np.nan,
        "Entry Time": "",
        "Exit Value": np.nan,
        "Exit Time": "",
    }

    if iter_df is None or iter_df.empty:
        return empty, state_df

    d = iter_df.copy()
    d["iteration"] = pd.to_numeric(d["iteration"], errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna(subset=["iteration", "close", "Option Symbol"])
    if d.empty:
        return empty, state_df

    pivot = d.pivot_table(index="iteration", columns="Option Symbol", values="close", aggfunc="last").sort_index()
    if pivot.shape[0] < 2 or pivot.shape[1] == 0:
        return empty, state_df

    prev = pivot.shift(1)
    all_up = (pivot > prev).all(axis=1)
    all_down = (pivot < prev).all(axis=1)

    existing = get_state_row(state_df if state_df is not None else pd.DataFrame(), state_key) if state_key else {}
    entered = str(existing.get("entered", "False")).strip().lower() in {"1", "true", "yes", "y"}
    down_streak = int(safe_float(existing.get("down_streak"), 0))
    entry_value = safe_float(existing.get("Entry Value"), np.nan)
    entry_time = "" if pd.isna(existing.get("Entry Time", np.nan)) else str(existing.get("Entry Time", ""))
    exit_value = safe_float(existing.get("Exit Value"), np.nan)
    exit_time = "" if pd.isna(existing.get("Exit Time", np.nan)) else str(existing.get("Exit Time", ""))
    chain_signal = str(existing.get("Chain Signal", "WAIT")).upper() if existing else "WAIT"
    last_processed_iter = int(safe_float(existing.get("last_iteration"), 0))

    new_iters = [int(x) for x in pivot.index if int(x) > last_processed_iter]
    if not new_iters and existing:
        return {
            "Chain Signal": chain_signal,
            "Entry Value": entry_value,
            "Entry Time": entry_time,
            "Exit Value": exit_value,
            "Exit Time": exit_time,
        }, state_df

    latest_iter = int(pivot.index.max())
    latest_row = d[d["iteration"] == latest_iter].sort_values(["timestamp", "Option Symbol"]).copy()
    latest_time = str(latest_row["timestamp"].iloc[0]) if "timestamp" in latest_row.columns and not latest_row.empty else ""
    latest_value = round(float(latest_row["close"].mean()), 2) if not latest_row.empty else np.nan

    for it in new_iters:
        is_up = bool(all_up.loc[it]) if pd.notna(all_up.loc[it]) else False
        is_down = bool(all_down.loc[it]) if pd.notna(all_down.loc[it]) else False
        row_it = d[d["iteration"] == it].sort_values(["timestamp", "Option Symbol"]).copy()
        row_time = str(row_it["timestamp"].iloc[0]) if "timestamp" in row_it.columns and not row_it.empty else ""
        row_value = round(float(row_it["close"].mean()), 2) if not row_it.empty else np.nan

        if not entered:
            if is_up:
                entered = True
                down_streak = 0
                entry_value = row_value
                entry_time = row_time
                exit_value = np.nan
                exit_time = ""
                chain_signal = "ENTER"
            else:
                chain_signal = "WAIT"
        else:
            if is_down:
                down_streak += 1
            else:
                down_streak = 0

            if down_streak >= 2:
                exit_value = row_value
                exit_time = row_time
                chain_signal = "EXIT"
                entered = False
                down_streak = 0
            else:
                chain_signal = "ENTER"
                exit_value = latest_value
                exit_time = latest_time

        last_processed_iter = max(last_processed_iter, it)

    if state_key:
        state_df = upsert_state_row(state_df if state_df is not None else pd.DataFrame(), state_key, {
            "entered": entered,
            "down_streak": down_streak,
            "Entry Value": None if pd.isna(entry_value) else float(entry_value),
            "Entry Time": entry_time,
            "Exit Value": None if pd.isna(exit_value) else float(exit_value),
            "Exit Time": exit_time,
            "Chain Signal": chain_signal,
            "last_iteration": int(last_processed_iter),
        })

    return {
        "Chain Signal": chain_signal,
        "Entry Value": entry_value,
        "Entry Time": entry_time,
        "Exit Value": exit_value,
        "Exit Time": exit_time,
    }, state_df


def send_single_email(subject: str, html_body: str, attachments: Optional[List[str]] = None) -> bool:
    sender = os.environ.get("SENDER_EMAIL")
    password = os.environ.get("SENDER_PASSWORD")
    recipients_raw = os.environ.get("RECIPIENT_EMAIL", "")
    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]

    if not sender or not password or not recipients:
        logger.warning("Email credentials/recipients not configured; skipping.")
        return False

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    for path in attachments or []:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as fh:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(fh.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(path)}"')
                msg.attach(part)
        except Exception as e:
            logger.warning("Could not attach %s: %s", path, e)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        logger.info("Email sent: %s", subject)
        return True
    except Exception as e:
        logger.error("Failed to send email %s: %s", subject, e)
        return False


def _prepare_email_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    rename_map = {k: v for k, v in OPTION_EMAIL_COL_RENAME.items() if k in out.columns}
    out = out.rename(columns=rename_map)
    if "Opt Symbol" in out.columns:
        out = out.drop(columns=["Opt Symbol"])
    return out


def compact_table_html(df: pd.DataFrame, title: str, max_rows: int) -> str:
    if df is None or df.empty:
        return f'<div class="section-card"><h3>{title}</h3><p class="muted">No rows</p></div>'

    df = _prepare_email_df(df)
    if "Chain Signal" in df.columns:
        df["Chain Signal"] = df["Chain Signal"].astype(str).map(
            lambda x: f'<span class="badge badge-enter">{x}</span>' if x.upper() == "ENTER" else (
                f'<span class="badge badge-exit">{x}</span>' if x.upper() == "EXIT" else f'<span class="badge badge-wait">{x}</span>'
            )
        )

    cols = [c for c in [
        "Underlying", "Option Type", "Strike", "LTP", "OI", "Volume", "OBV",
        "Liq Score", "Rank", "Entry", "Entry Time", "Exit", "Exit Time", "Chain Signal", "Chain Legs"
    ] if c in df.columns]
    body = df[cols].head(max_rows).to_html(index=False, border=0, escape=False, classes="mail-table")
    return f'<div class="section-card"><h3>{title}</h3>{body}</div>'


def build_chain_email_html(long_rows: List[Dict], short_rows: List[Dict]) -> str:
    ts = datetime.now().strftime("%d %b %Y, %H:%M")
    return f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f4f7fb; color: #172033; margin: 0; padding: 20px; }}
        .wrap {{ max-width: 1180px; margin: 0 auto; }}
        .hero {{ background: linear-gradient(135deg, #0f172a, #1d4ed8); color: white; padding: 18px 22px; border-radius: 14px; }}
        .hero h2 {{ margin: 0 0 6px 0; font-size: 24px; }}
        .hero p {{ margin: 0; color: #dbeafe; }}
        .section-card {{ background: white; border: 1px solid #dbe4f0; border-radius: 14px; padding: 14px 16px; margin-top: 16px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06); }}
        h3 {{ margin: 0 0 12px 0; color: #1e3a8a; }}
        .mail-table {{ width: 100%; border-collapse: collapse; font-size: 13px; overflow: hidden; }}
        .mail-table th {{ background: #2563eb; color: white; padding: 10px 8px; text-align: left; }}
        .mail-table td {{ padding: 8px; border-bottom: 1px solid #e5e7eb; }}
        .mail-table tr:nth-child(even) td {{ background: #f8fbff; }}
        .mail-table tr:hover td {{ background: #eef6ff; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
        .badge-enter {{ background: #dcfce7; color: #166534; }}
        .badge-exit {{ background: #fee2e2; color: #991b1b; }}
        .badge-wait {{ background: #f3f4f6; color: #374151; }}
        .muted {{ color: #64748b; }}
    </style>
    </head>
    <body>
        <div class="wrap">
            <div class="hero">
                <h2>Chain Signal Report - {ts}</h2>
                <p>Shortlist is reused when available. One underlying row near ATM. Entry is all legs up together. Exit is only after entry and 2 consecutive all-leg down checks.</p>
            </div>
            {compact_table_html(pd.DataFrame(long_rows), 'LONG CANDIDATES', EMAIL_MAX_ROWS_LONG)}
            {compact_table_html(pd.DataFrame(short_rows), 'SHORT CANDIDATES', EMAIL_MAX_ROWS_SHORT)}
        </div>
    </body>
    </html>
    """


def send_chain_signal_email(long_rows: List[Dict], short_rows: List[Dict], attachments: Optional[List[str]] = None) -> bool:
    if not long_rows and not short_rows:
        return False
    subject = f"Chain Signal Report - {datetime.now().strftime('%d %b %H:%M')}"
    return send_single_email(subject, build_chain_email_html(long_rows, short_rows), attachments)


def pick_best_representative(options_df: pd.DataFrame, underlying: str, opt_type: str) -> Dict:
    if options_df is None or options_df.empty:
        return {}

    temp = options_df.copy()
    temp = temp[(temp["Underlying"].astype(str) == str(underlying)) & (temp["Option Type"].astype(str).str.upper() == str(opt_type).upper())].copy()
    if temp.empty:
        return {}

    temp["Strike"] = pd.to_numeric(temp.get("Strike"), errors="coerce")
    temp = temp.dropna(subset=["Strike"])
    if temp.empty:
        return {}

    temp["EMAIL_RANK_SCORE"] = pd.to_numeric(temp.get("EMAIL_RANK_SCORE"), errors="coerce").fillna(-np.inf)
    temp["Volume"] = pd.to_numeric(temp.get("Volume"), errors="coerce").fillna(0)
    temp["OI"] = pd.to_numeric(temp.get("OI"), errors="coerce").fillna(0)
    temp["_atm_ref"] = temp["Strike"].median()
    temp["_atm_gap"] = (temp["Strike"] - temp["_atm_ref"]).abs()
    temp = temp.sort_values(["_atm_gap", "EMAIL_RANK_SCORE", "Volume", "OI", "Strike"], ascending=[True, False, False, False, True])
    return temp.iloc[0].to_dict()


def build_chain_rows_from_iters(options_df: pd.DataFrame, iteration_df: pd.DataFrame, state_df: Optional[pd.DataFrame] = None) -> Tuple[List[Dict], pd.DataFrame]:
    if options_df is None or options_df.empty or iteration_df is None or iteration_df.empty:
        return [], state_df

    required = {"Underlying", "Option Type", "Option Symbol", "close"}
    if not required.issubset(set(iteration_df.columns)):
        return [], state_df

    rows = []
    cur_state = state_df if state_df is not None else load_state_df()
    for (underlying, opt_type), grp in iteration_df.groupby(["Underlying", "Option Type"]):
        grp = grp.copy()
        sort_cols = [c for c in ["iteration", "timestamp", "Option Symbol"] if c in grp.columns]
        if sort_cols:
            grp = grp.sort_values(sort_cols).reset_index(drop=True)

        chain_legs = int(grp["Option Symbol"].astype(str).nunique()) if "Option Symbol" in grp.columns else 0
        if chain_legs < MIN_CHAIN_LEGS:
            continue

        state_key = f"{str(underlying).upper()}|{str(opt_type).upper()}"
        chain_sig, cur_state = chain_entry_exit_from_iters(grp, state_key=state_key, state_df=cur_state)
        rep = pick_best_representative(options_df, str(underlying), str(opt_type))

        strike = safe_float(rep.get("Strike"), np.nan) if rep else np.nan
        if pd.isna(strike) and "Strike" in grp.columns:
            strike_series = pd.to_numeric(grp["Strike"], errors="coerce").dropna()
            strike = float(strike_series.median()) if not strike_series.empty else np.nan

        close_series = pd.to_numeric(grp["close"], errors="coerce").dropna()
        fallback_ltp = close_series.iloc[-1] if not close_series.empty else np.nan

        rows.append({
            "Underlying": str(underlying),
            "Option Type": str(opt_type).upper(),
            "Strike": strike,
            "LTP": rep.get("LTP", fallback_ltp) if rep else fallback_ltp,
            "OI": rep.get("OI", np.nan) if rep else np.nan,
            "Volume": rep.get("Volume", np.nan) if rep else np.nan,
            "OBV": rep.get("OBV", np.nan) if rep else np.nan,
            "OI+Volume+OBV Score": rep.get("OI+Volume+OBV Score", np.nan) if rep else np.nan,
            "EMAIL_RANK_SCORE": rep.get("EMAIL_RANK_SCORE", np.nan) if rep else np.nan,
            "Entry Value": chain_sig.get("Entry Value", np.nan),
            "Entry Time": chain_sig.get("Entry Time", ""),
            "Exit Value": chain_sig.get("Exit Value", np.nan),
            "Exit Time": chain_sig.get("Exit Time", ""),
            "Chain Signal": chain_sig.get("Chain Signal", "WAIT"),
            "Chain Legs": chain_legs,
        })

    if not rows:
        return [], cur_state

    out = pd.DataFrame(rows).copy()
    out["EMAIL_RANK_SCORE"] = pd.to_numeric(out.get("EMAIL_RANK_SCORE"), errors="coerce").fillna(-np.inf)
    out["Chain Legs"] = pd.to_numeric(out.get("Chain Legs"), errors="coerce").fillna(0)
    signal_priority = {"ENTER": 0, "EXIT": 1, "WAIT": 2}
    out["_sig"] = out["Chain Signal"].astype(str).str.upper().map(signal_priority).fillna(9)
    out = out.sort_values(["EMAIL_RANK_SCORE", "Chain Legs", "_sig", "Underlying"], ascending=[False, False, True, True]).reset_index(drop=True)
    return out.drop(columns=["_sig"]).to_dict("records"), cur_state


def scansymbol(symbol: str) -> Optional[Dict]:
    eq = format_eq_symbol(symbol)
    daily_df = gethistory(eq, "D", DAILY_LOOKBACK_DAYS)
    intra_df = gethistory(eq, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        return None
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    summary["Symbol"] = symbol
    return summary


def load_or_build_shortlist() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    asit_path = find_latest_asit_csv(".")
    if not asit_path:
        raise RuntimeError("No asit*.csv found for shortlist generation.")

    logger.info("Using ASIT shortlist source: %s", asit_path)
    long_seed_df, short_seed_df = load_asit_shortlist(asit_path, topn=TOP_N_UNDERLYINGS)
    if long_seed_df.empty and short_seed_df.empty:
        raise RuntimeError(f"No bullish/bearish shortlist rows found in {asit_path}")

    summary_df = pd.concat(
        [
            long_seed_df.assign(Side="long"),
            short_seed_df.assign(Side="short"),
        ],
        ignore_index=True,
    )

    summary_df.to_csv(SUMMARY_PATH, index=False)
    long_seed_df.to_csv(LONG_SEED_PATH, index=False)
    short_seed_df.to_csv(SHORT_SEED_PATH, index=False)
    return summary_df, long_seed_df, short_seed_df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    initfyers()

    summary_df, long_seed_df, short_seed_df = load_or_build_shortlist()
    summary_df.to_csv(SUMMARY_PATH, index=False)

    long_df, long_iter_df = buildoptioncandidates(long_seed_df, side="long")
    short_df, short_iter_df = buildoptioncandidates(short_seed_df, side="short")

    long_df.to_csv(LONG_CANDIDATES_PATH, index=False)
    short_df.to_csv(SHORT_CANDIDATES_PATH, index=False)

    iteration_df = pd.concat([long_iter_df, short_iter_df], ignore_index=True) if (not long_iter_df.empty or not short_iter_df.empty) else pd.DataFrame()
    if iteration_df.empty:
        iteration_df = pd.DataFrame(columns=[
            "iteration", "Underlying", "Option Type", "Strike", "Option Symbol",
            "timestamp", "windowminutes", "windowstart", "windowend",
            "currentwindowscore", "previoustradingdaysametimescore",
            "windowdelta", "windowsignal", "close"
        ])
    iteration_df.to_csv(ITER_PATH, index=False)

    state_df = load_state_df()

    options_df = pd.concat([long_df, short_df], ignore_index=True) if (not long_df.empty or not short_df.empty) else pd.DataFrame()
    chain_rows, state_df = build_chain_rows_from_iters(options_df, iteration_df, state_df=state_df)
    save_state_df(state_df)

    chain_df = pd.DataFrame(chain_rows)
    if chain_df.empty:
        long_rows = []
        short_rows = []
    else:
        long_rows = chain_df[chain_df["Option Type"].astype(str).str.upper() == "CE"].to_dict("records")
        short_rows = chain_df[chain_df["Option Type"].astype(str).str.upper() == "PE"].to_dict("records")

    pd.DataFrame(long_rows).to_csv(CHAIN_LONG_PATH, index=False)
    pd.DataFrame(short_rows).to_csv(CHAIN_SHORT_PATH, index=False)

    send_chain_signal_email(
        long_rows,
        short_rows,
        attachments=[SUMMARY_PATH, LONG_SEED_PATH, SHORT_SEED_PATH, LONG_CANDIDATES_PATH, SHORT_CANDIDATES_PATH, ITER_PATH, CHAIN_LONG_PATH, CHAIN_SHORT_PATH, STATE_CSV_PATH],
    )


if __name__ == "__main__":
    main()

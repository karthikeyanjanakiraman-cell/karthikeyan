import os
import json
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
IVP_LOOKBACK_DAYS = 252
OPTION_PAIRS_TO_KEEP = int(os.environ.get("OPTION_PAIRS_TO_KEEP", "5"))
SIGNAL_WINDOW_MINUTES = int(os.environ.get("SIGNAL_WINDOW_MINUTES", "5"))
ITERATIONS_TO_KEEP = int(os.environ.get("ITERATIONS_TO_KEEP", "75"))
SECTORS_DIR = os.environ.get("SECTORS_DIR", "sectors")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
MIN_OPTION_LTP = float(os.environ.get("MIN_OPTION_LTP", "5"))
MIN_ATM_CHAIN_VOLUME = int(os.environ.get("MIN_ATM_CHAIN_VOLUME", "1000"))
MIN_OPTION_DAY_VOLUME = int(os.environ.get("MIN_OPTION_DAY_VOLUME", "1000"))
PER_SYMBOL_SLEEP_SEC = float(os.environ.get("PER_SYMBOL_SLEEP_SEC", "0.10"))
EMAIL_MAX_ROWS_LONG = int(os.environ.get("EMAIL_MAX_ROWS_LONG", "25"))
EMAIL_MAX_ROWS_SHORT = int(os.environ.get("EMAIL_MAX_ROWS_SHORT", "25"))
TOP_N_UNDERLYINGS = int(os.environ.get("TOP_N_UNDERLYINGS", "20"))
T30_MIN = int(os.environ.get("T30_MIN", "2"))
DAILY_STATE_FILE = os.path.join(OUTPUT_DIR, "chain_signal_state.json")

OPTION_EMAIL_COLS = [
    "ATM Strike",
    "Underlying",
    "Option Type",
    "Option Symbol",
    "Strike",
    "LTP",
    "% Change",
    "OI",
    "Volume",
    "OBV",
    "OI+Volume+OBV Score",
    "EMAIL_RANK_SCORE",
    "Rank Delta",
    "Cumulative ADX",
    "5m_Signal",
    "15m_Signal",
    "30m_Signal",
    "60m_Signal",
    "Bull_Signal",
    "Bear_Signal",
    "Overall_Signal",
    "Price_Lead_Status",
    "IVP",
    "Volatility State",
    "Last Iteration Time",
]

OPTION_EMAIL_COL_RENAME = {
    "OI+Volume+OBV Score": "Liq Score",
    "EMAIL_RANK_SCORE": "Rank",
    "% Change": "% Chg",
    "Last Iteration Time": "Time",
    "Price_Lead_Status": "Lead",
    "Volatility State": "Vol State",
    "Option Symbol": "Opt Symbol",
}

ATM_CHAIN_EMAIL_COLS = [
    "Underlying",
    "Option Type",
    "ATM Strike",
    "ATM LTP",
    "Qualified Strikes",
    "All Tracked Strikes",
    "Qualified Count",
    "Tracked Count",
    "Chain Signal",
    "Entry Time",
    "Last Check",
]

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


def discover_sector_csvs(rootdir: str = SECTORS_DIR) -> List[str]:
    if not os.path.isdir(rootdir):
        return []
    files = []
    for dirpath, _, filenames in os.walk(rootdir):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                files.append(os.path.join(dirpath, fname))
    return sorted(set(files))


def load_fno_symbols_from_sectors(rootdir: str = SECTORS_DIR) -> List[str]:
    symbols = set()
    for path in discover_sector_csvs(rootdir):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning("Failed reading sector csv %s: %s", path, e)
            continue
        lowered = {str(c).strip().lower(): c for c in df.columns}
        sym_col = next((lowered[k] for k in ["symbol", "symbols", "ticker", "tradingsymbol"] if k in lowered), None)
        if not sym_col:
            continue
        vals = df[sym_col].dropna().astype(str).str.strip().str.upper()
        vals = vals[~vals.isin(["", "NAN", "NONE", "NULL"])]
        symbols.update(vals.tolist())
    return sorted(symbols)


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


def compute_ivp(history_df: pd.DataFrame, min_bars: int = 10) -> Tuple[float, str]:
    if history_df is None or history_df.empty or len(history_df) < min_bars:
        return np.nan, "Neutral Vol"
    close = pd.to_numeric(history_df["close"], errors="coerce")
    high = pd.to_numeric(history_df["high"], errors="coerce")
    low = pd.to_numeric(history_df["low"], errors="coerce")
    proxy = ((high - low) / close.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
    if proxy.empty:
        return np.nan, "Neutral Vol"
    lookback = proxy.tail(min(IVP_LOOKBACK_DAYS, len(proxy)))
    current = float(lookback.iloc[-1])
    ivp = round((lookback.lt(current).sum() / len(lookback)) * 100.0, 2)
    if ivp < 30:
        return ivp, "Buyer Zone"
    if ivp > 50:
        return ivp, "Avoid Buy Premium"
    return ivp, "Neutral Vol"


def score_label(delta: float) -> str:
    if pd.isna(delta):
        return "Neutral"
    if delta >= 1:
        return "Buy"
    if delta <= -1:
        return "Sell"
    return "Neutral"


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
    same_time_rows = prev_day_df[
        (prev_day_df["timestamp"].dt.hour == endts.hour) &
        (prev_day_df["timestamp"].dt.minute == endts.minute)
    ]
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
    df = intra_df.copy().sort_values("timestamp").reset_index(drop=True)
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    delta = close.diff().fillna(0.0)
    avg_gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    upmove = high.diff()
    downmove = -low.diff()
    plusdm = np.where((upmove > downmove) & (upmove > 0), upmove, 0.0)
    minusdm = np.where((downmove > upmove) & (downmove > 0), downmove, 0.0)
    atr = pd.Series(tr).rolling(14, min_periods=14).mean()
    plusdi = 100 * pd.Series(plusdm).rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    minusdi = 100 * pd.Series(minusdm).rolling(14, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = 100 * (plusdi - minusdi).abs() / (plusdi + minusdi).replace(0, np.nan)
    adx = dx.rolling(14, min_periods=14).mean().fillna(0)

    typical = (high + low + close) / 3.0
    cumvol = volume.cumsum().replace(0, np.nan)
    vwap = (typical * volume).cumsum() / cumvol.ffill().fillna(close)
    vwap_std = (typical - vwap).rolling(20, min_periods=5).std().replace(0, np.nan)
    vwap_z = ((close - vwap) / vwap_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    prev_close = safe_float(daily_df["close"].iloc[-2] if daily_df is not None and len(daily_df) >= 2 else np.nan, np.nan)
    ltp = safe_float(close.iloc[-1], np.nan)
    pct_change = ((ltp - prev_close) / prev_close * 100.0) if pd.notna(prev_close) and prev_close != 0 else 0.0

    current_win = intraday_window_score(df)
    prev_win = previous_trading_day_same_time_score(df)
    win_signal = compare_window_signal(current_win, prev_win)[1]
    iteration_history = build_iteration_history(df)

    bull = (
        int(pct_change >= 0) +
        int(safe_float(vwap_z.iloc[-1], 0) > 0.3) +
        int(safe_float(plusdi.iloc[-1], 0) > safe_float(minusdi.iloc[-1], 0)) +
        int(safe_float(rsi.iloc[-1], 50) > 55) +
        int(win_signal.startswith("Buy")) * 2
    )
    bear = (
        int(pct_change <= 0) +
        int(safe_float(vwap_z.iloc[-1], 0) < -0.3) +
        int(safe_float(minusdi.iloc[-1], 0) > safe_float(plusdi.iloc[-1], 0)) +
        int(safe_float(rsi.iloc[-1], 50) < 45) +
        int(win_signal.startswith("Sell")) * 2
    )

    rank_delta = bull - bear
    ivp, vol_state = compute_ivp(daily_df, min_bars=10)
    last_ts = pd.to_datetime(df["timestamp"].iloc[-1])

    return {
        "LTP": round(ltp, 2),
        "% Change": round(pct_change, 2),
        "5m_Signal": score_label(rank_delta),
        "15m_Signal": win_signal,
        "30m_Signal": score_label(rank_delta * 0.8),
        "60m_Signal": score_label(rank_delta * 0.7),
        "Bull_Signal": score_label(bull),
        "Bear_Signal": score_label(-bear),
        "Overall_Signal": score_label(rank_delta),
        "Price_Lead_Status": "NORMAL",
        "IVP": ivp,
        "Volatility State": vol_state,
        "Last Iteration Time": last_ts.strftime("%H:%M"),
        "Bull Rank": bull,
        "Bear Rank": bear,
        "Rank Delta": rank_delta,
        "Cumulative ADX": round(safe_float(adx.iloc[-1], np.nan), 2),
        "Iteration History": iteration_history,
    }


def choose_top_candidates(summary_df: pd.DataFrame, topn: int = TOP_N_UNDERLYINGS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    rank_delta = pd.to_numeric(summary_df.get("Rank Delta", 0), errors="coerce").fillna(0)
    long_df = summary_df[rank_delta > 0].copy()
    short_df = summary_df[rank_delta < 0].copy()

    long_df = long_df.sort_values(["Rank Delta", "Cumulative ADX", "% Change"], ascending=[False, False, False]).head(topn)
    short_df = short_df.sort_values(["Rank Delta", "Cumulative ADX", "% Change"], ascending=[True, False, True]).head(topn)
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
        logger.warning("fetch_optionpairs skipped no client symbol=%s", symbol)
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
    used_symbol = None
    for chain_symbol in chain_attempts:
        try:
            chain_res = client.optionchain({"symbol": chain_symbol, "strikecount": 50})
            used_symbol = chain_symbol
            break
        except Exception as e:
            logger.warning("optionchain failed symbol=%s err=%s", chain_symbol, e)

    if chain_res is None:
        return pd.DataFrame()

    chain = parse_chain_records(chain_res)
    logger.info("optionchain symbol=%s used=%s raw_records=%d", symbol, used_symbol, len(chain))

    rows = []
    for item in chain:
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
        logger.warning("No usable option legs parsed symbol=%s used=%s", symbol, used_symbol)
        return pd.DataFrame()

    oc = pd.DataFrame(rows)
    if pd.isna(ltp):
        ltp = safe_float(oc["Strike"].median(), np.nan)
    step = nearest_step(ltp if pd.notna(ltp) else oc["Strike"].median())
    atm = round(ltp / step) * step if pd.notna(ltp) and step else oc["Strike"].median()
    strikes = sorted(oc["Strike"].dropna().unique(), key=lambda x: abs(x - atm))[:max(paircount, 1)]
    out = oc[oc["Strike"].isin(strikes)].copy().sort_values(["Strike", "Option Type"]).reset_index(drop=True)

    logger.info("optionpairs symbol=%s atm=%s step=%s kept_rows=%d strikes=%s", symbol, atm, step, len(out), strikes)
    return out


def scansingleoption(option_symbol: str, option_type: str, strike: float, underlying: str) -> Optional[Dict]:
    hist_symbol = normalize_option_symbol(option_symbol)
    daily_df = gethistory(hist_symbol, "D", DAILY_LOOKBACK_DAYS)
    intra_df = gethistory(hist_symbol, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        logger.warning("HISTORY EMPTY underlying=%s symbol=%s daily=%d intra=%d", underlying, hist_symbol, len(daily_df), len(intra_df))
        return None

    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        logger.warning("SUMMARY EMPTY underlying=%s symbol=%s", underlying, hist_symbol)
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
        np.log1p(max(safe_float(oi, 0), 0)) * 0.45 +
        np.log1p(max(safe_float(volume, 0), 0)) * 0.35 +
        np.log1p(max(abs(safe_float(obv, 0)), 0)) * 0.20,
        4
    )


def rank_option_candidates(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["__liq"] = pd.to_numeric(out.get("OI+Volume+OBV Score", 0), errors="coerce").fillna(0)
    out["__rd"] = pd.to_numeric(out.get("Rank Delta", 0), errors="coerce").fillna(0)
    out["__adx"] = pd.to_numeric(out.get("Cumulative ADX", 0), errors="coerce").fillna(0)
    out["__pct"] = pd.to_numeric(out.get("% Change", 0), errors="coerce").fillna(0)
    opt_type = out["Option Type"].astype(str).str.upper() if "Option Type" in out.columns else pd.Series("", index=out.index)

    if side == "long":
        type_bonus = np.where(opt_type.eq("CE"), 0.30, 0.10)
        out["EMAIL_RANK_SCORE"] = out["__liq"] * 0.40 + out["__rd"] * 0.30 + out["__adx"] * 0.18 + out["__pct"] * 0.10 + type_bonus
        out = out.sort_values(["EMAIL_RANK_SCORE", "__liq", "__rd", "__adx", "__pct"], ascending=[False, False, False, False, False])
    else:
        type_bonus = np.where(opt_type.eq("PE"), 0.30, 0.10)
        out["EMAIL_RANK_SCORE"] = out["__liq"] * 0.40 + (-out["__rd"]) * 0.30 + out["__adx"] * 0.18 + (-out["__pct"]) * 0.10 + type_bonus
        out = out.sort_values(["EMAIL_RANK_SCORE", "__liq", "__rd", "__adx", "__pct"], ascending=[False, False, True, False, True])

    return out.reset_index(drop=True)


def buildoptioncandidates(candidates_df: pd.DataFrame, side: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if candidates_df is None or candidates_df.empty or "Symbol" not in candidates_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    iter_rows = []
    total_stats = []

    for underlying in candidates_df["Symbol"].dropna().astype(str):
        pair_df = fetch_optionpairs(underlying)
        if pair_df.empty:
            logger.warning("SKIP underlying=%s reason=empty_optionpairs", underlying)
            total_stats.append((underlying, 0, 0, 0, 0, 0))
            continue

        req_type = "CE" if side == "long" else "PE"
        type_df = pair_df[pair_df["Option Type"] == req_type].copy()
        if type_df.empty:
            logger.warning("SKIP underlying=%s reason=no_%s_legs", underlying, req_type)
            total_stats.append((underlying, len(pair_df), 0, 0, 0, 0))
            continue

        atm_strike = sorted(type_df["Strike"].dropna().unique(), key=lambda x: abs(x - type_df["Strike"].median()))[0]
        atm_rows = type_df[type_df["Strike"] == atm_strike]
        atm_vol = safe_float(atm_rows["Chain Volume"].max() if not atm_rows.empty else 0, 0)

        logger.info("UNDERLYING %s side=%s pair_rows=%d type_rows=%d atm=%s atmvol=%.0f", underlying, side, len(pair_df), len(type_df), atm_strike, atm_vol)

        if atm_vol < MIN_ATM_CHAIN_VOLUME:
            logger.warning("SKIP underlying=%s reason=atm_chain_volume atmvol=%.0f min=%d", underlying, atm_vol, MIN_ATM_CHAIN_VOLUME)
            total_stats.append((underlying, len(pair_df), len(type_df), 0, 0, 0))
            continue

        passed_history = 0
        passed_ltp = 0
        passed_volume = 0

        for _, row in type_df.iterrows():
            strike = safe_float(row.get("Strike"), np.nan)
            opt_type = str(row.get("Option Type", "")).upper()
            sym = str(row.get("Option Symbol", "")).strip()

            scanned = scansingleoption(sym, opt_type, strike, underlying)
            if not scanned:
                continue
            passed_history += 1

            scanned["OI"] = safe_float(row.get("OI"), np.nan)
            scanned["Volume"] = max(safe_float(scanned.get("Volume", 0), 0), safe_float(row.get("Chain Volume", 0), 0))
            scanned["OI+Volume+OBV Score"] = option_liquidity_score(scanned.get("OI", 0), scanned.get("Volume", 0), scanned.get("OBV", 0))
            scanned["ATM Strike"] = atm_strike

            if safe_float(scanned.get("LTP", 0.0), 0.0) < MIN_OPTION_LTP:
                logger.warning("SKIP symbol=%s reason=ltp ltp=%.2f min=%.2f", sym, safe_float(scanned.get("LTP", 0.0), 0.0), MIN_OPTION_LTP)
                continue
            passed_ltp += 1

            if safe_float(scanned.get("Volume", 0), 0) < MIN_OPTION_DAY_VOLUME:
                logger.warning("SKIP symbol=%s reason=volume vol=%.0f min=%d", sym, safe_float(scanned.get("Volume", 0), 0), MIN_OPTION_DAY_VOLUME)
                continue
            passed_volume += 1

            rows.append(scanned)

            hist = scanned.get("Iteration History")
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                temp = hist.copy()
                temp.insert(0, "Option Symbol", sym)
                temp.insert(1, "Underlying", underlying)
                temp.insert(2, "Strike", strike)
                temp.insert(3, "Option Type", opt_type)
                temp.insert(4, "ATM Strike", atm_strike)
                iter_rows.append(temp)

        total_stats.append((underlying, len(pair_df), len(type_df), passed_history, passed_ltp, passed_volume))
        logger.info("SUMMARY underlying=%s pair_rows=%d type_rows=%d history_ok=%d ltp_ok=%d final_ok=%d", underlying, len(pair_df), len(type_df), passed_history, passed_ltp, passed_volume)

    if total_stats:
        stats_df = pd.DataFrame(total_stats, columns=["Underlying", "pair_rows", "type_rows", "history_ok", "ltp_ok", "final_ok"])
        stats_df.to_csv(os.path.join(OUTPUT_DIR, f"debug_stats_{side}.csv"), index=False)

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    out = pd.DataFrame(rows)
    out = rank_option_candidates(out, side)

    final_cols = [c for c in OPTION_EMAIL_COLS if c in out.columns]
    final_out = out[final_cols].reset_index(drop=True)

    iter_df = pd.DataFrame()
    if iter_rows and not final_out.empty:
        all_iters = pd.concat(iter_rows, ignore_index=True)
        all_iters = all_iters[all_iters["Option Symbol"].isin(final_out["Option Symbol"])].copy()
        sort_cols = [c for c in ["Underlying", "Option Type", "Strike", "Option Symbol", "iteration"] if c in all_iters.columns]
        if sort_cols:
            all_iters = all_iters.sort_values(sort_cols).reset_index(drop=True)
        group_cols = [c for c in ["Underlying", "Option Type", "Strike", "Option Symbol"] if c in all_iters.columns]
        if group_cols and not all_iters.empty:
            all_iters["iteration"] = all_iters.groupby(group_cols).cumcount() + 1
        if "iteration" in all_iters.columns:
            all_iters["iteration"] = pd.to_numeric(all_iters["iteration"], errors="coerce").astype("Int64")
        all_iters = all_iters[all_iters["iteration"].between(1, ITERATIONS_TO_KEEP)]
        iter_df = all_iters.reset_index(drop=True)

    return final_out, iter_df


def _fmt_num(v) -> str:
    x = safe_float(v, np.nan)
    if pd.isna(x):
        return ""
    return str(int(x)) if float(x).is_integer() else f"{x:.2f}".rstrip("0").rstrip(".")


def _fmt_num_list(values: List[object]) -> str:
    cleaned = []
    for v in values:
        x = safe_float(v, np.nan)
        if pd.notna(x):
            cleaned.append(x)
    cleaned = sorted(set(cleaned))
    return ", ".join(_fmt_num(v) for v in cleaned)


def build_atm_chain_rows(options_df: pd.DataFrame, iteration_df: pd.DataFrame) -> List[Dict]:
    if options_df is None or options_df.empty or iteration_df is None or iteration_df.empty:
        return []

    opt = options_df.copy()
    itr = iteration_df.copy()

    if "Option Symbol" not in opt.columns or "Option Symbol" not in itr.columns:
        return []

    opt["Option Symbol"] = opt["Option Symbol"].astype(str).str.strip()
    opt["Strike"] = pd.to_numeric(opt["Strike"], errors="coerce")
    if "ATM Strike" in opt.columns:
        opt["ATM Strike"] = pd.to_numeric(opt["ATM Strike"], errors="coerce")

    itr["Option Symbol"] = itr["Option Symbol"].astype(str).str.strip()
    itr["Strike"] = pd.to_numeric(itr.get("Strike"), errors="coerce")
    itr["iteration"] = pd.to_numeric(itr.get("iteration"), errors="coerce")
    itr["close"] = pd.to_numeric(itr.get("close"), errors="coerce")
    itr = itr.sort_values(["Underlying", "Option Type", "Option Symbol", "iteration"]).reset_index(drop=True)

    itr["prev_close"] = itr.groupby(["Underlying", "Option Type", "Option Symbol"])["close"].shift(1)
    itr["is_plus"] = itr["close"] > itr["prev_close"]

    out_rows = []

    for (underlying, option_type), grp in opt.groupby(["Underlying", "Option Type"], dropna=False):
        grp = grp.dropna(subset=["Strike"]).sort_values("Strike").copy()
        if grp.empty:
            continue

        symbols = grp["Option Symbol"].dropna().astype(str).str.strip().unique().tolist()
        tracked_count = len(symbols)
        if tracked_count == 0:
            continue

        atm_candidates = grp["ATM Strike"].dropna().tolist() if "ATM Strike" in grp.columns else []
        atm_strike = safe_float(atm_candidates[0], np.nan) if atm_candidates else safe_float(grp["Strike"].median(), np.nan)

        itr_grp = itr[
            (itr["Underlying"] == underlying) &
            (itr["Option Type"] == option_type) &
            (itr["Option Symbol"].isin(symbols))
        ].copy()
        if itr_grp.empty:
            continue

        latest_iter = pd.to_numeric(itr_grp["iteration"], errors="coerce").dropna().max()
        if pd.isna(latest_iter):
            continue

        latest = itr_grp[itr_grp["iteration"] == latest_iter].copy()
        latest["is_plus"] = latest["is_plus"].fillna(False)

        qualified_symbols = latest.loc[latest["is_plus"], "Option Symbol"].astype(str).str.strip().unique().tolist()
        qualified_strikes = grp[grp["Option Symbol"].isin(qualified_symbols)]["Strike"].dropna().tolist()
        qualified_count = len(qualified_symbols)

        all_qualified_now = tracked_count > 0 and qualified_count == tracked_count

        by_time = (
            itr_grp.assign(is_plus=itr_grp["is_plus"].fillna(False))
            .groupby("timestamp", dropna=False)
            .agg(
                symbol_count=("Option Symbol", "nunique"),
                qualified_count=("is_plus", "sum"),
            )
            .reset_index()
            .sort_values("timestamp")
        )

        first_full = by_time[
            (by_time["symbol_count"] >= tracked_count) &
            (by_time["qualified_count"] >= tracked_count)
        ]

        entry_time = ""
        if all_qualified_now and not first_full.empty:
            entry_time = str(first_full.iloc[0]["timestamp"])

        atm_pick = grp.iloc[(grp["Strike"] - atm_strike).abs().argsort()[:1]].copy()
        atm_ltp = safe_float(atm_pick["LTP"].iloc[0], np.nan) if not atm_pick.empty and "LTP" in atm_pick.columns else np.nan

        last_check = ""
        if "timestamp" in latest.columns and not latest["timestamp"].dropna().empty:
            last_check = str(latest["timestamp"].dropna().iloc[-1])

        out_rows.append({
            "Underlying": underlying,
            "Option Type": option_type,
            "ATM Strike": round(atm_strike, 2) if pd.notna(atm_strike) else np.nan,
            "ATM LTP": round(atm_ltp, 2) if pd.notna(atm_ltp) else np.nan,
            "Qualified Strikes": _fmt_num_list(qualified_strikes),
            "All Tracked Strikes": _fmt_num_list(grp["Strike"].dropna().tolist()),
            "Qualified Count": qualified_count,
            "Tracked Count": tracked_count,
            "Chain Signal": "ENTER" if all_qualified_now else "WAIT",
            "Entry Time": entry_time,
            "Last Check": last_check,
        })

    return sorted(
        out_rows,
        key=lambda r: (
            str(r.get("Underlying", "")),
            str(r.get("Option Type", "")),
            safe_float(r.get("ATM Strike"), 0),
        ),
    )


def load_daily_state() -> Dict:
    today = datetime.now().strftime("%Y-%m-%d")
    if os.path.exists(DAILY_STATE_FILE):
        try:
            with open(DAILY_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict) and state.get("date") == today:
                return state
        except Exception as e:
            logger.warning("Could not read state file: %s", e)
    return {"date": today, "rows": []}


def save_daily_state(state: Dict):
    try:
        os.makedirs(os.path.dirname(DAILY_STATE_FILE) or ".", exist_ok=True)
        with open(DAILY_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str)
    except Exception as e:
        logger.warning("Could not save chain state: %s", e)


def update_sticky_rows(state: Dict, new_rows: List[Dict]) -> List[Dict]:
    existing_raw = state.get("rows", {})
    existing = {}

    if isinstance(existing_raw, list):
        for r in existing_raw:
            if isinstance(r, dict):
                key = f"{r.get('Underlying')}_{r.get('Option Type')}_{r.get('ATM Strike')}"
                existing[key] = dict(r)
    elif isinstance(existing_raw, dict):
        existing = {k: dict(v) for k, v in existing_raw.items() if isinstance(v, dict)}

    for row in new_rows:
        row = dict(row)
        key = f"{row.get('Underlying')}_{row.get('Option Type')}_{row.get('ATM Strike')}"

        if key not in existing:
            if row.get("Chain Signal") == "ENTER":
                if not row.get("Entry Time"):
                    row["Entry Time"] = row.get("Last Check") or datetime.now().strftime("%H:%M")
                existing[key] = row
        else:
            keep = existing[key]
            for col in ATM_CHAIN_EMAIL_COLS:
                if col in row and row.get(col) not in [None, ""]:
                    keep[col] = row[col]
            if not keep.get("Entry Time"):
                keep["Entry Time"] = row.get("Entry Time") or row.get("Last Check") or datetime.now().strftime("%H:%M")
            existing[key] = keep

    state["rows"] = list(existing.values())
    save_daily_state(state)

    return sorted(
        existing.values(),
        key=lambda r: (
            r.get("Entry Time", "99:99"),
            str(r.get("Underlying", "")),
            str(r.get("Option Type", "")),
        ),
    )


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


def compact_table_html(df: pd.DataFrame, title: str, max_rows: int) -> str:
    if df is None or df.empty:
        return f"<h3>{title}</h3><p>No rows</p>"

    renamed = df.rename(columns=OPTION_EMAIL_COL_RENAME).copy()
    cols = [c for c in [
        "ATM Strike",
        "Underlying",
        "Option Type",
        "Opt Symbol",
        "Strike",
        "LTP",
        "% Chg",
        "OI",
        "Volume",
        "OBV",
        "Liq Score",
        "Rank",
        "Rank Delta",
        "Cumulative ADX",
        "5m_Signal",
        "15m_Signal",
        "Overall_Signal",
        "IVP",
        "Time",
    ] if c in renamed.columns]

    renamed = renamed[cols].head(max_rows).copy()
    header = "".join(f"<th>{c}</th>" for c in cols)

    rows_html = []
    for _, row in renamed.iterrows():
        cells = []
        for c in cols:
            val = "" if pd.isna(row[c]) else str(row[c])
            cells.append(f"<td>{val}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")

    body = "".join(rows_html)
    return f"<h3>{title}</h3><table border='1' cellspacing='0' cellpadding='4'><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def build_atm_chain_table_html(rows: List[Dict], title: str) -> str:
    df = pd.DataFrame(rows)
    if df.empty:
        return f"<h3>{title}</h3><p>No ATM chain signals.</p>"

    cols = [c for c in ATM_CHAIN_EMAIL_COLS if c in df.columns]
    header = "".join(f"<th>{c}</th>" for c in cols)

    body_rows = []
    for _, row in df[cols].iterrows():
        cells = []
        for c in cols:
            val = "" if pd.isna(row[c]) else str(row[c])
            style = ""
            if c == "Chain Signal":
                if val == "ENTER":
                    style = " style='background:#d4edda;color:#155724;font-weight:bold'"
                else:
                    style = " style='background:#e2e3e5;color:#383d41;font-weight:bold'"
            cells.append(f"<td{style}>{val}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        f"<h3>{title}</h3>"
        f"<table border='1' cellspacing='0' cellpadding='4'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table>"
    )


def build_chain_email_html(long_rows: List[Dict], short_rows: List[Dict]) -> str:
    ts = datetime.now().strftime("%d %b %Y, %H:%M")
    note = "Only ATM summary rows are shown. Entry Time is the first timestamp when all tracked consecutive strikes qualified together."
    return (
        "<html><body>"
        f"<p>Chain Signal Report - {ts}</p>"
        f"<p>{note}</p>"
        f"{build_atm_chain_table_html(long_rows, 'LONG ATM CHAIN CE')}"
        f"{build_atm_chain_table_html(short_rows, 'SHORT ATM CHAIN PE')}"
        "</body></html>"
    )


def build_cepebuy_rows(options_df: pd.DataFrame, iteration_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    if options_df is None or options_df.empty or iteration_df is None or iteration_df.empty:
        return [], []
    if "Option Symbol" not in iteration_df.columns or "windowsignal" not in iteration_df.columns:
        return [], []

    iter_map = {sym: grp for sym, grp in iteration_df.groupby("Option Symbol")}
    ce_rows, pe_rows = [], []

    for _, row in options_df.iterrows():
        sym = str(row.get("Option Symbol", "")).strip()
        if not sym or sym not in iter_map:
            continue

        grp = iter_map[sym]
        signals = [str(v).strip().title() for v in grp["windowsignal"].tolist()]
        last11 = signals[-11:]
        if len(last11) < 11:
            continue

        buy_count = sum(s.startswith("Buy") for s in last11)
        sell_count = sum(s.startswith("Sell") for s in last11)
        opt_type = str(row.get("Option Type", "")).upper()

        trade_signal = "CE BUY" if opt_type == "CE" and buy_count >= 8 else "PE BUY" if opt_type == "PE" and sell_count >= 8 else ""
        if not trade_signal:
            continue

        out = {
            "Underlying": row.get("Underlying", ""),
            "Option Type": row.get("Option Type", ""),
            "Option Symbol": sym,
            "Strike": row.get("Strike", np.nan),
            "LTP": row.get("LTP", np.nan),
            "% Change": row.get("% Change", np.nan),
            "Rank Delta": row.get("Rank Delta", np.nan),
            "Last Iteration Time": row.get("Last Iteration Time", ""),
            "Trade Signal": trade_signal,
            "Buy Count": buy_count,
            "Sell Count": sell_count,
        }

        if trade_signal == "CE BUY":
            ce_rows.append(out)
        else:
            pe_rows.append(out)

    return ce_rows, pe_rows


def build_cepebuy_email_html(ce_rows: List[Dict], pe_rows: List[Dict]) -> str:
    ts = datetime.now().strftime("%d %b %Y, %H:%M")

    def table(rows, title):
        df = pd.DataFrame(rows)
        if df.empty:
            return f"<h3>{title}</h3><p>No momentum buy signals.</p>"
        cols = [c for c in [
            "Underlying",
            "Option Type",
            "Option Symbol",
            "Strike",
            "LTP",
            "% Change",
            "Rank Delta",
            "Last Iteration Time",
            "Trade Signal",
            "Buy Count",
            "Sell Count",
        ] if c in df.columns]
        header = "".join(f"<th>{c}</th>" for c in cols)
        body = "".join(
            f"<tr>{''.join(f'<td>{row[c]}</td>' for c in cols)}</tr>"
            for _, row in df[cols].iterrows()
        )
        return f"<h3>{title}</h3><table border='1' cellspacing='0' cellpadding='4'><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"

    return (
        "<html><body>"
        f"<p>CE PE Momentum Buy Report - {ts}</p>"
        "<p>Rule: 8 of last 11 same-direction windows.</p>"
        f"{table(ce_rows, 'CE BUY MOMENTUM SIGNALS')}"
        f"{table(pe_rows, 'PE BUY MOMENTUM SIGNALS')}"
        "</body></html>"
    )


def send_cepebuy_email(ce_rows: List[Dict], pe_rows: List[Dict], attachments: Optional[List[str]] = None) -> bool:
    logger.info("CE BUY %d PE BUY %d", len(ce_rows), len(pe_rows))
    if not ce_rows and not pe_rows:
        return False
    subject = f"CE PE Momentum Buy Report - {datetime.now().strftime('%d %b %H:%M')}"
    html = build_cepebuy_email_html(ce_rows, pe_rows)
    return send_single_email(subject, html, attachments)


def scansymbol(symbol: str) -> Optional[Dict]:
    eq = format_eq_symbol(symbol)
    daily_df = gethistory(eq, "D", DAILY_LOOKBACK_DAYS)
    intra_df = gethistory(eq, "5", INTRADAY_LOOKBACK_DAYS)
    if daily_df.empty or intra_df.empty:
        logger.warning("Underlying history empty symbol=%s daily=%d intra=%d", symbol, len(daily_df), len(intra_df))
        return None
    summary = summarize_intraday(intra_df, daily_df)
    if not summary:
        return None
    summary["Symbol"] = symbol
    return summary


def send_chain_signal_email(long_rows: List[Dict], short_rows: List[Dict], attachments: Optional[List[str]] = None) -> bool:
    subject = f"Chain Signal Report - {datetime.now().strftime('%d %b %H:%M')}"
    html = build_chain_email_html(long_rows, short_rows)
    return send_single_email(subject, html, attachments)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    initfyers()

    symbols = load_fno_symbols_from_sectors(SECTORS_DIR)
    if not symbols:
        raise RuntimeError("No F&O symbols found in sector CSVs.")

    rows = []
    for i, symbol in enumerate(symbols, start=1):
        logger.info("%d/%d Scanning %s", i, len(symbols), symbol)
        row = scansymbol(symbol)
        if row:
            rows.append(row)
        time.sleep(PER_SYMBOL_SLEEP_SEC)

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        raise RuntimeError("No symbols returned usable market data.")

    summary_df = summary_df.sort_values(["Rank Delta", "% Change"], ascending=[False, False]).reset_index(drop=True)

    summary_path = os.path.join(OUTPUT_DIR, "fo_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    long_seed_df, short_seed_df = choose_top_candidates(summary_df, topn=TOP_N_UNDERLYINGS)
    long_df, long_iter_df = buildoptioncandidates(long_seed_df, side="long")
    short_df, short_iter_df = buildoptioncandidates(short_seed_df, side="short")

    iteration_df = (
        pd.concat([long_iter_df, short_iter_df], ignore_index=True)
        if (not long_iter_df.empty or not short_iter_df.empty)
        else pd.DataFrame()
    )

    long_path = os.path.join(OUTPUT_DIR, "fo_long_candidates.csv")
    short_path = os.path.join(OUTPUT_DIR, "fo_short_candidates.csv")
    iter_path = os.path.join(OUTPUT_DIR, "fo_iteration_history.csv")

    long_df.to_csv(long_path, index=False)
    short_df.to_csv(short_path, index=False)

    if iteration_df.empty:
        iteration_df = pd.DataFrame(columns=[
            "iteration", "Underlying", "Option Type", "Strike", "Option Symbol",
            "timestamp", "windowminutes", "windowstart", "windowend",
            "currentwindowscore", "previoustradingdaysametimescore",
            "windowdelta", "windowsignal", "close"
        ])
    iteration_df.to_csv(iter_path, index=False)

    logger.info("LONG %d rows SHORT %d rows Iter %d rows", len(long_df), len(short_df), len(iteration_df))

    combined_options_df = (
        pd.concat([long_df, short_df], ignore_index=True)
        if (not long_df.empty or not short_df.empty)
        else pd.DataFrame()
    )

    atm_rows_now = build_atm_chain_rows(combined_options_df, iteration_df)

    state = load_daily_state()
    sticky_rows = update_sticky_rows(state, atm_rows_now)

    long_rows = [r for r in sticky_rows if str(r.get("Option Type", "")).upper() == "CE"]
    short_rows = [r for r in sticky_rows if str(r.get("Option Type", "")).upper() == "PE"]

    chain_long_path = os.path.join(OUTPUT_DIR, "chain_long.csv")
    chain_short_path = os.path.join(OUTPUT_DIR, "chain_short.csv")
    atm_chain_path = os.path.join(OUTPUT_DIR, "atm_chain.csv")

    pd.DataFrame(long_rows).to_csv(chain_long_path, index=False)
    pd.DataFrame(short_rows).to_csv(chain_short_path, index=False)
    pd.DataFrame(sticky_rows).to_csv(atm_chain_path, index=False)

    send_chain_signal_email(
        long_rows,
        short_rows,
        attachments=[summary_path, long_path, short_path, iter_path, atm_chain_path]
    )

    ce_buy_rows, pe_buy_rows = build_cepebuy_rows(combined_options_df, iteration_df)
    send_cepebuy_email(
        ce_buy_rows,
        pe_buy_rows,
        attachments=[chain_long_path, chain_short_path, atm_chain_path]
    )


if __name__ == "__main__":
    main()

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

fyers = None

OPTION_EMAIL_COLS = [
    "Underlying", "Option Type", "Option Symbol", "Strike", "LTP", "% Change", "OI", "Volume", "OBV",
    "OI+Volume+OBV Score", "EMAIL_RANK_SCORE", "Entry Value", "Entry Time", "Exit Value", "Exit Time",
    "Cumulative ADX", "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal", "Bull_Signal", "Bear_Signal",
    "Overall_Signal", "Price_Lead_Status", "IVP", "Volatility State", "Last Iteration Time",
]

OPTION_EMAIL_COL_RENAME = {
    "OI+Volume+OBV Score": "Liq Score",
    "EMAIL_RANK_SCORE": "Rank",
    "% Change": "% Chg",
    "Last Iteration Time": "Time",
    "Price_Lead_Status": "Lead",
    "Volatility State": "Vol State",
    "Option Symbol": "Opt Symbol",
    "Entry Value": "Entry",
    "Exit Value": "Exit",
}


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

    bull = int(pct_change >= 0) + int(safe_float(vwap_z.iloc[-1], 0) > 0.3) + int(safe_float(plusdi.iloc[-1], 0) > safe_float(minusdi.iloc[-1], 0)) + int(safe_float(rsi.iloc[-1], 50) > 55) + int(win_signal.startswith("Buy")) * 2
    bear = int(pct_change <= 0) + int(safe_float(vwap_z.iloc[-1], 0) < -0.3) + int(safe_float(minusdi.iloc[-1], 0) > safe_float(plusdi.iloc[-1], 0)) + int(safe_float(rsi.iloc[-1], 50) < 45) + int(win_signal.startswith("Sell")) * 2
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
    chain_attempts = [str(s).strip() for s in chain_attempts]
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
    return round(np.log1p(max(safe_float(oi, 0), 0)) * 0.45 + np.log1p(max(safe_float(volume, 0), 0)) * 0.35 + np.log1p(max(abs(safe_float(obv, 0)), 0)) * 0.20, 4)


def rank_option_candidates(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["__liq"] = pd.to_numeric(out.get("OI+Volume+OBV Score", 0), errors="coerce").fillna(0)
    out["__adx"] = pd.to_numeric(out.get("Cumulative ADX", 0), errors="coerce").fillna(0)
    out["__pct"] = pd.to_numeric(out.get("% Change", 0), errors="coerce").fillna(0)
    opt_type = out["Option Type"].astype(str).str.upper() if "Option Type" in out.columns else pd.Series("", index=out.index)
    if side == "long":
        type_bonus = np.where(opt_type.eq("CE"), 0.30, 0.10)
        out["EMAIL_RANK_SCORE"] = out["__liq"] * 0.40 + out["__adx"] * 0.18 + out["__pct"] * 0.10 + type_bonus
        out = out.sort_values(["EMAIL_RANK_SCORE", "__liq", "__adx", "__pct"], ascending=[False, False, False, False])
    else:
        type_bonus = np.where(opt_type.eq("PE"), 0.30, 0.10)
        out["EMAIL_RANK_SCORE"] = out["__liq"] * 0.40 + out["__adx"] * 0.18 + (-out["__pct"]) * 0.10 + type_bonus
        out = out.sort_values(["EMAIL_RANK_SCORE", "__liq", "__adx", "__pct"], ascending=[False, False, False, True])
    return out.reset_index(drop=True)


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
            continue
        for _, row in type_df.iterrows():
            strike = safe_float(row.get("Strike"), np.nan)
            opt_type = str(row.get("Option Type", "")).upper()
            sym = str(row.get("Option Symbol", "")).strip()
            scanned = scansingleoption(sym, opt_type, strike, underlying)
            if not scanned:
                continue
            scanned["OI"] = safe_float(row.get("OI"), np.nan)
            scanned["Volume"] = max(safe_float(scanned.get("Volume"), 0), safe_float(row.get("Chain Volume"), 0))
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
            all_iters = all_iters.sort_values(group_cols + (["timestamp"] if "timestamp" in all_iters.columns else [])).reset_index(drop=True)
            all_iters["iteration"] = all_iters.groupby(group_cols).cumcount() + 1
            all_iters["iteration"] = pd.to_numeric(all_iters["iteration"], errors="coerce").astype("Int64")
            all_iters = all_iters[all_iters["iteration"].between(1, ITERATIONS_TO_KEEP)]
        iter_df = all_iters.reset_index(drop=True)
    return final_out, iter_df


def chain_entry_exit_from_iters(iter_df: pd.DataFrame) -> Dict[str, object]:
    if iter_df is None or iter_df.empty:
        return {"Chain Signal": "WAIT", "Entry Value": np.nan, "Entry Time": "", "Exit Value": np.nan, "Exit Time": ""}
    d = iter_df.copy()
    d["iteration"] = pd.to_numeric(d["iteration"], errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna(subset=["iteration", "close", "Option Symbol"])
    if d.empty:
        return {"Chain Signal": "WAIT", "Entry Value": np.nan, "Entry Time": "", "Exit Value": np.nan, "Exit Time": ""}
    pivot = d.pivot_table(index="iteration", columns="Option Symbol", values="close", aggfunc="last").sort_index()
    if pivot.shape[0] < 2 or pivot.shape[1] == 0:
        return {"Chain Signal": "WAIT", "Entry Value": np.nan, "Entry Time": "", "Exit Value": np.nan, "Exit Time": ""}

    prev = pivot.shift(1)
    all_up = (pivot > prev).all(axis=1)
    entry_iters = all_up[all_up].index.tolist()
    if not entry_iters:
        return {"Chain Signal": "WAIT", "Entry Value": np.nan, "Entry Time": "", "Exit Value": np.nan, "Exit Time": ""}

    entry_iter = entry_iters[0]
    entry_row = d[d["iteration"] == entry_iter].sort_values(["timestamp", "Option Symbol"]).copy()
    entry_time = str(entry_row["timestamp"].iloc[0]) if "timestamp" in entry_row.columns and not entry_row.empty else ""
    entry_value = round(float(entry_row["close"].mean()), 2) if not entry_row.empty else np.nan

    exit_iter = None
    for it in sorted([x for x in pivot.index if x > entry_iter]):
        prev_it = it - 1
        if prev_it not in pivot.index:
            continue
        if (pivot.loc[it] > pivot.loc[prev_it]).all():
            continue
        exit_iter = it
        break

    if exit_iter is not None:
        exit_row = d[d["iteration"] == exit_iter].sort_values(["timestamp", "Option Symbol"]).copy()
        exit_time = str(exit_row["timestamp"].iloc[0]) if "timestamp" in exit_row.columns and not exit_row.empty else ""
        exit_value = round(float(exit_row["close"].mean()), 2) if not exit_row.empty else np.nan
        return {"Chain Signal": "EXIT", "Entry Value": entry_value, "Entry Time": entry_time, "Exit Value": exit_value, "Exit Time": exit_time}

    last_row = d[d["iteration"] == pivot.index.max()].sort_values(["timestamp", "Option Symbol"]).copy()
    last_time = str(last_row["timestamp"].iloc[0]) if "timestamp" in last_row.columns and not last_row.empty else ""
    last_value = round(float(last_row["close"].mean()), 2) if not last_row.empty else np.nan
    return {"Chain Signal": "ENTER", "Entry Value": entry_value, "Entry Time": entry_time, "Exit Value": last_value, "Exit Time": last_time}


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
    return out


def compact_table_html(df: pd.DataFrame, title: str, max_rows: int) -> str:
    if df is None or df.empty:
        return f"<div class="section-card"><h3>{title}</h3><p class="muted">No rows</p></div>"
    df = _prepare_email_df(df)
    if "Chain Signal" in df.columns:
        df["Chain Signal"] = df["Chain Signal"].astype(str).map(
            lambda x: f'<span class="badge badge-enter">{x}</span>' if x.upper() == "ENTER" else (
                f'<span class="badge badge-exit">{x}</span>' if x.upper() == "EXIT" else f'<span class="badge badge-wait">{x}</span>'
            )
        )
    cols = [c for c in ["Underlying", "Option Type", "Strike", "LTP", "% Chg", "OI", "Volume", "OBV", "Liq Score", "Rank", "Entry", "Entry Time", "Exit", "Exit Time", "Cumulative ADX", "5m_Signal", "15m_Signal", "Overall_Signal", "IVP", "Time", "Chain Signal"] if c in df.columns]
    body = df[cols].head(max_rows).to_html(index=False, border=0, escape=False, classes="mail-table")
    return f"<div class="section-card"><h3>{title}</h3>{body}</div>"


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
                <p>Color-coded chain summary for long and short candidates.</p>
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
    html = build_chain_email_html(long_rows, short_rows)
    return send_single_email(subject, html, attachments)


def build_cepebuy_rows(options_df: pd.DataFrame, iteration_df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
    if options_df is None or options_df.empty or iteration_df is None or iteration_df.empty:
        return [], []
    required = {"Underlying", "Option Type", "Strike", "Option Symbol", "close"}
    if not required.issubset(set(iteration_df.columns)):
        return [], []

    ce_rows, pe_rows = [], []
    group_cols = ["Underlying", "Option Type", "Strike"]

    for _, grp in iteration_df.groupby(group_cols):
        grp = grp.copy()
        sort_cols = [c for c in ["iteration", "timestamp", "Option Symbol"] if c in grp.columns]
        if sort_cols:
            grp = grp.sort_values(sort_cols).reset_index(drop=True)

        chain_sig = chain_entry_exit_from_iters(grp)
        if str(chain_sig.get("Chain Signal", "WAIT")).upper() not in ("ENTER", "EXIT"):
            continue

        first = grp.iloc[0]
        opt_type = str(first.get("Option Type", "")).upper()
        strike = safe_float(first.get("Strike", np.nan), np.nan)
        underlying = first.get("Underlying", "")

        strike_series = pd.to_numeric(options_df.get("Strike"), errors="coerce") if "Strike" in options_df.columns else pd.Series(dtype=float)
        opt_match = options_df[
            (options_df["Underlying"].astype(str) == str(underlying)) &
            (options_df["Option Type"].astype(str).str.upper() == opt_type) &
            (strike_series == strike)
        ].copy()
        if opt_match.empty:
            ltp = pd.to_numeric(grp["close"], errors="coerce").dropna().iloc[-1] if pd.to_numeric(grp["close"], errors="coerce").dropna().shape[0] else np.nan
            pct_chg = np.nan
        else:
            sort_pref = [c for c in ["Volume", "OI+Volume+OBV Score", "EMAIL_RANK_SCORE"] if c in opt_match.columns]
            if sort_pref:
                opt_match = opt_match.sort_values(sort_pref, ascending=[False] * len(sort_pref))
            row = opt_match.iloc[0]
            ltp = row.get("LTP", np.nan)
            pct_chg = row.get("% Change", np.nan)

        out = {
            "Underlying": underlying,
            "Option Type": opt_type,
            "Strike": strike,
            "LTP": ltp,
            "% Change": pct_chg,
            "Entry Value": chain_sig.get("Entry Value", np.nan),
            "Entry Time": chain_sig.get("Entry Time", ""),
            "Exit Value": chain_sig.get("Exit Value", np.nan),
            "Exit Time": chain_sig.get("Exit Time", ""),
            "Trade Signal": f"{opt_type} CHAIN",
            "Chain Signal": chain_sig.get("Chain Signal", "WAIT"),
            "Chain Legs": int(grp["Option Symbol"].nunique()) if "Option Symbol" in grp.columns else 0,
        }
        if opt_type == "CE":
            ce_rows.append(out)
        elif opt_type == "PE":
            pe_rows.append(out)

    return ce_rows, pe_rows


def build_cepebuy_email_html(ce_rows: List[Dict], pe_rows: List[Dict]) -> str:
    ts = datetime.now().strftime("%d %b %Y, %H:%M")

    def style_signal(val: str) -> str:
        s = str(val).upper()
        if s == "ENTER":
            return f'<span class="badge badge-enter">{s}</span>'
        if s == "EXIT":
            return f'<span class="badge badge-exit">{s}</span>'
        return f'<span class="badge badge-wait">{s}</span>'

    def table(rows, title, accent):
        df = pd.DataFrame(rows)
        if df.empty:
            return f"<div class="section-card"><h3 style="color:{accent};">{title}</h3><p class="muted">No chain-based eligible strikes.</p></div>"
        if "Chain Signal" in df.columns:
            df["Chain Signal"] = df["Chain Signal"].map(style_signal)
        cols = [c for c in ["Underlying", "Option Type", "Strike", "LTP", "% Change", "Entry Value", "Entry Time", "Exit Value", "Exit Time", "Trade Signal", "Chain Signal", "Chain Legs"] if c in df.columns]
        html = df[cols].to_html(index=False, border=0, escape=False, classes="mail-table")
        return f"<div class="section-card"><h3 style="color:{accent};">{title}</h3>{html}</div>"

    return f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; background: #f4f7fb; color: #172033; margin: 0; padding: 20px; }}
        .wrap {{ max-width: 1180px; margin: 0 auto; }}
        .hero {{ background: linear-gradient(135deg, #7c3aed, #2563eb); color: white; padding: 18px 22px; border-radius: 14px; }}
        .hero h2 {{ margin: 0 0 6px 0; font-size: 24px; }}
        .hero p {{ margin: 0; color: #ede9fe; }}
        .section-card {{ background: white; border: 1px solid #dbe4f0; border-radius: 14px; padding: 14px 16px; margin-top: 16px; box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06); }}
        .mail-table {{ width: 100%; border-collapse: collapse; font-size: 13px; overflow: hidden; }}
        .mail-table th {{ background: #1d4ed8; color: white; padding: 10px 8px; text-align: left; }}
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
                <h2>CE PE Chain Eligibility Report - {ts}</h2>
                <p>Strike is included only when grouped chain iteration movement marks it as eligible. Option Symbol is hidden from the email.</p>
            </div>
            {table(ce_rows, 'CE CHAIN ELIGIBLE STRIKES', '#2563eb')}
            {table(pe_rows, 'PE CHAIN ELIGIBLE STRIKES', '#7c3aed')}
        </div>
    </body>
    </html>
    """


def send_cepebuy_email(ce_rows: List[Dict], pe_rows: List[Dict], attachments: Optional[List[str]] = None) -> bool:
    if not ce_rows and not pe_rows:
        return False
    subject = f"CE PE Chain Eligibility Report - {datetime.now().strftime('%d %b %H:%M')}"
    return send_single_email(subject, build_cepebuy_email_html(ce_rows, pe_rows), attachments)


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


def chain_row_from_group(grp: pd.DataFrame) -> Dict:
    sig = chain_entry_exit_from_iters(grp)
    first = grp.iloc[0]
    return {
        "Underlying": first.get("Underlying", ""),
        "Option Type": first.get("Option Type", ""),
        "Strike": first.get("Strike", np.nan),
        "Option Symbol": first.get("Option Symbol", ""),
        "Entry Value": sig.get("Entry Value", np.nan),
        "Entry Time": sig.get("Entry Time", ""),
        "Exit Value": sig.get("Exit Value", np.nan),
        "Exit Time": sig.get("Exit Time", ""),
        "Chain Signal": sig.get("Chain Signal", "WAIT"),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    initfyers()
    symbols = load_fno_symbols_from_sectors(SECTORS_DIR)
    if not symbols:
        raise RuntimeError("No F&O symbols found in sector CSVs.")

    rows = []
    for symbol in symbols:
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

    iteration_df = pd.concat([long_iter_df, short_iter_df], ignore_index=True) if (not long_iter_df.empty or not short_iter_df.empty) else pd.DataFrame()
    long_path = os.path.join(OUTPUT_DIR, "fo_long_candidates.csv")
    short_path = os.path.join(OUTPUT_DIR, "fo_short_candidates.csv")
    iter_path = os.path.join(OUTPUT_DIR, "fo_iteration_history.csv")

    long_df.to_csv(long_path, index=False)
    short_df.to_csv(short_path, index=False)
    if iteration_df.empty:
        iteration_df = pd.DataFrame(columns=["iteration", "Underlying", "Option Type", "Strike", "Option Symbol", "timestamp", "windowminutes", "windowstart", "windowend", "currentwindowscore", "previoustradingdaysametimescore", "windowdelta", "windowsignal", "close"])
    iteration_df.to_csv(iter_path, index=False)

    long_merged = []
    short_merged = []
    if not long_iter_df.empty:
        cols = [c for c in ["Underlying", "Option Type", "Strike"] if c in long_iter_df.columns]
        for _, grp in long_iter_df.groupby(cols):
            long_merged.append(chain_row_from_group(grp))
    if not short_iter_df.empty:
        cols = [c for c in ["Underlying", "Option Type", "Strike"] if c in short_iter_df.columns]
        for _, grp in short_iter_df.groupby(cols):
            short_merged.append(chain_row_from_group(grp))

    chain_long_path = os.path.join(OUTPUT_DIR, "chain_long.csv")
    chain_short_path = os.path.join(OUTPUT_DIR, "chain_short.csv")
    pd.DataFrame(long_merged).to_csv(chain_long_path, index=False)
    pd.DataFrame(short_merged).to_csv(chain_short_path, index=False)

    send_chain_signal_email(long_merged, short_merged, attachments=[summary_path, long_path, short_path, iter_path])

    combined_options_df = pd.concat([long_df, short_df], ignore_index=True) if (not long_df.empty or not short_df.empty) else pd.DataFrame()
    ce_buy_rows, pe_buy_rows = build_cepebuy_rows(combined_options_df, iteration_df)
    send_cepebuy_email(ce_buy_rows, pe_buy_rows, attachments=[chain_long_path, chain_short_path])


if __name__ == "__main__":
    main()

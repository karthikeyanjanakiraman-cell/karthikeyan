#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py

Optimized Intraday F&O scanner via Fyers API with email alerts.
- FIXED: Added explicit TurningDiff calculation to resolve the KeyError crash.
- FIXED: Target breakout/breakdown prices are anchored to Yesterday's Close to 
  form perfectly flat, unmoving horizontal execution thresholds.
- FIXED: Breakout tracking evaluates price action ANYTIME during the whole day.
- Fixed 15-minute Gamma Hulk anchor level decay using forward-fill.
- Realigned multi-timeframe engine for native 15-minute bars.
- Prioritized Gamma Hulk confirmations at the top of candidate sorting.
- Eliminated ARIMA fitting loop to prevent live market execution lag.
"""

import os
import sys
import logging
import warnings
from datetime import datetime, timedelta, time
import time as time_module
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fyers_apiv3 import fyersModel

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


class UTF8Formatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        record.msg = msg.encode("ascii", "ignore").decode("ascii")
        return super().format(record)


logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = UTF8Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)

warnings.filterwarnings("ignore")

DAILY_LOOKBACK_DAYS = 60
INTRADAY_LOOKBACK_DAYS = 135  
IVP_LOOKBACK_DAYS = 60
INDEX_SOFT_BOOST_WEIGHT = 0.0
HISTORY_API_MAX_SPAN_DAYS = 99
FYERS_RATE_LIMIT_SLEEP = 0.31
FYERS_RETRY_SLEEP = 2.0
FYERS_MAX_RETRIES = 2

fyers: Optional[fyersModel.FyersModel] = None

smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
smtp_port = int(os.environ.get("SMTP_PORT", "587"))
sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
sender_password = os.environ.get("SENDER_PASSWORD", "password")
recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")

EMAIL_DISPLAY_COLS = [
    "Symbol",
    "LTP",
    "% Change",
    "ROC_14",          
    "ROC_6M_Peak",     
    "ROC_6M_Peak_Price",    
    "ROC_6M_Bottom",   
    "ROC_6M_Bottom_Price",  
    "MTF_15m",
    "MTF_30m",
    "MTF_60m",
    "MTF_SCORE",
    "MTF_ALIGN",
    "Last Iteration Time",
]


def compute_gamma_hulk_roc(intra_df: pd.DataFrame, prev_close: Optional[float] = None) -> pd.DataFrame:
    """
    15-minute Gamma Hulk engine:
    - Extracts a TRUE static horizontal 6-month boundary line from past data.
    - Anchors price targets to Yesterday's Close to maintain flat daily lines.
    - Evaluates breakouts/breakdowns based on the price crossing flat thresholds ANYTIME.
    """
    df = intra_df.copy().sort_values("timestamp").reset_index(drop=True)

    required_cols = {"timestamp", "high", "low", "close"}
    if df.empty or not required_cols.issubset(df.columns):
        df["ROC_14"] = np.nan
        df["ROC_6M_Peak"] = np.nan
        df["ROC_6M_Peak_Price"] = np.nan
        df["ROC_6M_Bottom"] = np.nan
        df["ROC_6M_Bottom_Price"] = np.nan
        df["Gamma_Long_Breakout"] = False
        df["Gamma_Short_Breakdown"] = False
        df["Gamma_Long_Confirmed"] = False
        df["Gamma_Short_Confirmed"] = False
        df["Gamma_Breakout_High"] = np.nan
        df["Gamma_Breakout_Low"] = np.nan
        return df

    for col in ["high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    df["ROC_14"] = ((df["close"] - df["close"].shift(1)) / df["close"].shift(1)) * 100.0

    df["_date_only"] = pd.to_datetime(df["timestamp"]).dt.date
    unique_dates = sorted(df["_date_only"].dropna().unique())
    
    if len(unique_dates) >= 2:
        current_date = unique_dates[-1]
        historical_data = df[df["_date_only"] < current_date]
        if not historical_data.empty and historical_data["ROC_14"].notna().sum() > 2:
            static_peak = float(historical_data["ROC_14"].max())
            static_bottom = float(historical_data["ROC_14"].min())
        else:
            static_peak = float(df["ROC_14"].rolling(500, min_periods=2).max().shift(1).iloc[-1])
            static_bottom = float(df["ROC_14"].rolling(500, min_periods=2).min().shift(1).iloc[-1])
    else:
        static_peak = float(df["ROC_14"].max())
        static_bottom = float(df["ROC_14"].min())

    df["ROC_6M_Peak"] = static_peak
    df["ROC_6M_Bottom"] = static_bottom

    if prev_close and float(prev_close) > 0:
        df["ROC_6M_Peak_Price"] = float(prev_close) * (1 + df["ROC_6M_Peak"] / 100.0)
        df["ROC_6M_Bottom_Price"] = float(prev_close) * (1 + df["ROC_6M_Bottom"] / 100.0)
    else:
        df["ROC_6M_Peak_Price"] = df["close"].shift(1) * (1 + df["ROC_6M_Peak"] / 100.0)
        df["ROC_6M_Bottom_Price"] = df["close"].shift(1) * (1 + df["ROC_6M_Bottom"] / 100.0)

    prev_close_price = df["close"].shift(1)
    is_long_breach = (df["close"] > df["ROC_6M_Peak_Price"]) & (prev_close_price.fillna(df["close"]) <= df["ROC_6M_Peak_Price"])
    is_short_breach = (df["close"] < df["ROC_6M_Bottom_Price"]) & (prev_close_price.fillna(df["close"]) >= df["ROC_6M_Bottom_Price"])

    df["Gamma_Breakout_High"] = np.where(is_long_breach, df["high"], np.nan)
    df["Gamma_Breakout_Low"] = np.where(is_short_breach, df["low"], np.nan)

    df["Gamma_Breakout_High"] = df["Gamma_Breakout_High"].ffill()
    df["Gamma_Breakout_Low"] = df["Gamma_Breakout_Low"].ffill()

    long_trigger_high = df["Gamma_Breakout_High"].shift(1)
    short_trigger_low = df["Gamma_Breakout_Low"].shift(1)
    df["Gamma_Long_Confirmed"] = long_trigger_high.notna() & (df["close"] > long_trigger_high)
    df["Gamma_Short_Confirmed"] = short_trigger_low.notna() & (df["close"] < short_trigger_low)

    df["Gamma_Long_Breakout"] = df["close"] > df["ROC_6M_Peak_Price"]
    df["Gamma_Short_Breakdown"] = df["close"] < df["ROC_6M_Bottom_Price"]

    df.drop(columns=["_date_only"], inplace=True)
    return df


def build_signals_from_raw_directional(detail_df) -> dict:
    nan = float("nan")
    out = {
        k: nan for k in (
            "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal",
            "Bull_Signal", "Bear_Signal", "Overall_Signal"
        )
    }
    if detail_df is None or detail_df.empty: return out

    df = detail_df.copy()
    if "Iteration No" in df.columns: df = df.sort_values("Iteration No")

    vals = pd.to_numeric(df["Directional"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0: return out

    last = vals.size - 1
    def raw_at(offset: int) -> float:
        i = last - offset
        if i < 0: i = 0
        return float(vals[i])

    out["5m_Signal"] = round(raw_at(0), 4)
    out["15m_Signal"] = round(raw_at(3) if last >= 3 else raw_at(0), 4)
    out["30m_Signal"] = round(raw_at(6) if last >= 6 else raw_at(0), 4)
    out["60m_Signal"] = round(raw_at(12) if last >= 12 else raw_at(0), 4)
    out["Bull_Signal"] = round(float(vals[vals > 0].max()) if (vals > 0).any() else 0.0, 4)
    out["Bear_Signal"] = round(abs(float(vals[vals < 0].min())) if (vals < 0).any() else 0.0, 4)
    out["Overall_Signal"] = round(raw_at(0), 4)
    return out


def classify_mtf_from_window(win, eps: float = 1e-9) -> float:
    s = pd.Series(win).dropna().astype(float)
    if len(s) < 3: return float("nan")
    diff1 = s.diff().dropna()
    if len(diff1) < 2: return float("nan")
    x = np.arange(len(s), dtype=float)
    slope = float(np.polyfit(x, s.values, 1)[0])
    net_move = float(s.iloc[-1] - s.iloc[0])
    turning = float(np.mean(np.abs(np.diff(s.values, n=2)))) if len(s) >= 3 else 0.0
    stability = float(np.std(s.values))
    score = (slope + net_move) + (0.25 * float(np.clip(diff1, 0, None).sum())) - (0.50 * turning) - (0.10 * stability)
    return 1.0 if score > eps else -1.0 if score < -eps else 0.0


def build_mtf_alignment(detail_df: pd.DataFrame) -> Dict[str, object]:
    out = {
        "MTF_5m": float("nan"), "MTF_15m": float("nan"), "MTF_30m": float("nan"),
        "MTF_60m": float("nan"), "MTF_SCORE": float("nan"), "MTF_ALIGN": "NA",
    }
    if detail_df is None or detail_df.empty or "Iteration Change" not in detail_df.columns: return out

    df = detail_df.copy().sort_values("Iteration No").reset_index(drop=True)
    series = pd.to_numeric(df["Iteration Change"], errors="coerce").dropna().astype(float)
    if len(series) < 3: return out

    def classify_from_tail(s: pd.Series, bars: int) -> float:
        if len(s) < bars: return float("nan")
        return classify_mtf_from_window(s.tail(bars))

    mtf_5 = float("nan")                                                       
    mtf_15 = classify_from_tail(series, 3)                                     
    mtf_30 = classify_from_tail(series.iloc[1::2].reset_index(drop=True), 3)   
    mtf_60 = classify_from_tail(series.iloc[3::4].reset_index(drop=True), 3)   

    available = [v for v in [mtf_15, mtf_30, mtf_60] if pd.notna(v)]
    if not available: return out

    score = float(np.nansum([mtf_15, mtf_30, mtf_60]))
    align = "LONG" if all(v == 1.0 for v in available) else "SHORT" if all(v == -1.0 for v in available) else "MIXED"

    out.update({
        "MTF_5m": mtf_5, "MTF_15m": mtf_15, "MTF_30m": mtf_30,
        "MTF_60m": mtf_60, "MTF_SCORE": score, "MTF_ALIGN": align,
    })
    return out


def add_dual_engine_matrix(detail_df: pd.DataFrame, eps: float = 1e-4) -> pd.DataFrame:
    if detail_df is None or detail_df.empty: return pd.DataFrame()
    out = detail_df.copy()
    out["Turning"] = pd.to_numeric(out["Turning"], errors="coerce").fillna(0.0)
    out["CumsumPlus"] = pd.to_numeric(out["CumsumPlus"], errors="coerce").fillna(0.0)
    out["Iteration No"] = pd.to_numeric(out["Iteration No"], errors="coerce")
    out = out.dropna(subset=["Iteration No"]).sort_values(["Symbol", "Iteration No"]).reset_index(drop=True)

    grouped = out.groupby("Symbol", group_keys=False)
    out["Current_Step"] = grouped["CumsumPlus"].diff().fillna(0.0)
    out["Prior_Step"] = grouped["Current_Step"].shift(1).fillna(0.0)
    out["CumsumDiff"] = out["Current_Step"]
    
    out["Prior_Turning"] = grouped["Turning"].shift(1).fillna(0.0)
    # FIX: Calculated TurningDiff to safely eliminate the index KeyError
    out["TurningDiff"] = out["Turning"] - out["Prior_Turning"]
    
    out["Friction_Expanding"] = (out["Turning"] > out["Prior_Turning"]) & (out["Turning"] > eps)

    cond_pristine = (out["Current_Step"] > eps) & (out["Prior_Step"] <= eps) & (~out["Friction_Expanding"])
    cond_exhaustion = (out["Current_Step"] <= eps) & out["Friction_Expanding"]
    cond_trap = (out["Current_Step"] > eps) & out["Friction_Expanding"]
    cond_pause = (out["Current_Step"].abs() <= eps) & (~out["Friction_Expanding"])
    cond_active = (out["Current_Step"] > eps) & (out["Prior_Step"] > eps) & (~out["Friction_Expanding"])

    out["Dual Engine State"] = np.select(
        [cond_pristine, cond_exhaustion, cond_trap, cond_pause, cond_active],
        ["PRISTINE_BREAKOUT", "TRUE_EXHAUSTION", "CHURNING_FAKEOUT", "HEALTHY_PAUSE", "ACTIVE_CONTINUATION"], default="TRANSITION",
    )
    out["Trade Action"] = np.select(
        [out["Dual Engine State"] == "PRISTINE_BREAKOUT", out["Dual Engine State"] == "TRUE_EXHAUSTION", out["Dual Engine State"] == "CHURNING_FAKEOUT", out["Dual Engine State"] == "HEALTHY_PAUSE", out["Dual Engine State"] == "ACTIVE_CONTINUATION"],
        ["ENTRY", "EXIT", "BLOCK_ENTRY", "HOLD", "HOLD"], default="WAIT",
    )
    out["Turning Regime"] = np.where(out["Friction_Expanding"], "EXPANDING_FRICTION", "LOW_FRICTION")
    out["Diff Status"] = out["Dual Engine State"]
    return out


def merge_dual_engine_latest(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or summary_df.empty or detail_df is None or detail_df.empty: return summary_df
    latest = detail_df.sort_values(["Symbol", "Iteration No"]).groupby("Symbol", as_index=False).tail(1)[["Symbol", "CumsumDiff", "TurningDiff", "Turning Regime", "Dual Engine State", "Trade Action"]]
    return summary_df.merge(latest, on="Symbol", how="left")


def init_fyers():
    global fyers
    try:
        client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
        if not client_id or not access_token: return
        fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
        logger.info("INIT FyersModel initialized successfully.")
    except Exception as e:
        logger.warning(f"INIT Failed: {e}")


def load_fno_symbols_from_sectors(root_dir: str = "sectors") -> List[str]:
    symbols = set()
    configured_dir = os.environ.get("SECTORS_DIR", root_dir)
    if not os.path.isdir(configured_dir): return []

    for dirpath, _, filenames in os.walk(configured_dir):
        for fname in filenames:
            if not fname.lower().endswith(".csv"): continue
            try:
                df = pd.read_csv(os.path.join(dirpath, fname))
                col = next((c for c in df.columns if c.lower() in ["symbol", "symbols", "ticker"]), None)
                if col is None: continue
                for s in df[col].dropna().astype(str):
                    s = s.strip()
                    if s: symbols.add(s)
            except Exception: pass
    return sorted(symbols)


def format_fyers_symbol(symbol: str) -> str:
    return symbol if symbol.startswith("NSE:") and symbol.endswith("-EQ") else f"NSE:{symbol}-EQ"


def get_fyers_history(symbol: str, resolution: str, days_back: int) -> Optional[pd.DataFrame]:
    if not fyers: return None
    try:
        now = datetime.now()
        start_date, end_date = (now - timedelta(days=days_back)).date(), now.date()
        all_candles, current_start = [], start_date

        while current_start <= end_date:
            current_end = min(current_start + timedelta(days=HISTORY_API_MAX_SPAN_DAYS), end_date)
            data = {"symbol": symbol, "resolution": resolution, "date_format": "1", "range_from": current_start.strftime("%Y-%m-%d"), "range_to": current_end.strftime("%Y-%m-%d"), "cont_flag": "1"}

            for attempt in range(1, FYERS_MAX_RETRIES + 1):
                res = fyers.history(data=data)
                if res and res.get("s") == "ok":
                    candles = res.get("candles", [])
                    if candles: all_candles.extend(candles)
                    break
                if isinstance(res, dict) and res.get("code") == 429 and attempt < FYERS_MAX_RETRIES:
                    time_module.sleep(FYERS_RETRY_SLEEP * attempt)
                else: break

            time_module.sleep(FYERS_RATE_LIMIT_SLEEP)
            current_start = current_end + timedelta(days=1)

        if all_candles:
            df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            df.sort_values("timestamp", inplace=True)
            df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            return df.reset_index(drop=True)
        return None
    except Exception as e:
        logger.error(f"FYERS Error fetching {resolution} data for {symbol}: {e}")
        return None


def compute_iv_proxies(daily_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if daily_df is None or daily_df.empty or len(daily_df) < 30: return {"IVP": np.nan, "Volatility State": "Neutral Vol"}
    df = daily_df.copy().sort_values(daily_df.columns[0]).reset_index(drop=True)
    close, high, low = pd.to_numeric(df["close"], errors="coerce").astype(float), pd.to_numeric(df["high"], errors="coerce").astype(float), pd.to_numeric(df["low"], errors="coerce").astype(float)

    iv_proxy = ((high - low) / close.replace(0, np.nan) * 100.0).dropna()
    if iv_proxy.empty: return {"IVP": np.nan, "Volatility State": "Neutral Vol"}

    lookback = iv_proxy.tail(min(IVP_LOOKBACK_DAYS, len(iv_proxy)))
    ivp = round((lookback.lt(float(lookback.iloc[-1])).sum() / len(lookback)) * 100, 2)
    return {"IVP": ivp, "Volatility State": "Buyer Zone" if ivp < 30 else "Avoid Buy Premium" if ivp > 50 else "Neutral Vol"}


def compute_cumulative_directional_metrics(curr_df: pd.DataFrame) -> pd.DataFrame:
    df = curr_df.copy().sort_values("time").reset_index(drop=True)
    if df.empty: return df

    h, l, c, o = df["high"].astype(float).to_numpy(), df["low"].astype(float).to_numpy(), df["close"].astype(float).to_numpy(), df["open"].astype(float).to_numpy()
    n = len(df)
    tr, plus_dm, minus_dm = np.zeros(n), np.zeros(n), np.zeros(n)

    tr[0] = max(h[0] - l[0], 0)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        plus_dm[i] = up if up > dn and up > 0 else 0.0
        minus_dm[i] = dn if dn > up and dn > 0 else 0.0

    out = []
    qualified = 0
    for i in range(n):
        if i == 0:
            out.append([np.nan, np.nan, np.nan, np.nan, "0/0", 0.0])
            continue
        prior_kers = [abs(c[j] - o[0]) / (abs(c[0] - o[0]) + np.sum(np.abs(np.diff(c[: j + 1])))) for j in range(1, i + 1) if (abs(c[0] - o[0]) + np.sum(np.abs(np.diff(c[: j + 1])))) > 0]
        tr_sum = tr[1 : i + 1].sum()
        pdi = 100 * plus_dm[1 : i + 1].sum() / tr_sum if tr_sum > 0 else 0.0
        mdi = 100 * minus_dm[1 : i + 1].sum() / tr_sum if tr_sum > 0 else 0.0

        dxs = []
        for k in range(1, i + 1):
            ktr = tr[1 : k + 1].sum()
            kpdi = 100 * plus_dm[1 : k + 1].sum() / ktr if ktr > 0 else 0.0
            kmdi = 100 * minus_dm[1 : k + 1].sum() / ktr if ktr > 0 else 0.0
            if (kpdi + kmdi) > 0: dxs.append(100 * abs(kpdi - kmdi) / (kpdi + kmdi))

        if c[i] > c[i - 1]: qualified += 1
        out.append([float(np.mean(prior_kers)) if prior_kers else np.nan, pdi, mdi, float(np.mean(dxs)) if dxs else np.nan, f"{qualified}/{i + 1}", qualified / (i + 1)])

    return pd.concat([df, pd.DataFrame(out, columns=["Cumulative KER", "Cumulative DI", "Cumulative -DI", "Cumulative ADX", "Survival Score", "SurvivalNum"])], axis=1)


def compute_cumulative_flow_metrics(curr_df: pd.DataFrame) -> pd.DataFrame:
    df = curr_df.copy().sort_values("time").reset_index(drop=True)
    if df.empty: return df

    close, high, low, volume = pd.to_numeric(df["close"], errors="coerce").astype(float), pd.to_numeric(df["high"], errors="coerce").astype(float), pd.to_numeric(df["low"], errors="coerce").astype(float), pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype(float)
    delta = close.diff().fillna(0.0)
    avg_gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    avg_loss = (-delta).clip(lower=0.0).rolling(14, min_periods=14).mean()

    rsi = 100 - (100 / (1 + (avg_gain / avg_loss.replace(0, float("nan")))))
    rsi = rsi.mask((avg_loss == 0) & avg_loss.notna(), 100.0).fillna(0.0)

    obv = (close.diff().fillna(0.0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * volume).cumsum()
    typical_price = (high + low + close) / 3.0
    cum_vol = volume.cumsum().replace(0, float("nan"))
    vwap = ((typical_price * volume).cumsum() / cum_vol).fillna(0.0)

    vwap_std = np.sqrt((volume * (typical_price - vwap) ** 2).cumsum() / cum_vol).fillna(0.0)
    vwap_z_score = np.where(vwap_std > 0, (close - vwap) / vwap_std, 0.0)

    return pd.concat([df, pd.DataFrame({"Cumulative RSI": rsi, "Cumulative OBV": obv, "Cumulative VWAP": vwap, "VWAP Z-Score": vwap_z_score})], axis=1)


def compute_price_lead_metrics(curr_df: pd.DataFrame) -> pd.DataFrame:
    df = curr_df.copy().sort_values("time").reset_index(drop=True)
    if df.empty: return pd.DataFrame()

    df["range"] = (df["high"] - df["low"]).clip(lower=0.0)
    df["range_expansion"] = df["range"] / df["range"].rolling(5, min_periods=3).mean().replace(0, np.nan)
    df["volume_expansion"] = df["volume"] / df["volume"].rolling(5, min_periods=3).mean().replace(0, np.nan)

    df["mid"] = (df["high"] + df["low"]) / 2.0
    df["delta"] = np.where(df["close"] > df["mid"], df["volume"], np.where(df["close"] < df["mid"], -df["volume"], 0.0))
    cvd_change = df["delta"].cumsum().diff().abs().fillna(0.0)
    df["delta_expansion"] = cvd_change / cvd_change.rolling(5, min_periods=3).mean().replace(0, np.nan)

    df["price_leading_flag"] = (df["range_expansion"] >= 1.5) & (df["volume_expansion"] <= 1.0) & (df["delta_expansion"] <= 1.0)
    
    streak, run = [], 0
    for flag in df["price_leading_flag"].astype(bool):
        run = run + 1 if flag else 0
        streak.append(run)
    df["price_lead_streak"] = streak
    df["Price_Lead_Status"] = np.select(
        [df["price_leading_flag"] & (df["price_lead_streak"] >= 3), df["price_leading_flag"] & (df["price_lead_streak"] >= 2), df["price_leading_flag"]],
        ["STRONG_PRICE_LEAD_FADE", "PRICE_LEADING_FADE_RISK", "EARLY_PRICE_LEAD"], default="NORMAL"
    )
    return df[["time", "range_expansion", "volume_expansion", "delta_expansion", "price_leading_flag", "price_lead_streak", "Price_Lead_Status"]]


def price_stats_from_series(prices: pd.Series) -> dict:
    p = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if len(p) < 3: return {"Directional": np.nan, "Turning": np.nan, "Stability": np.nan, "Balanced": np.nan, "CumsumPlus": np.nan}
    x = np.arange(len(p), dtype=float)
    slope = float(np.polyfit(x, p.values, 1)[0])
    directional = slope + float(p.iloc[-1] - p.iloc[0])
    turning = float(np.mean(np.abs(np.diff(p.values, n=2))))
    return {"Directional": directional, "Turning": turning, "Stability": float(np.std(p.values)), "Balanced": directional - turning + float(np.std(p.values)), "CumsumPlus": float(np.sum(np.clip(np.diff(p.values), 0, None)))}


def kalman_signal_from_series(prices: pd.Series, q: float = 1e-3) -> float:
    p = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if len(p) < 5: return float("nan")
    arr, x, P = p.to_numpy(), p.iloc[0], 1.0
    r = float(p.diff().dropna().var()) + 1e-6
    for y in arr:
        P += q
        K = P / (P + r)
        x += K * (y - x)
        P *= (1.0 - K)
    return round((arr[-1] - x) / (float(p.std()) or 1e-6), 4)


def compute_iteration_volume_profile(intra_df: Optional[pd.DataFrame], prev_close: Optional[float] = None) -> Tuple[Dict, pd.DataFrame]:
    if intra_df is None or intra_df.empty: return {}, pd.DataFrame()

    df = intra_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["time_only"] = pd.to_datetime(df["timestamp"]).dt.time
    
    df = compute_gamma_hulk_roc(df, prev_close=prev_close)

    dates = sorted(df["date"].unique())
    if len(dates) < 2: return {}, pd.DataFrame()

    current_date = dates[-1]
    curr_df = df[df["date"] == current_date].copy().sort_values("time_only")
    hist_df_10 = df[df["date"].isin(dates[-11:-1] if len(dates) >= 11 else dates[:-1])].copy()
    hist_df_20 = df[df["date"].isin(dates[-21:-1] if len(dates) >= 21 else dates[:-1])].copy()

    if curr_df.empty: return {}, pd.DataFrame()

    curr_df["cum_vol"] = curr_df["volume"].cumsum()
    curr_df["Iteration Change"] = ((pd.to_numeric(curr_df["close"], errors="coerce") - float(prev_close)) / float(prev_close) * 100.0) if prev_close else 0.0

    work_df = curr_df.copy()
    work_df["time"] = pd.to_datetime(work_df["date"].astype(str) + " " + work_df["time_only"].astype(str))

    metric_df = compute_cumulative_directional_metrics(work_df[["time", "open", "high", "low", "close", "volume"]].copy())
    flow_df = compute_cumulative_flow_metrics(work_df[["time", "high", "low", "close", "volume"]].copy())
    price_lead_df = compute_price_lead_metrics(work_df[["time", "open", "high", "low", "close", "volume"]].copy())

    rows = []
    total_iters = 0
    last_cum_vol = last_rvol10 = last_rvol20 = 0.0
    last_iter_mins = last_iter_time = None

    for i in range(len(curr_df)):
        total_iters += 1
        row = curr_df.iloc[i]
        t = row["time_only"]
        cum_vol = float(row["cum_vol"])

        h10 = hist_df_10[hist_df_10["time_only"] <= t]
        rvol10 = cum_vol / h10.groupby("date")["volume"].sum().mean() if not h10.empty and h10.groupby("date")["volume"].sum().mean() > 0 else 0.0

        h20 = hist_df_20[hist_df_20["time_only"] <= t]
        rvol20 = cum_vol / h20.groupby("date")["volume"].sum().mean() if not h20.empty and h20.groupby("date")["volume"].sum().mean() > 0 else 0.0

        iter_mins = int((datetime.combine(current_date, t) - datetime.combine(current_date, time(9, 15))).total_seconds() / 60)
        ps = price_stats_from_series(curr_df["Iteration Change"].iloc[: i + 1])

        rows.append({
            "Iteration No": total_iters, "Iteration Minutes": iter_mins, "Iteration Time": t.strftime("%H:%M"),
            "LTP": float(row["close"]), "Iteration Change": float(curr_df["Iteration Change"].iloc[i]), "Current Volume": cum_vol,
            "Directional": ps["Directional"], "Turning": ps["Turning"], "Stability": ps["Stability"], "Balanced": ps["Balanced"], "CumsumPlus": ps.get("CumsumPlus"),
            "10 Day Relative Volume": rvol10, "20 Day Relative Volume": rvol20,
            "ROC_14": float(row.get("ROC_14", np.nan)),               
            "ROC_6M_Peak": float(row.get("ROC_6M_Peak", np.nan)),     
            "ROC_6M_Peak_Price": float(row.get("ROC_6M_Peak_Price", np.nan)), 
            "ROC_6M_Bottom": float(row.get("ROC_6M_Bottom", np.nan)), 
            "ROC_6M_Bottom_Price": float(row.get("ROC_6M_Bottom_Price", np.nan)), 
            "Gamma_Long_Breakout": bool(row.get("Gamma_Long_Breakout", False)), "Gamma_Short_Breakdown": bool(row.get("Gamma_Short_Breakdown", False)),
            "Gamma_Long_Confirmed": bool(row.get("Gamma_Long_Confirmed", False)), "Gamma_Short_Confirmed": bool(row.get("Gamma_Short_Confirmed", False)),
            "Gamma_Breakout_High": float(row.get("Gamma_Breakout_High", np.nan)), "Gamma_Breakout_Low": float(row.get("Gamma_Breakout_Low", np.nan)),
            "Cumulative RSI": float(flow_df["Cumulative RSI"].iloc[i]) if not flow_df.empty else float("nan"),
            "Cumulative OBV": float(flow_df["Cumulative OBV"].iloc[i]) if not flow_df.empty else float("nan"),
            "Cumulative VWAP": float(flow_df["Cumulative VWAP"].iloc[i]) if not flow_df.empty else float("nan"),
            "VWAP Z-Score": float(flow_df["VWAP Z-Score"].iloc[i]) if not flow_df.empty else float("nan"),
            "Price_Lead_Status": str(price_lead_df["Price_Lead_Status"].iloc[i]) if not price_lead_df.empty else "NORMAL",
        })
        last_cum_vol, last_rvol10, last_rvol20 = cum_vol, rvol10, rvol20
        last_iter_mins, last_iter_time = iter_mins, t.strftime("%H:%M")

    detail_df = pd.DataFrame(rows)
    ltp = float(curr_df["close"].iloc[-1]) if not curr_df.empty else np.nan
    hod = float(curr_df["high"].max()) if not curr_df.empty else float("nan")

    obv_30m_delta = float(flow_df["Cumulative OBV"].iloc[-1]) - float(flow_df["Cumulative OBV"].iloc[-7]) if len(flow_df) >= 7 else 0.0
    rsi_30m_delta = float(flow_df["Cumulative RSI"].iloc[-1]) - float(flow_df["Cumulative RSI"].iloc[-7]) if len(flow_df) >= 7 else 0.0

    final_ps = price_stats_from_series(curr_df["Iteration Change"])
    summary = {
        "LTP": ltp, "Directional": final_ps["Directional"], "Turning": final_ps["Turning"], "Stability": final_ps["Stability"],
        "Balanced": final_ps["Balanced"], "CumsumPlus": final_ps.get("CumsumPlus"), "ARIMA Signal": np.nan,
        "Kalman Signal": kalman_signal_from_series(curr_df["Iteration Change"]),
        "Current Volume": last_cum_vol, "10 Day Relative Volume": last_rvol10, "20 Day Relative Volume": last_rvol20,
        "ROC_14": float(curr_df["ROC_14"].iloc[-1]) if "ROC_14" in curr_df.columns else np.nan,               
        "ROC_6M_Peak": float(curr_df["ROC_6M_Peak"].iloc[-1]) if "ROC_6M_Peak" in curr_df.columns else np.nan,     
        "ROC_6M_Peak_Price": float(curr_df["ROC_6M_Peak_Price"].iloc[-1]) if "ROC_6M_Peak_Price" in curr_df.columns else np.nan, 
        "ROC_6M_Bottom": float(curr_df["ROC_6M_Bottom"].iloc[-1]) if "ROC_6M_Bottom" in curr_df.columns else np.nan, 
        "ROC_6M_Bottom_Price": float(curr_df["ROC_6M_Bottom_Price"].iloc[-1]) if "ROC_6M_Bottom_Price" in curr_df.columns else np.nan, 
        "Gamma_Long_Breakout": bool(curr_df["Gamma_Long_Breakout"].iloc[-1]) if "Gamma_Long_Breakout" in curr_df.columns else False,
        "Gamma_Short_Breakdown": bool(curr_df["Gamma_Short_Breakdown"].iloc[-1]) if "Gamma_Short_Breakdown" in curr_df.columns else False,
        "Gamma_Long_Confirmed": bool(curr_df["Gamma_Long_Confirmed"].iloc[-1]) if "Gamma_Long_Confirmed" in curr_df.columns else False,
        "Gamma_Short_Confirmed": bool(curr_df["Gamma_Short_Confirmed"].iloc[-1]) if "Gamma_Short_Confirmed" in curr_df.columns else False,
        "Gamma_Breakout_High": float(curr_df["Gamma_Breakout_High"].iloc[-1]) if "Gamma_Breakout_High" in curr_df.columns else np.nan,
        "Gamma_Breakout_Low": float(curr_df["Gamma_Breakout_Low"].iloc[-1]) if "Gamma_Breakout_Low" in curr_df.columns else np.nan,
        "Cumulative RSI": float(flow_df["Cumulative RSI"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Cumulative OBV": float(flow_df["Cumulative OBV"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Cumulative VWAP": float(flow_df["Cumulative VWAP"].iloc[-1]) if not flow_df.empty else float("nan"),
        "VWAP Z-Score": float(flow_df["VWAP Z-Score"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Total Iterations": total_iters, "Last Iteration Minutes": last_iter_mins, "Last Iteration Time": last_iter_time,
        "Cumulative KER": float(metric_df["Cumulative KER"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative DI": float(metric_df["Cumulative DI"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative -DI": float(metric_df["Cumulative -DI"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative ADX": float(metric_df["Cumulative ADX"].iloc[-1]) if not metric_df.empty else np.nan,
        "Survival Score": str(metric_df["Survival Score"].iloc[-1]) if not metric_df.empty else "0/0",
        "SurvivalNum": float(metric_df["SurvivalNum"].iloc[-1]) if not metric_df.empty else 0.0,
        "HOD": hod, "StrikeDistance": ((hod - ltp) / hod if hod else 1.0),
        "Last5mVolume": float(curr_df["volume"].iloc[-1]) if not curr_df.empty else 0.0,
        "Volume1hAvg5m": float(curr_df["volume"].tail(12).mean()) if not curr_df.empty else 0.0,
        "OBV30mDelta": obv_30m_delta, "RSI30mDelta": rsi_30m_delta,
        "Price_Lead_Status": str(price_lead_df["Price_Lead_Status"].iloc[-1]) if not price_lead_df.empty else "NORMAL",
    }

    summary.update(build_signals_from_raw_directional(detail_df))
    summary.update(build_mtf_alignment(detail_df))
    return summary, detail_df


def scan_fno_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = load_fno_symbols_from_sectors()
    if not symbols: return pd.DataFrame(), pd.DataFrame()

    rows, iteration_rows = [], []
    total = len(symbols)

    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"CORE [{idx}/{total}] Processing {sym}")
        fyers_sym = format_fyers_symbol(sym)

        daily_df = get_fyers_history(fyers_sym, resolution="D", days_back=max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
        intra_df = get_fyers_history(fyers_sym, resolution="15", days_back=INTRADAY_LOOKBACK_DAYS)

        prev_close = float(daily_df["close"].iloc[-2]) if (daily_df is not None and len(daily_df) >= 2) else None
        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df, prev_close)
        iv_info = compute_iv_proxies(daily_df)

        ltp = iter_summary.get("LTP")
        pct_change = ((ltp - prev_close) / prev_close * 100) if (ltp is not None and prev_close and prev_close != 0) else 0.0

        if not iter_detail.empty:
            iter_detail.insert(0, "Symbol", sym)
            iter_detail.insert(1, "% Change", pct_change)
            iteration_rows.append(iter_detail)

        rows.append({
            "Symbol": sym, "LTP": ltp, "% Change": pct_change,
            "Directional": iter_summary.get("Directional"), "Turning": iter_summary.get("Turning"),
            "Stability": iter_summary.get("Stability"), "Balanced": iter_summary.get("Balanced"), "CumsumPlus": iter_summary.get("CumsumPlus"),
            "ROC_14": iter_summary.get("ROC_14"), "ROC_6M_Peak": iter_summary.get("ROC_6M_Peak"), "ROC_6M_Peak_Price": iter_summary.get("ROC_6M_Peak_Price"), 
            "ROC_6M_Bottom": iter_summary.get("ROC_6M_Bottom"), "ROC_6M_Bottom_Price": iter_summary.get("ROC_6M_Bottom_Price"), 
            "Gamma_Long_Breakout": iter_summary.get("Gamma_Long_Breakout", False), "Gamma_Short_Breakdown": iter_summary.get("Gamma_Short_Breakdown", False),
            "Gamma_Long_Confirmed": iter_summary.get("Gamma_Long_Confirmed", False), "Gamma_Short_Confirmed": iter_summary.get("Gamma_Short_Confirmed", False),
            "Gamma_Breakout_High": iter_summary.get("Gamma_Breakout_High"), "Gamma_Breakout_Low": iter_summary.get("Gamma_Breakout_Low"),
            "ARIMA Signal": iter_summary.get("ARIMA Signal"), "Kalman Signal": iter_summary.get("Kalman Signal"),
            "5m_Signal": iter_summary.get("5m_Signal"), "15m_Signal": iter_summary.get("15m_Signal"), "30m_Signal": iter_summary.get("30m_Signal"), "60m_Signal": iter_summary.get("60m_Signal"),
            "MTF_5m": iter_summary.get("MTF_5m"), "MTF_15m": iter_summary.get("MTF_15m"), "MTF_30m": iter_summary.get("MTF_30m"), "MTF_60m": iter_summary.get("MTF_60m"),
            "MTF_SCORE": iter_summary.get("MTF_SCORE"), "MTF_ALIGN": iter_summary.get("MTF_ALIGN"),
            "Bull_Signal": iter_summary.get("Bull_Signal"), "Bear_Signal": iter_summary.get("Bear_Signal"), "Overall_Signal": iter_summary.get("Overall_Signal"),
            "Current Volume": iter_summary.get("Current Volume"), "10 Day Relative Volume": iter_summary.get("10 Day Relative Volume"), "20 Day Relative Volume": iter_summary.get("20 Day Relative Volume"),
            "Cumulative RSI": iter_summary.get("Cumulative RSI"), "Cumulative OBV": iter_summary.get("Cumulative OBV"), "Cumulative VWAP": iter_summary.get("Cumulative VWAP"), "VWAP Z-Score": iter_summary.get("VWAP Z-Score"),
            "Total Iterations": iter_summary.get("Total Iterations"), "Last Iteration Minutes": iter_summary.get("Last Iteration Minutes"), "Last Iteration Time": iter_summary.get("Last Iteration Time"),
            "Cumulative KER": iter_summary.get("Cumulative KER"), "Cumulative DI": iter_summary.get("Cumulative DI"), "Cumulative -DI": iter_summary.get("Cumulative -DI"), "Cumulative ADX": iter_summary.get("Cumulative ADX"),
            "Survival Score": iter_summary.get("Survival Score"), "SurvivalNum": iter_summary.get("SurvivalNum"),
            "HOD": iter_summary.get("HOD"), "StrikeDistance": iter_summary.get("StrikeDistance"),
            "Last5mVolume": iter_summary.get("Last5mVolume"), "Volume1hAvg5m": iter_summary.get("Volume1hAvg5m"),
            "OBV30mDelta": iter_summary.get("OBV30mDelta"), "RSI30mDelta": iter_summary.get("RSI30mDelta"),
            "Price_Lead_Status": iter_summary.get("Price_Lead_Status", "NORMAL"), "IVP": iv_info.get("IVP"), "Volatility State": iv_info.get("Volatility State"),
        })

    return pd.DataFrame(rows), (pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame())


def derive_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()

    def score_bull(row):
        score = 0
        if pd.notna(row.get("% Change")) and row.get("% Change") > 0: score += 2
        if pd.notna(row.get("VWAP Z-Score")) and row.get("VWAP Z-Score") >= 0.30: score += 2
        if pd.notna(row.get("Cumulative DI")) and pd.notna(row.get("Cumulative -DI")) and row.get("Cumulative DI") > row.get("Cumulative -DI"): score += 2
        if pd.notna(row.get("Cumulative ADX")) and row.get("Cumulative ADX") >= 20: score += 1
        if pd.notna(row.get("Cumulative RSI")) and row.get("Cumulative RSI") >= 55: score += 1
        return min(score, 13)

    def score_bear(row):
        score = 0
        if pd.notna(row.get("% Change")) and row.get("% Change") < 0: score += 2
        if pd.notna(row.get("VWAP Z-Score")) and row.get("VWAP Z-Score") <= -0.30: score += 2
        if pd.notna(row.get("Cumulative DI")) and pd.notna(row.get("Cumulative -DI")) and row.get("Cumulative -DI") > row.get("Cumulative DI"): score += 2
        if pd.notna(row.get("Cumulative ADX")) and row.get("Cumulative ADX") >= 20: score += 1
        if pd.notna(row.get("Cumulative RSI")) and row.get("Cumulative RSI") <= 45: score += 1
        return min(score, 13)

    out["Bull Rank"] = out.apply(score_bull, axis=1)
    out["Bear Rank"] = out.apply(score_bear, axis=1)
    out["Rank Delta"] = out["Bull Rank"] - out["Bear Rank"]
    return out


def add_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    if "Trade Action" in out.columns:
        out["Entry Allowed"] = out["Trade Action"].eq("ENTRY")
        out["Hold Allowed"] = out["Trade Action"].eq("HOLD")
        out["Exit Now"] = out["Trade Action"].eq("EXIT")
    return out


def signal_color(label) -> str:
    mapping = {
        "Buyer Zone": "#33691e", "Neutral Vol": "#4b5563", "Avoid Buy Premium": "#7a5c00",
        "LOW_FRICTION": "#166534", "EXPANDING_FRICTION": "#991b1b", "PRISTINE_BREAKOUT": "#15803d",
        "HEALTHY_PAUSE": "#0f766e", "CHURNING_FAKEOUT": "#b45309", "TRUE_EXHAUSTION": "#b91c1c",
        "ACTIVE_CONTINUATION": "#16a34a", "TRANSITION": "#374151", "ENTRY": "#15803d",
        "HOLD": "#0f766e", "BLOCK_ENTRY": "#b45309", "EXIT": "#b91c1c", "WAIT": "#4b5563",
        "STRONG_PRICE_LEAD_FADE": "#7f1d1d", "PRICE_LEADING_FADE_RISK": "#b45309", "EARLY_PRICE_LEAD": "#374151",
    }
    return mapping.get(str(label).strip(), "#374151")


def format_value(col: str, val):
    if pd.isna(val): return ""
    if col == "% Change": return f"{float(val):.2f}%"
    if col in ["IVP", "ROC_14", "ROC_6M_Peak", "ROC_6M_Peak_Price", "ROC_6M_Bottom", "ROC_6M_Bottom_Price"]: return f"{float(val):.2f}"
    if isinstance(val, (int, float, np.integer, np.floating)): return f"{float(val):.4f}"
    return str(val)


def build_candidate_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty: return pd.DataFrame(columns=EMAIL_DISPLAY_COLS), pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    base = df.copy()
    for c in ["Directional", "Turning", "Stability", "Balanced", "CumsumPlus", "10 Day Relative Volume", "Last5mVolume", "ROC_14", "ROC_6M_Peak", "ROC_6M_Peak_Price", "ROC_6M_Bottom", "ROC_6M_Bottom_Price", "% Change", "MTF_SCORE"]:
        if c in base.columns: base[c] = pd.to_numeric(base[c], errors="coerce")

    def prep_side_df(dfside: pd.DataFrame, side: str) -> pd.DataFrame:
        if dfside.empty: return dfside
        out = dfside.copy()

        if side == "long":
            out = out[out["Directional"] > 0].copy()
            if out.empty: return out
            out["Gamma_Long_Confirmed"] = out["Gamma_Long_Confirmed"].fillna(False).astype(int)
            out["Gamma_Long_Breakout"] = out["Gamma_Long_Breakout"].fillna(False).astype(int)
            return out.sort_values(['Gamma_Long_Confirmed', 'Gamma_Long_Breakout', 'MTF_SCORE', 'ROC_14'], ascending=[False, False, False, False], na_position='last')
        else:
            out = out[out["Directional"] < 0].copy()
            if out.empty: return out
            out["Gamma_Short_Confirmed"] = out["Gamma_Short_Confirmed"].fillna(False).astype(int)
            out["Gamma_Short_Breakdown"] = out["Gamma_Short_Breakdown"].fillna(False).astype(int)
            return out.sort_values(['Gamma_Short_Confirmed', 'Gamma_Short_Breakdown', 'MTF_SCORE', 'ROC_14'], ascending=[False, False, True, True], na_position='last')

    long_df = prep_side_df(base, "long").drop_duplicates(subset=["Symbol"]).head(15)
    short_df = prep_side_df(base, "short").drop_duplicates(subset=["Symbol"]).head(15)
    cols = [c for c in EMAIL_DISPLAY_COLS if c in base.columns]
    return (long_df[cols] if not long_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)), (short_df[cols] if not short_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS))


def load_iteration_history(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df is None or detail_df.empty: return pd.DataFrame()
    df = detail_df.copy()
    if "Iteration No" not in df.columns: return pd.DataFrame()

    last15_iters = sorted(df["Iteration No"].dropna().astype(int).unique())[-15:]
    df = df[df["Iteration No"].astype(int).isin(last15_iters)].copy()

    long_top = df[df["Directional"] > 0].sort_values(["Iteration No", "Directional"], ascending=[True, False]).groupby("Iteration No").head(1).assign(Side="Long")
    short_top = df[df["Directional"] < 0].sort_values(["Iteration No", "Directional"], ascending=[True, True]).groupby("Iteration No").head(1).assign(Side="Short")
    
    out = pd.concat([long_top, short_top], ignore_index=True, sort=False)
    if out.empty: return out

    out = out.sort_values(["Iteration No", "Side"]).reset_index(drop=True)
    out["Iteration"] = out["Iteration No"].astype(str) + " @ " + out["Iteration Time"].astype(str)
    out["First Occurrence"] = out["Iteration"]
    out["Latest"] = out["Iteration"]
    return out


def build_history_table(history_df: pd.DataFrame, side: str) -> str:
    if history_df is None or history_df.empty: return "No history yet."
    df = history_df[history_df["Side"].astype(str).str.lower() == side.lower()].copy()
    if df.empty: return "No history yet."

    cols = ["First Occurrence", "Latest", "Symbol", "LTP", "% Change", "Directional", "Turning", "Stability", "Balanced", "CumsumPlus", "Iteration Time"]
    df = df.tail(15)[cols].copy()

    header = "".join(f'<th style="padding:8px;border:1px solid #4b5563;background:#111827;color:#f9fafb;">{c}</th>' for c in cols)
    body_rows = []
    for _, r in df.iterrows():
        tds = "".join(f'<td style="padding:6px 8px;border:1px solid #4b5563;color:#e5e7eb;background:{"#14532d" if float(r["Directional"]) > 0 else "#7f1d1d" if float(r["Directional"]) < 0 else "#030712"}">{format_value(c, r[c])}</td>' for c in cols)
        body_rows.append(f"<tr>{tds}</tr>")
    return f'<h3 style="color:{"#22c55e" if side.lower()=="long" else "#ef4444"};margin:12px 0 6px 0;">Top 1 {side.title()} - Last 15 Iterations</h3><table style="border-collapse:collapse;width:100%;background:#030712;"><thead><tr>{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def build_html_table(df: pd.DataFrame, title: str, max_rows: int = 15) -> str:
    if df is None or df.empty: return f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3><div style="padding:12px;background:#111827;color:#d1d5db;">No candidates found.</div>'
    df_slice = df.head(max_rows).copy()
    cols = [c for c in EMAIL_DISPLAY_COLS if c in df_slice.columns]

    header_cells = "".join(f'<th style="padding:8px;border:1px solid #4b5563;background:#111827;color:#f9fafb;">{c}</th>' for c in cols)
    body_rows = []
    for _, row in df_slice.iterrows():
        tds = "".join(f'<td style="padding:6px 8px;border:1px solid #4b5563;color:#e5e7eb;background:{signal_color(row[col]) if col in ["Volatility State", "Price_Lead_Status"] else "#14532d" if col=="% Change" and float(row[col])>0 else "#7f1d1d" if col=="% Change" and float(row[col])<0 else "#030712"}">{format_value(col, row[col])}</td>' for col in cols)
        body_rows.append(f"<tr>{tds}</tr>")
    return f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3><table style="border-collapse:collapse;width:100%;background:#030712;"><thead><tr>{header_cells}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def send_email_with_tables(long_df: pd.DataFrame, short_df: pd.DataFrame, history_df: pd.DataFrame, csv_filename: str = "", detail_csv_filename: str = "") -> bool:
    try:
        scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_body = f"""
        <html>
        <body style="background:#030712;color:#e5e7eb;padding:20px;font-family:Arial,sans-serif;">
            <h2 style="color:#facc15;">Intraday Gamma Hulk Execution Alert</h2>
            <div style="color:#cbd5e1;font-size:14px;margin-bottom:18px;">Scan completed at {scan_time}</div>
            {build_html_table(long_df, "Current Long Candidates")}
            {build_history_table(history_df, "long")}
            <div style="height:28px;"></div>
            {build_html_table(short_df, "Current Short Candidates")}
            {build_history_table(history_df, "short")}
        </body>
        </html>
        """
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = sender_email, recipient_email, f"Intraday Gamma Hulk Alert - {scan_time}"
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        for fname in [csv_filename, detail_csv_filename]:
            if fname and os.path.exists(fname):
                with open(fname, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f'attachment; filename={os.path.basename(fname)}')
                    msg.attach(part)

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        logger.info("Email sent successfully.")
        return True
    except Exception as e:
        logger.error(f"EMAIL Error: {e}")
        return False


def save_outputs(summary_df: pd.DataFrame, detail_df: pd.DataFrame, prefix: str = "scan") -> Tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    s_csv, d_csv = f"{prefix}_summary_{ts}.csv", f"{prefix}_detail_{ts}.csv"
    summary_df.to_csv(s_csv, index=False)
    detail_df.to_csv(d_csv, index=False)
    return s_csv, d_csv


def main():
    init_fyers()
    if not fyers:
        logger.error("Fyers not initialized. Exiting.")
        return

    summary_df, detail_df = scan_fno_universe()
    if summary_df.empty:
        logger.warning("No summary data produced.")
        return

    detail_df = add_dual_engine_matrix(detail_df)
    summary_df = merge_dual_engine_latest(summary_df, detail_df)
    summary_df = derive_rank_columns(summary_df)
    summary_df = add_signal_columns(summary_df)

    summary_csv, detail_csv = save_outputs(summary_df, detail_df, prefix="fno")
    long_df, short_df = build_candidate_tables(summary_df)
    history_df = load_iteration_history(detail_df)

    send_email_with_tables(
        long_df=long_df, short_df=short_df, history_df=history_df,
        csv_filename=summary_csv, detail_csv_filename=detail_csv
    )


if __name__ == "__main__":
    main()

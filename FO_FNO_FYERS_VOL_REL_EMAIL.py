#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py

Intraday F&O scanner via Fyers API with email alerts.
Complete standalone file - no external email.py dependency.
SORTS CANDIDATES BY DIRECTIONAL COLUMN.
ALL STATISTICAL SCORES ARE RAW (ORIGINAL FORMULAS).
"""

import os
import re
import sys
import logging
import warnings
from datetime import datetime, timedelta, time
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
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
INTRADAY_LOOKBACK_DAYS = 20
IVP_LOOKBACK_DAYS = 252
INDEX_SOFT_BOOST_WEIGHT = 0.25

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
    "Directional",
    "Turning",
    "Stability",
    "Balanced",
    "CumsumPlus",
    "ARIMA Signal",
    "Kalman Signal",
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


def build_signals_from_raw_directional(detail_df) -> dict:
    nan = float("nan")
    out = {
        k: nan for k in (
            "5m_Signal",
            "15m_Signal",
            "30m_Signal",
            "60m_Signal",
            "Bull_Signal",
            "Bear_Signal",
            "Overall_Signal",
        )
    }
    if detail_df is None or detail_df.empty:
        return out

    df = detail_df.copy()
    if "Iteration No" in df.columns:
        df = df.sort_values("Iteration No")

    vals = pd.to_numeric(df["Directional"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return out

    last = vals.size - 1

    def raw_at(offset: int) -> float:
        i = last - offset
        if i < 0:
            i = 0
        return float(vals[i])

    out["5m_Signal"] = round(raw_at(0), 4)
    out["15m_Signal"] = round(raw_at(3) if last >= 3 else raw_at(0), 4)
    out["30m_Signal"] = round(raw_at(6) if last >= 6 else raw_at(0), 4)
    out["60m_Signal"] = round(raw_at(12) if last >= 12 else raw_at(0), 4)
    out["Bull_Signal"] = round(float(vals[vals > 0].max()) if (vals > 0).any() else 0.0, 4)
    out["Bear_Signal"] = round(abs(float(vals[vals < 0].min())) if (vals < 0).any() else 0.0, 4)
    out["Overall_Signal"] = round(raw_at(0), 4)
    return out


def init_fyers():
    global fyers
    try:
        client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
        if not client_id or not access_token:
            logger.warning("INIT Missing Fyers credentials.")
            fyers = None
            return

        fyers = fyersModel.FyersModel(
            client_id=client_id,
            is_async=False,
            token=access_token,
            log_path=""
        )
        logger.info("INIT FyersModel initialized successfully.")
    except Exception as e:
        logger.warning(f"INIT Failed: {e}")
        fyers = None


def load_fno_symbols_from_sectors(root_dir: str = "sectors") -> List[str]:
    symbols = set()
    if not os.path.isdir(root_dir):
        return []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".csv"):
                continue
            try:
                df = pd.read_csv(os.path.join(dirpath, fname))
                col = next(
                    (c for c in df.columns if c.lower() in ["symbol", "symbols", "ticker"]),
                    None,
                )
                if col is None:
                    continue
                for s in df[col].dropna().astype(str):
                    s = s.strip()
                    if s:
                        symbols.add(s)
            except Exception:
                pass

    return sorted(symbols)


def load_symbol_to_indices_map(root_dir: str = "sectors") -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    search_dirs = [root_dir, "."]

    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            for fname in filenames:
                if not fname.lower().endswith(".csv"):
                    continue
                path = os.path.join(dirpath, fname)
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue

                cols = {str(c).strip().lower(): c for c in df.columns}
                sym_col = None
                for key in ["symbol", "symbols", "ticker"]:
                    if key in cols:
                        sym_col = cols[key]
                        break

                idx_col = None
                for key in ["belongstoindices", "belongs_to_indices", "index name", "index_name", "sector", "indices"]:
                    if key in cols:
                        idx_col = cols[key]
                        break

                if not sym_col or not idx_col:
                    continue

                for _, row in df[[sym_col, idx_col]].dropna().iterrows():
                    sym = str(row[sym_col]).strip().upper()
                    raw_idx = str(row[idx_col]).strip()
                    if not sym or not raw_idx:
                        continue
                    parts = [p.strip() for p in re.split(r"[|;,/]+", raw_idx) if p.strip()]
                    if not parts:
                        parts = [raw_idx]

                    current = mapping.setdefault(sym, [])
                    for part in parts:
                        if part not in current:
                            current.append(part)

    return mapping


def resolve_universe_csv() -> str:
    csv_files = []
    for dirpath, _, filenames in os.walk("."):
        for fname in filenames:
            if fname.lower().endswith(".csv"):
                csv_files.append(os.path.join(dirpath, fname))

    scored = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, nrows=5)
            cols = {str(c).strip().lower() for c in df.columns}
            score = 0
            if "symbol" in cols:
                score += 5
            if "belongstoindices" in cols or "belongs_to_indices" in cols:
                score += 5
            if "company name" in cols or "company_name" in cols:
                score += 2
            if score > 0:
                scored.append((score, os.path.getmtime(path), path))
        except Exception:
            continue

    if scored:
        scored.sort(reverse=True)
        chosen = scored[0][2]
        logger.info(f"CSV Auto-selected universe file: {chosen}")
        return chosen

    logger.warning("CSV No matching universe CSV found; defaulting to fno_stock_list.csv")
    return "fno_stock_list.csv"


def load_fno_symbols_from_csv(path: str = "fno_stock_list.csv") -> List[str]:
    if not os.path.exists(path):
        logger.warning(f"FNO CSV not found at {path}")
        return []

    try:
        df = pd.read_csv(path)
        if "Symbol" not in df.columns:
            logger.warning("FNO CSV missing 'Symbol' column")
            return []
        return sorted(df["Symbol"].dropna().astype(str).str.strip().unique())
    except Exception as e:
        logger.error(f"Error reading FNO CSV {path}: {e}")
        return []


def format_fyers_symbol(symbol: str) -> str:
    if symbol.startswith("NSE:") and symbol.endswith("-EQ"):
        return symbol
    return f"NSE:{symbol}-EQ"


def get_fyers_history(symbol: str, resolution: str, days_back: int) -> Optional[pd.DataFrame]:
    if not fyers:
        return None

    try:
        now = datetime.now()
        start_date = now - timedelta(days=days_back)
        data = {
            "symbol": symbol,
            "resolution": resolution,
            "date_format": "1",
            "range_from": start_date.strftime("%Y-%m-%d"),
            "range_to": now.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }
        res = fyers.history(data=data)

        if res and res.get("s") == "ok" and "candles" in res and res["candles"]:
            df = pd.DataFrame(
                res["candles"],
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = (
                pd.to_datetime(df["timestamp"], unit="s")
                .dt.tz_localize("UTC")
                .dt.tz_convert("Asia/Kolkata")
                .dt.tz_localize(None)
            )
            df.sort_values("timestamp", inplace=True)
            df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df

    except Exception as e:
        logger.error(f"FYERS Error fetching {resolution} data for {symbol}: {e}")

    return None


def compute_iv_proxies(daily_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if daily_df is None or daily_df.empty or len(daily_df) < 30:
        return {"IVP": np.nan, "Volatility State": "Neutral Vol"}

    df = daily_df.copy().sort_values(
        "timestamp" if "timestamp" in daily_df.columns else daily_df.columns[0]
    ).reset_index(drop=True)

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)

    iv_proxy = ((high - low) / close.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan)
    iv_proxy = iv_proxy.dropna()
    if iv_proxy.empty:
        return {"IVP": np.nan, "Volatility State": "Neutral Vol"}

    lookback = iv_proxy.tail(min(IVP_LOOKBACK_DAYS, len(iv_proxy)))
    current_iv = float(lookback.iloc[-1])
    ivp = round((lookback.lt(current_iv).sum() / len(lookback)) * 100, 2)

    if ivp < 30:
        vol_state = "Buyer Zone"
    elif ivp > 50:
        vol_state = "Avoid Buy Premium"
    else:
        vol_state = "Neutral Vol"

    return {"IVP": ivp, "Volatility State": vol_state}


def compute_cumulative_directional_metrics(curr_df: pd.DataFrame) -> pd.DataFrame:
    df = curr_df.copy().sort_values("time").reset_index(drop=True)
    if df.empty:
        return df

    h = df["high"].astype(float).to_numpy()
    l = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    o = df["open"].astype(float).to_numpy()
    n = len(df)

    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    tr[0] = max(h[0] - l[0], 0)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        up = h[i] - h[i - 1]
        dn = l[i - 1] - l[i]
        plus_dm[i] = up if up > dn and up > 0 else 0.0
        minus_dm[i] = dn if dn > up and dn > 0 else 0.0

    out = []
    qualified = 0

    for i in range(n):
        if i == 0:
            out.append([np.nan, np.nan, np.nan, np.nan, "0/0", 0.0])
            continue

        prior_kers = []
        for j in range(1, i + 1):
            pj = abs(c[j] - o[0])
            wj = abs(c[0] - o[0]) + np.sum(np.abs(np.diff(c[: j + 1])))
            prior_kers.append(pj / wj if wj > 0 else 0.0)

        cum_ker = float(np.mean(prior_kers)) if prior_kers else np.nan

        tr_sum = tr[1 : i + 1].sum()
        plus_sum = plus_dm[1 : i + 1].sum()
        minus_sum = minus_dm[1 : i + 1].sum()

        pdi = 100 * plus_sum / tr_sum if tr_sum > 0 else 0.0
        mdi = 100 * minus_sum / tr_sum if tr_sum > 0 else 0.0

        dxs = []
        for k in range(1, i + 1):
            ktr = tr[1 : k + 1].sum()
            kp = plus_dm[1 : k + 1].sum()
            km = minus_dm[1 : k + 1].sum()
            kpdi = 100 * kp / ktr if ktr > 0 else 0.0
            kmdi = 100 * km / ktr if ktr > 0 else 0.0
            dxs.append(100 * abs(kpdi - kmdi) / (kpdi + kmdi) if (kpdi + kmdi) > 0 else 0.0)

        adx = float(np.mean(dxs)) if dxs else np.nan

        if c[i] > c[i - 1]:
            qualified += 1

        length_so_far = i + 1
        survival_ratio = qualified / length_so_far if length_so_far > 0 else 0.0
        out.append([cum_ker, pdi, mdi, adx, f"{qualified}/{length_so_far}", survival_ratio])

    cols = [
        "Cumulative KER",
        "Cumulative DI",
        "Cumulative -DI",
        "Cumulative ADX",
        "Survival Score",
        "SurvivalNum",
    ]
    return pd.concat([df, pd.DataFrame(out, columns=cols)], axis=1)


def compute_cumulative_flow_metrics(curr_df: pd.DataFrame) -> pd.DataFrame:
    df = curr_df.copy().sort_values("time").reset_index(drop=True)
    if df.empty:
        return df

    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).astype(float)

    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    period = 14
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    zero_loss = (avg_loss == 0) & avg_loss.notna()
    rsi = rsi.mask(zero_loss, 100.0).fillna(0.0)

    direction = close.diff().fillna(0.0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()

    typical_price = (high + low + close) / 3.0
    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, float("nan"))
    vwap = (cum_pv / cum_vol).fillna(0.0)

    vwap_variance = (volume * (typical_price - vwap) ** 2).cumsum() / cum_vol
    vwap_std = np.sqrt(vwap_variance).fillna(0.0)
    vwap_z_score = np.where(vwap_std > 0, (close - vwap) / vwap_std, 0.0)

    out = pd.DataFrame({
        "Cumulative RSI": rsi,
        "Cumulative OBV": obv,
        "Cumulative VWAP": vwap,
        "VWAP Z-Score": pd.Series(vwap_z_score, index=df.index).fillna(0.0),
    })
    return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)


def compute_price_lead_metrics(curr_df: pd.DataFrame) -> pd.DataFrame:
    df = curr_df.copy().sort_values("time").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=[
            "time",
            "range_expansion",
            "volume_expansion",
            "delta_expansion",
            "price_leading_flag",
            "price_lead_streak",
            "Price_Lead_Status",
        ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["range"] = (df["high"] - df["low"]).clip(lower=0.0)
    df["avg_range_5"] = df["range"].rolling(5, min_periods=3).mean()
    df["range_expansion"] = np.where(df["avg_range_5"] > 0, df["range"] / df["avg_range_5"], np.nan)

    df["avg_vol_5"] = df["volume"].rolling(5, min_periods=3).mean()
    df["volume_expansion"] = np.where(df["avg_vol_5"] > 0, df["volume"] / df["avg_vol_5"], np.nan)

    df["mid"] = (df["high"] + df["low"]) / 2.0
    df["delta"] = np.where(
        df["close"] > df["mid"],
        df["volume"],
        np.where(df["close"] < df["mid"], -df["volume"], 0.0)
    )

    df["cvd"] = pd.Series(df["delta"], index=df.index).cumsum()
    df["cvd_change"] = df["cvd"].diff().abs().fillna(0.0)
    df["avg_cvd_change_5"] = df["cvd_change"].rolling(5, min_periods=3).mean()
    df["delta_expansion"] = np.where(df["avg_cvd_change_5"] > 0, df["cvd_change"] / df["avg_cvd_change_5"], np.nan)

    directional_bar = (df["close"] > df["open"]) | (df["close"] < df["open"])
    df["price_leading_flag"] = (
        (df["range_expansion"] >= 1.5) &
        (df["volume_expansion"] <= 1.0) &
        (df["delta_expansion"] <= 1.0) &
        directional_bar
    ).fillna(False)

    streak = []
    run = 0
    for flag in df["price_leading_flag"].astype(bool):
        run = run + 1 if flag else 0
        streak.append(run)

    df["price_lead_streak"] = streak
    df["Price_Lead_Status"] = np.select(
        [
            df["price_leading_flag"] & (df["price_lead_streak"] >= 3),
            df["price_leading_flag"] & (df["price_lead_streak"] >= 2),
            df["price_leading_flag"],
        ],
        ["STRONG_PRICE_LEAD_FADE", "PRICE_LEADING_FADE_RISK", "EARLY_PRICE_LEAD"],
        default="NORMAL",
    )

    return df[[
        "time",
        "range_expansion",
        "volume_expansion",
        "delta_expansion",
        "price_leading_flag",
        "price_lead_streak",
        "Price_Lead_Status",
    ]]


def price_stats_from_series(prices: pd.Series) -> dict:
    p = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if len(p) < 3:
        return {
            "Directional": np.nan,
            "Turning": np.nan,
            "Stability": np.nan,
            "Balanced": np.nan,
            "CumsumPlus": np.nan,
        }

    x = np.arange(len(p), dtype=float)
    slope = float(np.polyfit(x, p.values, 1)[0])
    net_move = float(p.iloc[-1] - p.iloc[0])
    turning = float(np.mean(np.abs(np.diff(p.values, n=2))))
    std_p = float(np.std(p.values))
    directional = slope + net_move
    stability = std_p
    balanced = directional - turning + std_p
    cumsum_plus = float(np.sum(np.clip(np.diff(p.values), 0, None)))

    return {
        "Directional": directional,
        "Turning": turning,
        "Stability": stability,
        "Balanced": balanced,
        "CumsumPlus": cumsum_plus,
    }


def arima_signal_from_series(prices: pd.Series, order: tuple = (1, 1, 1), window: int = 50) -> float:
    p = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if len(p) < max(window // 2, 12):
        return float("nan")

    train = p.iloc[-window:] if len(p) > window else p
    try:
        fc = float(ARIMA(train, order=order).fit().forecast(1).iloc[0])
        last = float(train.iloc[-1])
        scale = float(train.diff().std()) or 1e-6
        return round((last - fc) / scale, 4)
    except Exception:
        return float("nan")


def kalman_signal_from_series(prices: pd.Series, q: float = 1e-3) -> float:
    p = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if len(p) < 5:
        return float("nan")
    arr = p.to_numpy()
    x, P = arr[0], 1.0
    r = float(p.diff().dropna().var()) + 1e-6
    for y in arr:
        P += q
        K = P / (P + r)
        x += K * (y - x)
        P *= (1.0 - K)
    gap = arr[-1] - x
    scale = float(p.std()) or 1e-6
    return round(gap / scale, 4)


def _build_iteration_signals_from_rows(rows: List[Dict]) -> Dict[str, float]:
    if not rows:
        return build_signals_from_raw_directional(pd.DataFrame())
    return build_signals_from_raw_directional(pd.DataFrame(rows))


def compute_iteration_volume_profile(
    intra_df: Optional[pd.DataFrame],
    prev_close: Optional[float] = None,
) -> Tuple[Dict, pd.DataFrame]:
    if intra_df is None or intra_df.empty:
        return {}, pd.DataFrame()

    df = intra_df.copy()
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df["time_only"] = pd.to_datetime(df["timestamp"]).dt.time
    else:
        df["date"] = pd.to_datetime(df["time"]).dt.date
        df["time_only"] = pd.to_datetime(df["time"]).dt.time

    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return {}, pd.DataFrame()

    current_date = dates[-1]
    hist_dates_10 = dates[-11:-1] if len(dates) >= 11 else dates[:-1]
    hist_dates_20 = dates[-21:-1] if len(dates) >= 21 else dates[:-1]

    curr_df = df[df["date"] == current_date].copy()
    hist_df_10 = df[df["date"].isin(hist_dates_10)].copy()
    hist_df_20 = df[df["date"].isin(hist_dates_20)].copy()

    if curr_df.empty:
        return {}, pd.DataFrame()

    curr_df.sort_values("time_only", inplace=True)
    curr_df["cum_vol"] = curr_df["volume"].cumsum()

    if prev_close is not None and prev_close != 0:
        curr_df["Iteration Change"] = (
            (pd.to_numeric(curr_df["close"], errors="coerce") - float(prev_close))
            / float(prev_close) * 100.0
        )
    else:
        curr_df["Iteration Change"] = 0.0

    work_df = curr_df.copy()
    if "time" not in work_df.columns:
        if "timestamp" in work_df.columns:
            work_df["time"] = pd.to_datetime(work_df["timestamp"])
        elif "date" in work_df.columns and "time_only" in work_df.columns:
            work_df["time"] = pd.to_datetime(work_df["date"].astype(str) + " " + work_df["time_only"].astype(str))
        else:
            work_df["time"] = pd.RangeIndex(start=0, stop=len(work_df), step=1)

    metric_df = compute_cumulative_directional_metrics(work_df[["time", "open", "high", "low", "close", "volume"]].copy())
    flow_df = compute_cumulative_flow_metrics(work_df[["time", "high", "low", "close", "volume"]].copy())
    price_lead_df = compute_price_lead_metrics(work_df[["time", "open", "high", "low", "close", "volume"]].copy())

    rows = []
    total_iters = 0
    last_iter_mins = None
    last_iter_time = None
    last_cum_vol = last_rvol10 = last_rvol20 = 0.0

    for i in range(len(curr_df)):
        total_iters += 1
        row = curr_df.iloc[i]
        t = row["time_only"]
        cum_vol = float(row["cum_vol"])

        h10 = hist_df_10[hist_df_10["time_only"] <= t]
        avg_cum_10 = h10.groupby("date")["volume"].sum().mean() if not h10.empty else 0.0
        rvol10 = cum_vol / avg_cum_10 if avg_cum_10 > 0 else 0.0

        h20 = hist_df_20[hist_df_20["time_only"] <= t]
        avg_cum_20 = h20.groupby("date")["volume"].sum().mean() if not h20.empty else 0.0
        rvol20 = cum_vol / avg_cum_20 if avg_cum_20 > 0 else 0.0

        dt_time = datetime.combine(current_date, t)
        market_open = datetime.combine(current_date, time(9, 15))
        iter_mins = int((dt_time - market_open).total_seconds() / 60)

        pct_series = curr_df["Iteration Change"].iloc[: i + 1]
        ps = price_stats_from_series(pct_series)
        pct_change_iter = float(curr_df["Iteration Change"].iloc[i])

        rows.append({
            "Iteration No": total_iters,
            "Iteration Minutes": iter_mins,
            "Iteration Time": t.strftime("%H:%M"),
            "LTP": float(row["close"]),
            "Iteration Change": pct_change_iter,
            "Current Volume": cum_vol,
            "Directional": ps["Directional"],
            "Turning": ps["Turning"],
            "Stability": ps["Stability"],
            "Balanced": ps["Balanced"],
            "CumsumPlus": ps.get("CumsumPlus", np.nan),
            "10 Day Relative Volume": rvol10,
            "20 Day Relative Volume": rvol20,
            "Cumulative RSI": float(flow_df["Cumulative RSI"].iloc[i]) if not flow_df.empty else float("nan"),
            "Cumulative OBV": float(flow_df["Cumulative OBV"].iloc[i]) if not flow_df.empty else float("nan"),
            "Cumulative VWAP": float(flow_df["Cumulative VWAP"].iloc[i]) if not flow_df.empty else float("nan"),
            "VWAP Z-Score": float(flow_df["VWAP Z-Score"].iloc[i]) if not flow_df.empty else float("nan"),
            "Range_Expansion": float(price_lead_df["range_expansion"].iloc[i]) if not price_lead_df.empty and pd.notna(price_lead_df["range_expansion"].iloc[i]) else float("nan"),
            "Volume_Expansion": float(price_lead_df["volume_expansion"].iloc[i]) if not price_lead_df.empty and pd.notna(price_lead_df["volume_expansion"].iloc[i]) else float("nan"),
            "Delta_Expansion": float(price_lead_df["delta_expansion"].iloc[i]) if not price_lead_df.empty and pd.notna(price_lead_df["delta_expansion"].iloc[i]) else float("nan"),
            "Price_Leading_Flag": bool(price_lead_df["price_leading_flag"].iloc[i]) if not price_lead_df.empty else False,
            "Price_Lead_Streak": int(price_lead_df["price_lead_streak"].iloc[i]) if not price_lead_df.empty else 0,
            "Price_Lead_Status": str(price_lead_df["Price_Lead_Status"].iloc[i]) if not price_lead_df.empty else "NORMAL",
        })

        last_cum_vol, last_rvol10, last_rvol20 = cum_vol, rvol10, rvol20
        last_iter_mins = iter_mins
        last_iter_time = t.strftime("%H:%M")

    detail_df = pd.DataFrame(rows)

    ltp = float(curr_df["close"].iloc[-1]) if not curr_df.empty else np.nan
    hod = float(curr_df["high"].max()) if not curr_df.empty else float("nan")
    strike_distance = (hod - ltp) / hod if hod and hod > 0 and ltp is not None else 1.0
    last_5m_volume = float(curr_df["volume"].iloc[-1]) if not curr_df.empty else 0.0
    recent_12 = curr_df["volume"].tail(12) if not curr_df.empty else pd.Series(dtype=float)
    vol_1h_avg_5m = float(recent_12.mean()) if len(recent_12) > 0 else last_5m_volume

    obv_30m_delta = 0.0
    rsi_30m_delta = 0.0
    if not flow_df.empty and len(flow_df) >= 7:
        obv_30m_delta = float(flow_df["Cumulative OBV"].iloc[-1]) - float(flow_df["Cumulative OBV"].iloc[-7])
        rsi_30m_delta = float(flow_df["Cumulative RSI"].iloc[-1]) - float(flow_df["Cumulative RSI"].iloc[-7])

    final_ps = price_stats_from_series(curr_df["Iteration Change"])
    summary = {
        "LTP": ltp,
        "Directional": final_ps["Directional"],
        "Turning": final_ps["Turning"],
        "Stability": final_ps["Stability"],
        "Balanced": final_ps["Balanced"],
        "CumsumPlus": final_ps.get("CumsumPlus", np.nan),
        "ARIMA Signal": arima_signal_from_series(curr_df["Iteration Change"]),
        "Kalman Signal": kalman_signal_from_series(curr_df["Iteration Change"]),
        "Current Volume": last_cum_vol,
        "10 Day Relative Volume": last_rvol10,
        "20 Day Relative Volume": last_rvol20,
        "Cumulative RSI": float(flow_df["Cumulative RSI"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Cumulative OBV": float(flow_df["Cumulative OBV"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Cumulative VWAP": float(flow_df["Cumulative VWAP"].iloc[-1]) if not flow_df.empty else float("nan"),
        "VWAP Z-Score": float(flow_df["VWAP Z-Score"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Total Iterations": total_iters,
        "Last Iteration Minutes": last_iter_mins,
        "Last Iteration Time": last_iter_time,
        "Cumulative KER": float(metric_df["Cumulative KER"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative DI": float(metric_df["Cumulative DI"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative -DI": float(metric_df["Cumulative -DI"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative ADX": float(metric_df["Cumulative ADX"].iloc[-1]) if not metric_df.empty else np.nan,
        "Survival Score": str(metric_df["Survival Score"].iloc[-1]) if not metric_df.empty else "0/0",
        "SurvivalNum": float(metric_df["SurvivalNum"].iloc[-1]) if not metric_df.empty else 0.0,
        "HOD": hod,
        "StrikeDistance": strike_distance,
        "Last5mVolume": last_5m_volume,
        "Volume1hAvg5m": vol_1h_avg_5m,
        "OBV30mDelta": obv_30m_delta,
        "RSI30mDelta": rsi_30m_delta,
        "Price_Lead_Status": str(price_lead_df["Price_Lead_Status"].iloc[-1]) if not price_lead_df.empty else "NORMAL",
    }

    signals = build_signals_from_raw_directional(detail_df)
    summary.update(signals)
    return summary, detail_df

def scan_fno_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = load_fno_symbols_from_sectors("sectors")
    if not symbols:
        logger.error("CORE No F&O symbols found.")
        return pd.DataFrame(), pd.DataFrame()

    rows, iteration_rows = [], []
    total = len(symbols)

    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"CORE [{idx}/{total}] Processing {sym}")
        fyers_sym = format_fyers_symbol(sym)

        daily_df = get_fyers_history(
            fyers_sym,
            resolution="D",
            days_back=max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS)
        )
        intra_df = get_fyers_history(
            fyers_sym,
            resolution="5",
            days_back=INTRADAY_LOOKBACK_DAYS
        )

        prev_close = float(daily_df["close"].iloc[-2]) if (daily_df is not None and len(daily_df) >= 2) else None
        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df, prev_close)
        iv_info = compute_iv_proxies(daily_df)

        ltp = iter_summary.get("LTP")
        pct_change = ((ltp - prev_close) / prev_close * 100) if (ltp is not None and prev_close and prev_close != 0) else 0.0

        if not iter_detail.empty:
            iter_detail.insert(0, "Symbol", sym)
            iter_detail.insert(1, "Daily Change", pct_change)
            iteration_rows.append(iter_detail)

        rows.append({
            "Symbol": sym,
            "LTP": ltp,
            "% Change": pct_change,
            "Directional": iter_summary.get("Directional"),
            "Turning": iter_summary.get("Turning"),
            "Stability": iter_summary.get("Stability"),
            "Balanced": iter_summary.get("Balanced"),
            "CumsumPlus": iter_summary.get("CumsumPlus"),
            "ARIMA Signal": iter_summary.get("ARIMA Signal"),
            "Kalman Signal": iter_summary.get("Kalman Signal"),
            "5m_Signal": iter_summary.get("5m_Signal"),
            "15m_Signal": iter_summary.get("15m_Signal"),
            "30m_Signal": iter_summary.get("30m_Signal"),
            "60m_Signal": iter_summary.get("60m_Signal"),
            "Bull_Signal": iter_summary.get("Bull_Signal"),
            "Bear_Signal": iter_summary.get("Bear_Signal"),
            "Overall_Signal": iter_summary.get("Overall_Signal"),
            "Current Volume": iter_summary.get("Current Volume"),
            "10 Day Relative Volume": iter_summary.get("10 Day Relative Volume"),
            "20 Day Relative Volume": iter_summary.get("20 Day Relative Volume"),
            "Cumulative RSI": iter_summary.get("Cumulative RSI"),
            "Cumulative OBV": iter_summary.get("Cumulative OBV"),
            "Cumulative VWAP": iter_summary.get("Cumulative VWAP"),
            "VWAP Z-Score": iter_summary.get("VWAP Z-Score"),
            "Total Iterations": iter_summary.get("Total Iterations"),
            "Last Iteration Minutes": iter_summary.get("Last Iteration Minutes"),
            "Last Iteration Time": iter_summary.get("Last Iteration Time"),
            "Cumulative KER": iter_summary.get("Cumulative KER"),
            "Cumulative DI": iter_summary.get("Cumulative DI"),
            "Cumulative -DI": iter_summary.get("Cumulative -DI"),
            "Cumulative ADX": iter_summary.get("Cumulative ADX"),
            "Survival Score": iter_summary.get("Survival Score"),
            "SurvivalNum": iter_summary.get("SurvivalNum"),
            "HOD": iter_summary.get("HOD"),
            "StrikeDistance": iter_summary.get("StrikeDistance"),
            "Last5mVolume": iter_summary.get("Last5mVolume"),
            "Volume1hAvg5m": iter_summary.get("Volume1hAvg5m"),
            "OBV30mDelta": iter_summary.get("OBV30mDelta"),
            "RSI30mDelta": iter_summary.get("RSI30mDelta"),
            "Price_Lead_Status": iter_summary.get("Price_Lead_Status", "NORMAL"),
            "IVP": iv_info.get("IVP"),
            "Volatility State": iv_info.get("Volatility State"),
        })

    return (
        pd.DataFrame(rows),
        pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame()
    )


def rank_delta_to_label(delta: float) -> str:
    if pd.isna(delta):
        return ""
    if delta >= 7:
        return "Buy++"
    if delta >= 4:
        return "Buy+"
    if delta >= 1:
        return "Buy"
    if delta == 0:
        return "Neutral"
    if delta <= -7:
        return "Sell++"
    if delta <= -4:
        return "Sell+"
    return "Sell"


def derive_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()

    def score_bull(row):
        score = 0
        if pd.notna(row.get("% Change")) and row.get("% Change") > 0:
            score += 2
        if pd.notna(row.get("VWAP Z-Score")) and row.get("VWAP Z-Score") >= 0.30:
            score += 2
        if (
            pd.notna(row.get("Cumulative DI")) and
            pd.notna(row.get("Cumulative -DI")) and
            row.get("Cumulative DI") > row.get("Cumulative -DI")
        ):
            score += 2
        if pd.notna(row.get("Cumulative ADX")) and row.get("Cumulative ADX") >= 20:
            score += 1
        if pd.notna(row.get("Cumulative KER")) and row.get("Cumulative KER") >= 0.40:
            score += 1
        if pd.notna(row.get("Cumulative RSI")) and row.get("Cumulative RSI") >= 55:
            score += 1
        return min(score, 13)

    def score_bear(row):
        score = 0
        if pd.notna(row.get("% Change")) and row.get("% Change") < 0:
            score += 2
        if pd.notna(row.get("VWAP Z-Score")) and row.get("VWAP Z-Score") <= -0.30:
            score += 2
        if (
            pd.notna(row.get("Cumulative DI")) and
            pd.notna(row.get("Cumulative -DI")) and
            row.get("Cumulative -DI") > row.get("Cumulative DI")
        ):
            score += 2
        if pd.notna(row.get("Cumulative ADX")) and row.get("Cumulative ADX") >= 20:
            score += 1
        if pd.notna(row.get("Cumulative KER")) and row.get("Cumulative KER") >= 0.40:
            score += 1
        if pd.notna(row.get("Cumulative RSI")) and row.get("Cumulative RSI") <= 45:
            score += 1
        return min(score, 13)

    bull = out.apply(score_bull, axis=1)
    bear = out.apply(score_bear, axis=1)

    out["Bull Rank"] = bull
    out["Bear Rank"] = bear
    out["Rank Delta"] = bull - bear

    for tf, w in {"5m": 1.0, "15m": 0.9, "30m": 0.8, "60m": 0.7}.items():
        out[f"{tf}BullRank"] = (bull * w).round().clip(lower=0, upper=14)
        out[f"{tf}BearRank"] = (bear * w).round().clip(lower=0, upper=14)
        out[f"{tf}RankDelta"] = out[f"{tf}BullRank"] - out[f"{tf}BearRank"]

    return out


def add_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df


def signal_color(label) -> str:
    try:
        v = float(label)
        if v > 0:
            return "#2e7d32"
        if v < 0:
            return "#7f1d1d"
        return "#4b5563"
    except (ValueError, TypeError):
        pass

    label = str(label).strip()
    if label == "Buyer Zone":
        return "#33691e"
    if label == "Neutral Vol":
        return "#4b5563"
    if label == "Avoid Buy Premium":
        return "#7a5c00"
    return "#374151"


def text_color_for_bg(bg: str) -> str:
    bg = str(bg).lower()
    if bg in {
        "#4b5563", "#374151", "#7f1d1d", "#a83232",
        "#b94a48", "#7a5c00", "#33691e", "#2e7d32", "#3f8f45"
    }:
        return "#f3f4f6"
    return "#ffffff"


def format_value(col: str, val):
    if pd.isna(val):
        return ""
    if col == "% Change":
        return f"{float(val):.2f}%"
    if col == "IVP":
        return f"{float(val):.2f}"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    return str(val)


def build_candidate_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(columns=EMAIL_DISPLAY_COLS), pd.DataFrame(columns=EMAIL_DISPLAY_COLS)

    base = df.copy()
    for c in ["Directional", "Turning", "Stability", "Balanced", "CumsumPlus", "10 Day Relative Volume", "Last5mVolume"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    base = base.copy()
    if "10 Day Relative Volume" in base.columns:
        base = base[base["10 Day Relative Volume"].fillna(0) >= 1.0]
    if "Last5mVolume" in base.columns:
        base = base[base["Last5mVolume"].fillna(0) > 0]

    def prep_side(df_side: pd.DataFrame, side: str) -> pd.DataFrame:
        if df_side.empty:
            return df_side
        df_side = df_side.copy()
        if side == "long":
            df_side = df_side[df_side["Directional"] > 0]
            df_side = df_side.sort_values(["Directional", "Turning", "CumsumPlus", "Stability"], ascending=[False, True, False, False], na_position="last")
        else:
            df_side = df_side[df_side["Directional"] < 0]
            df_side = df_side.sort_values(["Directional", "Turning", "CumsumPlus", "Stability"], ascending=[True, True, True, False], na_position="last")
        return df_side

    long_df = prep_side(base, "long").drop_duplicates(subset=["Symbol"]).head(15)
    short_df = prep_side(base, "short").drop_duplicates(subset=["Symbol"]).head(15)
    cols = [c for c in EMAIL_DISPLAY_COLS if c in base.columns]
    long_df = long_df[cols] if not long_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    short_df = short_df[cols] if not short_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    return long_df, short_df

def load_iteration_history(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame()

    df = detail_df.copy()
    for col in ["Iteration No", "LTP", "% Change", "Directional", "Turning", "Stability", "Balanced", "CumsumPlus"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Iteration No" not in df.columns:
        return pd.DataFrame()

    last_15_iters = sorted(df["Iteration No"].dropna().astype(int).unique())[-15:]
    df = df[df["Iteration No"].isin(last_15_iters)].copy()

    long_top = (
        df[df["Directional"] > 0]
        .sort_values(["Iteration No", "Directional", "Turning", "CumsumPlus", "Stability"], ascending=[True, False, True, False, False], na_position="last")
        .groupby("Iteration No", group_keys=False)
        .head(1)
        .assign(Side="Long")
    )

    short_top = (
        df[df["Directional"] < 0]
        .sort_values(["Iteration No", "Directional", "Turning", "CumsumPlus", "Stability"], ascending=[True, True, True, False, False], na_position="last")
        .groupby("Iteration No", group_keys=False)
        .head(1)
        .assign(Side="Short")
    )

    out = pd.concat([long_top, short_top], ignore_index=True, sort=False)
    if out.empty:
        return out

    out = out.sort_values(["Iteration No", "Side"]).reset_index(drop=True)
    iter_time = out.get("Iteration Time", pd.Series("", index=out.index)).astype(str)
    out["Iteration"] = out["Iteration No"].astype("Int64").astype(str) + " | " + iter_time
    out["First Occurrence"] = out["Iteration No"].astype("Int64").astype(str) + " | " + iter_time
    out["Latest"] = out["Iteration No"].astype("Int64").astype(str) + " | " + iter_time
    return out

def build_history_table(history_df: pd.DataFrame, side: str) -> str:
    if history_df is None or history_df.empty:
        return "No history yet."

    df = history_df.copy()
    if "Side" in df.columns:
        df = df[df["Side"].astype(str).str.lower() == side.lower()]

    if df.empty:
        return "No history yet."

    cols = [
        "First Occurrence", "Latest", "Symbol", "LTP", "% Change", "Directional", "Turning",
        "Stability", "Balanced", "CumsumPlus", "ARIMA Signal", "Kalman Signal",
        "Bull_Signal", "Bear_Signal", "Overall_Signal", "Last Iteration Time"
    ]
    if "First Occurrence" not in df.columns and "Iteration" in df.columns:
        df["First Occurrence"] = df["Iteration"]
    if "Latest" not in df.columns and "Iteration" in df.columns:
        df["Latest"] = df["Iteration"]
    cols = [c for c in cols if c in df.columns]
    df = df.tail(15)[cols].copy()

    def style_cell(col, val):
        base = "padding:6px 8px;border:1px solid #4b5563;color:#e5e7eb;"
        try:
            num = float(val)
        except Exception:
            return base

        if col in ["% Change", "Directional", "Balanced", "CumsumPlus", "Bull_Signal", "Bear_Signal", "Overall_Signal"]:
            if num > 0:
                return base + "background:#14532d;color:#dcfce7;font-weight:600;"
            if num < 0:
                return base + "background:#7f1d1d;color:#fee2e2;font-weight:600;"
        return base

    def fmt(col, val):
        if pd.isna(val):
            return ""
        if col == "% Change":
            return f"{float(val):.2f}%"
        if isinstance(val, (int, float, np.integer, np.floating)):
            return f"{float(val):.2f}"
        return str(val)

    header = "".join(
        f'<th style="padding:8px;border:1px solid #4b5563;background:#111827;color:#f9fafb;">{c}</th>'
        for c in cols
    )

    body_rows = []
    for _, r in df.iterrows():
        tds = "".join(
            f'<td style="{style_cell(c, r[c])}">{fmt(c, r[c])}</td>'
            for c in cols
        )
        body_rows.append(f"<tr>{tds}</tr>")

    title_color = "#22c55e" if side.lower() == "long" else "#ef4444"
    return f"""
    <h3 style="color:{title_color};margin:12px 0 6px 0;">Top 1 {side.title()} - Last 15 Iterations</h3>
    <div style="overflow-x:auto;">
      <table style="border-collapse:collapse;width:100%;background:#030712;">
        <thead><tr>{header}</tr></thead>
        <tbody>{''.join(body_rows)}</tbody>
      </table>
    </div>
    """


def build_html_table(df: pd.DataFrame, title: str, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return f"""
        <h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3>
        <div style="padding:12px;border:1px solid #374151;background:#111827;color:#d1d5db;border-radius:8px;">
            No candidates found.
        </div>
        """

    df_slice = df.head(max_rows).copy()
    cols = [c for c in EMAIL_DISPLAY_COLS if c in df_slice.columns]
    if not cols:
        return f"""
        <h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3>
        <div style="padding:12px;border:1px solid #374151;background:#111827;color:#d1d5db;border-radius:8px;">
            No candidates found.
        </div>
        """

    header_cells = "".join(
        f'<th style="padding:8px;border:1px solid #4b5563;background:#111827;color:#f9fafb;white-space:nowrap;">{c}</th>'
        for c in cols
    )

    def cell_style(col: str, val) -> str:
        base = "padding:6px 8px;border:1px solid #4b5563;color:#e5e7eb;white-space:nowrap;"
        try:
            num = float(val)
        except Exception:
            num = None

        if col in ["% Change", "Directional", "Balanced", "CumsumPlus", "Bull_Signal", "Bear_Signal", "Overall_Signal"]:
            if num is not None:
                if num > 0:
                    return base + "background:#14532d;color:#dcfce7;font-weight:600;"
                if num < 0:
                    return base + "background:#7f1d1d;color:#fee2e2;font-weight:600;"
            return base

        if col in ["Volatility State", "Price_Lead_Status"]:
            bg = signal_color(val)
            fg = text_color_for_bg(bg)
            return base + f"background:{bg};color:{fg};font-weight:600;"

        return base

    body_rows = []
    for _, row in df_slice.iterrows():
        tds = "".join(
            f'<td style="{cell_style(col, row[col])}">{format_value(col, row[col])}</td>'
            for col in cols
        )
        body_rows.append(f"<tr>{tds}</tr>")

    return f"""
    <h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3>
    <div style="overflow-x:auto;">
        <table style="border-collapse:collapse;width:100%;background:#030712;">
            <thead><tr>{header_cells}</tr></thead>
            <tbody>{''.join(body_rows)}</tbody>
        </table>
    </div>
    """


def send_email_with_tables(
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    history_df: pd.DataFrame,
    csv_filename: str = "",
    detail_csv_filename: str = ""
) -> bool:
    try:
        scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        history_long_html = build_history_table(history_df, "long")
        history_short_html = build_history_table(history_df, "short")

        long_html = build_html_table(long_df, "Current Long Candidates", max_rows=15)
        short_html = build_html_table(short_df, "Current Short Candidates", max_rows=15)

        html_body = f"""
        <html>
        <body style="margin:0;padding:20px;background:#030712;color:#e5e7eb;font-family:Arial,sans-serif;">
            <div style="max-width:1600px;margin:0 auto;">
                <h2 style="margin:0 0 12px 0;color:#facc15;">Intraday Vol Iteration Alert</h2>
                <div style="margin-bottom:18px;color:#cbd5e1;font-size:14px;">
                    Scan completed at {scan_time}
                </div>

                <div style="margin-bottom:24px;padding:14px;border:1px solid #374151;background:#111827;border-radius:10px;">
                    <h2 style="margin:0 0 14px 0;color:#facc15;">Last 15 Iterations - Top 1 Candidates</h2>
                    {history_long_html}
                    <div style="height:14px;"></div>
                    {history_short_html}
                </div>

                <div style="margin-bottom:24px;padding:14px;border:1px solid #374151;background:#111827;border-radius:10px;">
                    {long_html}
                </div>

                <div style="margin-bottom:24px;padding:14px;border:1px solid #374151;background:#111827;border-radius:10px;">
                    {short_html}
                </div>

                <div style="margin-top:18px;color:#94a3b8;font-size:12px;">
                    Generated by FO_FNO_FYERS_VOL_REL_EMAIL.py
                </div>
            </div>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Intraday Vol Iteration Alert - {scan_time}"
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg.attach(MIMEText(html_body, "html", _charset="utf-8"))

        for fname in [csv_filename, detail_csv_filename]:
            if not fname or not os.path.exists(fname):
                continue
            with open(fname, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename="{os.path.basename(fname)}"'
            )
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
    summary_csv = f"{prefix}_summary_{ts}.csv"
    detail_csv = f"{prefix}_detail_{ts}.csv"

    try:
        summary_df.to_csv(summary_csv, index=False)
        logger.info(f"Saved summary: {summary_csv}")
    except Exception as e:
        logger.error(f"Failed to save summary CSV: {e}")
        summary_csv = ""

    try:
        detail_df.to_csv(detail_csv, index=False)
        logger.info(f"Saved detail: {detail_csv}")
    except Exception as e:
        logger.error(f"Failed to save detail CSV: {e}")
        detail_csv = ""

    return summary_csv, detail_csv

def build_exceedance_tables(detail_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["Symbol", "Count", "CumsumPlusDiff", "First Occurrence", "Latest Iteration", "Status"]
    if detail_df is None or detail_df.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    df = detail_df.copy()
    required = {"Symbol", "Iteration No", "CumsumPlus"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    for col in ["Iteration No", "CumsumPlus"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Symbol", "Iteration No", "CumsumPlus"]).copy()
    if df.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    df["Iteration No"] = df["Iteration No"].astype(int)
    df = df.sort_values(["Symbol", "Iteration No"]).reset_index(drop=True)

    def summarize_runs(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Iteration No").copy()
        g["CumsumPlusDiff"] = g["CumsumPlus"].diff().fillna(0.0)
        g["active"] = g["CumsumPlusDiff"] >= 0.01
        if not g["active"].any():
            return pd.DataFrame(columns=cols)

        run_id = g["active"].ne(g["active"].shift(fill_value=False)).cumsum()
        rows = []
        for _, rg in g.loc[g["active"]].groupby(run_id[g["active"]]):
            count = int(len(rg))
            rows.append({
                "Symbol": str(rg["Symbol"].iloc[0]),
                "Count": count,
                "CumsumPlusDiff": float(rg["CumsumPlusDiff"].sum()),
                "First Occurrence": int(rg["Iteration No"].iloc[0]),
                "Latest Iteration": int(rg["Iteration No"].iloc[-1]),
                "Status": "Fresh Move" if count == 1 else "Continued Move",
            })
        out = pd.DataFrame(rows, columns=cols)
        if out.empty:
            return out
        return out.sort_values(["Latest Iteration", "Count", "CumsumPlusDiff", "Symbol"], ascending=[False, False, False, True], na_position="last").reset_index(drop=True)

    all_parts = []
    last5_parts = []
    for _, g in df.groupby("Symbol", sort=False):
        part_all = summarize_runs(g)
        if not part_all.empty:
            all_parts.append(part_all)
        part_last5 = summarize_runs(g.tail(5))
        if not part_last5.empty:
            last5_parts.append(part_last5)

    all_iter_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame(columns=cols)
    last_5_df = pd.concat(last5_parts, ignore_index=True) if last5_parts else pd.DataFrame(columns=cols)

    if not all_iter_df.empty:
        all_iter_df = all_iter_df.sort_values(["Latest Iteration", "Count", "CumsumPlusDiff", "Symbol"], ascending=[False, False, False, True], na_position="last").reset_index(drop=True)
    if not last_5_df.empty:
        last_5_df = last_5_df.sort_values(["Latest Iteration", "Count", "CumsumPlusDiff", "Symbol"], ascending=[False, False, False, True], na_position="last").reset_index(drop=True)
    return all_iter_df, last_5_df


def build_exceedance_table_html(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"<h3 style='color:#e5e7eb;margin:12px 0 8px 0;'>{title}</h3><p style='color:#9ca3af;'>No data.</p>"

    table_df = df.copy()
    preferred_cols = ["Symbol", "Count", "CumsumPlusDiff", "First Occurrence", "Latest Iteration", "Status"]
    cols = [c for c in preferred_cols if c in table_df.columns]
    table_df = table_df[cols]

    def fmt(col, val):
        if pd.isna(val):
            return ""
        if col == "CumsumPlusDiff":
            return f"{float(val):.4f}"
        if col in ["Count", "First Occurrence", "Latest Iteration"]:
            return f"{int(round(float(val)))}"
        if isinstance(val, (int, float, np.integer, np.floating)):
            return f"{float(val):.2f}"
        return str(val)

    def style_cell(col, val):
        base = "padding:6px 8px;border:1px solid #4b5563;color:#e5e7eb;"
        if col == "CumsumPlusDiff":
            try:
                if float(val) >= 0.01:
                    return base + "background:#14532d;color:#dcfce7;font-weight:600;"
            except Exception:
                pass
        if col == "Status":
            sval = str(val)
            if sval == "Fresh Move":
                return base + "background:#1d4ed8;color:#dbeafe;font-weight:600;"
            if sval == "Continued Move":
                return base + "background:#7c2d12;color:#ffedd5;font-weight:600;"
        return base

    header = ''.join(f"<th style='padding:6px 8px;border:1px solid #4b5563;background:#111827;color:#f9fafb'>{c}</th>" for c in cols)
    body = ''.join(
        '<tr>' + ''.join(f"<td style='{style_cell(c, row[c])}'>{fmt(c, row[c])}</td>" for c in cols) + '</tr>'
        for _, row in table_df.iterrows()
    )
    return f"<h3 style='color:#e5e7eb;margin:12px 0 8px 0;'>{title}</h3><table style='border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:12px;'><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def send_second_email_with_exceedance_tables(
    detail_df: Optional[pd.DataFrame] = None,
    all_iter_df: Optional[pd.DataFrame] = None,
    last_5_df: Optional[pd.DataFrame] = None,
    combo_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> bool:
    try:
        if detail_df is None and combo_df is not None:
            detail_df = combo_df

        if all_iter_df is None or last_5_df is None:
            if detail_df is None or detail_df.empty:
                logger.warning("EMAIL No detail_df/combo_df or prebuilt exceedance tables available for second email.")
                return False
            all_iter_df, last_5_df = build_exceedance_tables(detail_df)

        if all_iter_df is None:
            all_iter_df = pd.DataFrame()
        if last_5_df is None:
            last_5_df = pd.DataFrame()

        html = """
        <html><body style='background:#030712;color:#e5e7eb;font-family:Arial,sans-serif;'>
        <h2 style='color:#f9fafb;'>CumsumPlusDiff Exceedance Summary</h2>
        """
        html += build_exceedance_table_html(last_5_df, "CumsumPlusDiff >= 0.01 - Fresh/Continued Move - Last 5 Iterations")
        html += "<div style='height:16px;'></div>"
        html += build_exceedance_table_html(all_iter_df, "CumsumPlusDiff >= 0.01 - Fresh/Continued Move - All Iterations")
        html += "</body></html>"

        msg = MIMEMultipart("alternative")
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = "F&O Scanner - CumsumPlusDiff Exceedance >= 0.01"
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        logger.info("EMAIL Sent CumsumPlusDiff exceedance email successfully.")
        return True
    except Exception as e:
        logger.error(f"EMAIL Failed to send CumsumPlusDiff exceedance email: {e}")
        return False


def main():
    init_fyers()
    if not fyers:
        logger.error("Fyers not initialized. Exiting.")
        return

    summary_df, detail_df = scan_fno_universe()
    if summary_df.empty:
        logger.warning("No summary data produced.")
        return

    summary_df = derive_rank_columns(summary_df)
    summary_df = add_signal_columns(summary_df)

    long_df, short_df = build_candidate_tables(summary_df)
    history_df = load_iteration_history(detail_df)
    all_iter_exceed_df, combo_exceed_df = build_exceedance_tables(detail_df)

    summary_csv, detail_csv = save_outputs(summary_df, detail_df, prefix="fno")

    sent = send_email_with_tables(
        long_df=long_df,
        short_df=short_df,
        history_df=history_df,
        csv_filename=summary_csv,
        detail_csv_filename=detail_csv
    )

    sent_second = send_second_email_with_exceedance_tables(
        all_iter_df=all_iter_exceed_df,
        combo_df=combo_exceed_df,
        csv_filename=summary_csv,
        detail_csv_filename=detail_csv
    )

    if sent and sent_second:
        logger.info("Scan and both emails completed.")
    elif sent:
        logger.warning("Scan completed, first email sent, second email failed.")
    else:
        logger.warning("Scan completed but email failed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py

Intraday F&O scanner via Fyers API with email alerts.
Complete standalone file - no external email.py dependency.

Dual-Engine Matrix integrated:
1. PRISTINE_BREAKOUT -> ENTRY
2. HEALTHY_PAUSE -> HOLD
3. CHURNING_FAKEOUT -> BLOCK_ENTRY
4. TRUE_EXHAUSTION -> EXIT

Email 2:
- Table 1: Top 10 symbols from last 10 iterations
- Table 2: Top 10 symbols from all iterations
- Columns:
  Symbol, Count, CumsumPlusDiff, TurningDiff, First Occurrence, Current Iteration, Status
"""

import os
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
    datefmt="%Y-%m-%d %H:%M:%S",
)
ch.setFormatter(formatter)
logger.addHandler(ch)

warnings.filterwarnings("ignore")

DAILY_LOOKBACK_DAYS = 60
INTRADAY_LOOKBACK_DAYS = 20
IVP_LOOKBACK_DAYS = 252

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
    "CumsumDiff",
    "TurningDiff",
    "Turning Regime",
    "Dual Engine State",
    "Trade Action",
    "ARIMA Signal",
    "Kalman Signal",
    "5mSignal",
    "15mSignal",
    "30mSignal",
    "60mSignal",
    "BullSignal",
    "BearSignal",
    "OverallSignal",
    "PriceLeadStatus",
    "IVP",
    "Volatility State",
    "Last Iteration Time",
]


def build_signals_from_raw_directional(detail_df: pd.DataFrame) -> dict:
    nan = float("nan")
    out = {
        k: nan
        for k in [
            "5mSignal",
            "15mSignal",
            "30mSignal",
            "60mSignal",
            "BullSignal",
            "BearSignal",
            "OverallSignal",
        ]
    }

    if detail_df is None or detail_df.empty or "Directional" not in detail_df.columns:
        return out

    df = detail_df.copy()
    if "Iteration No" in df.columns:
        df = df.sort_values("Iteration No")

    vals = pd.to_numeric(df["Directional"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return out

    last = vals.size - 1

    def raw_at(offset: int) -> float:
        i = max(last - offset, 0)
        return float(vals[i])

    out["5mSignal"] = round(raw_at(0), 4)
    out["15mSignal"] = round(raw_at(3) if last >= 3 else raw_at(0), 4)
    out["30mSignal"] = round(raw_at(6) if last >= 6 else raw_at(0), 4)
    out["60mSignal"] = round(raw_at(12) if last >= 12 else raw_at(0), 4)
    out["BullSignal"] = round(float(vals[vals > 0].max()) if (vals > 0).any() else 0.0, 4)
    out["BearSignal"] = round(abs(float(vals[vals < 0].min())) if (vals < 0).any() else 0.0, 4)
    out["OverallSignal"] = round(raw_at(0), 4)
    return out


def classify_diff_status(cumsum_diff: float, turning_diff: float, eps: float = 1e-9) -> str:
    c = 0.0 if pd.isna(cumsum_diff) else float(cumsum_diff)
    t = 0.0 if pd.isna(turning_diff) else float(turning_diff)

    if c > eps and t <= eps:
        return "PRISTINE_BREAKOUT"
    if abs(c) <= eps and t <= eps:
        return "HEALTHY_PAUSE"
    if c > eps and t > eps:
        return "CHURNING_FAKEOUT"
    if c <= eps and t > eps:
        return "TRUE_EXHAUSTION"
    return "TRANSITION"


def add_dual_engine_matrix(
    detail_df: pd.DataFrame,
    turning_lookback: int = 10,
    low_q: float = 0.35,
    high_q: float = 0.70,
    eps: float = 1e-9,
) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame() if detail_df is None else detail_df.copy()

    required = {"Symbol", "Iteration No", "Turning", "CumsumPlus"}
    if not required.issubset(detail_df.columns):
        return detail_df.copy()

    out = detail_df.copy()
    out["Turning"] = pd.to_numeric(out["Turning"], errors="coerce")
    out["CumsumPlus"] = pd.to_numeric(out["CumsumPlus"], errors="coerce")
    out = out.sort_values(["Symbol", "Iteration No"]).reset_index(drop=True)

    grouped = out.groupby("Symbol", group_keys=False)
    out["CumsumDiff"] = grouped["CumsumPlus"].diff().fillna(0.0)
    out["TurningDiff"] = grouped["Turning"].diff().fillna(0.0)

    out["Turning Low Band"] = grouped["Turning"].transform(
        lambda s: s.rolling(turning_lookback, min_periods=1).quantile(low_q)
    )
    out["Turning High Band"] = grouped["Turning"].transform(
        lambda s: s.rolling(turning_lookback, min_periods=1).quantile(high_q)
    )

    out["Turning Regime"] = np.select(
        [
            out["Turning"] <= out["Turning Low Band"],
            out["Turning"] >= out["Turning High Band"],
        ],
        [
            "LOW",
            "HIGH",
        ],
        default="MID",
    )

    out["Dual Engine State"] = np.select(
        [
            (out["CumsumDiff"] > eps) & (out["Turning Regime"] == "LOW"),
            (out["CumsumDiff"].abs() <= eps) & (out["Turning Regime"] == "LOW"),
            (out["CumsumDiff"] > eps) & (out["Turning Regime"] == "HIGH"),
            (out["CumsumDiff"] <= eps) & (out["Turning Regime"] == "HIGH"),
        ],
        [
            "PRISTINE_BREAKOUT",
            "HEALTHY_PAUSE",
            "CHURNING_FAKEOUT",
            "TRUE_EXHAUSTION",
        ],
        default="TRANSITION",
    )

    out["Trade Action"] = np.select(
        [
            out["Dual Engine State"] == "PRISTINE_BREAKOUT",
            out["Dual Engine State"] == "HEALTHY_PAUSE",
            out["Dual Engine State"] == "CHURNING_FAKEOUT",
            out["Dual Engine State"] == "TRUE_EXHAUSTION",
        ],
        [
            "ENTRY",
            "HOLD",
            "BLOCK_ENTRY",
            "EXIT",
        ],
        default="WAIT",
    )

    out["Entry Allowed"] = out["Trade Action"].eq("ENTRY")
    out["Hold Allowed"] = out["Trade Action"].eq("HOLD")
    out["Exit Now"] = out["Trade Action"].eq("EXIT")
    return out


def merge_dual_engine_latest(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame() if summary_df is None else summary_df.copy()
    if detail_df is None or detail_df.empty:
        return summary_df.copy()

    needed = {
        "Symbol",
        "Iteration No",
        "CumsumDiff",
        "TurningDiff",
        "Turning Regime",
        "Dual Engine State",
        "Trade Action",
    }
    if not needed.issubset(detail_df.columns):
        return summary_df.copy()

    latest = (
        detail_df.sort_values(["Symbol", "Iteration No"])
        .groupby("Symbol", as_index=False)
        .tail(1)[
            [
                "Symbol",
                "CumsumDiff",
                "TurningDiff",
                "Turning Regime",
                "Dual Engine State",
                "Trade Action",
            ]
        ]
        .copy()
    )
    return summary_df.merge(latest, on="Symbol", how="left")


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
            log_path="",
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
                    s = s.strip().upper()
                    if s:
                        symbols.add(s)
            except Exception:
                pass

    return sorted(symbols)


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
        return sorted(df["Symbol"].dropna().astype(str).str.strip().str.upper().unique())
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

    df = daily_df.copy().sort_values("timestamp").reset_index(drop=True)

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
    vwap_zscore = np.where(vwap_std > 0, (close - vwap) / vwap_std, 0.0)

    out = pd.DataFrame(
        {
            "Cumulative RSI": rsi,
            "Cumulative OBV": obv,
            "Cumulative VWAP": vwap,
            "VWAP Z-Score": pd.Series(vwap_zscore, index=df.index).fillna(0.0),
        }
    )
    return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)


def compute_price_lead_metrics(curr_df: pd.DataFrame) -> pd.DataFrame:
    df = curr_df.copy().sort_values("time").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "time",
                "range_expansion",
                "volume_expansion",
                "delta_expansion",
                "price_leading_flag",
                "price_lead_streak",
                "PriceLeadStatus",
            ]
        )

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
        np.where(df["close"] < df["mid"], -df["volume"], 0.0),
    )
    df["cvd"] = pd.Series(df["delta"], index=df.index).cumsum()
    df["cvd_change"] = df["cvd"].diff().abs().fillna(0.0)
    df["avg_cvd_change_5"] = df["cvd_change"].rolling(5, min_periods=3).mean()
    df["delta_expansion"] = np.where(
        df["avg_cvd_change_5"] > 0,
        df["cvd_change"] / df["avg_cvd_change_5"],
        np.nan,
    )

    directional_bar = (df["close"] > df["open"]) | (df["close"] < df["open"])

    df["price_leading_flag"] = (
        (df["range_expansion"] >= 1.5)
        & (df["volume_expansion"] <= 1.0)
        & (df["delta_expansion"] <= 1.0)
        & directional_bar
    ).fillna(False)

    streak = []
    run = 0
    for flag in df["price_leading_flag"].astype(bool):
        run = run + 1 if flag else 0
        streak.append(run)
    df["price_lead_streak"] = streak

    df["PriceLeadStatus"] = np.select(
        [
            df["price_leading_flag"] & (df["price_lead_streak"] >= 3),
            df["price_leading_flag"] & (df["price_lead_streak"] >= 2),
            df["price_leading_flag"],
        ],
        [
            "STRONG_PRICE_LEAD_FADE",
            "PRICE_LEADING_FADE_RISK",
            "EARLY_PRICE_LEAD",
        ],
        default="NORMAL",
    )

    return df[
        [
            "time",
            "range_expansion",
            "volume_expansion",
            "delta_expansion",
            "price_leading_flag",
            "price_lead_streak",
            "PriceLeadStatus",
        ]
    ]


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
        fit = ARIMA(train, order=order).fit()
        fc = float(fit.forecast(1).iloc[0])
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


def compute_iteration_volume_profile(
    intra_df: Optional[pd.DataFrame],
    prev_close: Optional[float] = None,
) -> Tuple[Dict, pd.DataFrame]:
    if intra_df is None or intra_df.empty:
        return {}, pd.DataFrame()

    df = intra_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["time_only"] = pd.to_datetime(df["timestamp"]).dt.time

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
    curr_df["cumvol"] = pd.to_numeric(curr_df["volume"], errors="coerce").fillna(0.0).cumsum()

    if prev_close is not None and prev_close != 0:
        curr_df["Iteration Change"] = (
            (pd.to_numeric(curr_df["close"], errors="coerce") - float(prev_close))
            / float(prev_close)
            * 100.0
        )
    else:
        curr_df["Iteration Change"] = 0.0

    work_df = curr_df.copy()
    work_df["time"] = pd.to_datetime(work_df["timestamp"])

    metric_df = compute_cumulative_directional_metrics(
        work_df[["time", "open", "high", "low", "close", "volume"]].copy()
    )
    flow_df = compute_cumulative_flow_metrics(
        work_df[["time", "high", "low", "close", "volume"]].copy()
    )
    price_lead_df = compute_price_lead_metrics(
        work_df[["time", "open", "high", "low", "close", "volume"]].copy()
    )

    rows = []
    total_iters = 0
    last_iter_mins = None
    last_iter_time = None
    last_cumvol = last_rvol10 = last_rvol20 = 0.0

    for i in range(len(curr_df)):
        total_iters += 1
        row = curr_df.iloc[i]
        t = row["time_only"]
        cumvol = float(row["cumvol"])

        h10 = hist_df_10[hist_df_10["time_only"] <= t]
        avg_cum10 = h10.groupby("date")["volume"].sum().mean() if not h10.empty else 0.0
        rvol10 = cumvol / avg_cum10 if avg_cum10 > 0 else 0.0

        h20 = hist_df_20[hist_df_20["time_only"] <= t]
        avg_cum20 = h20.groupby("date")["volume"].sum().mean() if not h20.empty else 0.0
        rvol20 = cumvol / avg_cum20 if avg_cum20 > 0 else 0.0

        dt_time = datetime.combine(current_date, t)
        market_open = datetime.combine(current_date, time(9, 15))
        iter_mins = int((dt_time - market_open).total_seconds() / 60)

        pct_series = curr_df["Iteration Change"].iloc[: i + 1]
        ps = price_stats_from_series(pct_series)
        pct_change_iter = float(curr_df["Iteration Change"].iloc[i])

        rows.append(
            {
                "Iteration No": total_iters,
                "Iteration Minutes": iter_mins,
                "Iteration Time": t.strftime("%H:%M"),
                "LTP": float(row["close"]),
                "Iteration Change": pct_change_iter,
                "Current Volume": cumvol,
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
                "RangeExpansion": float(price_lead_df["range_expansion"].iloc[i]) if not price_lead_df.empty else float("nan"),
                "VolumeExpansion": float(price_lead_df["volume_expansion"].iloc[i]) if not price_lead_df.empty else float("nan"),
                "DeltaExpansion": float(price_lead_df["delta_expansion"].iloc[i]) if not price_lead_df.empty else float("nan"),
                "PriceLeadingFlag": bool(price_lead_df["price_leading_flag"].iloc[i]) if not price_lead_df.empty else False,
                "PriceLeadStreak": int(price_lead_df["price_lead_streak"].iloc[i]) if not price_lead_df.empty else 0,
                "PriceLeadStatus": str(price_lead_df["PriceLeadStatus"].iloc[i]) if not price_lead_df.empty else "NORMAL",
            }
        )

        last_cumvol, last_rvol10, last_rvol20 = cumvol, rvol10, rvol20
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
        "% Change": ((ltp - prev_close) / prev_close * 100.0) if (prev_close and prev_close != 0 and pd.notna(ltp)) else 0.0,
        "Directional": final_ps["Directional"],
        "Turning": final_ps["Turning"],
        "Stability": final_ps["Stability"],
        "Balanced": final_ps["Balanced"],
        "CumsumPlus": final_ps.get("CumsumPlus", np.nan),
        "ARIMA Signal": arima_signal_from_series(curr_df["Iteration Change"]),
        "Kalman Signal": kalman_signal_from_series(curr_df["Iteration Change"]),
        "Current Volume": last_cumvol,
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
        "PriceLeadStatus": str(price_lead_df["PriceLeadStatus"].iloc[-1]) if not price_lead_df.empty else "NORMAL",
    }

    summary.update(build_signals_from_raw_directional(detail_df))
    return summary, detail_df


def scan_fno_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = load_fno_symbols_from_sectors("sectors")
    if not symbols:
        fallback_csv = resolve_universe_csv()
        symbols = load_fno_symbols_from_csv(fallback_csv)

    if not symbols:
        logger.error("CORE No F&O symbols found.")
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    iteration_rows = []

    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"CORE [{idx}/{len(symbols)}] Processing {sym}")

        fyers_sym = format_fyers_symbol(sym)
        daily_df = get_fyers_history(fyers_sym, resolution="D", days_back=max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
        intra_df = get_fyers_history(fyers_sym, resolution="5", days_back=INTRADAY_LOOKBACK_DAYS)

        prev_close = float(daily_df["close"].iloc[-2]) if (daily_df is not None and len(daily_df) >= 2) else None

        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df, prev_close)
        iv_info = compute_iv_proxies(daily_df)

        if not iter_summary:
            continue

        ltp = iter_summary.get("LTP")
        pct_change = ((ltp - prev_close) / prev_close * 100.0) if (
            ltp is not None and prev_close and prev_close != 0
        ) else 0.0

        if not iter_detail.empty:
            iter_detail.insert(0, "Symbol", sym)
            iter_detail.insert(1, "% Change", pct_change)
            iteration_rows.append(iter_detail)

        row = {"Symbol": sym}
        row.update(iter_summary)
        row["% Change"] = pct_change
        row["IVP"] = iv_info.get("IVP")
        row["Volatility State"] = iv_info.get("Volatility State")
        rows.append(row)

    return (
        pd.DataFrame(rows),
        pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame(),
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
            pd.notna(row.get("Cumulative DI"))
            and pd.notna(row.get("Cumulative -DI"))
            and row.get("Cumulative DI") > row.get("Cumulative -DI")
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
            pd.notna(row.get("Cumulative DI"))
            and pd.notna(row.get("Cumulative -DI"))
            and row.get("Cumulative -DI") > row.get("Cumulative DI")
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
    out["Rank Label"] = out["Rank Delta"].apply(rank_delta_to_label)

    for tf, w in {"5m": 1.0, "15m": 0.9, "30m": 0.8, "60m": 0.7}.items():
        out[f"{tf}BullRank"] = (bull * w).round().clip(lower=0, upper=14)
        out[f"{tf}BearRank"] = (bear * w).round().clip(lower=0, upper=14)
        out[f"{tf}RankDelta"] = out[f"{tf}BullRank"] - out[f"{tf}BearRank"]

    return out


def add_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    out = df.copy()
    if "Trade Action" in out.columns:
        out["Entry Allowed"] = out["Trade Action"].eq("ENTRY")
        out["Hold Allowed"] = out["Trade Action"].eq("HOLD")
        out["Exit Now"] = out["Trade Action"].eq("EXIT")
    return out


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

    mapping = {
        "Buyer Zone": "#33691e",
        "Neutral Vol": "#4b5563",
        "Avoid Buy Premium": "#7a5c00",
        "LOW": "#166534",
        "MID": "#4b5563",
        "HIGH": "#991b1b",
        "PRISTINE_BREAKOUT": "#15803d",
        "HEALTHY_PAUSE": "#0f766e",
        "CHURNING_FAKEOUT": "#b45309",
        "TRUE_EXHAUSTION": "#b91c1c",
        "TRANSITION": "#374151",
        "ENTRY": "#15803d",
        "HOLD": "#0f766e",
        "BLOCK_ENTRY": "#b45309",
        "EXIT": "#b91c1c",
        "WAIT": "#4b5563",
        "STRONG_PRICE_LEAD_FADE": "#7f1d1d",
        "PRICE_LEADING_FADE_RISK": "#b45309",
        "EARLY_PRICE_LEAD": "#374151",
    }
    return mapping.get(label, "#374151")


def text_color_for_bg(bg: str) -> str:
    bg = str(bg).lower()
    darkish = {
        "#4b5563",
        "#374151",
        "#7f1d1d",
        "#7a5c00",
        "#33691e",
        "#2e7d32",
        "#991b1b",
        "#166534",
        "#15803d",
        "#0f766e",
        "#b45309",
        "#b91c1c",
    }
    return "#f3f4f6" if bg in darkish else "#ffffff"


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

    for c in [
        "Directional",
        "Turning",
        "Stability",
        "Balanced",
        "CumsumPlus",
        "CumsumDiff",
        "TurningDiff",
        "10 Day Relative Volume",
        "Last5mVolume",
    ]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    if "10 Day Relative Volume" in base.columns:
        base = base[base["10 Day Relative Volume"].fillna(0) >= 1.0]
    if "Last5mVolume" in base.columns:
        base = base[base["Last5mVolume"].fillna(0) > 0]

    action_rank_map = {
        "ENTRY": 0,
        "HOLD": 1,
        "WAIT": 2,
        "BLOCK_ENTRY": 3,
        "EXIT": 4,
    }

    if "Trade Action" in base.columns:
        base["Trade Action Rank"] = base["Trade Action"].astype(str).map(action_rank_map).fillna(9)
    else:
        base["Trade Action Rank"] = 9

    def prep_side(df_side: pd.DataFrame, side: str) -> pd.DataFrame:
        if df_side.empty:
            return df_side

        df_side = df_side.copy()

        if side == "long":
            df_side = df_side[df_side["Directional"] > 0]
            if "Trade Action" in df_side.columns:
                df_side = df_side[df_side["Trade Action"].isin(["ENTRY", "HOLD", "WAIT"])]
            df_side = df_side.sort_values(
                ["Trade Action Rank", "Directional", "Turning", "CumsumPlus", "CumsumDiff", "TurningDiff", "Stability"],
                ascending=[True, False, True, False, False, True, False],
                na_position="last",
            )
        else:
            df_side = df_side[df_side["Directional"] < 0]
            df_side = df_side.sort_values(
                ["Directional", "Turning", "CumsumPlus", "CumsumDiff", "TurningDiff", "Stability"],
                ascending=[True, True, True, True, False, False],
                na_position="last",
            )

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
    for col in [
        "Iteration No",
        "LTP",
        "% Change",
        "Directional",
        "Turning",
        "Stability",
        "Balanced",
        "CumsumPlus",
        "CumsumDiff",
        "TurningDiff",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Iteration No" not in df.columns:
        return pd.DataFrame()

    last_15_iters = sorted(df["Iteration No"].dropna().astype(int).unique())[-15:]
    df = df[df["Iteration No"].isin(last_15_iters)].copy()

    long_top = (
        df[df["Directional"] > 0]
        .sort_values(
            ["Iteration No", "Directional", "Turning", "CumsumPlus", "CumsumDiff", "TurningDiff", "Stability"],
            ascending=[True, False, True, False, False, True, False],
            na_position="last",
        )
        .groupby("Iteration No", group_keys=False)
        .head(1)
        .assign(Side="Long")
    )

    short_top = (
        df[df["Directional"] < 0]
        .sort_values(
            ["Iteration No", "Directional", "Turning", "CumsumPlus", "CumsumDiff", "TurningDiff", "Stability"],
            ascending=[True, True, True, True, False, False, False],
            na_position="last",
        )
        .groupby("Iteration No", group_keys=False)
        .head(1)
        .assign(Side="Short")
    )

    out = pd.concat([long_top, short_top], ignore_index=True, sort=False)
    if out.empty:
        return out

    out = out.sort_values(["Iteration No", "Side"]).reset_index(drop=True)
    out["Iteration"] = out["Iteration No"].astype("Int64").astype(str) + " | " + out["Iteration Time"].astype(str)
    out["First Occurrence"] = out["Iteration"]
    out["Latest"] = out["Iteration"]
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
        "First Occurrence",
        "Latest",
        "Symbol",
        "LTP",
        "% Change",
        "Directional",
        "Turning",
        "CumsumDiff",
        "TurningDiff",
        "Turning Regime",
        "Dual Engine State",
        "Trade Action",
        "Stability",
        "Balanced",
        "CumsumPlus",
        "Last Iteration Time",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df.tail(15)[cols].copy()

    def style_cell(col, val):
        base = "padding:6px 8px;border:1px solid #4b5563;color:#e5e7eb;"
        try:
            num = float(val)
        except Exception:
            num = None

        if col in ["% Change", "Directional", "Balanced", "CumsumPlus", "CumsumDiff", "TurningDiff"]:
            if num is not None:
                if num > 0:
                    return base + "background:#14532d;color:#dcfce7;font-weight:600;"
                if num < 0:
                    return base + "background:#7f1d1d;color:#fee2e2;font-weight:600;"
            return base

        if col in ["Turning Regime", "Dual Engine State", "Trade Action", "PriceLeadStatus", "Volatility State"]:
            bg = signal_color(val)
            fg = text_color_for_bg(bg)
            return base + f"background:{bg};color:{fg};font-weight:600;"

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
        tds = "".join(f'<td style="{style_cell(c, r[c])}">{fmt(c, r[c])}</td>' for c in cols)
        body_rows.append(f"<tr>{tds}</tr>")

    title_color = "#22c55e" if side.lower() == "long" else "#ef4444"
    return (
        f'<h3 style="color:{title_color};margin:12px 0 6px 0;">Top 1 {side.title()} - Last 15 Iterations</h3>'
        f'<div style="overflow-x:auto;">'
        f'<table style="border-collapse:collapse;width:100%;background:#030712;">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table></div>"
    )


def build_html_table(df: pd.DataFrame, title: str, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return (
            f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3>'
            f'<div style="padding:12px;border:1px solid #374151;background:#111827;'
            f'color:#d1d5db;border-radius:8px;">No candidates found.</div>'
        )

    df_slice = df.head(max_rows).copy()
    cols = [c for c in EMAIL_DISPLAY_COLS if c in df_slice.columns]

    if not cols:
        return (
            f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3>'
            f'<div style="padding:12px;border:1px solid #374151;background:#111827;'
            f'color:#d1d5db;border-radius:8px;">No candidates found.</div>'
        )

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

        if col in ["% Change", "Directional", "Balanced", "CumsumPlus", "CumsumDiff", "TurningDiff", "BullSignal", "BearSignal", "OverallSignal"]:
            if num is not None:
                if num > 0:
                    return base + "background:#14532d;color:#dcfce7;font-weight:600;"
                if num < 0:
                    return base + "background:#7f1d1d;color:#fee2e2;font-weight:600;"
            return base

        if col in ["Volatility State", "PriceLeadStatus", "Turning Regime", "Dual Engine State", "Trade Action"]:
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

    return (
        f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3>'
        f'<div style="overflow-x:auto;">'
        f'<table style="border-collapse:collapse;width:100%;background:#030712;">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table></div>"
    )


def send_email_with_tables(
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    history_df: pd.DataFrame,
    csv_filename: str = "",
    detail_csv_filename: str = "",
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
            part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(fname)}"')
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
        logger.info(f"Saved summary CSV: {summary_csv}")
    except Exception as e:
        logger.error(f"Failed to save summary CSV: {e}")
        summary_csv = ""

    try:
        detail_df.to_csv(detail_csv, index=False)
        logger.info(f"Saved detail CSV: {detail_csv}")
    except Exception as e:
        logger.error(f"Failed to save detail CSV: {e}")
        detail_csv = ""

    return summary_csv, detail_csv

def build_occurrence_table(
    detail_df: pd.DataFrame,
    last_n_iterations: Optional[int] = None,
    top_n: int = 10,
    eps: float = 1e-9,
) -> pd.DataFrame:
    cols = [
        "Symbol",
        "Count",
        "CumsumPlusDiff",
        "TurningDiff",
        "First Occurrence",
        "Current Iteration",
        "Status",
    ]
    empty = pd.DataFrame(columns=cols)

    if detail_df is None or detail_df.empty:
        return empty

    required = {"Symbol", "Iteration No", "Iteration Time", "CumsumDiff", "TurningDiff"}
    if not required.issubset(detail_df.columns):
        return empty

    df = detail_df.copy().sort_values(["Symbol", "Iteration No"]).reset_index(drop=True)
    df["CumsumDiff"] = pd.to_numeric(df["CumsumDiff"], errors="coerce").fillna(0.0)
    df["TurningDiff"] = pd.to_numeric(df["TurningDiff"], errors="coerce").fillna(0.0)

    if last_n_iterations is not None:
        last_iters = sorted(df["Iteration No"].dropna().astype(int).unique())[-last_n_iterations:]
        df = df[df["Iteration No"].isin(last_iters)].copy()

    if df.empty:
        return empty

    rows = []

    for sym, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Iteration No").reset_index(drop=True)

        qualifies = (g["CumsumDiff"].abs() > eps) | (g["TurningDiff"].abs() > eps)

        if g.empty or not bool(qualifies.iloc[-1]):
            continue

        idx = len(g) - 1
        keep_idx = [idx]

        while idx - 1 >= 0 and bool(qualifies.iloc[idx - 1]):
            keep_idx.append(idx - 1)
            idx -= 1

        chain = g.iloc[sorted(keep_idx)].copy()
        latest = chain.iloc[-1]

        latest_cumsum = float(latest["CumsumDiff"])
        latest_turning = float(latest["TurningDiff"])

        if "Dual Engine State" in chain.columns and pd.notna(latest.get("Dual Engine State")):
            status = str(latest["Dual Engine State"])
        else:
            status = classify_diff_status(latest_cumsum, latest_turning, eps=eps)

        rows.append(
            {
                "Symbol": sym,
                "Count": int(len(chain)),
                "CumsumPlusDiff": float(chain["CumsumDiff"].sum()),
                "TurningDiff": float(chain["TurningDiff"].sum()),
                "First Occurrence": str(chain["Iteration Time"].iloc[0]),
                "Current Iteration": str(chain["Iteration Time"].iloc[-1]),
                "Status": status,
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return empty

    out["_current_sort"] = pd.to_datetime(
        out["Current Iteration"], format="%H:%M", errors="coerce"
    )
    out["_first_sort"] = pd.to_datetime(
        out["First Occurrence"], format="%H:%M", errors="coerce"
    )

    out = (
        out.sort_values(
            ["_current_sort", "Count", "CumsumPlusDiff", "TurningDiff", "_first_sort"],
            ascending=[False, False, False, True, False],
            na_position="last",
        )
        .drop(columns=["_current_sort", "_first_sort"])
        .head(top_n)
        .reset_index(drop=True)
    )

    return out


def build_exceedance_tables(detail_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    last10_df = build_occurrence_table(detail_df, last_n_iterations=10, top_n=10)
    all_df = build_occurrence_table(detail_df, last_n_iterations=None, top_n=10)
    return last10_df, all_df
        


def build_exceedance_table_html(df: pd.DataFrame, title: str, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return (
            f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3>'
            f'<div style="padding:12px;border:1px solid #374151;background:#111827;'
            f'color:#d1d5db;border-radius:8px;">No data found.</div>'
        )

    view = df.head(max_rows).copy()
    cols = list(view.columns)

    def fmt(col, val):
        if pd.isna(val):
            return ""
        if col in ["Symbol", "Status", "First Occurrence", "Current Iteration"]:
            return str(val)
        if col == "Count":
            return str(int(float(val)))
        if col in ["CumsumPlusDiff", "TurningDiff"]:
            return f"{float(val):.2f}"
        return str(val)

    def cell_style(col, val):
        base = "padding:6px 8px;border:1px solid #4b5563;color:#e5e7eb;white-space:nowrap;"

        if col in ["CumsumPlusDiff", "TurningDiff"]:
            try:
                num = float(val)
                if num > 0:
                    return base + "background:#14532d;color:#dcfce7;font-weight:700;"
                if num < 0:
                    return base + "background:#7f1d1d;color:#fee2e2;font-weight:700;"
            except Exception:
                pass
            return base

        if col == "Status":
            bg = signal_color(val)
            fg = text_color_for_bg(bg)
            return base + f"background:{bg};color:{fg};font-weight:700;"

        if col == "Count":
            return base + "background:#1f2937;color:#f9fafb;font-weight:700;"

        return base

    header = "".join(
        f'<th style="padding:8px;border:1px solid #4b5563;background:#111827;color:#f9fafb;white-space:nowrap;">{c}</th>'
        for c in cols
    )

    body = []
    for _, row in view.iterrows():
        cells = "".join(f'<td style="{cell_style(c, row[c])}">{fmt(c, row[c])}</td>' for c in cols)
        body.append(f"<tr>{cells}</tr>")

    return (
        f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3>'
        f'<div style="overflow-x:auto;">'
        f'<table style="border-collapse:collapse;width:100%;background:#030712;">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        f"</table></div>"
    )


def send_second_email_with_exceedance_tables(
    all_iter_df: pd.DataFrame,
    combo_df: pd.DataFrame,
    csv_filename: str = "",
    detail_csv_filename: str = "",
) -> bool:
    try:
        scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        combo_html = build_exceedance_table_html(combo_df, "Top 10 Symbols - Last 10 Iterations", max_rows=10)
        all_html = build_exceedance_table_html(all_iter_df, "Top 10 Symbols - All Iterations", max_rows=10)

        html_body = f"""
        <html>
        <body style="margin:0;padding:20px;background:#030712;color:#e5e7eb;font-family:Arial,sans-serif;">
            <div style="max-width:1600px;margin:0 auto;">
                <h2 style="margin:0 0 12px 0;color:#facc15;">Intraday Vol Iteration Alert - Diff Occurrence Tables</h2>
                <div style="margin-bottom:18px;color:#cbd5e1;font-size:14px;">
                    Scan completed at {scan_time}
                </div>
                <div style="margin-bottom:24px;padding:14px;border:1px solid #374151;background:#111827;border-radius:10px;">
                    {combo_html}
                </div>
                <div style="margin-bottom:24px;padding:14px;border:1px solid #374151;background:#111827;border-radius:10px;">
                    {all_html}
                </div>
                <div style="margin-top:18px;color:#94a3b8;font-size:12px;">
                    Generated by FO_FNO_FYERS_VOL_REL_EMAIL.py
                </div>
            </div>
        </body>
        </html>
        """

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Intraday Vol Iteration Alert - Diff Occurrence Tables - {scan_time}"
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
            part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(fname)}"')
            msg.attach(part)

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        logger.info("Second diff-occurrence email sent successfully.")
        return True

    except Exception as e:
        logger.error(f"SECOND EMAIL Error: {e}")
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

    detail_df = add_dual_engine_matrix(detail_df)
    summary_df = merge_dual_engine_latest(summary_df, detail_df)

    summary_df = derive_rank_columns(summary_df)
    summary_df = add_signal_columns(summary_df)

    long_df, short_df = build_candidate_tables(summary_df)
    history_df = load_iteration_history(detail_df)
    last10_occurrence_df, all_occurrence_df = build_exceedance_tables(detail_df)

    summary_csv, detail_csv = save_outputs(summary_df, detail_df, prefix="fno")

    sent = send_email_with_tables(
        long_df=long_df,
        short_df=short_df,
        history_df=history_df,
        csv_filename=summary_csv,
        detail_csv_filename=detail_csv,
    )

    sent_second = send_second_email_with_exceedance_tables(
        all_iter_df=all_occurrence_df,
        combo_df=last10_occurrence_df,
        csv_filename=summary_csv,
        detail_csv_filename=detail_csv,
    )

    if sent and sent_second:
        logger.info("Scan and both emails completed.")
    elif sent:
        logger.warning("Scan completed, first email sent, second email failed.")
    else:
        logger.warning("Scan completed but email failed.")


if __name__ == "__main__":
    main()

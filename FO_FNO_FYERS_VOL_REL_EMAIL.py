#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py

Intraday F&O scanner via Fyers API with email alerts.
Complete standalone file - no external email.py dependency.
SORTS CANDIDATES BY DIRECTIONAL COLUMN.
ALL STATISTICAL SCORES ARE RAW (ORIGINAL FORMULAS).
Now powered by Zero-Lag Dual Engine Row-vs-Row State Matrices.
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
    "CumsumDiff",
    "TurningDiff",
    "Turning Regime",
    "Dual Engine State",
    "Trade Action",
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


def classify_diff_status(cumsum_diff: float, turning_diff: float, prior_cumsum: float = 0.0, eps: float = 1e-4) -> str:
    c = 0.0 if pd.isna(cumsum_diff) else float(cumsum_diff)
    t = 0.0 if pd.isna(turning_diff) else float(turning_diff)
    p = 0.0 if pd.isna(prior_cumsum) else float(prior_cumsum)

    friction_expanding = t > eps

    if c > eps and p <= eps and not friction_expanding:
        return "PRISTINE_BREAKOUT"
    if c > eps and p > eps and not friction_expanding:
        return "ACTIVE_CONTINUATION"
    if c <= eps and friction_expanding:
        return "TRUE_EXHAUSTION"
    if c > eps and friction_expanding:
        return "CHURNING_FAKEOUT"
    if abs(c) <= eps and not friction_expanding:
        return "HEALTHY_PAUSE"
    return "TRANSITION"


def add_dual_engine_matrix(
    detail_df: pd.DataFrame,
    eps: float = 1e-4,
) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame() if detail_df is None else detail_df.copy()

    required = {"Symbol", "Iteration No", "Turning", "CumsumPlus"}
    if not required.issubset(detail_df.columns):
        return detail_df.copy()

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
    out["TurningDiff"] = out["Turning"] - out["Prior_Turning"]

    out["Friction_Expanding"] = (out["Turning"] > out["Prior_Turning"]) & (out["Turning"] > eps)

    cond_pristine = (out["Current_Step"] > eps) & (out["Prior_Step"] <= eps) & (~out["Friction_Expanding"])
    cond_exhaustion = (out["Current_Step"] <= eps) & out["Friction_Expanding"]
    cond_trap = (out["Current_Step"] > eps) & out["Friction_Expanding"]
    cond_pause = (out["Current_Step"].abs() <= eps) & (~out["Friction_Expanding"])
    cond_active = (out["Current_Step"] > eps) & (out["Prior_Step"] > eps) & (~out["Friction_Expanding"])

    out["Dual Engine State"] = np.select(
        [cond_pristine, cond_exhaustion, cond_trap, cond_pause, cond_active],
        ["PRISTINE_BREAKOUT", "TRUE_EXHAUSTION", "CHURNING_FAKEOUT", "HEALTHY_PAUSE", "ACTIVE_CONTINUATION"],
        default="TRANSITION",
    )

    out["Trade Action"] = np.select(
        [
            out["Dual Engine State"] == "PRISTINE_BREAKOUT",
            out["Dual Engine State"] == "TRUE_EXHAUSTION",
            out["Dual Engine State"] == "CHURNING_FAKEOUT",
            out["Dual Engine State"] == "HEALTHY_PAUSE",
            out["Dual Engine State"] == "ACTIVE_CONTINUATION"
        ],
        ["ENTRY", "EXIT", "BLOCK_ENTRY", "HOLD", "HOLD"],
        default="WAIT",
    )

    out["Turning Regime"] = np.where(out["Friction_Expanding"], "EXPANDING_FRICTION", "LOW_FRICTION")

    out["Entry Allowed"] = out["Trade Action"].eq("ENTRY")
    out["Hold Allowed"] = out["Trade Action"].eq("HOLD")
    out["Exit Now"] = out["Trade Action"].eq("EXIT")
    out["Diff Status"] = out["Dual Engine State"]

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
        .tail(1)[[
            "Symbol",
            "CumsumDiff",
            "TurningDiff",
            "Turning Regime",
            "Dual Engine State",
            "Trade Action",
        ]]
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
        "VWAP Z-Score": float(flow_df["VWAP Z-Score"].iloc[-1]) if not flow_df.empty else float

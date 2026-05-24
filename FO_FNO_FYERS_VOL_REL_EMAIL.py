#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py
Intraday F&O scanner via Fyers API with email alerts.
Complete standalone file ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â no external email.py dependency.
SORTS CANDIDATES BY DIRECTIONAL COLUMN.
ALL STATISTICAL SCORES ARE RAW (ORIGINAL FORMULAS).
"""
import os
import re
import sys
import logging
from datetime import datetime, timedelta, time
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
    "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)

DAILY_LOOKBACK_DAYS = 60
INTRADAY_LOOKBACK_DAYS = 20
IVP_LOOKBACK_DAYS = 252
INDEX_SOFT_BOOST_WEIGHT = 0.25

fyers: Optional[fyersModel.FyersModel] = None

# Email settings (adjust to your environment)
smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
smtp_port = int(os.environ.get("SMTP_PORT", "587"))
sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
sender_password = os.environ.get("SENDER_PASSWORD", "password")
recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")

EMAIL_DISPLAY_COLS = [
    "Symbol", "LTP", "% Change", "Directional", "Turning", "Stability", "Balanced",
    "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal",
    "Bull_Signal", "Bear_Signal", "Overall_Signal", "Price_Lead_Status", "IVP",
    "Volatility State", "Last Iteration Time",
]


def build_signals_from_raw_directional(detail_df) -> dict:
    nan = float('nan')
    out = {k: nan for k in (
        "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal",
        "Bull_Signal", "Bear_Signal", "Overall_Signal")}
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
            client_id=client_id, is_async=False, token=access_token, log_path=""
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
    search_dirs = [root_dir, '.']
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        for dirpath, _, filenames in os.walk(base):
            for fname in filenames:
                if not fname.lower().endswith('.csv'):
                    continue
                path = os.path.join(dirpath, fname)
                try:
                    df = pd.read_csv(path)
                except Exception:
                    continue
                cols = {str(c).strip().lower(): c for c in df.columns}
                sym_col = None
                for key in ['symbol', 'symbols', 'ticker']:
                    if key in cols:
                        sym_col = cols[key]
                        break
                idx_col = None
                for key in ['belongstoindices', 'belongs_to_indices', 'index name', 'index_name', 'sector', 'indices']:
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
                    parts = [p.strip() for p in re.split(r'[|;,/]+', raw_idx) if p.strip()]
                    if not parts:
                        parts = [raw_idx]
                    current = mapping.setdefault(sym, [])
                    for part in parts:
                        if part not in current:
                            current.append(part)
    return mapping


def resolve_universe_csv() -> str:
    csv_files = []
    for dirpath, _, filenames in os.walk('.'):
        for fname in filenames:
            if fname.lower().endswith('.csv'):
                csv_files.append(os.path.join(dirpath, fname))
    scored = []
    for path in csv_files:
        try:
            df = pd.read_csv(path, nrows=5)
            cols = {str(c).strip().lower() for c in df.columns}
            score = 0
            if 'symbol' in cols:
                score += 5
            if 'belongstoindices' in cols or 'belongs_to_indices' in cols:
                score += 5
            if 'company name' in cols or 'company_name' in cols:
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
    return 'fno_stock_list.csv'


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
    cols = ["Cumulative KER", "Cumulative +DI", "Cumulative -DI", "Cumulative ADX", "Survival Score", "Survival_Num"]
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
            "time", "range_expansion", "volume_expansion", "delta_expansion",
            "price_leading_flag", "price_lead_streak", "Price_Lead_Status",
        ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["range"] = (df["high"] - df["low"]).clip(lower=0.0)
    df["avg_range_5"] = df["range"].rolling(5, min_periods=3).mean()
    df["range_expansion"] = np.where(df["avg_range_5"] > 0, df["range"] / df["avg_range_5"], np.nan)
    df["avg_vol_5"] = df["volume"].rolling(5, min_periods=3).mean()
    df["volume_expansion"] = np.where(df["avg_vol_5"] > 0, df["volume"] / df["avg_vol_5"], np.nan)
    df["mid"] = (df["high"] + df["low"]) / 2.0
    df["delta"] = np.where(df["close"] > df["mid"], df["volume"], np.where(df["close"] < df["mid"], -df["volume"], 0.0))
    df["cvd"] = pd.Series(df["delta"], index=df.index).cumsum()
    df["cvd_change"] = df["cvd"].diff().abs().fillna(0.0)
    df["avg_cvd_change_5"] = df["cvd_change"].rolling(5, min_periods=3).mean()
    df["delta_expansion"] = np.where(df["avg_cvd_change_5"] > 0, df["cvd_change"] / df["avg_cvd_change_5"], np.nan)
    directional_bar = (df["close"] > df["open"]) | (df["close"] < df["open"])
    df["price_leading_flag"] = (
        (df["range_expansion"] >= 1.5) & (df["volume_expansion"] <= 1.0) &
        (df["delta_expansion"] <= 1.0) & directional_bar
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
        "time", "range_expansion", "volume_expansion", "delta_expansion",
        "price_leading_flag", "price_lead_streak", "Price_Lead_Status",
    ]]


def price_stats_from_series(prices: pd.Series) -> dict:
    p = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if len(p) < 3:
        return {"Directional": np.nan, "Turning": np.nan, "Stability": np.nan, "Balanced": np.nan}
    x = np.arange(len(p), dtype=float)
    slope = float(np.polyfit(x, p.values, 1)[0])
    net_move = float(p.iloc[-1] - p.iloc[0])
    turning = float(np.mean(np.abs(np.diff(p.values, n=2))))
    std_p = float(np.std(p.values))
    directional = slope + net_move
    stability = std_p
    balanced = directional - turning + std_p
    return {"Directional": directional, "Turning": turning, "Stability": stability, "Balanced": balanced}


def compute_iteration_volume_profile(intra_df: Optional[pd.DataFrame]) -> Tuple[Dict, pd.DataFrame]:
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
    last_cum_vol = last_rvol10 = last_rvol20 = 0
    for i in range(len(curr_df)):
        total_iters += 1
        row = curr_df.iloc[i]
        t = row["time_only"]
        cum_vol = float(row["cum_vol"])
        h10 = hist_df_10[hist_df_10["time_only"] <= t]
        avg_cum_10 = h10.groupby("date")["volume"].sum().mean() if not h10.empty else 0
        rvol10 = cum_vol / avg_cum_10 if avg_cum_10 > 0 else 0
        h20 = hist_df_20[hist_df_20["time_only"] <= t]
        avg_cum_20 = h20.groupby("date")["volume"].sum().mean() if not h20.empty else 0
        rvol20 = cum_vol / avg_cum_20 if avg_cum_20 > 0 else 0
        dt_time = datetime.combine(current_date, t)
        market_open = datetime.combine(current_date, time(9, 15))
        iter_mins = int((dt_time - market_open).total_seconds() / 60)
        price_series = curr_df["close"].iloc[: i + 1]
        ps = price_stats_from_series(price_series)
        rows.append({
            "Iteration No": total_iters,
            "Iteration Minutes": iter_mins,
            "Iteration Time": t.strftime("%H:%M"),
            "Current Volume": cum_vol,
            "Directional": ps["Directional"],
            "Turning": ps["Turning"],
            "Stability": ps["Stability"],
            "Balanced": ps["Balanced"],
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
    adx_now = float(metric_df["Cumulative ADX"].iloc[-1]) if not metric_df.empty else float("nan")
    ker_now = float(metric_df["Cumulative KER"].iloc[-1]) if not metric_df.empty else float("nan")
    final_ps = price_stats_from_series(curr_df["close"])
    summary = {
        "LTP": ltp, "Directional": final_ps["Directional"], "Turning": final_ps["Turning"],
        "Stability": final_ps["Stability"], "Balanced": final_ps["Balanced"],
        "Current Volume": last_cum_vol, "10 Day Relative Volume": last_rvol10,
        "20 Day Relative Volume": last_rvol20,
        "Cumulative RSI": float(flow_df["Cumulative RSI"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Cumulative OBV": float(flow_df["Cumulative OBV"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Cumulative VWAP": float(flow_df["Cumulative VWAP"].iloc[-1]) if not flow_df.empty else float("nan"),
        "VWAP Z-Score": float(flow_df["VWAP Z-Score"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Total Iterations": total_iters, "Last Iteration Minutes": last_iter_mins,
        "Last Iteration Time": last_iter_time,
        "Cumulative KER": float(metric_df["Cumulative KER"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative +DI": float(metric_df["Cumulative +DI"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative -DI": float(metric_df["Cumulative -DI"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative ADX": float(metric_df["Cumulative ADX"].iloc[-1]) if not metric_df.empty else np.nan,
        "Survival Score": str(metric_df["Survival Score"].iloc[-1]) if not metric_df.empty else "0/0",
        "Survival_Num": float(metric_df["Survival_Num"].iloc[-1]) if not metric_df.empty else 0.0,
        "HOD": hod, "Strike_Distance": strike_distance,
        "Last_5m_Volume": last_5m_volume, "Volume_1h_Avg_5m": vol_1h_avg_5m,
        "OBV_30m_Delta": obv_30m_delta, "RSI_30m_Delta": rsi_30m_delta,
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
        daily_df = get_fyers_history(fyers_sym, resolution="D", days_back=max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
        intra_df = get_fyers_history(fyers_sym, resolution="5", days_back=INTRADAY_LOOKBACK_DAYS)
        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df)
        iv_info = compute_iv_proxies(daily_df)
        prev_close = float(daily_df["close"].iloc[-2]) if (daily_df is not None and len(daily_df) >= 2) else None
        ltp = iter_summary.get("LTP")
        pct_change = ((ltp - prev_close) / prev_close * 100) if (ltp is not None and prev_close and prev_close != 0) else 0.0
        if not iter_detail.empty:
            iter_detail.insert(0, "Symbol", sym)
            iter_detail.insert(1, "% Change", pct_change)
            iteration_rows.append(iter_detail)
        rows.append({
            "Symbol": sym, "LTP": ltp, "% Change": pct_change,
            "Directional": iter_summary.get("Directional"), "Turning": iter_summary.get("Turning"),
            "Stability": iter_summary.get("Stability"), "Balanced": iter_summary.get("Balanced"),
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
            "Cumulative RSI": iter_summary.get("Cumulative RSI"), "Cumulative OBV": iter_summary.get("Cumulative OBV"),
            "Cumulative VWAP": iter_summary.get("Cumulative VWAP"), "VWAP Z-Score": iter_summary.get("VWAP Z-Score"),
            "Total Iterations": iter_summary.get("Total Iterations"),
            "Last Iteration Minutes": iter_summary.get("Last Iteration Minutes"),
            "Last Iteration Time": iter_summary.get("Last Iteration Time"),
            "Cumulative KER": iter_summary.get("Cumulative KER"), "Cumulative +DI": iter_summary.get("Cumulative +DI"),
            "Cumulative -DI": iter_summary.get("Cumulative -DI"), "Cumulative ADX": iter_summary.get("Cumulative ADX"),
            "Survival Score": iter_summary.get("Survival Score"), "Survival_Num": iter_summary.get("Survival_Num"),
            "HOD": iter_summary.get("HOD"), "Strike_Distance": iter_summary.get("Strike_Distance"),
            "Last_5m_Volume": iter_summary.get("Last_5m_Volume"), "Volume_1h_Avg_5m": iter_summary.get("Volume_1h_Avg_5m"),
            "OBV_30m_Delta": iter_summary.get("OBV_30m_Delta"), "RSI_30m_Delta": iter_summary.get("RSI_30m_Delta"),
            "Price_Lead_Status": iter_summary.get("Price_Lead_Status", "NORMAL"),
            "IVP": iv_info.get("IVP"), "Volatility State": iv_info.get("Volatility State"),
        })
    return (pd.DataFrame(rows), (pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame()))


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
        if pd.notna(row.get("Cumulative +DI")) and pd.notna(row.get("Cumulative -DI")) and row.get("Cumulative +DI") > row.get("Cumulative -DI"):
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
        if pd.notna(row.get("Cumulative +DI")) and pd.notna(row.get("Cumulative -DI")) and row.get("Cumulative -DI") > row.get("Cumulative +DI"):
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
        if v > 0: return "#2e7d32"
        if v < 0: return "#7f1d1d"
        return "#4b5563"
    except (ValueError, TypeError):
        pass
    label = str(label).strip()
    if label == "Buyer Zone": return "#33691e"
    if label == "Neutral Vol": return "#4b5563"
    if label == "Avoid Buy Premium": return "#7a5c00"
    return "#374151"


def text_color_for_bg(bg: str) -> str:
    bg = str(bg).lower()
    if bg in {"#4b5563", "#374151", "#7f1d1d", "#a83232", "#b94a48", "#7a5c00", "#33691e", "#2e7d32", "#3f8f45"}:
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


 def buildcandidatetables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(columns=EMAILDISPLAYCOLS), pd.DataFrame(columns=EMAILDISPLAYCOLS)

    base = df.copy()

    for c in [
        "Directional", "Turning", "BullSignal", "BearSignal", "OverallSignal",
        "5mSignal", "Stability", "Change", "Cumulative KER", "SurvivalNum",
        "Cumulative ADX", "Cumulative RSI", "Cumulative DI", "Cumulative -DI",
        "VWAP Z-Score"
    ]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # ratio used only for sorting
    base["_dir_turn_ratio"] = np.where(
        base["Turning"].notna() & (base["Turning"] != 0),
        base["Directional"].abs() / base["Turning"],
        np.nan
    )

    # keep long/short separation same as directional sign
    longdf = base[base["Directional"] > 0].copy() if "Directional" in base.columns else pd.DataFrame()
    shortdf = base[base["Directional"] < 0].copy() if "Directional" in base.columns else pd.DataFrame()

    # pure one-condition sort: Directional / Turning descending
    if not longdf.empty:
        longdf = (
            longdf.sort_values(by="_dir_turn_ratio", ascending=False, na_position="last")
                  .drop_duplicates(subset="Symbol")
                  .head(15)
        )

    if not shortdf.empty:
        shortdf = (
            shortdf.sort_values(by="_dir_turn_ratio", ascending=False, na_position="last")
                   .drop_duplicates(subset="Symbol")
                   .head(15)
        )

    if "_dir_turn_ratio" in longdf.columns:
        longdf = longdf.drop(columns=["_dir_turn_ratio"])
    if "_dir_turn_ratio" in shortdf.columns:
        shortdf = shortdf.drop(columns=["_dir_turn_ratio"])

    if not longdf.empty:
        longdf = longdf.drop(columns=[c for c in ["BearSignal", "OverallSignal"] if c in longdf.columns])
    if not shortdf.empty:
        shortdf = shortdf.drop(columns=[c for c in ["BullSignal", "OverallSignal"] if c in shortdf.columns])

    longcols = [c for c in EMAILDISPLAYCOLS if c in longdf.columns]
    shortcols = [c for c in EMAILDISPLAYCOLS if c in shortdf.columns]

    longdf = longdf[longcols] if not longdf.empty else pd.DataFrame(
        columns=[c for c in EMAILDISPLAYCOLS if c not in ["BearSignal", "OverallSignal"]]
    )
    shortdf = shortdf[shortcols] if not shortdf.empty else pd.DataFrame(
        columns=[c for c in EMAILDISPLAYCOLS if c not in ["BullSignal", "OverallSignal"]]
    )

    return longdf, shortdf           
                
def df_to_html_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return '<p style="color:#cbd5e1;font-family:Arial,sans-serif;">No candidates found.</p>'
    df_slice = df.head(max_rows).copy()
    cols = [c for c in EMAIL_DISPLAY_COLS if c in df_slice.columns]
    if not cols:
        return '<p style="color:#cbd5e1;font-family:Arial,sans-serif;">No candidates found.</p>'
    header_cells = ''.join(
        f'<th style="padding:8px 10px;border:1px solid #4b5563;background:#374151;color:#f9fafb;font-size:12px;font-weight:700;text-align:center;white-space:nowrap;">{c}</th>'
        for c in cols
    )
    rows_html = []
    signal_cols = {
        '5m_Signal', '15m_Signal', '30m_Signal', '60m_Signal',
        'Bull_Signal', 'Bear_Signal', 'Overall_Signal', 'Volatility State', 'Price_Lead_Status'
    }
    for _, row in df_slice.iterrows():
        cells = []
        for c in cols:
            val = format_value(c, row.get(c))
            bg = '#2f3542'
            fg = '#f3f4f6'
            extra = 'text-align:center;'
            if c in signal_cols:
                bg = signal_color(val)
                fg = text_color_for_bg(bg)
            elif c == 'Symbol':
                bg = '#374151'
                fg = '#f9fafb'
                extra = 'text-align:left;font-weight:700;'
            elif c == '% Change':
                try:
                    num = float(str(val).replace('%', ''))
                    if num > 0:
                        bg = '#2e7d32'; fg = '#e8f5e9'
                    elif num < 0:
                        bg = '#a83232'; fg = '#ffebee'
                    else:
                        bg = '#4b5563'; fg = '#f3f4f6'
                except Exception:
                    bg = '#4b5563'; fg = '#f3f4f6'
            elif c in {'LTP', 'IVP'}:
                bg = '#303643'
                fg = '#f3f4f6'
            else:
                bg = '#3b4252'
                fg = '#f3f4f6'
            cells.append(f'<td style="padding:7px 9px;border:1px solid #4b5563;background:{bg};color:{fg};font-size:12px;font-weight:600;white-space:nowrap;{extra}">{val}</td>')
        rows_html.append('<tr>' + ''.join(cells) + '</tr>')
    return ('<div style="overflow-x:auto;margin:10px 0 18px 0;">' '<table style="border-collapse:collapse;background:#1f2937;font-family:Arial,sans-serif;">' f'<thead><tr>{header_cells}</tr></thead>' f'<tbody>{"" .join(rows_html)}</tbody>' '</table></div>')


def send_email_with_tables(
    long_df: pd.DataFrame, short_df: pd.DataFrame,
    csv_filename: str, detail_csv_filename: str,
    index_long_df: pd.DataFrame = None, index_short_df: pd.DataFrame = None,
    index_iter_csv_filename: Optional[str] = None,
) -> bool:
    try:
        if index_long_df is None:
            index_long_df = pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
        if index_short_df is None:
            index_short_df = pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
        scan_time = datetime.now().strftime("%d %b %Y, %H:%M")
        index_long_html = df_to_html_table(index_long_df)
        index_short_html = df_to_html_table(index_short_df)
        stock_long_html = df_to_html_table(long_df)
        stock_short_html = df_to_html_table(short_df)
        html_body = f"""
<html>
  <body style="font-family: Arial, sans-serif; font-size: 13px; background:#111827; color:#e5e7eb;">
    <h2 style="color:#f9fafb;">Intraday Vol Iteration Alert</h2>
    <p style="color:#9ca3af;">Scan completed at {scan_time}.</p>
    <h3 style="color:#34d399;">Index Long Candidates</h3>
    {index_long_html}
    <h3 style="color:#f87171;">Index Short Candidates</h3>
    {index_short_html}
    <h3 style="color:#34d399;">Stock Long Candidates</h3>
    {stock_long_html}
    <h3 style="color:#f87171;">Stock Short Candidates</h3>
    {stock_short_html}
    <p style="color:#6b7280; font-size:11px; margin-top:20px;">Generated by FO_FNO_FYERS_VOL_REL_EMAIL.py</p>
  </body>
</html>
"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Intraday Scan ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â {scan_time}"
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg.attach(MIMEText(html_body, "html", _charset="utf-8"))
        for fname in [csv_filename, detail_csv_filename, index_iter_csv_filename]:
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
    summary_csv, detail_csv = save_outputs(summary_df, detail_df, prefix="fno")
    sent = send_email_with_tables(
        long_df=long_df, short_df=short_df,
        csv_filename=summary_csv, detail_csv_filename=detail_csv,
    )
    if sent:
        logger.info("Scan and email completed.")
    else:
        logger.warning("Scan completed but email failed.")


if __name__ == "__main__":
    main()

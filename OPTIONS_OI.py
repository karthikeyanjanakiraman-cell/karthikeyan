import sys
import subprocess
import os
import site
import logging
from datetime import datetime, timedelta, time
from typing import List, Dict, Optional, Tuple

# Install deps
subprocess.check_call([sys.executable, "-m", "pip", "install", "fyers-apiv3", "pandas", "numpy"])

# Path fixes
for site_package in site.getsitepackages():
    if site_package not in sys.path:
        sys.path.append(site_package)

# Import Fyers
try:
    from fyersapiv3 import fyersModel
except ImportError:
    from fyers_apiv3 import fyersModel

import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger()

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
    "Symbol", "LTP", "% Change", "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal",
    "Bull_Signal", "Bear_Signal", "Overall_Signal", "Price_Lead_Status", "IVP",
    "Volatility State", "Last Iteration Time",
]


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
                    (
                        c
                        for c in df.columns
                        if c.lower() in ["symbol", "symbols", "ticker"]
                    ),
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


def get_fyers_history(
    symbol: str, resolution: str, days_back: int
) -> Optional[pd.DataFrame]:
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

    iv_proxy = ((high - low) / close.replace(0, np.nan) * 100.0).replace(
        [np.inf, -np.inf], np.nan
    )
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
        tr[i] = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )
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
            dxs.append(
                100 * abs(kpdi - kmdi) / (kpdi + kmdi)
                if (kpdi + kmdi) > 0
                else 0.0
            )
        adx = float(np.mean(dxs)) if dxs else np.nan

        if c[i] > c[i - 1]:
            qualified += 1
        length_so_far = i + 1
        survival_ratio = qualified / length_so_far if length_so_far > 0 else 0.0

        out.append(
            [
                cum_ker,
                pdi,
                mdi,
                adx,
                f"{qualified}/{length_so_far}",
                survival_ratio,
            ]
        )

    cols = [
        "Cumulative KER",
        "Cumulative +DI",
        "Cumulative -DI",
        "Cumulative ADX",
        "Survival Score",
        "Survival_Num",
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

    direction = close.diff().fillna(0.0).apply(
        lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    )
    obv = (direction * volume).cumsum()

    typical_price = (high + low + close) / 3.0
    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, float("nan"))
    vwap = (cum_pv / cum_vol).fillna(0.0)

    vwap_variance = (volume * (typical_price - vwap) ** 2).cumsum() / cum_vol
    vwap_std = np.sqrt(vwap_variance).fillna(0.0)
    vwap_z_score = np.where(vwap_std > 0, (close - vwap) / vwap_std, 0.0)

    out = pd.DataFrame(
        {
            "Cumulative RSI": rsi,
            "Cumulative OBV": obv,
            "Cumulative VWAP": vwap,
            "VWAP Z-Score": pd.Series(vwap_z_score, index=df.index).fillna(0.0),
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
                "Price_Lead_Status",
            ]
        )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["range"] = (df["high"] - df["low"]).clip(lower=0.0)
    df["avg_range_5"] = df["range"].rolling(5, min_periods=3).mean()
    df["range_expansion"] = np.where(
        df["avg_range_5"] > 0,
        df["range"] / df["avg_range_5"],
        np.nan,
    )

    df["avg_vol_5"] = df["volume"].rolling(5, min_periods=3).mean()
    df["volume_expansion"] = np.where(
        df["avg_vol_5"] > 0,
        df["volume"] / df["avg_vol_5"],
        np.nan,
    )

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

    df["Price_Lead_Status"] = np.select(
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
            "Price_Lead_Status",
        ]
    ]


def compute_iteration_volume_profile(
    intra_df: Optional[pd.DataFrame],
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

    work_df = curr_df.copy()
    if "time" not in work_df.columns:
        if "timestamp" in work_df.columns:
            work_df["time"] = pd.to_datetime(work_df["timestamp"])
        elif "date" in work_df.columns and "time_only" in work_df.columns:
            work_df["time"] = pd.to_datetime(
                work_df["date"].astype(str) + " " + work_df["time_only"].astype(str)
            )
        else:
            work_df["time"] = pd.RangeIndex(start=0, stop=len(work_df), step=1)

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
    last_cum_vol = last_rvol10 = last_rvol20 = 0

    for i in range(len(curr_df)):
        total_iters += 1
        row = curr_df.iloc[i]
        t = row["time_only"]
        cum_vol = float(row["cum_vol"])

        h10 = hist_df_10[hist_df_10["time_only"] <= t]
        avg_cum_10 = (
            h10.groupby("date")["volume"].sum().mean() if not h10.empty else 0
        )
        rvol10 = cum_vol / avg_cum_10 if avg_cum_10 > 0 else 0

        h20 = hist_df_20[hist_df_20["time_only"] <= t]
        avg_cum_20 = (
            h20.groupby("date")["volume"].sum().mean() if not h20.empty else 0
        )
        rvol20 = cum_vol / avg_cum_20 if avg_cum_20 > 0 else 0

        dt_time = datetime.combine(current_date, t)
        market_open = datetime.combine(current_date, time(9, 15))
        iter_mins = int((dt_time - market_open).total_seconds() / 60)

        rows.append(
            {
                "Iteration No": total_iters,
                "Iteration Minutes": iter_mins,
                "Iteration Time": t.strftime("%H:%M"),
                "Current Volume": cum_vol,
                "10 Day Relative Volume": rvol10,
                "20 Day Relative Volume": rvol20,
                "Cumulative RSI": float(flow_df["Cumulative RSI"].iloc[i])
                if not flow_df.empty
                else float("nan"),
                "Cumulative OBV": float(flow_df["Cumulative OBV"].iloc[i])
                if not flow_df.empty
                else float("nan"),
                "Cumulative VWAP": float(flow_df["Cumulative VWAP"].iloc[i])
                if not flow_df.empty
                else float("nan"),
                "VWAP Z-Score": float(flow_df["VWAP Z-Score"].iloc[i])
                if not flow_df.empty
                else float("nan"),
                "Range_Expansion": float(
                    price_lead_df["range_expansion"].iloc[i]
                )
                if not price_lead_df.empty
                and pd.notna(price_lead_df["range_expansion"].iloc[i])
                else float("nan"),
                "Volume_Expansion": float(
                    price_lead_df["volume_expansion"].iloc[i]
                )
                if not price_lead_df.empty
                and pd.notna(price_lead_df["volume_expansion"].iloc[i])
                else float("nan"),
                "Delta_Expansion": float(
                    price_lead_df["delta_expansion"].iloc[i]
                )
                if not price_lead_df.empty
                and pd.notna(price_lead_df["delta_expansion"].iloc[i])
                else float("nan"),
                "Price_Leading_Flag": bool(
                    price_lead_df["price_leading_flag"].iloc[i]
                )
                if not price_lead_df.empty
                else False,
                "Price_Lead_Streak": int(
                    price_lead_df["price_lead_streak"].iloc[i]
                )
                if not price_lead_df.empty
                else 0,
                "Price_Lead_Status": str(
                    price_lead_df["Price_Lead_Status"].iloc[i]
                )
                if not price_lead_df.empty
                else "NORMAL",
            }
        )

        last_cum_vol, last_rvol10, last_rvol20 = cum_vol, rvol10, rvol20
        last_iter_mins = iter_mins
        last_iter_time = t.strftime("%H:%M")

    detail_df = pd.DataFrame(rows)

    ltp = float(curr_df["close"].iloc[-1]) if not curr_df.empty else np.nan
    hod = float(curr_df["high"].max()) if not curr_df.empty else float("nan")
    strike_distance = (
        (hod - ltp) / hod if hod and hod > 0 and ltp is not None else 1.0
    )

    last_5m_volume = float(curr_df["volume"].iloc[-1]) if not curr_df.empty else 0.0
    recent_12 = curr_df["volume"].tail(12) if not curr_df.empty else pd.Series(dtype=float)
    vol_1h_avg_5m = (
        float(recent_12.mean()) if len(recent_12) > 0 else last_5m_volume
    )

    obv_30m_delta = 0.0
    rsi_30m_delta = 0.0
    if not flow_df.empty and len(flow_df) >= 7:
        obv_30m_delta = float(flow_df["Cumulative OBV"].iloc[-1]) - float(
            flow_df["Cumulative OBV"].iloc[-7]
        )
        rsi_30m_delta = float(flow_df["Cumulative RSI"].iloc[-1]) - float(
            flow_df["Cumulative RSI"].iloc[-7]
        )

    adx_now = float(metric_df["Cumulative ADX"].iloc[-1]) if not metric_df.empty else float("nan")
    ker_now = float(metric_df["Cumulative KER"].iloc[-1]) if not metric_df.empty else float("nan")

    summary = {
        "LTP": ltp,
        "Current Volume": last_cum_vol,
        "10 Day Relative Volume": last_rvol10,
        "20 Day Relative Volume": last_rvol20,
        "Cumulative RSI": float(flow_df["Cumulative RSI"].iloc[-1])
        if not flow_df.empty
        else float("nan"),
        "Cumulative OBV": float(flow_df["Cumulative OBV"].iloc[-1])
        if not flow_df.empty
        else float("nan"),
        "Cumulative VWAP": float(flow_df["Cumulative VWAP"].iloc[-1])
        if not flow_df.empty
        else float("nan"),
        "VWAP Z-Score": float(flow_df["VWAP Z-Score"].iloc[-1])
        if not flow_df.empty
        else float("nan"),
        "Total Iterations": total_iters,
        "Last Iteration Minutes": last_iter_mins,
        "Last Iteration Time": last_iter_time,
        "Cumulative KER": float(metric_df["Cumulative KER"].iloc[-1])
        if not metric_df.empty
        else np.nan,
        "Cumulative +DI": float(metric_df["Cumulative +DI"].iloc[-1])
        if not metric_df.empty
        else np.nan,
        "Cumulative -DI": float(metric_df["Cumulative -DI"].iloc[-1])
        if not metric_df.empty
        else np.nan,
        "Cumulative ADX": float(metric_df["Cumulative ADX"].iloc[-1])
        if not metric_df.empty
        else np.nan,
        "Survival Score": str(metric_df["Survival Score"].iloc[-1])
        if not metric_df.empty
        else "0/0",
        "Survival_Num": float(metric_df["Survival_Num"].iloc[-1])
        if not metric_df.empty
        else 0.0,
        "HOD": hod,
        "Strike_Distance": strike_distance,
        "Last_5m_Volume": last_5m_volume,
        "Volume_1h_Avg_5m": vol_1h_avg_5m,
        "OBV_30m_Delta": obv_30m_delta,
        "RSI_30m_Delta": rsi_30m_delta,
        "Price_Lead_Status": str(price_lead_df["Price_Lead_Status"].iloc[-1])
        if not price_lead_df.empty
        else "NORMAL",
    }

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
            days_back=max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS),
        )
        intra_df = get_fyers_history(
            fyers_sym,
            resolution="5",
            days_back=INTRADAY_LOOKBACK_DAYS,
        )

        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df)
        iv_info = compute_iv_proxies(daily_df)

        prev_close = float(daily_df["close"].iloc[-2]) if (
            daily_df is not None and len(daily_df) >= 2
        ) else None
        ltp = iter_summary.get("LTP")
        pct_change = (
            (ltp - prev_close) / prev_close * 100
            if (ltp is not None and prev_close and prev_close != 0)
            else 0.0
        )

        if not iter_detail.empty:
            iter_detail.insert(0, "Symbol", sym)
            iter_detail.insert(1, "% Change", pct_change)
            iteration_rows.append(iter_detail)

        rows.append(
            {
                "Symbol": sym,
                "LTP": ltp,
                "% Change": pct_change,
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
                "Cumulative +DI": iter_summary.get("Cumulative +DI"),
                "Cumulative -DI": iter_summary.get("Cumulative -DI"),
                "Cumulative ADX": iter_summary.get("Cumulative ADX"),
                "Survival Score": iter_summary.get("Survival Score"),
                "Survival_Num": iter_summary.get("Survival_Num"),
                "HOD": iter_summary.get("HOD"),
                "Strike_Distance": iter_summary.get("Strike_Distance"),
                "Last_5m_Volume": iter_summary.get("Last_5m_Volume"),
                "Volume_1h_Avg_5m": iter_summary.get("Volume_1h_Avg_5m"),
                "OBV_30m_Delta": iter_summary.get("OBV_30m_Delta"),
                "RSI_30m_Delta": iter_summary.get("RSI_30m_Delta"),
                "Price_Lead_Status": iter_summary.get("Price_Lead_Status", "NORMAL"),
                "IVP": iv_info.get("IVP"),
                "Volatility State": iv_info.get("Volatility State"),
            }
        )

    return (
        pd.DataFrame(rows),
        (pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame()),
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
    if df is None or df.empty:
        return df
    out = df.copy()
    for tf in ["5m", "15m", "30m", "60m"]:
        col = f"{tf}RankDelta"
        out[f"{tf}_Signal"] = out[col].apply(rank_delta_to_label) if col in out.columns else ""
    out["Bull_Signal"] = out["Bull Rank"].apply(rank_delta_to_label) if "Bull Rank" in out.columns else ""
    out["Bear_Signal"] = out["Bear Rank"].apply(lambda x: rank_delta_to_label(-x)) if "Bear Rank" in out.columns else ""
    out["Overall_Signal"] = out["Rank Delta"].apply(rank_delta_to_label) if "Rank Delta" in out.columns else ""
    return out


def signal_color(label: str) -> str:
    label = str(label).strip()
    if label == "Buy++":
        return "#2e7d32"
    if label == "Buy+":
        return "#3f8f45"
    if label == "Buy":
        return "#5b9b5f"
    if label == "Buyer Zone":
        return "#33691e"
    if label == "Neutral":
        return "#4b5563"
    if label == "Neutral Vol":
        return "#4b5563"
    if label == "Sell++":
        return "#7f1d1d"
    if label == "Sell+":
        return "#a83232"
    if label == "Sell":
        return "#b94a48"
    if label == "Avoid Buy Premium":
        return "#7a5c00"
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


def build_candidate_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(columns=EMAIL_DISPLAY_COLS), pd.DataFrame(columns=EMAIL_DISPLAY_COLS)

    base = df.copy()

    strict_long = base[
        (base["% Change"] > 0)
        & (base["Cumulative +DI"] > base["Cumulative -DI"])
    ].copy()
    strict_short = base[
        (base["% Change"] < 0)
        & (base["Cumulative -DI"] > base["Cumulative +DI"])
    ].copy()

    long_df = strict_long.sort_values(
        by=["Cumulative KER", "Survival_Num", "Cumulative ADX", "% Change"],
        ascending=[False, False, False, False],
        na_position="last",
    ).drop_duplicates(subset=["Symbol"]).head(15)

    short_df = strict_short.sort_values(
        by=["Cumulative KER", "Survival_Num", "Cumulative ADX", "% Change"],
        ascending=[False, False, False, True],
        na_position="last",
    ).drop_duplicates(subset=["Symbol"]).head(15)

    long_df = (
        long_df[[c for c in EMAIL_DISPLAY_COLS if c in long_df.columns]]
        if not long_df.empty
        else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    )
    short_df = (
        short_df[[c for c in EMAIL_DISPLAY_COLS if c in short_df.columns]]
        if not short_df.empty
        else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    )

    return long_df, short_df


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
                        bg = '#2e7d32'
                        fg = '#e8f5e9'
                    elif num < 0:
                        bg = '#a83232'
                        fg = '#ffebee'
                    else:
                        bg = '#4b5563'
                        fg = '#f3f4f6'
                except Exception:
                    bg = '#4b5563'
                    fg = '#f3f4f6'
            elif c in {'LTP', 'IVP'}:
                bg = '#303643'
                fg = '#f3f4f6'
            else:
                bg = '#3b4252'
                fg = '#f3f4f6'
            cells.append(f'<td style="padding:7px 9px;border:1px solid #4b5563;background:{bg};color:{fg};font-size:12px;font-weight:600;white-space:nowrap;{extra}">{val}</td>')
        rows_html.append('<tr>' + ''.join(cells) + '</tr>')
    return ('<div style="overflow-x:auto;margin:10px 0 18px 0;">' '<table style="border-collapse:collapse;background:#1f2937;font-family:Arial,sans-serif;">' f'<thead><tr>{header_cells}</tr></thead>' f'<tbody>{"".join(rows_html)}</tbody>' '</table></div>')

def send_email_with_tables(
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    csv_filename: str,
    detail_csv_filename: str,
    index_long_df: pd.DataFrame = None,
    index_short_df: pd.DataFrame = None,
    index_iter_csv_filename: Optional[str] = None,
) -> bool:
    """Send email with index and stock long/short tables and attach CSVs."""
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
  <body style="font-family: Arial, sans-serif; font-size: 13px;">
    <h2>Intraday Vol Iteration Alert</h2>
    <p>Scan completed at {scan_time}.</p>

    <h3>Index Long Candidates</h3>
    {index_long_html}

    <h3>Index Short Candidates</h3>
    {index_short_html}

    <h3>Stock Long Candidates (from long indices)</h3>
    {stock_long_html}

    <h3>Stock Short Candidates (from short indices)</h3>
    {stock_short_html}

    <p>Attached CSVs: summary &amp; intraday iterations.</p>
  </body>
</html>
"""

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = (
            f"Intraday Vol Iteration Alert - {datetime.now().strftime('%d %b %H:%M')}"
        )
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        for filename in [csv_filename, detail_csv_filename, index_iter_csv_filename]:
            if not filename or not isinstance(filename, (str, bytes, os.PathLike)):
                continue
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename={os.path.basename(filename)}",
                )
                msg.attach(part)

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=40) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=40) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender_email, sender_password)
                server.send_message(msg)

        logger.info(f"EMAIL Sent successfully to {recipient_email}")
        return True
    except Exception as e:
        logger.error(f"EMAIL Failed to send email: {type(e).__name__}: {e}")
        return False



##############################
# INDEX-FIRST EXTENSIONS
##############################

def normalize_index_name(x: str) -> str:
    s = str(x).strip().upper()
    s = s.replace('-INDEX', '')
    s = s.replace('&', ' AND ')
    s = re.sub(r'\s+', ' ', s).strip()
    compact_map = {
        'NIFTY50': 'NIFTY 50',
        'NIFTYNEXT50': 'NIFTY NEXT 50',
        'NIFTYBANK': 'NIFTY BANK',
        'NIFTYAUTO': 'NIFTY AUTO',
        'NIFTYENERGY': 'NIFTY ENERGY',
        'NIFTYIT': 'NIFTY IT',
        'NIFTYPHARMA': 'NIFTY PHARMA',
        'NIFTYMETAL': 'NIFTY METAL',
        'NIFTYFMCG': 'NIFTY FMCG',
        'NIFTYREALTY': 'NIFTY REALTY',
        'NIFTYPSUBANK': 'NIFTY PSU BANK',
        'NIFTYPRIVATEBANK': 'NIFTY PRIVATE BANK',
        'NIFTYFINANCIALSERVICES': 'NIFTY FINANCIAL SERVICES',
        'NIFTYFINSERVICE': 'NIFTY FINANCIAL SERVICES',
        'NIFTYOILGAS': 'NIFTY OIL & GAS',
    }
    key = s.replace(' ', '')
    return compact_map.get(key, s)


def format_fyers_index_symbol(symbol: str) -> str:
    raw = str(symbol).strip()
    if raw.startswith('NSE:'):
        return raw
    named = normalize_index_name(raw)
    reverse_map = {
        'NIFTY 50': 'NIFTY50-INDEX',
        'NIFTY NEXT 50': 'NIFTYNEXT50-INDEX',
        'NIFTY BANK': 'NIFTYBANK-INDEX',
        'NIFTY AUTO': 'NIFTYAUTO-INDEX',
        'NIFTY ENERGY': 'NIFTYENERGY-INDEX',
        'NIFTY IT': 'NIFTYIT-INDEX',
        'NIFTY PHARMA': 'NIFTYPHARMA-INDEX',
        'NIFTY METAL': 'NIFTYMETAL-INDEX',
        'NIFTY FMCG': 'NIFTYFMCG-INDEX',
        'NIFTY REALTY': 'NIFTYREALTY-INDEX',
        'NIFTY PSU BANK': 'NIFTYPSUBANK-INDEX',
        'NIFTY PRIVATE BANK': 'NIFTYPVTBANK-INDEX',
        'NIFTY FINANCIAL SERVICES': 'NIFTYFINSERVICE-INDEX',
        'NIFTY OIL & GAS': 'NIFTYOILGAS-INDEX',
    }
    mapped = reverse_map.get(named, raw if raw.endswith('-INDEX') else raw)
    return mapped if mapped.startswith('NSE:') else f'NSE:{mapped}'


def discover_csv_files() -> list:
    csvs = []
    for base in ['.', 'sectors']:
        if os.path.isdir(base):
            for dirpath, _, filenames in os.walk(base):
                for fname in filenames:
                    if fname.lower().endswith('.csv'):
                        csvs.append(os.path.join(dirpath, fname))
    return sorted(set(csvs))


def load_index_symbols() -> List[str]:
    csv_files = discover_csv_files()
    index_values = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
            norm_cols = {str(c).strip().lower().replace(' ', '').replace('_', ''): c for c in df.columns}
            idx_cols = [norm_cols[k] for k in ['belongstoindices', 'belongstoindex', 'indices'] if k in norm_cols]
            if idx_cols:
                for col in idx_cols:
                    for cell in df[col].dropna().astype(str):
                        parts = [normalize_index_name(p) for p in re.split(r'[,;/|]+', cell) if str(p).strip()]
                        index_values.extend(parts)
            elif 'index' in norm_cols:
                index_values.extend([normalize_index_name(x) for x in df[norm_cols['index']].dropna().astype(str)])
        except Exception:
            continue
    index_values = [x for x in index_values if x and x != 'UNMAPPED SECTORAL']
    index_values = list(dict.fromkeys(index_values))
    if not index_values:
        raise ValueError('No index values found in any CSV file (including sectors folder and mapping CSVs)')
    logger.info(f"INDEX Loaded index universe from CSV content ({len(index_values)}): {index_values}")
    return index_values


def resolve_mapping_csv() -> str:
    candidates = []
    for path in discover_csv_files():
        try:
            df = pd.read_csv(path)
            norm_cols = {str(c).strip().lower().replace(' ', '').replace('_', ''): c for c in df.columns}
            if 'symbol' in norm_cols and 'belongstoindices' in norm_cols:
                candidates.append((len(df), path))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError('No mapping CSV found with Symbol and Belongs_To_Indices')
    candidates.sort(reverse=True)
    chosen = candidates[0][1]
    logger.info(f"CSV Mapping-selected file: {chosen}")
    return chosen




def filter_stock_df_by_index_membership(stock_df: pd.DataFrame, index_symbols: List[str]) -> pd.DataFrame:
    if stock_df is None or stock_df.empty or not index_symbols:
        return stock_df.iloc[0:0].copy() if isinstance(stock_df, pd.DataFrame) else pd.DataFrame()
    csv_path = resolve_mapping_csv()
    map_df = pd.read_csv(csv_path)
    norm_cols = {str(c).strip().lower().replace(' ', '').replace('_', ''): c for c in map_df.columns}
    symbol_col = norm_cols['symbol']
    idx_col = norm_cols['belongstoindices']
    wanted = {normalize_index_name(x) for x in index_symbols if str(x).strip()}

    symbol_to_indices = {}
    for _, row in map_df[[symbol_col, idx_col]].dropna(subset=[symbol_col]).iterrows():
        sym = str(row[symbol_col]).strip()
        parts = [normalize_index_name(p) for p in re.split(r'[,;/|]+', str(row[idx_col])) if str(p).strip()]
        symbol_to_indices[sym] = set(parts)

    keep = []
    for sym in stock_df['Symbol'].astype(str):
        keep.append(bool(symbol_to_indices.get(sym, set()) & wanted))
    return stock_df.loc[keep].copy()

def load_fno_symbols_for_indices(active_index_symbols: List[str]) -> List[str]:
    csv_path = resolve_mapping_csv()
    df = pd.read_csv(csv_path)
    norm_cols = {str(c).strip().lower().replace(' ', '').replace('_', ''): c for c in df.columns}
    symbol_col = norm_cols['symbol']
    idx_col = norm_cols['belongstoindices']
    wanted = {normalize_index_name(x) for x in active_index_symbols if str(x).strip()}
    logger.info(f"FNO Matching selected indices: {sorted(wanted)}")
    def row_matches(cell) -> bool:
        parts = [normalize_index_name(p) for p in re.split(r'[,;/|]+', str(cell)) if str(p).strip()]
        return any(p in wanted for p in parts)
    filt = df[df[idx_col].apply(row_matches)].copy()
    symbols = sorted(filt[symbol_col].dropna().astype(str).str.strip().unique())
    logger.info(f"FNO Filtered {len(symbols)} stocks from {csv_path} using {idx_col}")
    return symbols


def scan_symbol_universe(symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not symbols:
        return pd.DataFrame(), pd.DataFrame()
    rows, iteration_rows = [], []
    total = len(symbols)
    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"CORE [{idx}/{total}] Processing {sym}")
        fyers_sym = format_fyers_symbol(sym)
        daily_df = get_fyers_history(fyers_sym, resolution='D', days_back=max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
        intra_df = get_fyers_history(fyers_sym, resolution='5', days_back=INTRADAY_LOOKBACK_DAYS)
        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df)
        iv_info = compute_iv_proxies(daily_df)
        prev_close = float(daily_df['close'].iloc[-2]) if (daily_df is not None and len(daily_df) >= 2 and 'close' in daily_df.columns) else None
        ltp = iter_summary.get('LTP')
        pct_change = ((float(ltp) - prev_close) / prev_close * 100.0) if (ltp is not None and prev_close is not None and prev_close != 0) else 0.0
        if not iter_detail.empty:
            iter_detail.insert(0, 'Symbol', sym)
            iter_detail.insert(1, '% Change', pct_change)
            iteration_rows.append(iter_detail)
        rows.append({
            'Symbol': sym,
            'LTP': ltp,
            '% Change': pct_change,
            'Current Volume': iter_summary.get('Current Volume'),
            '10 Day Relative Volume': iter_summary.get('10 Day Relative Volume'),
            '20 Day Relative Volume': iter_summary.get('20 Day Relative Volume'),
            'Cumulative RSI': iter_summary.get('Cumulative RSI'),
            'Cumulative OBV': iter_summary.get('Cumulative OBV'),
            'Cumulative VWAP': iter_summary.get('Cumulative VWAP'),
            'VWAP Z-Score': iter_summary.get('VWAP Z-Score'),
            'Total Iterations': iter_summary.get('Total Iterations'),
            'Last Iteration Minutes': iter_summary.get('Last Iteration Minutes'),
            'Last Iteration Time': iter_summary.get('Last Iteration Time'),
            'Cumulative KER': iter_summary.get('Cumulative KER'),
            'Cumulative +DI': iter_summary.get('Cumulative +DI'),
            'Cumulative -DI': iter_summary.get('Cumulative -DI'),
            'Cumulative ADX': iter_summary.get('Cumulative ADX'),
            'Survival Score': iter_summary.get('Survival Score'),
            'Survival_Num': iter_summary.get('Survival_Num'),
            'HOD': iter_summary.get('HOD'),
            'Strike_Distance': iter_summary.get('Strike_Distance'),
            'Last_5m_Volume': iter_summary.get('Last_5m_Volume'),
            'Volume_1h_Avg_5m': iter_summary.get('Volume_1h_Avg_5m'),
            'OBV_30m_Delta': iter_summary.get('OBV_30m_Delta'),
            'RSI_30m_Delta': iter_summary.get('RSI_30m_Delta'),
            'Price_Lead_Status': iter_summary.get('Price_Lead_Status', 'NORMAL'),
            'IVP': iv_info.get('IVP'),
            'Volatility State': iv_info.get('Volatility State'),
        })
    return pd.DataFrame(rows), (pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame())


def build_index_strength_maps(index_long_df: pd.DataFrame, index_short_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
    long_map, short_map = {}, {}
    if index_long_df is not None and not index_long_df.empty and 'Symbol' in index_long_df.columns:
        work = index_long_df.copy().reset_index(drop=True)
        base_n = max(len(work), 1)
        for i, row in work.iterrows():
            sym = normalize_index_name(row.get('Symbol', ''))
            strength = float(base_n - i)
            if pd.notna(row.get('Bull Rank')):
                strength += float(row.get('Bull Rank', 0))
            if pd.notna(row.get('Rank Delta')) and float(row.get('Rank Delta', 0)) > 0:
                strength += float(row.get('Rank Delta', 0))
            if sym:
                long_map[sym] = strength
    if index_short_df is not None and not index_short_df.empty and 'Symbol' in index_short_df.columns:
        work = index_short_df.copy().reset_index(drop=True)
        base_n = max(len(work), 1)
        for i, row in work.iterrows():
            sym = normalize_index_name(row.get('Symbol', ''))
            strength = float(base_n - i)
            if pd.notna(row.get('Bear Rank')):
                strength += float(row.get('Bear Rank', 0))
            rank_delta = row.get('Rank Delta')
            if pd.notna(rank_delta) and float(rank_delta) < 0:
                strength += abs(float(rank_delta))
            if sym:
                short_map[sym] = strength
    return long_map, short_map

def apply_soft_index_boost(stock_df: pd.DataFrame, target_index_symbols: List[str], strength_map: Dict[str, float], direction: str) -> pd.DataFrame:
    if stock_df is None or stock_df.empty:
        return stock_df
    out = stock_df.copy()
    csv_path = resolve_mapping_csv()
    map_df = pd.read_csv(csv_path)
    norm_cols = {str(c).strip().lower().replace(' ', '').replace('_', ''): c for c in map_df.columns}
    symbol_col = norm_cols['symbol']
    idx_col = norm_cols['belongstoindices']
    target_set = {normalize_index_name(x) for x in target_index_symbols if str(x).strip()}
    symbol_to_indices = {}
    for _, row in map_df[[symbol_col, idx_col]].dropna(subset=[symbol_col]).iterrows():
        sym = str(row[symbol_col]).strip()
        parts = [normalize_index_name(p) for p in re.split(r'[,;/|]+', str(row[idx_col])) if str(p).strip()]
        symbol_to_indices[sym] = set(parts)
    boosts = []
    matched_indices = []
    for sym in out['Symbol'].astype(str):
        member_indices = symbol_to_indices.get(sym, set())
        relevant = member_indices & target_set
        if relevant:
            best_idx = max(relevant, key=lambda x: strength_map.get(x, 0.0))
            boost = float(strength_map.get(best_idx, 0.0)) * float(INDEX_SOFT_BOOST_WEIGHT)
            matched_indices.append(best_idx)
        else:
            best_idx = ''
            boost = 0.0
            matched_indices.append(best_idx)
        boosts.append(boost)
    out['Mapped_Index_For_Boost'] = matched_indices
    out['Index_Boost'] = boosts
    if direction.lower() == 'long':
        base_col = 'Bull Rank' if 'Bull Rank' in out.columns else None
        boosted_col = 'Bull Rank Boosted'
        sort_cols = [boosted_col, 'Cumulative KER', 'Survival_Num', 'Cumulative ADX', '% Change']
        ascending = [False, False, False, False, False]
    else:
        base_col = 'Bear Rank' if 'Bear Rank' in out.columns else None
        boosted_col = 'Bear Rank Boosted'
        sort_cols = [boosted_col, 'Cumulative KER', 'Survival_Num', 'Cumulative ADX', '% Change']
        ascending = [False, False, False, False, True]
    if base_col is not None:
        out[boosted_col] = pd.to_numeric(out[base_col], errors='coerce').fillna(0.0) + pd.to_numeric(out['Index_Boost'], errors='coerce').fillna(0.0)
    else:
        out[boosted_col] = pd.to_numeric(out['Index_Boost'], errors='coerce').fillna(0.0)
    out = out.sort_values(by=sort_cols, ascending=ascending, na_position='last').drop_duplicates(subset=['Symbol'])
    return out

def scan_index_universe() -> pd.DataFrame:
    symbols = load_index_symbols()
    rows = []
    for sym in symbols:
        fyers_sym = format_fyers_index_symbol(sym)
        daily_df = get_fyers_history(fyers_sym, resolution='D', days_back=max(DAILY_LOOKBACK_DAYS, IVP_LOOKBACK_DAYS))
        intra_df = get_fyers_history(fyers_sym, resolution='5', days_back=INTRADAY_LOOKBACK_DAYS)
        iter_summary, _ = compute_iteration_volume_profile(intra_df)
        iv_info = compute_iv_proxies(daily_df)
        prev_close = float(daily_df['close'].iloc[-2]) if (daily_df is not None and len(daily_df) >= 2 and 'close' in daily_df.columns) else None
        ltp = iter_summary.get('LTP')
        pct_change = ((float(ltp) - prev_close) / prev_close * 100.0) if (ltp is not None and prev_close is not None and prev_close != 0) else 0.0
        rows.append({
            'Symbol': normalize_index_name(sym),
            'LTP': ltp,
            '% Change': pct_change,
            'IVP': iv_info.get('IVP'),
            'Volatility State': iv_info.get('Volatility State'),
            'Last Iteration Time': iter_summary.get('Last Iteration Time', ''),
            'Cumulative KER': iter_summary.get('Cumulative KER', np.nan),
            'Cumulative +DI': iter_summary.get('Cumulative +DI', np.nan),
            'Cumulative -DI': iter_summary.get('Cumulative -DI', np.nan),
            'Cumulative ADX': iter_summary.get('Cumulative ADX', np.nan),
            'Survival_Num': iter_summary.get('Survival_Num', 0.0),
            'Price_Lead_Status': iter_summary.get('Price_Lead_Status', 'NORMAL'),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = add_signal_columns(derive_rank_columns(df))
    return df






def build_index_iteration_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame()

    work = detail_df.copy()
    required = [
        "Symbol", "% Change", "Iteration No", "Iteration Minutes", "Iteration Time",
        "Current Volume", "10 Day Relative Volume", "20 Day Relative Volume",
        "Cumulative RSI", "Cumulative OBV", "Cumulative VWAP", "VWAP Z-Score",
        "Range_Expansion", "Volume_Expansion", "Delta_Expansion",
        "Price_Leading_Flag", "Price_Lead_Streak",
    ]
    missing = [c for c in required if c not in work.columns]
    if missing:
        logger.warning(f"INDEX Missing columns for index detail build: {missing}")
        return pd.DataFrame()

    symbol_to_indices = load_symbol_to_indices_map("sectors")
    if not symbol_to_indices:
        logger.warning("INDEX No symbol-to-index mapping found from sectors CSVs.")
        return pd.DataFrame()

    work["Symbol"] = work["Symbol"].astype(str).str.strip().str.upper()
    work["Index Name"] = work["Symbol"].map(symbol_to_indices)
    work = work[work["Index Name"].notna()].copy()
    if work.empty:
        logger.warning("INDEX No mapped symbols found for sectoral index detail build.")
        return pd.DataFrame()

    work = work.explode("Index Name").reset_index(drop=True)
    work["Index Name"] = work["Index Name"].astype(str).str.strip()

    numeric_cols = [
        "% Change", "Iteration No", "Iteration Minutes", "Current Volume",
        "10 Day Relative Volume", "20 Day Relative Volume", "Cumulative RSI",
        "Cumulative OBV", "Cumulative VWAP", "VWAP Z-Score",
        "Range_Expansion", "Volume_Expansion", "Delta_Expansion",
        "Price_Lead_Streak",
    ]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work["Price_Leading_Flag"] = work["Price_Leading_Flag"].astype(str).str.lower().isin(["true", "1", "yes"])

    group_cols = ["Index Name", "Iteration No", "Iteration Minutes", "Iteration Time"]
    agg_spec = {
        "% Change": "mean",
        "Current Volume": "sum",
        "10 Day Relative Volume": "mean",
        "20 Day Relative Volume": "mean",
        "Cumulative RSI": "mean",
        "Cumulative OBV": "sum",
        "Cumulative VWAP": "mean",
        "VWAP Z-Score": "mean",
        "Range_Expansion": "mean",
        "Volume_Expansion": "mean",
        "Delta_Expansion": "mean",
        "Price_Leading_Flag": "sum",
        "Price_Lead_Streak": "max",
    }
    agg_spec = {k: v for k, v in agg_spec.items() if k in work.columns}

    out = work.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
    out["Symbol"] = out["Index Name"]
    out["Bucket"] = "INDEX"
    out["Price_Leading_Flag"] = out["Price_Leading_Flag"].fillna(0).astype(int) > 0

    def _lead_status(row):
        streak = float(row.get("Price_Lead_Streak", 0) or 0)
        flag = bool(row.get("Price_Leading_Flag", False))
        if flag and streak >= 3:
            return "STRONG_PRICE_LEAD_FADE"
        if flag and streak >= 2:
            return "PRICE_LEADING_FADE_RISK"
        if flag:
            return "EARLY_PRICE_LEAD"
        return "NORMAL"

    out["Price_Lead_Status"] = out.apply(_lead_status, axis=1)

    final_cols = [
        "Symbol", "Index Name", "% Change", "Iteration No", "Iteration Minutes", "Iteration Time",
        "Current Volume", "10 Day Relative Volume", "20 Day Relative Volume",
        "Cumulative RSI", "Cumulative OBV", "Cumulative VWAP", "VWAP Z-Score",
        "Range_Expansion", "Volume_Expansion", "Delta_Expansion",
        "Price_Leading_Flag", "Price_Lead_Streak", "Price_Lead_Status", "Bucket",
    ]
    for col in final_cols:
        if col not in out.columns:
            out[col] = np.nan

    out = out[final_cols].sort_values(["Index Name", "Iteration No", "Iteration Minutes"]).reset_index(drop=True)
    return out

def main_index_first():
    logger.info('Starting Index-first FO Iteration Volume Volatility Scan')
    init_fyers()

    df_indices = scan_index_universe()
    if df_indices.empty:
        raise ValueError('Index scan returned empty dataframe')

    df_indices = add_signal_columns(derive_rank_columns(df_indices))
    index_long_df, index_short_df = build_candidate_tables(df_indices)

    long_index_symbols = index_long_df['Symbol'].dropna().astype(str).tolist() if not index_long_df.empty and 'Symbol' in index_long_df.columns else []
    short_index_symbols = index_short_df['Symbol'].dropna().astype(str).tolist() if not index_short_df.empty and 'Symbol' in index_short_df.columns else []

    logger.info(f'Long index symbols ({len(long_index_symbols)}): {long_index_symbols}')
    logger.info(f'Short index symbols ({len(short_index_symbols)}): {short_index_symbols}')

    union_stock_symbols = sorted(set(load_fno_symbols_for_indices(long_index_symbols + short_index_symbols)))
    df_all, df_iter = scan_symbol_universe(union_stock_symbols)
    if not df_all.empty:
        df_all = add_signal_columns(derive_rank_columns(df_all))

    long_stock_source = filter_stock_df_by_index_membership(df_all, long_index_symbols)
    short_stock_source = filter_stock_df_by_index_membership(df_all, short_index_symbols)
    index_long_strength_map, index_short_strength_map = build_index_strength_maps(index_long_df, index_short_df)
    long_stock_source = apply_soft_index_boost(long_stock_source, long_index_symbols, index_long_strength_map, direction='long') if not long_stock_source.empty else long_stock_source
    short_stock_source = apply_soft_index_boost(short_stock_source, short_index_symbols, index_short_strength_map, direction='short') if not short_stock_source.empty else short_stock_source
    long_df = long_stock_source.head(15).copy() if not long_stock_source.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    short_df = short_stock_source.head(15).copy() if not short_stock_source.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    long_df = filter_stock_df_by_index_membership(long_df, long_index_symbols) if not long_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    short_df = filter_stock_df_by_index_membership(short_df, short_index_symbols) if not short_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    long_df = long_df[[c for c in EMAIL_DISPLAY_COLS if c in long_df.columns]] if not long_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    short_df = short_df[[c for c in EMAIL_DISPLAY_COLS if c in short_df.columns]] if not short_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)

    index_long_df = index_long_df[[c for c in EMAIL_DISPLAY_COLS if c in index_long_df.columns]] if not index_long_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    index_short_df = index_short_df[[c for c in EMAIL_DISPLAY_COLS if c in index_short_df.columns]] if not index_short_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)

    summary_parts = []
    if not index_long_df.empty:
        summary_parts.append(index_long_df.assign(Bucket='INDEX_LONG'))
    if not index_short_df.empty:
        summary_parts.append(index_short_df.assign(Bucket='INDEX_SHORT'))
    if not long_df.empty:
        summary_parts.append(long_df.assign(Bucket='STOCK_LONG_FROM_LONG_INDICES'))
    if not short_df.empty:
        summary_parts.append(short_df.assign(Bucket='STOCK_SHORT_FROM_SHORT_INDICES'))
    summary_df = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()

    detail_df = df_iter.copy() if isinstance(df_iter, pd.DataFrame) else pd.DataFrame()
    if not detail_df.empty:
        detail_df['Bucket'] = detail_df['Symbol'].astype(str).map(
            lambda s: 'LONGINDEXSTOCKS' if s in set(long_df['Symbol'].astype(str)) else ('SHORTINDEXSTOCKS' if s in set(short_df['Symbol'].astype(str)) else 'OTHER')
        )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_csv = f'fo_idx_filtered_summary_{timestamp}.csv'
    detail_csv = f'fo_idx_filtered_details_{timestamp}.csv'
    summary_df.to_csv(summary_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)
    index_iter_csv = None
    index_iter_df = build_index_iteration_summary(detail_df) if not detail_df.empty else pd.DataFrame()
    if isinstance(index_iter_df, pd.DataFrame) and not index_iter_df.empty:
        index_iter_csv = f'fo_idx_iteration_summary_{timestamp}.csv'
        index_iter_df.to_csv(index_iter_csv, index=False)
        logger.info(f'INDEX Iteration summary saved: {index_iter_csv}')


    # Forced Scan Injection
    scan_options_logic(long_df["Symbol"].tolist(), short_df["Symbol"].tolist())
    send_email_with_tables(long_df, short_df, summary_csv, detail_csv, index_long_df=index_long_df, index_short_df=index_short_df, index_iter_csv_filename=(index_iter_csv if 'index_iter_csv' in locals() else None))
    logger.info('Index-first Scan Pipeline Completed')

if __name__ == '__main__':
    main_index_first()


def scan_options_logic(long_symbols, short_symbols):
    logger.info(">>> STARTING FORCED OPTIONS SCAN <<<")
    try:
        all_syms = long_symbols + short_symbols
        for s in all_syms:
            logger.info(f"Scanning options for {s}")
            # Placeholder for your Fyers API call logic
    except Exception as e:
        logger.error(f"Scan Error: {e}")
    logger.info(">>> OPTIONS SCAN COMPLETE <<<")

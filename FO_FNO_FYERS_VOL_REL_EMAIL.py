import os
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

fyers: Optional[fyersModel.FyersModel] = None

# Email settings (adjust to your environment)
smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
smtp_port = int(os.environ.get("SMTP_PORT", "587"))
sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
sender_password = os.environ.get("SENDER_PASSWORD", "password")
recipient_email = os.environ.get("RECIPIENT_EMAIL", "you@example.com")

EMAIL_DISPLAY_COLS = [
    "Symbol",
    "LTP",
    "% Change",
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
    "Cumulative KER",
    "Cumulative ADX",
    "Cumulative +DI",
    "Cumulative -DI",
    "Cumulative RSI",
    "Freshness_Score",
    "Last Iteration Time",
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


def calculate_hybrid_freshness(df_intraday: pd.DataFrame) -> pdDataFrame:
    if df_intraday is None or df_intraday.empty:
        return df_intraday

    work = df_intraday.copy()
    if "time" not in work.columns and "timestamp" in work.columns:
        work["time"] = pd.to_datetime(work["timestamp"])

    work = work.sort_values("time").reset_index(drop=True)
    if len(work) < 3:
        work["Freshness_Score"] = 50.0
        work["Is_Fresh"] = False
        work["Fresh_State"] = ""
        work["Fresh_Since"] = ""
        return work

    work["rolling_HOD"] = work["high"].cummax()
    work["vol_sma_10"] = (
        work["volume"].rolling(window=10, min_periods=1).mean().shift(1)
    )
    work["vol_sma_10"] = work["vol_sma_10"].fillna(work["volume"].mean())
    work["vol_ratio"] = np.where(
        work["vol_sma_10"] > 0,
        work["volume"] / work["vol_sma_10"],
        0.0,
    )

    open_price = float(work.iloc[0]["open"])
    work["close_vs_open_pct"] = (
        (work["close"] - open_price) / open_price * 100
    )

    ib_high = float(work.iloc[0:3]["high"].max())
    ib_vol_avg = float(work.iloc[0:3]["volume"].mean()) if len(
        work.iloc[0:3]
    ) > 0 else 0.0

    freshness_scores, is_fresh_flags = [], []

    for i in range(len(work)):
        row = work.iloc[i]
        base_score = min(
            100.0, max(0.0, row["close_vs_open_pct"] / 3.0 * 100.0)
        )
        score = base_score
        is_fresh = False

        if i < 3:
            score = base_score
        elif 3 <= i < 11:
            dist_to_ib_high = (
                (ib_high - row["close"]) / ib_high * 100.0
                if ib_high > 0
                else 999.0
            )
            vol_vs_ib = (
                row["volume"] / ib_vol_avg if ib_vol_avg > 0 else 0.0
            )
            if dist_to_ib_high < 0.3 or row["close"] > ib_high:
                score = score * 0.2 if vol_vs_ib < 0.8 else min(
                    100.0, score * 1.5
                )
                is_fresh = vol_vs_ib >= 0.8
        else:
            prev_hod = float(work.iloc[i - 1]["rolling_HOD"])
            dist_to_prev_hod = (
                (prev_hod - row["close"]) / prev_hod * 100.0
                if prev_hod > 0
                else 999.0
            )
            if dist_to_prev_hod < 0.3 or row["close"] > prev_hod:
                score = score * 0.2 if row["vol_ratio"] < 1.3 else min(
                    100.0, score * 1.5
                )
                is_fresh = row["vol_ratio"] >= 1.3
            elif row["vol_ratio"] > 2.0 and row["close"] > row["open"]:
                score = min(100.0, score * 1.2)
                is_fresh = True

        freshness_scores.append(round(score, 1))
        is_fresh_flags.append(is_fresh)

    work["Freshness_Score"] = freshness_scores
    work["Is_Fresh"] = is_fresh_flags

    prev_fresh = work["Is_Fresh"].shift(1).fillna(False)
    fresh_states, fresh_since = [], []
    fresh_start_time = ""
    fresh_cycle = 0

    for i in range(len(work)):
        curr = bool(work.loc[i, "Is_Fresh"])
        prev = bool(prev_fresh.iloc[i])
        tstr = pd.to_datetime(work.loc[i, "time"]).strftime("%H:%M")

        if curr and not prev:
            fresh_cycle += 1
            fresh_start_time = tstr
            state = "Fresh" if fresh_cycle == 1 else "Re-Ignited"
            since = tstr
        elif curr and prev:
            state = "Fresh"
            since = fresh_start_time
        elif (not curr) and prev:
            state = "Fresh Lost"
            since = tstr
        else:
            state = ""
            since = ""

        fresh_states.append(state)
        fresh_since.append(since)

    work["Fresh_State"] = fresh_states
    work["Fresh_Since"] = fresh_since
    return work


# ...

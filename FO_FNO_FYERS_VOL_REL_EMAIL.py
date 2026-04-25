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

# TODO: set these according to your email setup
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


def calculate_hybrid_freshness(df_intraday: pd.DataFrame) -> pd.DataFrame:
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

    df = calculate_hybrid_freshness(intra_df.copy())
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

    metric_df = compute_cumulative_directional_metrics(
        curr_df[["time", "open", "high", "low", "close", "volume"]].copy()
    )
    flow_df = compute_cumulative_flow_metrics(
        curr_df[["time", "high", "low", "close", "volume"]].copy()
    )
    price_lead_df = compute_price_lead_metrics(
        curr_df[["time", "open", "high", "low", "close", "volume"]].copy()
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
                "Freshness_Score": float(curr_df["Freshness_Score"].iloc[i])
                if "Freshness_Score" in curr_df.columns
                else float("nan"),
                "Is_Fresh": bool(curr_df["Is_Fresh"].iloc[i])
                if "Is_Fresh" in curr_df.columns
                else False,
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

    fresh_score = float(curr_df["Freshness_Score"].iloc[-1]) if "Freshness_Score" in curr_df.columns else 0.0
    is_fresh = bool(fresh_score >= 60.0) and bool(adx_now > 20.0) and bool(ker_now > 0.40)

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
        "Freshness_Score": float(fresh_score),
        "Fresh_State": str(df["Fresh_State"].iloc[-1])
        if "Fresh_State" in df.columns
        else "",
        "Fresh_Since": str(df["Fresh_Since"].iloc[-1])
        if "Fresh_Since" in df.columns
        else "",
        "Is_Fresh": bool(is_fresh),
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
                "Freshness_Score": iter_summary.get("Freshness_Score"),
                "Fresh_State": iter_summary.get("Fresh_State"),
                "Fresh_Since": iter_summary.get("Fresh_Since"),
                "Is_Fresh": iter_summary.get("Is_Fresh"),
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
        if bool(row.get("Is_Fresh", False)):
            score += 1
        if pd.notna(row.get("Cumulative KER")) and row.get("Cumulative KER") >= 0.40:
            score += 1
        if pd.notna(row.get("Cumulative RSI")) and row.get("Cumulative RSI") >= 55:
            score += 1
        if pd.notna(row.get("Freshness_Score")) and row.get("Freshness_Score") >= 60:
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
        if bool(row.get("Is_Fresh", False)):
            score += 1
        if pd.notna(row.get("Cumulative KER")) and row.get("Cumulative KER") >= 0.40:
            score += 1
        if pd.notna(row.get("Cumulative RSI")) and row.get("Cumulative RSI") <= 45:
            score += 1
        if pd.notna(row.get("Freshness_Score")) and row.get("Freshness_Score") >= 60:
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
        return "#0b6623"
    if label == "Buy+":
        return "#1e8449"
    if label == "Buy":
        return "#27ae60"
    if label == "Buyer Zone":
        return "#145a32"
    if label == "Neutral":
        return "#555555"
    if label == "Neutral Vol":
        return "#555555"
    if label == "Sell++":
        return "#922b21"
    if label == "Sell+":
        return "#c0392b"
    if label == "Sell":
        return "#e74c3c"
    if label == "Avoid Buy Premium":
        return "#7d6608"
    return "#2c2c2c"


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
        return "<p>No candidates found.</p>"

    df_slice = df.head(max_rows).copy()
    cols = [c for c in EMAIL_DISPLAY_COLS if c in df_slice.columns]
    if not cols:
        return "<p>No candidates found.</p>"

    header_cells = "".join(f"<th>{c}</th>" for c in cols)
    header = f"<tr>{header_cells}</tr>"

    row_html = []
    for _, row in df_slice.iterrows():
        cells = "".join(f"<td>{format_value(col, row.get(col))}</td>" for col in cols)
        row_html.append(f"<tr>{cells}</tr>")
    body = "
".join(row_html)

    table_html = (
        "<table border='1' cellspacing='0' cellpadding='3' "
        "style='border-collapse: collapse; font-size: 12px;'>"
        f"{header}{body}</table>"
    )
    return table_html


def send_email_with_tables(
    long_df: pd.DataFrame,
    short_df: pd.DataFrame,
    csv_filename: str,
    detail_csv_filename: str,
    index_long_df: pd.DataFrame = None,
    index_short_df: pd.DataFrame = None,
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

        for filename in [csv_filename, detail_csv_filename]:
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

# Map indices to their F&O constituents (symbols must match your sector CSVs)
INDEX_CONSTITUENTS = {
    "NIFTY50-INDEX": [
        # TODO: fill with actual NIFTY 50 F&O symbols; examples:
        "RELIANCE",
        "HDFCBANK",
        "ICICIBANK",
        "INFY",
        "TCS",
        "ITC",
        "AXISBANK",
        "LT",
        "KOTAKBANK",
        "SBIN",
    ],
    "NIFTYBANK-INDEX": [
        # TODO: fill BANKNIFTY basket symbols
        "HDFCBANK",
        "ICICIBANK",
        "AXISBANK",
        "KOTAKBANK",
        "SBIN",
        "INDUSINDBK",
        "FEDERALBNK",
        "BANDHANBNK",
        "PNB",
    ],
}


def load_index_symbols() -> List[str]:
    """Return index symbols to scan on Fyers."""
    return [
        "NIFTY50-INDEX",
        "NIFTYBANK-INDEX",
    ]


def format_fyers_index_symbol(symbol: str) -> str:
    """Format index symbol for Fyers (no -EQ suffix)."""
    if symbol.startswith("NSE:"):
        return symbol
    return f"NSE:{symbol}"


def scan_index_universe() -> pd.DataFrame:
    """Scan configured indices using same metric stack as F&O universe."""
    symbols = load_index_symbols()
    if not symbols:
        logger.error("INDEX No index symbols configured.")
        return pd.DataFrame()

    rows = []
    total = len(symbols)

    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"INDEX [{idx}/{total}] Processing {sym}")
        fyers_sym = format_fyers_index_symbol(sym)

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

        iter_summary, _ = compute_iteration_volume_profile(intra_df)
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
                "Freshness_Score": iter_summary.get("Freshness_Score"),
                "Fresh_State": iter_summary.get("Fresh_State"),
                "Fresh_Since": iter_summary.get("Fresh_Since"),
                "Is_Fresh": iter_summary.get("Is_Fresh"),
                "IVP": iv_info.get("IVP"),
                "Volatility State": iv_info.get("Volatility State"),
            }
        )

    df_idx = pd.DataFrame(rows)
    df_idx = derive_rank_columns(df_idx)
    df_idx = add_signal_columns(df_idx)
    return df_idx


def load_fno_symbols_for_indices(active_index_symbols: List[str]) -> List[str]:
    """Return F&O symbols that belong to any of the given indices."""
    all_fno = set(load_fno_symbols_from_sectors("sectors"))
    selected = set()
    for idx_sym in active_index_symbols:
        members = INDEX_CONSTITUENTS.get(idx_sym, [])
        for s in members:
            if s in all_fno:
                selected.add(s)
    return sorted(selected)


def scan_symbol_universe(symbols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generic scanner using existing F&O logic for a given list of symbols."""
    if not symbols:
        logger.error("CORE No symbols to scan.")
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

        if iter_detail is not None and not iter_detail.empty:
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
                "Freshness_Score": iter_summary.get("Freshness_Score"),
                "Fresh_State": iter_summary.get("Fresh_State"),
                "Fresh_Since": iter_summary.get("Fresh_Since"),
                "Is_Fresh": iter_summary.get("Is_Fresh"),
                "IVP": iv_info.get("IVP"),
                "Volatility State": iv_info.get("Volatility State"),
            }
        )

    df_all = pd.DataFrame(rows)
    df_iter = (
        pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame()
    )
    return df_all, df_iter


def main_index_first():
    """Entry point: index-first, direction-aligned stock selection."""
    logger.info("Starting Index-first F&O Iteration Volume Volatility Scan")
    init_fyers()

    # 1) Scan indices and get top long / short
    df_indices = scan_index_universe()
    index_long_df, index_short_df = build_candidate_tables(df_indices)

    top_long_indices = index_long_df["Symbol"].head(2).tolist()
    top_short_indices = index_short_df["Symbol"].head(2).tolist()

    logger.info(f"Top long indices: {top_long_indices}")
    logger.info(f"Top short indices: {top_short_indices}")

    # 2) Build direction-aligned F&O baskets
    long_side_symbols = load_fno_symbols_for_indices(top_long_indices)
    short_side_symbols = load_fno_symbols_for_indices(top_short_indices)

    # 3) Scan long basket and keep only long candidates
    df_long_all, df_long_iter = scan_symbol_universe(long_side_symbols)
    if not df_long_all.empty:
        df_long_all = derive_rank_columns(df_long_all)
        df_long_all = add_signal_columns(df_long_all)
        long_long_df, _ = build_candidate_tables(df_long_all)
    else:
        long_long_df = pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
        df_long_iter = pd.DataFrame()

    # 4) Scan short basket and keep only short candidates
    df_short_all, df_short_iter = scan_symbol_universe(short_side_symbols)
    if not df_short_all.empty:
        df_short_all = derive_rank_columns(df_short_all)
        df_short_all = add_signal_columns(df_short_all)
        _, short_short_df = build_candidate_tables(df_short_all)
    else:
        short_short_df = pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
        df_short_iter = pd.DataFrame()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = f"fo_idx_filtered_summary_{timestamp}.csv"
    detail_csv = f"fo_idx_filtered_details_{timestamp}.csv"

    # Merge iterations for export convenience
    df_all = pd.concat([df_long_all, df_short_all], ignore_index=True) if (
        not df_long_all.empty or not df_short_all.empty
    ) else pd.DataFrame()
    df_iter = pd.concat([df_long_iter, df_short_iter], ignore_index=True) if (
        not df_long_iter.empty or not df_short_iter.empty
    ) else pd.DataFrame()

    if not df_all.empty:
        df_all.to_csv(summary_csv, index=False)
    else:
        pd.DataFrame().to_csv(summary_csv, index=False)

    if not df_iter.empty:
        df_iter.to_csv(detail_csv, index=False)
    else:
        pd.DataFrame().to_csv(detail_csv, index=False)

    # Send email including index and stock tables
    send_email_with_tables(
        long_long_df,
        short_short_df,
        summary_csv,
        detail_csv,
        index_long_df=index_long_df,
        index_short_df=index_short_df,
    )

    logger.info("Index-first Scan Pipeline Completed")


if __name__ == "__main__":
    main_index_first()

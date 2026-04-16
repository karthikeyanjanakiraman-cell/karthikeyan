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
formatter = UTF8Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

DAILY_LOOKBACK_DAYS = 60
INTRADAY_LOOKBACK_DAYS = 20
DAILY_VOL_THRESHOLD = 1.0
DAILY_VOLUME_THRESHOLD = 1.0
KER_CUTOFF = 0.50

fyers: Optional[fyersModel.FyersModel] = None


def init_fyers():
    global fyers
    try:
        client_id = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
        access_token = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
        if not client_id or not access_token:
            logger.warning("INIT Missing Fyers credentials in environment. Check GitHub Secrets. Proceeding with empty data.")
            fyers = None
            return
        fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
        logger.info("INIT FyersModel initialized successfully.")
    except Exception as e:
        logger.warning(f"INIT Failed to initialize FyersModel: {e}. Proceeding without API connection.")
        fyers = None


def load_fno_symbols_from_sectors(root_dir: str = "sectors") -> List[str]:
    symbols = set()
    if not os.path.isdir(root_dir):
        logger.warning(f"FNO Sectors folder '{root_dir}' not found, returning empty list.")
        return []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".csv"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                df = pd.read_csv(fpath)
                col = None
                for c in df.columns:
                    if c.lower() in ["symbol", "symbols", "ticker"]:
                        col = c
                        break
                if col is None:
                    continue
                for s in df[col].dropna().astype(str):
                    s = s.strip()
                    if s:
                        symbols.add(s)
            except Exception as e:
                logger.warning(f"FNO Error reading {fpath}: {e}")

    symbols_list = sorted(symbols)
    logger.info(f"FNO Loaded {len(symbols_list)} unique F&O symbols.")
    return symbols_list


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
            df = pd.DataFrame(res["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            df.sort_values("timestamp", inplace=True)
            df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df
        return None
    except Exception as e:
        logger.error(f"FYERS Error fetching {resolution} data for {symbol}: {e}")
        return None


def compute_volatility_pair(daily_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    if daily_df is None or daily_df.empty or len(daily_df) < 11:
        return {}
    df = daily_df.copy()
    df["DailyVolatility"] = df["high"] - df["low"]
    current_vol = float(df["DailyVolatility"].iloc[-1])
    avg_10d_vol = float(df["DailyVolatility"].iloc[-11:-1].mean())
    vol_exp = current_vol / avg_10d_vol if avg_10d_vol > 0 else 0.0
    return {
        "Current Daily Volatility": current_vol,
        "Avg Daily Volatility": avg_10d_vol,
        "Daily Volatility Expansion": vol_exp,
    }


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

        path = abs(c[i] - o[0])
        walked = abs(c[0] - o[0]) + np.sum(np.abs(np.diff(c[: i + 1])))
        ker = path / walked if walked > 0 else 0.0

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

        # Count up-iterations based on close > previous close
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
    rs = avg_gain / avg_loss.replace(0, float('nan'))
    rsi = 100 - (100 / (1 + rs))
    zero_loss = (avg_loss == 0) & avg_loss.notna()
    rsi = rsi.mask(zero_loss, 100.0).fillna(0.0)

    direction = close.diff().fillna(0.0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()

    typical_price = (high + low + close) / 3.0
    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, float('nan'))
    vwap = (cum_pv / cum_vol).fillna(0.0)

    cum_pv2 = ((typical_price ** 2) * volume).cumsum()
    vwap_variance = (cum_pv2 / cum_vol) - (vwap ** 2)
    vwap_variance = vwap_variance.clip(lower=0.0)
    vwap_std = np.sqrt(vwap_variance).fillna(0.0)

    out = pd.DataFrame({
        "Cumulative RSI": rsi,
        "Cumulative OBV": obv,
        "Cumulative VWAP": vwap,
        "VWAP StdDev": vwap_std,
    })
    return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

def compute_iteration_volume_profile(intra_df: Optional[pd.DataFrame]) -> Tuple[Dict, pd.DataFrame]:
    if intra_df is None or intra_df.empty:
        return {}, pd.DataFrame()

    df = intra_df.copy()
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
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

    curr_df.sort_values("time", inplace=True)
    curr_df["cum_vol"] = curr_df["volume"].cumsum()
    metric_df = compute_cumulative_directional_metrics(curr_df[["time", "open", "high", "low", "close", "volume"]].copy())
    flow_df = compute_cumulative_flow_metrics(curr_df[["time", "high", "low", "close", "volume"]].copy())
    ltp = float(curr_df["close"].iloc[-1])

    rows = []
    total_iters = 0
    last_iter_mins = None
    last_iter_time = None
    last_cum_vol = 0
    last_rvol10 = 0
    last_rvol20 = 0
    last_dvolexp = 0
    avg_daily_vol_10 = hist_df_10.groupby("date")["volume"].sum().mean() if not hist_df_10.empty else 0

    for i in range(len(curr_df)):
        total_iters += 1
        row = curr_df.iloc[i]
        t = row["time"]
        cum_vol = float(row["cum_vol"])

        h10 = hist_df_10[hist_df_10["time"] <= t]
        avg_cum_10 = h10.groupby("date")["volume"].sum().mean() if not h10.empty else 0
        rvol10 = cum_vol / avg_cum_10 if avg_cum_10 > 0 else 0

        h20 = hist_df_20[hist_df_20["time"] <= t]
        avg_cum_20 = h20.groupby("date")["volume"].sum().mean() if not h20.empty else 0
        rvol20 = cum_vol / avg_cum_20 if avg_cum_20 > 0 else 0

        dvolexp = cum_vol / avg_daily_vol_10 if avg_daily_vol_10 > 0 else 0
        dt_time = datetime.combine(current_date, t)
        market_open = datetime.combine(current_date, time(9, 15))
        iter_mins = int((dt_time - market_open).total_seconds() / 60)

        rows.append({
            "Iteration No": total_iters,
            "Iteration Minutes": iter_mins,
            "Iteration Time": t.strftime("%H:%M"),
            "Current Volume": cum_vol,
            "10 Day Relative Volume": rvol10,
            "20 Day Relative Volume": rvol20,
            "Daily Volume Expansion": dvolexp,
            "Cumulative RSI": float(flow_df["Cumulative RSI"].iloc[i]) if not flow_df.empty else float("nan"),
            "Cumulative OBV": float(flow_df["Cumulative OBV"].iloc[i]) if not flow_df.empty else float("nan"),
            "Cumulative VWAP": float(flow_df["Cumulative VWAP"].iloc[i]) if not flow_df.empty else float("nan"),
            "VWAP StdDev": float(flow_df["VWAP StdDev"].iloc[i]) if not flow_df.empty else float("nan"),
            "VWAP Z-Score": (float(curr_df["close"].iloc[i]) - float(flow_df["Cumulative VWAP"].iloc[i])) / float(flow_df["VWAP StdDev"].iloc[i]) if not flow_df.empty and float(flow_df["VWAP StdDev"].iloc[i]) > 0 else 0.0,
        })

        last_cum_vol = cum_vol
        last_rvol10 = rvol10
        last_rvol20 = rvol20
        last_dvolexp = dvolexp
        last_iter_mins = iter_mins
        last_iter_time = t.strftime("%H:%M")

    detail_df = pd.DataFrame(rows)
    summary = {
        "LTP": ltp,
        "Current Volume": last_cum_vol,
        "10 Day Relative Volume": last_rvol10,
        "20 Day Relative Volume": last_rvol20,
        "Daily Volume Expansion": last_dvolexp,
        "Cumulative RSI": float(flow_df["Cumulative RSI"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Cumulative OBV": float(flow_df["Cumulative OBV"].iloc[-1]) if not flow_df.empty else float("nan"),
        "Cumulative VWAP": float(flow_df["Cumulative VWAP"].iloc[-1]) if not flow_df.empty else float("nan"),
        "VWAP StdDev": float(flow_df["VWAP StdDev"].iloc[-1]) if not flow_df.empty else float("nan"),
        "VWAP Z-Score": (ltp - float(flow_df["Cumulative VWAP"].iloc[-1])) / float(flow_df["VWAP StdDev"].iloc[-1]) if not flow_df.empty and float(flow_df["VWAP StdDev"].iloc[-1]) > 0 else 0.0,
        "Total Iterations": total_iters,
        "Last Iteration Minutes": last_iter_mins,
        "Last Iteration Time": last_iter_time,
        "Cumulative KER": float(metric_df["Cumulative KER"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative +DI": float(metric_df["Cumulative +DI"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative -DI": float(metric_df["Cumulative -DI"].iloc[-1]) if not metric_df.empty else np.nan,
        "Cumulative ADX": float(metric_df["Cumulative ADX"].iloc[-1]) if not metric_df.empty else np.nan,
        "Survival Score": str(metric_df["Survival Score"].iloc[-1]) if not metric_df.empty else "0/0",
        "Survival_Num": float(metric_df["Survival_Num"].iloc[-1]) if not metric_df.empty else 0.0,
    }
    return summary, detail_df


def scan_fno_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = load_fno_symbols_from_sectors("sectors")
    if not symbols:
        logger.error("CORE No F&O symbols found. Exiting.")
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    iteration_rows = []
    total = len(symbols)

    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"CORE [{idx}/{total}] Processing {sym}")
        fyers_sym = format_fyers_symbol(sym)
        daily_df = get_fyers_history(fyers_sym, resolution="D", days_back=DAILY_LOOKBACK_DAYS)
        intra_df = get_fyers_history(fyers_sym, resolution="5", days_back=INTRADAY_LOOKBACK_DAYS)

        vol_info = compute_volatility_pair(daily_df)
        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df)

        if daily_df is not None and len(daily_df) >= 2:
            prev_close = float(daily_df["close"].iloc[-2])
        else:
            prev_close = None

        ltp = iter_summary.get("LTP")
        pct_change = ((ltp - prev_close) / prev_close * 100) if (ltp is not None and prev_close and prev_close != 0) else 0.0
        daily_vol_exp = vol_info.get("Daily Volatility Expansion")
        daily_volume_exp = iter_summary.get("Daily Volume Expansion")
        rvol_10d = iter_summary.get("10 Day Relative Volume")
        ease_of_movement = abs(pct_change) / rvol_10d if (pct_change is not None and rvol_10d and rvol_10d > 0) else None

        if not iter_detail.empty:
            if daily_vol_exp is not None and daily_vol_exp > DAILY_VOL_THRESHOLD:
                iter_detail["Above DV and DVol"] = iter_detail["Daily Volume Expansion"].gt(DAILY_VOLUME_THRESHOLD)
            else:
                iter_detail["Above DV and DVol"] = False
            above_count = int(iter_detail["Above DV and DVol"].sum())
            iter_detail.insert(0, "Symbol", sym)
            iter_detail.insert(1, "% Change", pct_change)
            iter_detail.insert(2, "Daily Volatility Expansion", daily_vol_exp)
            iteration_rows.append(iter_detail)
        else:
            above_count = 0

        total_iterations = int(iter_summary.get("Total Iterations") or 0)
        above_ratio = above_count / total_iterations if total_iterations > 0 else 0.0

        rows.append({
            "Symbol": sym,
            "LTP": ltp,
            "% Change": pct_change,
            "Current Daily Volatility": vol_info.get("Current Daily Volatility"),
            "Avg Daily Volatility": vol_info.get("Avg Daily Volatility"),
            "Daily Volatility Expansion": daily_vol_exp,
            "Current Volume": iter_summary.get("Current Volume"),
            "10 Day Relative Volume": iter_summary.get("10 Day Relative Volume"),
            "20 Day Relative Volume": iter_summary.get("20 Day Relative Volume"),
            "Daily Volume Expansion": daily_volume_exp,
            "Cumulative RSI": iter_summary.get("Cumulative RSI"),
            "Cumulative OBV": iter_summary.get("Cumulative OBV"),
            "Cumulative VWAP": iter_summary.get("Cumulative VWAP"),
            "VWAP StdDev": iter_summary.get("VWAP StdDev"),
            "VWAP Z-Score": iter_summary.get("VWAP Z-Score"),
            "Ease of Movement": ease_of_movement,
            "Total Iterations": total_iterations,
            "Above Threshold Iterations": above_count,
            "Above Threshold Ratio": above_ratio,
            "Last Iteration Minutes": iter_summary.get("Last Iteration Minutes"),
            "Last Iteration Time": iter_summary.get("Last Iteration Time"),
            "Cumulative KER": iter_summary.get("Cumulative KER"),
            "Cumulative +DI": iter_summary.get("Cumulative +DI"),
            "Cumulative -DI": iter_summary.get("Cumulative -DI"),
            "Cumulative ADX": iter_summary.get("Cumulative ADX"),
            "Survival Score": iter_summary.get("Survival Score"),
            "Survival_Num": iter_summary.get("Survival_Num"),
        })

    summary_df = pd.DataFrame(rows)
    iteration_df = pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame()
    return summary_df, iteration_df


DISPLAY_COLS = [
    "Symbol",
    "LTP",
    "% Change",
    "Daily Volatility Expansion",
    "10 Day Relative Volume",
    "20 Day Relative Volume",
    "Daily Volume Expansion",
    "Cumulative RSI",
    "Cumulative OBV",
    "Cumulative VWAP",
    "VWAP Z-Score",
    "Cumulative KER",
    "Cumulative +DI",
    "Cumulative -DI",
    "Cumulative ADX",
    "Survival Score",
    "Ease of Movement",
    "Above Threshold Iterations",
    "Last Iteration Minutes",
    "Last Iteration Time",
]

EMAIL_DISPLAY_COLS = [
    "Symbol",
    "LTP",
    "% Change",
    "Daily Volatility Expansion",
    "Daily Volume Expansion",
    "Cumulative RSI",
    "Cumulative OBV",
    "Cumulative VWAP",
    "VWAP Z-Score",
    "Cumulative KER",
    "Cumulative +DI",
    "Cumulative -DI",
    "Cumulative ADX",
    "Survival Score",
    "Last Iteration Time",
]


def build_candidate_tables(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_all is None or df_all.empty:
        return pd.DataFrame(columns=DISPLAY_COLS), pd.DataFrame(columns=DISPLAY_COLS)

    base = df_all.copy()
    if "Daily Volatility Expansion" in base.columns and "Daily Volume Expansion" in base.columns:
        filtered = base[
            (base["Daily Volatility Expansion"] > DAILY_VOL_THRESHOLD)
            & (base["Daily Volume Expansion"] > DAILY_VOLUME_THRESHOLD)
        ].copy()
        if not filtered.empty:
            base = filtered

    def _sort_long(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.sort_values(
            by=["Cumulative KER", "Survival_Num", "Cumulative ADX", "% Change"],
            ascending=[False, False, False, False],
            na_position="last",
        )

    def _sort_short(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        return df.sort_values(
            by=["Cumulative KER", "Survival_Num", "Cumulative ADX", "% Change"],
            ascending=[False, False, False, True],
            na_position="last",
        )

    if {"% Change", "Cumulative +DI", "Cumulative -DI"}.issubset(base.columns):
        strict_long = base[(base["% Change"] > 0) & (base["Cumulative +DI"] > base["Cumulative -DI"])].copy()
        strict_short = base[(base["% Change"] < 0) & (base["Cumulative -DI"] > base["Cumulative +DI"])].copy()
        fallback_long = base[base["% Change"] > 0].copy()
        fallback_short = base[base["% Change"] < 0].copy()
    else:
        strict_long = base.copy()
        strict_short = base.copy()
        fallback_long = base.copy()
        fallback_short = base.copy()

    long_df = _sort_long(strict_long)
    short_df = _sort_short(strict_short)

    extra_long = _sort_long(fallback_long[~fallback_long["Symbol"].isin(long_df["Symbol"])])
    long_df = pd.concat([long_df, extra_long])

    extra_short = _sort_short(fallback_short[~fallback_short["Symbol"].isin(short_df["Symbol"])])
    short_df = pd.concat([short_df, extra_short])

    long_df = long_df.drop_duplicates(subset=["Symbol"])
    short_df = short_df.drop_duplicates(subset=["Symbol"])

    return long_df[DISPLAY_COLS].copy(), short_df[DISPLAY_COLS].copy()


def format_value(col: str, val):
    if pd.isna(val):
        return ""
    if col in [
        "LTP",
        "Current Daily Volatility",
        "Avg Daily Volatility",
        "Daily Volatility Expansion",
        "10 Day Relative Volume",
        "20 Day Relative Volume",
        "Daily Volume Expansion",
        "Cumulative RSI",
        "Cumulative OBV",
        "Cumulative VWAP",
        "VWAP StdDev",
        "VWAP Z-Score",
        "Ease of Movement",
        "Cumulative KER",
        "Cumulative +DI",
        "Cumulative -DI",
        "Cumulative ADX",
    ]:
        return f"{float(val):.2f}"
    if col == "% Change":
        return f"{float(val):.2f}%"
    if col in ["Current Volume", "Cumulative OBV"]:
        return f"{int(float(val)):,}"
    if col == "Above Threshold Ratio":
        return f"{float(val) * 100:.2f}%"
    if col in ["Total Iterations", "Above Threshold Iterations", "Last Iteration Minutes"]:
        return f"{int(val)}"
    return str(val)


def df_to_html_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "<p>No candidates found.</p>"

    df_slice = df.head(max_rows).copy()
    cols = [c for c in EMAIL_DISPLAY_COLS if c in df_slice.columns]

    header_html = "".join(
        f'<th style="padding:8px;border:1px solid #d0d0d0;background:#f5f5f5;text-align:left;">{col}</th>'
        for col in cols
    )

    body_rows = []
    for _, row in df_slice.iterrows():
        cells = "".join(
            f'<td style="padding:8px;border:1px solid #d0d0d0;white-space:nowrap;">{format_value(col, row.get(col))}</td>'
            for col in cols
        )
        body_rows.append(f"<tr>{cells}</tr>")

    return (
        '<div style="overflow-x:auto; margin:12px 0 20px 0;">'
        '<table style="border-collapse:collapse;width:100%;font-family:Arial,sans-serif;font-size:13px;">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        '</table></div>'
    )


def _first_env(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and str(value).strip() != "":
            return value.strip()
    return None


def send_email_with_tables(long_df: pd.DataFrame, short_df: pd.DataFrame, csv_filename: str, detail_csv_filename: str) -> bool:
    try:
        sender_email_keys = ["SENDER_EMAIL", "EMAIL_USER", "GMAIL_USER", "SMTP_USERNAME", "MAIL_USERNAME", "FROM_EMAIL", "EMAIL_FROM"]
        sender_password_keys = ["SENDER_APP_PASSWORD", "SENDER_PASSWORD", "EMAIL_PASSWORD", "EMAIL_PASS", "GMAIL_APP_PASSWORD", "GMAIL_PASSWORD", "SMTP_PASSWORD", "MAIL_PASSWORD", "APP_PASSWORD", "EMAIL_APP_PASSWORD", "PASSWORD"]
        recipient_keys = ["RECIPIENT_EMAIL", "TO_EMAIL", "ALERT_EMAIL", "MAIL_TO", "EMAIL_TO"]

        sender_email = _first_env(*sender_email_keys)
        sender_app_password = _first_env(*sender_password_keys)
        recipient_email = _first_env(*recipient_keys) or sender_email
        smtp_host = _first_env("SMTP_HOST", "MAIL_SERVER", "EMAIL_HOST") or "smtp.gmail.com"
        smtp_port = _first_env("SMTP_PORT", "MAIL_PORT", "EMAIL_PORT") or "587"

        present_sender = [k for k in sender_email_keys if os.environ.get(k)]
        present_password = [k for k in sender_password_keys if os.environ.get(k)]
        present_recipient = [k for k in recipient_keys if os.environ.get(k)]
        logger.info(f"EMAIL Env sender keys present: {present_sender}")
        logger.info(f"EMAIL Env password keys present: {present_password}")
        logger.info(f"EMAIL Env recipient keys present: {present_recipient}")

        long_table = df_to_html_table(long_df, max_rows=15)
        short_table = df_to_html_table(short_df, max_rows=15)
        html_body = f"""
        <html>
        <body style="font-family:Arial,sans-serif;font-size:14px;color:#222;">
            <h2 style="margin-bottom:8px;">Intraday Vol Iteration Alert</h2>
            <p style="margin:0 0 12px 0;">Scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}.</p>
            <p style="margin:0 0 12px 0;">Filters applied: Daily Volatility Expansion &gt; {DAILY_VOL_THRESHOLD} and Daily Volume Expansion &gt; {DAILY_VOLUME_THRESHOLD}.</p>
            <p style="margin:0 0 18px 0;"><b>Ranking:</b> Cumulative KER descending, then Survival Score, then Cumulative ADX. Longs require +DI &gt; -DI, shorts prefer -DI &gt; +DI but will be filled up to 15 from the strongest negative movers if strict matches are fewer.</p>
            <h3 style="margin:18px 0 8px 0;">Long Candidates Top 15</h3>
            {long_table}
            <h3 style="margin:18px 0 8px 0;">Short Candidates Top 15</h3>
            {short_table}
            <p style="margin-top:18px;">Full scan summary and detailed iteration data are attached as CSV files.</p>
        </body>
        </html>
        """

        if not sender_email or not recipient_email:
            logger.error("EMAIL Missing sender or recipient email in environment.")
            return False

        if not sender_app_password:
            fallback_html = csv_filename.replace('.csv', '_email_preview.html')
            with open(fallback_html, 'w', encoding='utf-8') as f:
                f.write(html_body)
            logger.warning(f"EMAIL Password secret missing. Saved email preview HTML instead: {fallback_html}")
            return False

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = f"Intraday Vol Iteration Alert - {datetime.now().strftime('%d %b %H:%M')}"
        msg.attach(MIMEText(html_body, "html"))

        if os.path.exists(csv_filename):
            with open(csv_filename, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(csv_filename)}")
                msg.attach(part)

        if os.path.exists(detail_csv_filename):
            with open(detail_csv_filename, "rb") as f:
                part2 = MIMEBase("application", "octet-stream")
                part2.set_payload(f.read())
                encoders.encode_base64(part2)
                part2.add_header("Content-Disposition", f"attachment; filename={os.path.basename(detail_csv_filename)}")
                msg.attach(part2)

        server = smtplib.SMTP(smtp_host, int(smtp_port))
        server.starttls()
        server.login(sender_email, sender_app_password)
        server.send_message(msg)
        server.quit()
        logger.info(f"EMAIL Sent successfully to {recipient_email}")
        return True
    except Exception as e:
        logger.error(f"EMAIL Failed to send email: {e}")
        return False


def main():
    logger.info("Starting F&O Iteration Volume Volatility Scan")
    init_fyers()
    df_all, df_iter = scan_fno_universe()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_csv = f"fo_fyers_iteration_summary_{timestamp}.csv"
    detail_csv = f"fo_fyers_iteration_details_{timestamp}.csv"

    if df_all is not None and not df_all.empty:
        df_all.to_csv(summary_csv, index=False)
        logger.info(f"OUTPUT Saved summary scan results to {summary_csv}")
        long_df, short_df = build_candidate_tables(df_all)
    else:
        logger.warning("OUTPUT Summary dataframe is empty.")
        pd.DataFrame(columns=DISPLAY_COLS).to_csv(summary_csv, index=False)
        long_df, short_df = pd.DataFrame(columns=DISPLAY_COLS), pd.DataFrame(columns=DISPLAY_COLS)

    if df_iter is not None and not df_iter.empty:
        df_iter.to_csv(detail_csv, index=False)
        logger.info(f"OUTPUT Saved detailed iteration results to {detail_csv}")
    else:
        logger.warning("OUTPUT Iteration details dataframe is empty.")
        pd.DataFrame(columns=[
            "Symbol", "% Change", "Daily Volatility Expansion", "Iteration No", "Iteration Minutes",
            "Iteration Time", "Current Volume", "10 Day Relative Volume", "20 Day Relative Volume",
            "Daily Volume Expansion", "Cumulative RSI", "Cumulative OBV", "Cumulative VWAP", "Above DV and DVol"
        ]).to_csv(detail_csv, index=False)

    send_email_with_tables(long_df, short_df, summary_csv, detail_csv)
    logger.info("Scan Pipeline Completed")


if __name__ == "__main__":
    main()

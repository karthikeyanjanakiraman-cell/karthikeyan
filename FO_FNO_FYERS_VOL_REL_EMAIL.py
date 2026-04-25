import os
import sys
import logging
from datetime import datetime, timedelta, time
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fyersapiv3 import fyersModel
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
formatter = UTF8Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)

DAILY_LOOKBACK_DAYS = 60
INTRADAY_LOOKBACK_DAYS = 20
IVP_LOOKBACK_DAYS = 252
EMAIL_DISPLAY_COLS = [
    "Symbol", "LTP", "Change", "5mSignal", "15mSignal", "30mSignal", "60mSignal",
    "BullSignal", "BearSignal", "OverallSignal", "PriceLeadStatus", "IVP",
    "Volatility State", "Last Iteration Time"
]

fyers: Optional[fyersModel.FyersModel] = None

smtp_host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
smtp_port = int(os.environ.get("SMTP_PORT", 587))
sender_email = os.environ.get("SENDER_EMAIL", "you@example.com")
sender_password = os.environ.get("SENDER_PASSWORD", "password")
recipient_email = os.environ.get("RECIPIENT_EMAIL", sender_email)


def init_fyers():
    global fyers
    try:
        client_id = os.environ.get("CLIENT_ID")
        access_token = os.environ.get("ACCESS_TOKEN")
        if not client_id or not access_token:
            logger.warning("INIT Missing Fyers credentials")
            fyers = None
            return
        fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
        logger.info("INIT FyersModel initialized successfully")
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
                col = next((c for c in df.columns if c.lower() in ["symbol", "symbols", "ticker"]), None)
                if col is None:
                    continue
                for s in df[col].dropna().astype(str):
                    s = s.strip()
                    if s:
                        symbols.add(s)
            except Exception:
                pass
    return sorted(symbols)


def load_fno_symbols_from_csv(path: str = "fno_stock_list.csv") -> List[str]:
    if not os.path.exists(path):
        logger.warning(f"FNO CSV not found at {path}")
        return []
    try:
        df = pd.read_csv(path)
        if "Symbol" not in df.columns:
            logger.warning("FNO CSV missing Symbol column")
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
            df = pd.DataFrame(res["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = (
                pd.to_datetime(df["timestamp"], unit="s")
                .dt.tz_localize("UTC")
                .dt.tz_convert("Asia/Kolkata")
                .dt.tz_localize(None)
            )
            df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last").reset_index(drop=True)
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
    iv_proxy = ((high - low) / close.replace(0, np.nan) * 100.0).replace([np.inf, -np.inf], np.nan).dropna()
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
    out, qualified = [], 0
    for i in range(n):
        if i == 0:
            out.append((np.nan, np.nan, np.nan, np.nan, "0/0", 0.0))
            continue
        prior_kers = []
        for j in range(1, i + 1):
            pj = abs(c[j] - o[0])
            wj = abs(c[0] - o[0]) + np.sum(np.abs(np.diff(c[: j + 1])))
            prior_kers.append(pj / wj if wj != 0 else 0.0)
        cum_ker = float(np.mean(prior_kers)) if prior_kers else np.nan
        tr_sum = tr[1 : i + 1].sum()
        plus_sum = plus_dm[1 : i + 1].sum()
        minus_sum = minus_dm[1 : i + 1].sum()
        pdi = 100 * plus_sum / tr_sum if tr_sum != 0 else 0.0
        mdi = 100 * minus_sum / tr_sum if tr_sum != 0 else 0.0
        dxs = []
        for k in range(1, i + 1):
            ktr = tr[1 : k + 1].sum()
            kp = plus_dm[1 : k + 1].sum()
            km = minus_dm[1 : k + 1].sum()
            kpdi = 100 * kp / ktr if ktr != 0 else 0.0
            kmdi = 100 * km / ktr if ktr != 0 else 0.0
            dxs.append(100 * abs(kpdi - kmdi) / (kpdi + kmdi) if (kpdi + kmdi) != 0 else 0.0)
        adx = float(np.mean(dxs)) if dxs else np.nan
        if c[i] > c[i - 1]:
            qualified += 1
        length_so_far = i + 1
        survival_ratio = qualified / length_so_far if length_so_far else 0.0
        out.append((cum_ker, pdi, mdi, adx, f"{qualified}/{length_so_far}", survival_ratio))
    cols = ["Cumulative KER", "Cumulative DI", "Cumulative -DI", "Cumulative ADX", "Survival Score", "SurvivalNum"]
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
    loss = -delta.clip(upper=0.0)
    period = 14
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).mask((avg_loss == 0) & avg_loss.notna(), 100.0).fillna(0.0)
    direction = close.diff().fillna(0.0).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    obv = direction * volume
    obv = obv.cumsum()
    typical_price = (high + low + close) / 3.0
    cum_pv = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum().replace(0, np.nan)
    vwap = (cum_pv / cum_vol).fillna(0.0)
    vwap_variance = (volume * (typical_price - vwap) ** 2).cumsum() / cum_vol
    vwap_std = np.sqrt(vwap_variance.fillna(0.0))
    vwap_z = pd.Series(np.where(vwap_std > 0, (close - vwap) / vwap_std, 0.0), index=df.index)
    out = pd.DataFrame({
        "Cumulative RSI": rsi,
        "Cumulative OBV": obv,
        "Cumulative VWAP": vwap,
        "VWAP Z-Score": vwap_z.fillna(0.0),
    })
    return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)


def compute_price_lead_metrics(curr_df: pd.DataFrame) -> pd.DataFrame:
    df = curr_df.copy().sort_values("time").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["time", "rangeexpansion", "volumeexpansion", "deltaexpansion", "priceleadingflag", "priceleadstreak", "PriceLeadStatus"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["range"] = (df["high"] - df["low"]).clip(lower=0.0)
    df["avg_range_5"] = df["range"].rolling(5, min_periods=3).mean()
    df["rangeexpansion"] = np.where(df["avg_range_5"] > 0, df["range"] / df["avg_range_5"], np.nan)
    df["avg_vol_5"] = df["volume"].rolling(5, min_periods=3).mean()
    df["volumeexpansion"] = np.where(df["avg_vol_5"] > 0, df["volume"] / df["avg_vol_5"], np.nan)
    df["mid"] = (df["high"] + df["low"]) / 2.0
    delta = np.where(df["close"] > df["mid"], df["volume"], np.where(df["close"] < df["mid"], -df["volume"], 0.0))
    cvd = pd.Series(delta, index=df.index).cumsum()
    cvd_change = cvd.diff().abs().fillna(0.0)
    avg_cvd_change_5 = cvd_change.rolling(5, min_periods=3).mean()
    df["deltaexpansion"] = np.where(avg_cvd_change_5 > 0, cvd_change / avg_cvd_change_5, np.nan)
    directional_bar = df["close"] != df["open"]
    df["priceleadingflag"] = ((df["rangeexpansion"] >= 1.5) & (df["volumeexpansion"] >= 1.0) & (df["deltaexpansion"] >= 1.0) & directional_bar).fillna(False)
    streak, run = [], 0
    for flag in df["priceleadingflag"].astype(bool):
        run = run + 1 if flag else 0
        streak.append(run)
    df["priceleadstreak"] = streak
    df["PriceLeadStatus"] = np.select(
        [
            df["priceleadingflag"] & (df["priceleadstreak"] >= 3),
            df["priceleadingflag"] & (df["priceleadstreak"] == 2),
            df["priceleadingflag"],
        ],
        ["STRONG_PRICE_LEAD_FADE", "PRICE_LEADING_FADE_RISK", "EARLY_PRICE_LEAD"],
        default="NORMAL",
    )
    return df[["time", "rangeexpansion", "volumeexpansion", "deltaexpansion", "priceleadingflag", "priceleadstreak", "PriceLeadStatus"]]


def calculate_hybrid_freshness(df_intraday: pd.DataFrame) -> pd.DataFrame:
    if df_intraday is None or df_intraday.empty:
        return df_intraday
    work = df_intraday.copy()
    if "time" not in work.columns and "timestamp" in work.columns:
        work["time"] = pd.to_datetime(work["timestamp"])
    work = work.sort_values("time").reset_index(drop=True)
    if len(work) < 3:
        work["FreshnessScore"] = 50.0
        work["IsFresh"] = False
        work["FreshState"] = ""
        work["FreshSince"] = ""
        return work
    work["rollingHOD"] = work["high"].cummax()
    work["volSMA10"] = work["volume"].rolling(window=10, min_periods=1).mean().shift(1)
    work["volSMA10"] = work["volSMA10"].fillna(work["volume"].mean())
    work["volRatio"] = np.where(work["volSMA10"] > 0, work["volume"] / work["volSMA10"], 0.0)
    open_price = float(work.iloc[0]["open"])
    work["closeVsOpenPct"] = (work["close"] - open_price) / open_price * 100
    ib_high = float(work.iloc[:3]["high"].max())
    ib_vol_avg = float(work.iloc[:3]["volume"].mean()) if len(work.iloc[:3]) > 0 else 0.0
    freshness_scores, is_fresh_flags = [], []
    for i in range(len(work)):
        row = work.iloc[i]
        base_score = min(100.0, max(0.0, row["closeVsOpenPct"] / 3.0 * 100.0))
        score = base_score
        is_fresh = False
        if i < 3:
            score = base_score
        elif 3 <= i <= 11:
            dist_to_ib_high = (ib_high - row["close"]) / ib_high * 100.0 if ib_high > 0 else 999.0
            vol_vs_ib = row["volume"] / ib_vol_avg if ib_vol_avg > 0 else 0.0
            if dist_to_ib_high <= 0.3 or row["close"] >= ib_high:
                score = score * 0.2 if vol_vs_ib < 0.8 else min(100.0, score * 1.5)
                is_fresh = vol_vs_ib >= 0.8
        else:
            prev_hod = float(work.iloc[i - 1]["rollingHOD"])
            dist_to_prev_hod = (prev_hod - row["close"]) / prev_hod * 100.0 if prev_hod > 0 else 999.0
            if dist_to_prev_hod <= 0.3 or row["close"] >= prev_hod:
                score = score * 0.2 if row["volRatio"] < 1.3 else min(100.0, score * 1.5)
                is_fresh = row["volRatio"] >= 1.3
            elif row["volRatio"] >= 2.0 and row["close"] > row["open"]:
                score = min(100.0, score * 1.2)
                is_fresh = True
        freshness_scores.append(round(score, 1))
        is_fresh_flags.append(is_fresh)
    work["FreshnessScore"] = freshness_scores
    work["IsFresh"] = is_fresh_flags
    prev_fresh = work["IsFresh"].shift(1).fillna(False)
    fresh_states, fresh_since = [], []
    fresh_start_time, fresh_cycle = "", 0
    for i in range(len(work)):
        curr = bool(work.loc[i, "IsFresh"])
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
        elif not curr and prev:
            state = "Fresh Lost"
            since = tstr
        else:
            state = ""
            since = ""
        fresh_states.append(state)
        fresh_since.append(since)
    work["FreshState"] = fresh_states
    work["FreshSince"] = fresh_since
    return work


def rank_delta_to_label(delta: float) -> str:
    if pd.isna(delta):
        return ""
    if delta >= 7:
        return "Buy++"
    if delta >= 4:
        return "Buy+"
    if delta >= 1:
        return "Buy"
    if delta >= 0:
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
        if pd.notna(row.get("Change")) and row.get("Change") > 0:
            score += 2
        if pd.notna(row.get("VWAP Z-Score")) and row.get("VWAP Z-Score") >= 0.30:
            score += 2
        if pd.notna(row.get("Cumulative DI")) and pd.notna(row.get("Cumulative -DI")) and row.get("Cumulative DI") > row.get("Cumulative -DI"):
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
        if pd.notna(row.get("Change")) and row.get("Change") < 0:
            score += 2
        if pd.notna(row.get("VWAP Z-Score")) and row.get("VWAP Z-Score") <= -0.30:
            score += 2
        if pd.notna(row.get("Cumulative DI")) and pd.notna(row.get("Cumulative -DI")) and row.get("Cumulative -DI") > row.get("Cumulative DI"):
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
    weights = {"5m": 1.0, "15m": 0.9, "30m": 0.8, "60m": 0.7}
    for tf, w in weights.items():
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
        out[f"{tf}Signal"] = out[col].apply(rank_delta_to_label) if col in out.columns else ""
    out["BullSignal"] = out["Bull Rank"].apply(rank_delta_to_label) if "Bull Rank" in out.columns else ""
    out["BearSignal"] = out["Bear Rank"].apply(lambda x: rank_delta_to_label(-x)) if "Bear Rank" in out.columns else ""
    out["OverallSignal"] = out["Rank Delta"].apply(rank_delta_to_label) if "Rank Delta" in out.columns else ""
    return out


def format_value(col, val):
    if pd.isna(val):
        return ""
    if col == "Change":
        return f"{float(val):.2f}%"
    if col == "IVP":
        return f"{float(val):.2f}"
    if isinstance(val, (int, float, np.integer, np.floating)):
        return f"{float(val):.2f}"
    return str(val)


def df_to_html_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "<p>No candidates found.</p>"
    df_slice = df.head(max_rows).copy()
    cols = [c for c in EMAIL_DISPLAY_COLS if c in df_slice.columns]
    if not cols:
        return "<p>No candidates found.</p>"
    header_cells = "".join(f"<th>{c}</th>" for c in cols)
    rows = []
    for _, row in df_slice.iterrows():
        cells = "".join(f"<td>{format_value(col, row.get(col))}</td>" for col in cols)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table border='1' cellspacing='0' cellpadding='3' style='border-collapse:collapse;font-size:12px;'><tr>{header_cells}</tr>{''.join(rows)}</table>"


def build_candidate_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(columns=EMAIL_DISPLAY_COLS), pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    base = df.copy()
    strict_long = base[(base["Change"] > 0) & (base["Cumulative DI"] > base["Cumulative -DI"])].copy()
    strict_short = base[(base["Change"] < 0) & (base["Cumulative -DI"] > base["Cumulative DI"])].copy()
    long_df = strict_long.sort_values(by=["Cumulative KER", "SurvivalNum", "Cumulative ADX", "Change"], ascending=[False, False, False, False], na_position="last").drop_duplicates(subset="Symbol").head(15)
    short_df = strict_short.sort_values(by=["Cumulative KER", "SurvivalNum", "Cumulative ADX", "Change"], ascending=[False, False, False, True], na_position="last").drop_duplicates(subset="Symbol").head(15)
    long_df = long_df[[c for c in EMAIL_DISPLAY_COLS if c in long_df.columns]] if not long_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    short_df = short_df[[c for c in EMAIL_DISPLAY_COLS if c in short_df.columns]] if not short_df.empty else pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    return long_df, short_df


def send_email_with_tables(long_df: pd.DataFrame, short_df: pd.DataFrame, csv_filename: str, detail_csv_filename: str) -> bool:
    try:
        scan_time = datetime.now().strftime("%d %b %Y, %H:%M")
        html = f"""
        <html><body>
        <h2>Intraday Vol Iteration Alert - {scan_time}</h2>
        <h3>Long Candidates</h3>
        {df_to_html_table(long_df)}
        <br/>
        <h3>Short Candidates</h3>
        {df_to_html_table(short_df)}
        <br/>
        <p>Scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}.</p>
        </body></html>
        """
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = f"Intraday Vol Iteration Alert - {scan_time}"
        msg.attach(MIMEText(html, "html"))
        for path in [csv_filename, detail_csv_filename]:
            if path and os.path.exists(path):
                with open(path, "rb") as attachment:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(path)}")
                msg.attach(part)
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        logger.info("EMAIL Sent successfully")
        return True
    except Exception as e:
        logger.error(f"EMAIL Failed: {e}")
        return False


def main():
    logger.info("Starting FO/FNO FYERS VOL REL EMAIL fixed script")
    init_fyers()
    logger.info("Script structure repaired. Reattach your symbol universe and run scan pipeline.")


if __name__ == "__main__":
    main()

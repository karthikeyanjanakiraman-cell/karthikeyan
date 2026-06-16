#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py

Optimized Intraday F&O scanner via Fyers API with email alerts.
- NEW STRATEGY: 6-Month Volume Climax Bands.
- Identifies the highest volume day in the last 6 months.
- The High of that day = Top Band (Resistance / Long Trigger).
- The Low of that day = Bottom Band (Support / Short Trigger / Stop Loss).
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

DAILY_LOOKBACK_DAYS = 135  # ~6 Months of trading days
INTRADAY_LOOKBACK_DAYS = 30
IVP_LOOKBACK_DAYS = 60
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

# Updated Email Columns to reflect your Climax Band Strategy
EMAIL_DISPLAY_COLS = [
    "Symbol",
    "LTP",
    "% Change",
    "Top_Band",       # The High of the Climax Day (Resistance)
    "Bottom_Band",    # The Low of the Climax Day (Support)
    "Climax_Date",    # The exact date the massive volume occurred
    "MTF_15m",
    "MTF_30m",
    "MTF_60m",
    "MTF_SCORE",
    "Last Iteration Time",
]


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


def price_stats_from_series(prices: pd.Series) -> dict:
    p = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if len(p) < 3: return {"Directional": np.nan, "Turning": np.nan, "Stability": np.nan, "Balanced": np.nan, "CumsumPlus": np.nan}
    x = np.arange(len(p), dtype=float)
    slope = float(np.polyfit(x, p.values, 1)[0])
    directional = slope + float(p.iloc[-1] - p.iloc[0])
    turning = float(np.mean(np.abs(np.diff(p.values, n=2))))
    return {"Directional": directional, "Turning": turning, "Stability": float(np.std(p.values)), "Balanced": directional - turning + float(np.std(p.values)), "CumsumPlus": float(np.sum(np.clip(np.diff(p.values), 0, None)))}


def compute_iteration_volume_profile(intra_df: Optional[pd.DataFrame], prev_close: Optional[float], top_band: float, bottom_band: float, climax_date: str) -> Tuple[Dict, pd.DataFrame]:
    if intra_df is None or intra_df.empty: return {}, pd.DataFrame()

    df = intra_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["time_only"] = pd.to_datetime(df["timestamp"]).dt.time
    
    dates = sorted(df["date"].unique())
    if len(dates) < 2: return {}, pd.DataFrame()

    current_date = dates[-1]
    curr_df = df[df["date"] == current_date].copy().sort_values("time_only")
    
    if curr_df.empty: return {}, pd.DataFrame()

    curr_df["Iteration Change"] = ((pd.to_numeric(curr_df["close"], errors="coerce") - float(prev_close)) / float(prev_close) * 100.0) if prev_close else 0.0

    rows = []
    total_iters = 0
    last_iter_time = None

    for i in range(len(curr_df)):
        total_iters += 1
        row = curr_df.iloc[i]
        t = row["time_only"]

        ps = price_stats_from_series(curr_df["Iteration Change"].iloc[: i + 1])

        rows.append({
            "Iteration No": total_iters, "Iteration Time": t.strftime("%H:%M"),
            "LTP": float(row["close"]), "Iteration Change": float(curr_df["Iteration Change"].iloc[i]),
            "Directional": ps["Directional"], "Turning": ps["Turning"], "Stability": ps["Stability"], "Balanced": ps["Balanced"], "CumsumPlus": ps.get("CumsumPlus"),
            "Top_Band": top_band,
            "Bottom_Band": bottom_band,
            "Climax_Date": climax_date
        })
        last_iter_time = t.strftime("%H:%M")

    detail_df = pd.DataFrame(rows)
    ltp = float(curr_df["close"].iloc[-1]) if not curr_df.empty else np.nan

    final_ps = price_stats_from_series(curr_df["Iteration Change"])
    summary = {
        "LTP": ltp, "Directional": final_ps["Directional"], "Turning": final_ps["Turning"], "Stability": final_ps["Stability"],
        "Balanced": final_ps["Balanced"], "CumsumPlus": final_ps.get("CumsumPlus"), 
        "Top_Band": top_band,
        "Bottom_Band": bottom_band,
        "Climax_Date": climax_date,
        "Total Iterations": total_iters, "Last Iteration Time": last_iter_time,
    }

    summary.update(build_signals_from_raw_directional(detail_df))
    summary.update(build_mtf_alignment(detail_df))
    return summary, detail_df


def scan_fno_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = load_fno_symbols_from_sectors(root_dir="sectors")
    if not symbols: return pd.DataFrame(), pd.DataFrame()

    rows, iteration_rows = [], []
    total = len(symbols)

    for idx, sym in enumerate(symbols, start=1):
        logger.info(f"CORE [{idx}/{total}] Processing {sym}")
        fyers_sym = format_fyers_symbol(sym)

        daily_df = get_fyers_history(fyers_sym, resolution="D", days_back=DAILY_LOOKBACK_DAYS)
        intra_df = get_fyers_history(fyers_sym, resolution="15", days_back=INTRADAY_LOOKBACK_DAYS)

        prev_close = None
        top_band, bottom_band, climax_date = float("inf"), float("-inf"), "N/A"

        if daily_df is not None and len(daily_df) >= 3:
            daily_df["_date_parsed"] = pd.to_datetime(daily_df["timestamp"]).dt.date
            today_date = datetime.now().date()
            
            hist_daily = daily_df[daily_df["_date_parsed"] < today_date].copy()
            if not hist_daily.empty:
                prev_close = float(hist_daily["close"].iloc[-1])
                
                # YOUR LOGIC: Find the single day with the highest volume in the last 6 months
                max_vol_idx = hist_daily["volume"].idxmax()
                climax_day = hist_daily.loc[max_vol_idx]
                
                # Extract the High (Top Band) and Low (Bottom Band) of that specific day
                top_band = float(climax_day["high"])
                bottom_band = float(climax_day["low"])
                climax_date = str(climax_day["_date_parsed"])
        
        if prev_close is None and daily_df is not None and not daily_df.empty:
            prev_close = float(daily_df["close"].iloc[-1])

        iter_summary, iter_detail = compute_iteration_volume_profile(intra_df, prev_close, top_band, bottom_band, climax_date)

        ltp = iter_summary.get("LTP")
        pct_change = ((ltp - prev_close) / prev_close * 100) if (ltp is not None and prev_close and prev_close != 0) else 0.0

        if not iter_detail.empty:
            iter_detail.insert(0, "Symbol", sym)
            iter_detail.insert(1, "% Change", pct_change)
            iteration_rows.append(iter_detail)

        rows.append({
            "Symbol": sym, "LTP": ltp, "% Change": pct_change,
            "Top_Band": iter_summary.get("Top_Band"),
            "Bottom_Band": iter_summary.get("Bottom_Band"),
            "Climax_Date": iter_summary.get("Climax_Date"),
            "Directional": iter_summary.get("Directional"), "Turning": iter_summary.get("Turning"),
            "Stability": iter_summary.get("Stability"), "Balanced": iter_summary.get("Balanced"), "CumsumPlus": iter_summary.get("CumsumPlus"),
            "5m_Signal": iter_summary.get("5m_Signal"), "15m_Signal": iter_summary.get("15m_Signal"), "30m_Signal": iter_summary.get("30m_Signal"), "60m_Signal": iter_summary.get("60m_Signal"),
            "MTF_5m": iter_summary.get("MTF_5m"), "MTF_15m": iter_summary.get("MTF_15m"), "MTF_30m": iter_summary.get("MTF_30m"), "MTF_60m": iter_summary.get("MTF_60m"),
            "MTF_SCORE": iter_summary.get("MTF_SCORE"), "MTF_ALIGN": iter_summary.get("MTF_ALIGN"),
            "Total Iterations": iter_summary.get("Total Iterations"), "Last Iteration Time": iter_summary.get("Last Iteration Time"),
        })

    return pd.DataFrame(rows), (pd.concat(iteration_rows, ignore_index=True) if iteration_rows else pd.DataFrame())


def format_value(col: str, val):
    if pd.isna(val) or val == float("inf") or val == float("-inf"): return ""
    if col == "% Change": return f"{float(val):.2f}%"
    if col in ["Top_Band", "Bottom_Band"]: return f"{float(val):.2f}"
    if col == "Climax_Date": return str(val)
    if isinstance(val, (int, float, np.integer, np.floating)): return f"{float(val):.4f}"
    return str(val)


def build_candidate_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty: return pd.DataFrame(columns=EMAIL_DISPLAY_COLS), pd.DataFrame(columns=EMAIL_DISPLAY_COLS)
    base = df.copy()
    
    for c in ["LTP", "Top_Band", "Bottom_Band", "% Change", "MTF_SCORE"]:
        if c in base.columns: base[c] = pd.to_numeric(base[c], errors="coerce")

    def prep_side_df(dfside: pd.DataFrame, side: str) -> pd.DataFrame:
        if dfside.empty: return dfside
        out = dfside.copy()

        if side == "long":
            # YOUR PURE LOGIC: Long if Live Price breaks above the Top Band of the Climax Day
            if "LTP" in out.columns and "Top_Band" in out.columns:
                out = out[out["LTP"] > out["Top_Band"]].copy()
            if out.empty: return out
            return out.sort_values(['MTF_SCORE'], ascending=[False], na_position='last')
        
        else:
            # YOUR PURE LOGIC: Short if Live Price breaches below the Bottom Band of the Climax Day
            if "LTP" in out.columns and "Bottom_Band" in out.columns:
                out = out[out["LTP"] < out["Bottom_Band"]].copy()
            if out.empty: return out
            return out.sort_values(['MTF_SCORE'], ascending=[True], na_position='last')

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
        try:
            f_dir = float(r["Directional"])
            bg = "#14532d" if f_dir > 0 else "#7f1d1d" if f_dir < 0 else "#030712"
        except Exception:
            bg = "#030712"
            
        tds = "".join(f'<td style="padding:6px 8px;border:1px solid #4b5563;color:#e5e7eb;background:{bg}">{format_value(c, r[c])}</td>' for c in cols)
        body_rows.append(f"<tr>{tds}</tr>")
    return f'<h3 style="color:{"#22c55e" if side.lower()=="long" else "#ef4444"};margin:12px 0 6px 0;">Top 1 {side.title()} - Last 15 Iterations</h3><table style="border-collapse:collapse;width:100%;background:#030712;"><thead><tr>{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def build_html_table(df: pd.DataFrame, title: str, max_rows: int = 15) -> str:
    if df is None or df.empty: return f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3><div style="padding:12px;background:#111827;color:#d1d5db;">No breakout candidates found.</div>'
    df_slice = df.head(max_rows).copy()
    cols = [c for c in EMAIL_DISPLAY_COLS if c in df_slice.columns]

    header_cells = "".join(f'<th style="padding:8px;border:1px solid #4b5563;background:#111827;color:#f9fafb;">{c}</th>' for c in cols)
    body_rows = []
    for _, row in df_slice.iterrows():
        tds = []
        for col in cols:
            if col == "% Change":
                try:
                    f_pct = float(row[col])
                    bg = "#14532d" if f_pct > 0 else "#7f1d1d" if f_pct < 0 else "#030712"
                except Exception:
                    bg = "#030712"
            else:
                bg = "#030712"
                
            tds.append(f'<td style="padding:6px 8px;border:1px solid #4b5563;color:#e5e7eb;background:{bg}">{format_value(col, row[col])}</td>')
        body_rows.append(f"<tr>{''.join(tds)}</tr>")
    return f'<h3 style="color:#f9fafb;margin:14px 0 8px 0;">{title}</h3><table style="border-collapse:collapse;width:100%;background:#030712;"><thead><tr>{header_cells}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def send_email_with_tables(long_df: pd.DataFrame, short_df: pd.DataFrame, history_df: pd.DataFrame, csv_filename: str = "", detail_csv_filename: str = "") -> bool:
    try:
        scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_body = f"""
        <html>
        <body style="background:#030712;color:#e5e7eb;padding:20px;font-family:Arial,sans-serif;">
            <h2 style="color:#facc15;">Volume Climax Band Execution Alert</h2>
            <div style="color:#cbd5e1;font-size:14px;margin-bottom:18px;">Scan completed at {scan_time}</div>
            {build_html_table(long_df, "Confirmed Long Climax Breakouts")}
            {build_history_table(history_df, "long")}
            <div style="height:28px;"></div>
            {build_html_table(short_df, "Confirmed Short Climax Breakdowns")}
            {build_history_table(history_df, "short")}
        </body>
        </html>
        """
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = sender_email, recipient_email, f"Climax Band Alert - {scan_time}"
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

    summary_csv, detail_csv = save_outputs(summary_df, detail_df, prefix="fno")
    long_df, short_df = build_candidate_tables(summary_df)
    history_df = load_iteration_history(detail_df)

    send_email_with_tables(
        long_df=long_df, short_df=short_df, history_df=history_df,
        csv_filename=summary_csv, detail_csv_filename=detail_csv
    )


if __name__ == "__main__":
    main()

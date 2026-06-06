#!/usr/bin/env python3
"""
FO_FNO_FYERS_VOL_REL_EMAIL.py

Intraday F&O scanner via Fyers API with email alerts.
Full, uncompressed implementation.
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

# --- Setup Logging ---
class UTF8Formatter(logging.Formatter):
    def format(self, record):
        msg = record.getMessage()
        record.msg = msg.encode("ascii", "ignore").decode("ascii")
        return super().format(record)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers(): logger.handlers.clear()
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(UTF8Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(ch)
warnings.filterwarnings("ignore")

# --- Globals ---
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
    "Symbol", "LTP", "% Change", "Directional", "Turning", "Stability", "Balanced", 
    "CumsumPlus", "CumsumDiff", "TurningDiff", "Turning Regime", "Dual Engine State", 
    "Trade Action", "5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal", 
    "Price_Lead_Status", "IVP", "Volatility State", "Last Iteration Time"
]

def build_signals_from_raw_directional(detail_df) -> dict:
    out = {k: float("nan") for k in ("5m_Signal", "15m_Signal", "30m_Signal", "60m_Signal", "Bull_Signal", "Bear_Signal", "Overall_Signal")}
    if detail_df is None or detail_df.empty: return out
    df = detail_df.sort_values("Iteration No") if "Iteration No" in detail_df.columns else detail_df.copy()
    vals = pd.to_numeric(df["Directional"], errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0: return out
    last = vals.size - 1
    def raw_at(offset): return float(vals[max(0, last - offset)])
    out.update({
        "5m_Signal": round(raw_at(0), 4), "15m_Signal": round(raw_at(3) if last >= 3 else raw_at(0), 4),
        "30m_Signal": round(raw_at(6) if last >= 6 else raw_at(0), 4), "60m_Signal": round(raw_at(12) if last >= 12 else raw_at(0), 4),
        "Bull_Signal": round(float(vals[vals > 0].max()) if (vals > 0).any() else 0.0, 4),
        "Bear_Signal": round(abs(float(vals[vals < 0].min())) if (vals < 0).any() else 0.0, 4),
        "Overall_Signal": round(raw_at(0), 4)
    })
    return out

def add_dual_engine_matrix(detail_df, eps=1e-4):
    if detail_df is None or detail_df.empty or not {"Symbol", "Iteration No", "Turning", "CumsumPlus"}.issubset(detail_df.columns): return detail_df if detail_df is not None else pd.DataFrame()
    out = detail_df.copy()
    out[["Turning", "CumsumPlus"]] = out[["Turning", "CumsumPlus"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out["Iteration No"] = pd.to_numeric(out["Iteration No"], errors="coerce")
    out = out.dropna(subset=["Iteration No"]).sort_values(["Symbol", "Iteration No"]).reset_index(drop=True)
    g = out.groupby("Symbol", group_keys=False)
    out["Current_Step"] = g["CumsumPlus"].diff().fillna(0.0)
    out["Prior_Step"] = out["Current_Step"].shift(1).fillna(0.0)
    out["CumsumDiff"] = out["Current_Step"]
    out["Prior_Turning"] = g["Turning"].shift(1).fillna(0.0)
    out["TurningDiff"] = out["Turning"] - out["Prior_Turning"]
    out["Friction_Expanding"] = (out["Turning"] > out["Prior_Turning"]) & (out["Turning"] > eps)
    c1, c2, c3, c4, c5 = (out["Current_Step"] > eps) & (out["Prior_Step"] <= eps) & (~out["Friction_Expanding"]), (out["Current_Step"] <= eps) & out["Friction_Expanding"], (out["Current_Step"] > eps) & out["Friction_Expanding"], (out["Current_Step"].abs() <= eps) & (~out["Friction_Expanding"]), (out["Current_Step"] > eps) & (out["Prior_Step"] > eps) & (~out["Friction_Expanding"])
    out["Dual Engine State"] = np.select([c1, c2, c3, c4, c5], ["PRISTINE_BREAKOUT", "TRUE_EXHAUSTION", "CHURNING_FAKEOUT", "HEALTHY_PAUSE", "ACTIVE_CONTINUATION"], default="TRANSITION")
    out["Trade Action"] = np.select([out["Dual Engine State"] == "PRISTINE_BREAKOUT", out["Dual Engine State"] == "TRUE_EXHAUSTION", out["Dual Engine State"] == "CHURNING_FAKEOUT", out["Dual Engine State"] == "HEALTHY_PAUSE", out["Dual Engine State"] == "ACTIVE_CONTINUATION"], ["ENTRY", "EXIT", "BLOCK_ENTRY", "HOLD", "HOLD"], default="WAIT")
    out["Turning Regime"] = np.where(out["Friction_Expanding"], "EXPANDING_FRICTION", "LOW_FRICTION")
    out["Entry Allowed"], out["Hold Allowed"], out["Exit Now"], out["Diff Status"] = out["Trade Action"].eq("ENTRY"), out["Trade Action"].eq("HOLD"), out["Trade Action"].eq("EXIT"), out["Dual Engine State"]
    return out

def merge_dual_engine_latest(summary_df, detail_df):
    if summary_df is None or summary_df.empty or detail_df is None or detail_df.empty or not {"Symbol", "CumsumDiff", "TurningDiff", "Dual Engine State"}.issubset(detail_df.columns): return summary_df.copy() if summary_df is not None else pd.DataFrame()
    latest = detail_df.sort_values(["Symbol", "Iteration No"]).groupby("Symbol", as_index=False).tail(1)[["Symbol", "CumsumDiff", "TurningDiff", "Turning Regime", "Dual Engine State", "Trade Action"]]
    return summary_df.merge(latest, on="Symbol", how="left")

def init_fyers():
    global fyers
    c_id, a_t = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID"), os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
    if c_id and a_t:
        try: fyers = fyersModel.FyersModel(client_id=c_id, is_async=False, token=a_t, log_path=""); logger.info("INIT FyersModel successful.")
        except Exception as e: logger.warning(f"INIT Failed: {e}"); fyers = None

def load_fno_symbols_from_sectors(root_dir="sectors"):
    syms = set()
    if not os.path.isdir(root_dir): return []
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(dp, fn))
                    col = next((c for c in df.columns if c.lower() in ["symbol", "symbols", "ticker"]), None)
                    if col: syms.update([s.strip() for s in df[col].dropna().astype(str) if s.strip()])
                except: pass
    return sorted(syms)

def get_fyers_history(symbol, resolution, days_back):
    if not fyers: return None
    try:
        n = datetime.now(); sd = n - timedelta(days=days_back)
        res = fyers.history(data={"symbol": symbol, "resolution": resolution, "date_format": "1", "range_from": sd.strftime("%Y-%m-%d"), "range_to": n.strftime("%Y-%m-%d"), "cont_flag": "1"})
        if res and res.get("s") == "ok" and res.get("candles"):
            df = pd.DataFrame(res["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
            return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    except: pass
    return None

def compute_iv_proxies(df):
    if df is None or len(df) < 30: return {"IVP": np.nan, "Volatility State": "Neutral Vol"}
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    c, h, l = pd.to_numeric(df["close"], errors="coerce"), pd.to_numeric(df["high"], errors="coerce"), pd.to_numeric(df["low"], errors="coerce")
    iv = ((h - l) / c.replace(0, np.nan) * 100).dropna()
    if iv.empty: return {"IVP": np.nan, "Volatility State": "Neutral Vol"}
    lb = iv.tail(min(IVP_LOOKBACK_DAYS, len(iv)))
    ivp = round((lb.lt(float(lb.iloc[-1])).sum() / len(lb)) * 100, 2)
    return {"IVP": ivp, "Volatility State": "Buyer Zone" if ivp < 30 else ("Avoid Buy Premium" if ivp > 50 else "Neutral Vol")}

def price_stats_from_series(p_series):
    p = pd.to_numeric(p_series, errors="coerce").dropna().astype(float)
    if len(p) < 3: return {"Directional": np.nan, "Turning": np.nan, "Stability": np.nan, "Balanced": np.nan, "CumsumPlus": np.nan}
    x, v = np.arange(len(p), dtype=float), p.values
    sl, nm, t, std = float(np.polyfit(x, v, 1)[0]), float(v[-1] - v[0]), float(np.mean(np.abs(np.diff(v, n=2)))), float(np.std(v))
    return {"Directional": sl + nm, "Turning": t, "Stability": std, "Balanced": sl + nm - t + std, "CumsumPlus": float(np.sum(np.clip(np.diff(v), 0, None)))}

def compute_iteration_volume_profile(intra_df, prev_close):
    if intra_df is None or intra_df.empty: return {}, pd.DataFrame()
    df = intra_df.copy(); df["time_only"], df["date"] = pd.to_datetime(df["timestamp" if "timestamp" in df.columns else "time"]).dt.time, pd.to_datetime(df["timestamp" if "timestamp" in df.columns else "time"]).dt.date
    dates = sorted(df["date"].unique())
    if len(dates) < 2: return {}, pd.DataFrame()
    curr_df = df[df["date"] == dates[-1]].copy().sort_values("time_only")
    curr_df["IterChange"] = ((pd.to_numeric(curr_df["close"], errors="coerce") - prev_close) / prev_close * 100.0) if prev_close else 0.0
    rows = []
    for i in range(len(curr_df)):
        r, t = curr_df.iloc[i], curr_df.iloc[i]["time_only"]
        ps = price_stats_from_series(curr_df["IterChange"].iloc[:i+1])
        rows.append({"Iteration No": i+1, "Iteration Time": t.strftime("%H:%M"), "LTP": float(r["close"]), "Directional": ps["Directional"], "Turning": ps["Turning"], "Stability": ps["Stability"], "Balanced": ps["Balanced"], "CumsumPlus": ps.get("CumsumPlus", np.nan)})
    detail_df = pd.DataFrame(rows)
    fps = price_stats_from_series(curr_df["IterChange"])
    summ = {"LTP": float(curr_df["close"].iloc[-1]), "Directional": fps["Directional"], "Turning": fps["Turning"], "Stability": fps["Stability"], "Balanced": fps["Balanced"], "CumsumPlus": fps.get("CumsumPlus", np.nan), "Last Iteration Time": curr_df.iloc[-1]["time_only"].strftime("%H:%M")}
    summ.update(build_signals_from_raw_directional(detail_df))
    return summ, detail_df

def scan_fno_universe():
    symbols, rows, iter_rows = load_fno_symbols_from_sectors("sectors"), [], []
    for sym in symbols:
        ddf, idf = get_fyers_history(f"NSE:{sym}-EQ", "D", 60), get_fyers_history(f"NSE:{sym}-EQ", "5", 20)
        pc = float(ddf["close"].iloc[-2]) if ddf is not None and len(ddf) >= 2 else None
        summ, dtl = compute_iteration_volume_profile(idf, pc)
        if not dtl.empty: dtl.insert(0, "Symbol", sym); iter_rows.append(dtl)
        summ["Symbol"], summ["% Change"] = sym, ((summ.get("LTP", 0) - pc) / pc * 100) if pc and summ.get("LTP") else 0.0
        summ.update(compute_iv_proxies(ddf)); rows.append(summ)
    return pd.DataFrame(rows), pd.concat(iter_rows, ignore_index=True) if iter_rows else pd.DataFrame()

def derive_rank_columns(df):
    if df is None or df.empty: return df
    out = df.copy()
    bull = out.apply(lambda r: min(13, sum([2 if r.get("% Change", 0) > 0 else 0, 2 if r.get("Directional", 0) > 0 else 0])), axis=1)
    bear = out.apply(lambda r: min(13, sum([2 if r.get("% Change", 0) < 0 else 0, 2 if r.get("Directional", 0) < 0 else 0])), axis=1)
    out["Bull Rank"], out["Bear Rank"], out["Rank Delta"] = bull, bear, bull - bear
    return out

def build_candidate_tables(df):
    if df is None or df.empty: return pd.DataFrame(), pd.DataFrame()
    base = df.copy()
    for c in ["Directional", "Turning", "Stability", "CumsumPlus"]:
        if c in base.columns: base[c] = pd.to_numeric(base[c], errors="coerce")
    def prep(d, sd):
        if d.empty: return d
        d = d[d["Directional"] > 0] if sd=="long" else d[d["Directional"] < 0]
        return d.sort_values(["Directional", "Turning"], ascending=[sd!="long", True]).drop_duplicates(subset=["Symbol"]).head(15)
    return prep(base, "long"), prep(base, "short")

def build_occurrence_table(detail_df, time_window="all", top_n=15, eps=1e-4):
    cols = ["Symbol", "Count", "CumsumPlusDiff", "TurningDiff", "First Occurrence", "Current Iteration", "Status"]
    if detail_df is None or detail_df.empty: return pd.DataFrame(columns=cols)
    df = detail_df.dropna(subset=["Iteration No"]).sort_values(["Symbol", "Iteration No"]).reset_index(drop=True)
    max_iter = int(df["Iteration No"].max()) if not df.empty else 0
    valid_states = {"PRISTINE_BREAKOUT", "ACTIVE_CONTINUATION", "HEALTHY_PAUSE"}
    # Anchor to live bar
    latest_iter = df[df["Iteration No"] == max_iter]
    active_symbols = latest_iter[latest_iter["Dual Engine State"].isin(valid_states)]["Symbol"].unique()
    df = df[df["Symbol"].isin(active_symbols)]
    if time_window == "last_10": df = df[df["Iteration No"] > max_iter - 10]
    all_records = []
    # FORWARD ITERATION: Evaluate every bar in the window
    for sym, g in df.groupby("Symbol", sort=False):
        g = g.sort_values("Iteration No").reset_index(drop=True)
        for i in range(len(g)):
            row = g.iloc[i]
            state = str(row["Dual Engine State"]).strip()
            if state not in valid_states: continue
            chain_idx = []
            idx = i
            while idx >= 0:
                st = str(g["Dual Engine State"].iloc[idx]).strip()
                if st not in valid_states: break
                chain_idx.append(idx)
                if st == "PRISTINE_BREAKOUT": break
                idx -= 1
            if not chain_idx: continue
            chain = g.iloc[sorted(chain_idx)]
            if chain["CumsumDiff"].max() <= eps: continue
            all_records.append({
                "Symbol": sym, "Count": len(chain), "CumsumPlusDiff": float(row["CumsumDiff"]), "TurningDiff": float(row["TurningDiff"]),
                "First Occurrence": str(chain.iloc[0]["Iteration Time"]), "Current Iteration": str(row["Iteration Time"]), "Status": state, "Iteration No": int(row["Iteration No"])
            })
    rec_df = pd.DataFrame(all_records)
    if rec_df.empty: return pd.DataFrame(columns=cols)
    best_records = []
    for sym, grp in rec_df.groupby("Symbol"):
        best_records.append(grp.sort_values(["CumsumPlusDiff", "TurningDiff", "Count"], ascending=[False, True, False]).iloc[0])
    return pd.DataFrame(best_records).sort_values(["CumsumPlusDiff", "TurningDiff", "Count"], ascending=[False, True, False])[cols].head(top_n).reset_index(drop=True)

def build_exceedance_tables(detail_df):
    return build_occurrence_table(detail_df, time_window="last_10", top_n=15), build_occurrence_table(detail_df, time_window="all", top_n=15)

def build_html_table(df, title):
    if df is None or df.empty: return f"<h3>{title}</h3><div>No data.</div>"
    cols = list(df.columns)
    th = "".join(f"<th>{c}</th>" for c in cols)
    tr = "".join(f"<tr>{''.join(f'<td>{r[c]}</td>' for c in cols)}</tr>" for _, r in df.iterrows())
    return f"<h3>{title}</h3><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"

def send_email_with_tables(html1, html2, s_csv, d_csv):
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = f"Intraday Alert {datetime.now().strftime('%H:%M')}", sender_email, recipient_email
    msg.attach(MIMEText(f"<html><body>{html1}{html2}</body></html>", "html"))
    for f in [s_csv, d_csv]:
        if f and os.path.exists(f):
            part = MIMEBase("application", "octet-stream"); part.set_payload(open(f, "rb").read()); encoders.encode_base64(part); part.add_header("Content-Disposition", f'attachment; filename="{f}"'); msg.attach(part)
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server: server.starttls(); server.login(sender_email, sender_password); server.sendmail(sender_email, recipient_email, msg.as_string())
    except: pass

def main():
    init_fyers()
    summ_df, dtl_df = scan_fno_universe()
    if summ_df.empty: return
    dtl_df = add_dual_engine_matrix(dtl_df)
    summ_df = merge_dual_engine_latest(summ_df, dtl_df)
    long_df, short_df = build_candidate_tables(summ_df)
    r10_df, all_df = build_exceedance_tables(dtl_df)
    s_csv, d_csv = "summary.csv", "detail.csv"
    summ_df.to_csv(s_csv); dtl_df.to_csv(d_csv)
    send_email_with_tables(build_html_table(long_df, "Longs") + build_html_table(short_df, "Shorts"), build_html_table(r10_df, "Last 10") + build_html_table(all_df, "All Time"), s_csv, d_csv)

if __name__ == "__main__": main()

#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI STRATEGY - PRODUCTION v16.6 - FRACTAL VOLUME PHYSICS
FIXED: DATA TYPE CASTING | SILENT THREAD ERRORS EXPOSED | FLOODGATES OPEN
═══════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import logging
import smtplib
import traceback
import threading
from io import StringIO
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from fyers_apiv3 import fyersModel

# ===== LOGGING SETUP =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ===== ENVIRONMENTAL CREDENTIALS =====
CLIENT_ID = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

FYERS_FO_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"

try:
    if not CLIENT_ID or not ACCESS_TOKEN:
        raise ValueError("Missing CLIENT_ID or ACCESS_TOKEN in environment variables.")
    fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False, log_path="")
    logger.info("[OK] Fyers API connected successfully")
except Exception as e:
    logger.error(f"[FATAL] Fyers API Authentication Error: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# ROBUST RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, max_per_sec=4, max_per_min=140):
        self.min_interval = 1.0 / max_per_sec
        self.max_per_min = max_per_min
        self.timestamps = []
        self.lock = threading.Lock()

    def wait(self):
        sleep_time = 0.0
        with self.lock:
            now = time.monotonic()
            self.timestamps = [ts for ts in self.timestamps if now - ts < 60.0]
            if self.timestamps:
                elapsed = now - self.timestamps[-1]
                if elapsed < self.min_interval:
                    sleep_time = self.min_interval - elapsed
            if len(self.timestamps) >= self.max_per_min:
                wait_for_min = 60.0 - (now - self.timestamps[0])
                if wait_for_min > sleep_time:
                    sleep_time = wait_for_min
            self.timestamps.append(now + sleep_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

rate_limiter = RateLimiter()

def call_with_retries(func, *args, **kwargs):
    backoff = 2  
    for attempt in range(1, 4):  
        try:
            rate_limiter.wait()
            res = func(*args, **kwargs)
            if isinstance(res, dict) and res.get('s') == 'error':
                err_msg = res.get('message', '').lower()
                if 'limit' in err_msg:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    logger.error(f"[API ERROR] Fyers returned: {res}")
                    return None
            return res
        except Exception as e:
            time.sleep(backoff)
            backoff *= 2  
    return None

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# MACRO DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def get_live_fno_symbols():
    try:
        logger.info("[INIT] Fetching Live F&O Universe...")
        text = requests.get(FYERS_FO_MASTER_URL, timeout=10).text
        raw = pd.read_csv(StringIO(text), header=None)
        underlying = raw[13].astype(str).str.strip().str.upper().dropna().unique()
        symbols = [f"NSE:{sym}-EQ" for sym in underlying if len(sym) >= 2 and sym not in {"", "SYMBOL"} and "NIFTY" not in sym and "SENSEX" not in sym]
        return sorted(set(symbols))
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch F&O master list: {e}")
        return []

def get_15m_history_6m(symbol):
    df_list = []
    now_dt = pd.Timestamp.now(tz="Asia/Kolkata")
    
    ranges = [
        (now_dt - timedelta(days=180), now_dt - timedelta(days=90)),
        (now_dt - timedelta(days=90), now_dt)
    ]
    
    for r_start, r_end in ranges:
        payload = {
            "symbol": symbol, "resolution": "15", "date_format": 1,
            "range_from": r_start.strftime("%Y-%m-%d"), 
            "range_to": r_end.strftime("%Y-%m-%d"), 
            "cont_flag": 1
        }
        # Passing payload directly as a positional argument (fixes strict API signature issues)
        res = call_with_retries(fyers.history, payload)
        
        if res and isinstance(res, dict) and 'candles' in res and res['candles']:
            chunk_df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_list.append(chunk_df)
            
    if not df_list: return None
    
    df = pd.concat(df_list, ignore_index=True)
    
    # BULLETPROOF DATA CASTING: Force all columns to numeric to prevent silent Pandas String-Math crashes
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# FRACTAL TAPE PULSE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def get_fractal_decay_score(df_session):
    total_vol = df_session['volume'].sum()
    if total_vol < 10000: return 1.0 
    
    def get_time_for_vol(vol_target):
        cum_vol = df_session['volume'].cumsum()
        idx = np.searchsorted(cum_vol.values, vol_target)
        if idx >= len(df_session): return len(df_session) * 15
        return (idx + 1) * 15

    try:
        t50 = get_time_for_vol(total_vol * 0.5)
        t75 = get_time_for_vol(total_vol * 0.75) - t50
        t87 = get_time_for_vol(total_vol * 0.875) - (t50 + t75)
        
        decay_macro = t75 / t50 if t50 > 0 else 1.0
        decay_micro = t87 / t75 if t75 > 0 else 1.0
        return (decay_macro + decay_micro) / 2
    except:
        return 1.0

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# PHYSICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def process_volume_physics(df):
    if df is None or len(df) < 10: return None
    
    df['date'] = df['timestamp'].dt.date
    
    daily_vol = df.groupby('date')['volume'].sum()
    valid_dates = daily_vol[daily_vol > 10000].index
    if len(valid_dates) == 0: return None
    
    last_date = valid_dates[-1] 
    
    df['vol_diff'] = df['volume'].diff().fillna(0)
    df['vol_violence'] = df['vol_diff'].abs()               
    df['vol_accel'] = df['vol_diff'].clip(lower=0)          
    
    df_session = df[df['date'] == last_date].copy()
    if df_session.empty: return None
    
    T_session = len(df_session) * 15 
    if T_session == 0: T_session = 15
    
    target_mass = df_session['volume'].sum()
    target_violence = df_session['vol_violence'].sum()
    target_accel = df_session['vol_accel'].sum()
    
    record_mass, record_violence, record_accel = T_session, T_session, T_session
    
    for d in valid_dates[:-1]:
        df_day = df[df['date'] == d]
        if df_day.empty: continue
        
        cum_mass = df_day['volume'].cumsum().values
        cum_violence = df_day['vol_violence'].cumsum().values
        cum_accel = df_day['vol_accel'].cumsum().values
        
        idx_mass = np.argmax(cum_mass >= target_mass) if np.any(cum_mass >= target_mass) else -1
        if idx_mass != -1: record_mass = min(record_mass, (idx_mass + 1) * 15)
                
        idx_violence = np.argmax(cum_violence >= target_violence) if np.any(cum_violence >= target_violence) else -1
        if idx_violence != -1: record_violence = min(record_violence, (idx_violence + 1) * 15)
                
        idx_accel = np.argmax(cum_accel >= target_accel) if np.any(cum_accel >= target_accel) else -1
        if idx_accel != -1: record_accel = min(record_accel, (idx_accel + 1) * 15)

    rank_mass = (record_mass / T_session) * 15.0
    rank_violence = (record_violence / T_session) * 15.0
    rank_accel = (record_accel / T_session) * 15.0
    
    net_rank_base = (rank_mass + rank_violence + rank_accel) / 3.0
    
    decay_score = get_fractal_decay_score(df_session)
    if decay_score > 1.2: 
        net_rank_final = net_rank_base / decay_score
    else:
        net_rank_final = net_rank_base

    df_session['typical_price'] = (df_session['high'] + df_session['low'] + df_session['close']) / 3.0
    vwap = (df_session['volume'] * df_session['typical_price']).sum() / target_mass if target_mass > 0 else df_session['close'].iloc[-1]
    direction = 1 if df_session['close'].iloc[-1] >= vwap else -1
    
    return {
        'rank_mass': rank_mass,
        'rank_violence': rank_violence,
        'rank_accel': rank_accel,
        'decay_score': decay_score,
        'net_rank': net_rank_final * direction,
        'target_mass': target_mass,
        'direction': 'BULLISH' if direction == 1 else 'BEARISH',
        'ltp': df_session['close'].iloc[-1],
        'session_date': last_date
    }

def scan_symbol(symbol):
    df = get_15m_history_6m(symbol)
    res = process_volume_physics(df)
    
    if not res or res['target_mass'] == 0: return None
    
    avg_ratio = abs(res['net_rank']) / 15.0
    
    if avg_ratio >= 0.80: status = "🔥 Apex Breakout"
    elif avg_ratio >= 0.60: status = "🎯 Extreme Force"
    elif avg_ratio >= 0.30: status = "⚖️ Institutional Flow"
    elif avg_ratio >= 0.10: status = "🔄 Standard Flow"
    else: status = "💤 Flat / Decayed"

    return {
        'Symbol': symbol.replace('NSE:', '').replace('-EQ', ''),
        'Net_Rank': res['net_rank'],
        'Decay': res['decay_score'],
        'Trend': res['direction'],
        'Status': status,
        'Mass_Rank': res['rank_mass'],
        'Violence_Rank': res['rank_violence'],
        'Accel_Rank': res['rank_accel'],
        'Cur_Vol': f"{int(res['target_mass']):,}",
        'LTP': res['ltp']
    }

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# EMAIL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def get_status_colors(status):
    if "Apex Breakout" in status: return "#e91e63", "#fff"
    if "Extreme Force" in status: return "#9c27b0", "#fff"
    if "Institutional Flow" in status: return "#2196f3", "#fff"
    if "Standard Flow" in status: return "#009688", "#fff"
    return "#555555", "#fff"

def generate_html_table(df, side):
    if df.empty:
        return f"<p style='color:#aaaaaa; font-size: 13px;'>No anomalous volume physics detected.</p>"
        
    html = """
    <table style="width:100%; border-collapse: collapse; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #ffffff; text-align: center; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.5);">
        <thead>
            <tr style="background-color: #2d2d30; border-bottom: 2px solid #3e3e42;">
                <th style="padding: 12px 8px;">Symbol</th>
                <th style="padding: 12px 8px;">Kinetic State</th>
                <th style="padding: 12px 8px;">Mass Rk</th>
                <th style="padding: 12px 8px;">Viol Rk</th>
                <th style="padding: 12px 8px;">Accel Rk</th>
                <th style="padding: 12px 8px;">Tape Pulse (Decay)</th>
                <th style="padding: 12px 8px; font-size: 14px; color: #00e676;">Net Speed</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for idx, row in df.reset_index(drop=True).iterrows():
        bg_color = "#1e1e1e" if idx % 2 == 0 else "#252526"
        bg, fg = get_status_colors(row['Status'])
        badge_html = f"<span style='background-color:{bg}; color:{fg}; padding:4px 8px; border-radius:12px; font-size:12px; font-weight:bold; white-space:nowrap;'>{row['Status']}</span>"
        pulse_color = "#ff5252" if row['Decay'] > 1.2 else "#69f0ae" if row['Decay'] < 1.0 else "#e0e0e0"
        
        html += f"""
            <tr style="background-color: {bg_color}; border-bottom: 1px solid #333333;">
                <td style="padding: 10px 8px; font-weight: bold; color: #ffffff;">{row['Symbol']}</td>
                <td style="padding: 10px 8px;">{badge_html}</td>
                <td style="padding: 10px 8px; color: #64b5f6;">{row['Mass_Rank']:.1f}</td>
                <td style="padding: 10px 8px; color: #ff9800;">{row['Violence_Rank']:.1f}</td>
                <td style="padding: 10px 8px; color: #4caf50;">{row['Accel_Rank']:.1f}</td>
                <td style="padding: 10px 8px; font-weight: bold; color: {pulse_color};">{row['Decay']:.2f}</td>
                <td style="padding: 10px 8px; font-weight:bold; font-size: 16px; color: #00e676;">{abs(row['Net_Rank']):.2f}</td>
            </tr>
        """
    html += "</tbody></table>"
    return html

def send_email_report(results_list):
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
        logger.warning("[EMAIL] Missing SMTP configurations.")
        return

    df = pd.DataFrame(results_list)
    bulls = df[df['Trend'] == 'BULLISH'].sort_values(by=['Net_Rank'], ascending=False).head(15)
    bears = df[df['Trend'] == 'BEARISH'].sort_values(by=['Net_Rank'], ascending=True).head(15)

    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"] = SENDER_EMAIL, RECIPIENT_EMAIL
    msg["Subject"] = f"ASIT Physics Matrix - {datetime.now().strftime('%d %b %H:%M')}"

    html = f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; padding: 20px;">
      <div style="text-align: center; margin-bottom: 30px;">
          <h2 style="color: #00e676; margin-bottom: 5px;">ASIT FRACTAL PHYSICS ENGINE (v16.6)</h2>
      </div>
      <h3 style="color: #4caf50;">🚀 Institutional Inflows (Above VWAP)</h3>
      {generate_html_table(bulls, "BULLISH")}
      <br><br>
      <h3 style="color: #f44336;">🔻 Institutional Outflows (Below VWAP)</h3>
      {generate_html_table(bears, "BEARISH")}
    </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        logger.info(f"[EMAIL] ASIT Matrix successfully dispatched.")
    except Exception as e:
        logger.error(f"[EMAIL] SMTP Transmission Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# EXECUTION CORE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("[LAUNCH] ASIT v16.6 - FRACTAL VOLUME PHYSICS")
    print("=" * 80)
    
    symbols = get_live_fno_symbols()
    if not symbols: 
        logger.error("[TERMINATE] No targets found.")
        sys.exit(1)
        
    logger.info(f"Loaded {len(symbols)} F&O Symbols. Beginning Physics Analysis...")
    
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(scan_symbol, sym): sym for sym in symbols}
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
                if res: results.append(res)
                if idx % 25 == 0: logger.info(f"[SCAN] Physics processed: {idx}/{len(symbols)}")
            except Exception as e:
                # NO MORE SILENT FAILURES! This will print exactly why a stock crashed.
                logger.error(f"[FATAL] Exception inside thread for {futures[future]}: {e}")
                traceback.print_exc() 

    if results:
        send_email_report(results)
        print("=" * 80)
        print(f"[SUCCESS] Ranked {len(results)} symbols. Dispatched.")
        print("=" * 80)
    else:
        logger.error("[WARNING] 0 symbols returned. Check API errors above.")

if __name__ == "__main__":
    main()

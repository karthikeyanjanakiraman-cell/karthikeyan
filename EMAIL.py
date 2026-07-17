#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI STRATEGY - PRODUCTION v15.0 - PURE VOLUME PHYSICS
ZERO LAG | NO PRICE INDICATORS | 6-MONTH TAPE ACCUMULATION ENGINE
Pillars: 1. Mass (Total Vol) | 2. Violence (Vol Volatility) | 3. Acceleration (Vol ROC)
═══════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import logging
import smtplib
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
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
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
                if 'limit' in err_msg: raise ValueError(f"Rate Limit: {err_msg}")
                else: return None
            return res
        except Exception:
            time.sleep(backoff)
            backoff *= 2  
    return None

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# MACRO DATA FETCHING (180 DAYS)
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
    
    # 2 Chunks to get exactly 6 months of Intraday data without breaking API limits
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
        res = call_with_retries(fyers.history, data=payload)
        
        if res and isinstance(res, dict) and 'candles' in res and res['candles']:
            chunk_df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_list.append(chunk_df)
            
    if not df_list: return None
    
    df = pd.concat(df_list, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# PURE VOLUME PHYSICS ENGINE (THE 3 PILLARS)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def process_volume_physics(df):
    if df is None or len(df) < 10: return None
    
    df['date'] = df['timestamp'].dt.date
    dates = df['date'].unique()
    today = dates[-1]
    
    # Pillar Base Calculations
    df['vol_diff'] = df['volume'].diff().fillna(0)
    df['vol_violence'] = df['vol_diff'].abs()               # Violence: Absolute fluctuations
    df['vol_accel'] = df['vol_diff'].clip(lower=0)          # Acceleration: Only positive thrusts
    
    # --- TODAY'S TARGETS (From 9:15 AM to Current Time) ---
    df_today = df[df['date'] == today]
    if df_today.empty: return None
    
    T_today = len(df_today) * 15 # Current Time in minutes since 9:15 AM
    if T_today == 0: T_today = 15
    
    target_mass = df_today['volume'].sum()
    target_violence = df_today['vol_violence'].sum()
    target_accel = df_today['vol_accel'].sum()
    
    # Set default records to Today's time (Assumes today is the record unless proven otherwise)
    record_mass = T_today
    record_violence = T_today
    record_accel = T_today
    
    # --- THE 6-MONTH TAPE READER ---
    for d in dates[:-1]:
        df_day = df[df['date'] == d]
        if df_day.empty: continue
        
        # Cumulative flow for this historical day
        cum_mass = df_day['volume'].cumsum().values
        cum_violence = df_day['vol_violence'].cumsum().values
        cum_accel = df_day['vol_accel'].cumsum().values
        
        # Find the absolute fastest time this historical day hit today's targets
        idx_mass = np.argmax(cum_mass >= target_mass) if np.any(cum_mass >= target_mass) else -1
        if idx_mass != -1:
            t_mass = (idx_mass + 1) * 15
            if t_mass < record_mass: record_mass = t_mass
                
        idx_violence = np.argmax(cum_violence >= target_violence) if np.any(cum_violence >= target_violence) else -1
        if idx_violence != -1:
            t_violence = (idx_violence + 1) * 15
            if t_violence < record_violence: record_violence = t_violence
                
        idx_accel = np.argmax(cum_accel >= target_accel) if np.any(cum_accel >= target_accel) else -1
        if idx_accel != -1:
            t_accel = (idx_accel + 1) * 15
            if t_accel < record_accel: record_accel = t_accel

    # --- VELOCITY SCORING ---
    # Ratio = Record Shortest Time / Current Time
    ratio_mass = record_mass / T_today
    ratio_violence = record_violence / T_today
    ratio_accel = record_accel / T_today
    
    rank_mass = ratio_mass * 15.0
    rank_violence = ratio_violence * 15.0
    rank_accel = ratio_accel * 15.0
    
    # --- DIRECTION: DAILY VWAP ---
    df_today = df_today.copy()
    df_today['typical_price'] = (df_today['high'] + df_today['low'] + df_today['close']) / 3.0
    df_today['vol_price'] = df_today['volume'] * df_today['typical_price']
    
    total_vol = df_today['volume'].sum()
    vwap = df_today['vol_price'].sum() / total_vol if total_vol > 0 else df_today['close'].iloc[-1]
    direction = 1 if df_today['close'].iloc[-1] >= vwap else -1
    
    # Net Rank Engine Output
    net_rank = ((rank_mass + rank_violence + rank_accel) / 3.0) * direction
    
    return {
        'rank_mass': rank_mass,
        'rank_violence': rank_violence,
        'rank_accel': rank_accel,
        'net_rank': net_rank,
        'target_mass': target_mass,
        'direction': 'BULLISH' if direction == 1 else 'BEARISH',
        'ltp': df_today['close'].iloc[-1]
    }

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# SCANNER ROUTINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def scan_symbol(symbol):
    df = get_15m_history_6m(symbol)
    res = process_volume_physics(df)
    
    if not res: return None
    if res['target_mass'] == 0: return None
    
    # Ensure it's performing at at least 30% of its historical maximum capability
    avg_ratio = abs(res['net_rank']) / 15.0
    if avg_ratio < 0.30: return None
    
    if avg_ratio >= 0.90: status = "🔥 Apex Breakout"
    elif avg_ratio >= 0.70: status = "🎯 Extreme Force"
    elif avg_ratio >= 0.50: status = "⚖️ Institutional Accumulation"
    else: status = "🔄 Standard Flow"

    return {
        'Symbol': symbol,
        'Net_Rank': round(res['net_rank'], 2),
        'Trend': res['direction'],
        'Status': status,
        'Mass_Rank': round(res['rank_mass'], 2),
        'Violence_Rank': round(res['rank_violence'], 2),
        'Accel_Rank': round(res['rank_accel'], 2),
        'Cur_Vol': f"{int(res['target_mass']):,}",
        'LTP': round(res['ltp'], 2)
    }

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# HTML & EMAIL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def get_status_colors(status):
    if "Apex Breakout" in status: return "#e91e63", "#fff"
    if "Extreme Force" in status: return "#9c27b0", "#fff"
    if "Accumulation" in status: return "#2196f3", "#fff"
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
                <th style="padding: 12px 8px;">Vol Mass</th>
                <th style="padding: 12px 8px; color: #64b5f6;">Mass Rank</th>
                <th style="padding: 12px 8px; color: #ff9800;">Violence Rank</th>
                <th style="padding: 12px 8px; color: #4caf50;">Accel Rank</th>
                <th style="padding: 12px 8px;">LTP</th>
                <th style="padding: 12px 8px; font-size: 14px; color: #00e676;">Net Speed</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for idx, row in df.reset_index(drop=True).iterrows():
        bg_color = "#1e1e1e" if idx % 2 == 0 else "#252526"
        
        bg, fg = get_status_colors(row['Status'])
        badge_html = f"<span style='background-color:{bg}; color:{fg}; padding:4px 8px; border-radius:12px; font-size:12px; font-weight:bold; white-space:nowrap;'>{row['Status']}</span>"
        
        html += f"""
            <tr style="background-color: {bg_color}; border-bottom: 1px solid #333333;">
                <td style="padding: 10px 8px; font-weight: bold; color: #ffffff;">{row['Symbol'].replace('NSE:', '').replace('-EQ', '')}</td>
                <td style="padding: 10px 8px;">{badge_html}</td>
                <td style="padding: 10px 8px; color: #e0e0e0;">{row['Cur_Vol']}</td>
                <td style="padding: 10px 8px; font-weight: bold; color: #64b5f6;">{row['Mass_Rank']}</td>
                <td style="padding: 10px 8px; font-weight: bold; color: #ff9800;">{row['Violence_Rank']}</td>
                <td style="padding: 10px 8px; font-weight: bold; color: #4caf50;">{row['Accel_Rank']}</td>
                <td style="padding: 10px 8px; font-weight: 600;">{row['LTP']:.2f}</td>
                <td style="padding: 10px 8px; font-weight:bold; font-size: 16px; color: #00e676;">{abs(row['Net_Rank']):.2f}</td>
            </tr>
        """
    html += "</tbody></table>"
    return html

def send_email_report(results_list):
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
        logger.warning("[EMAIL] Missing SMTP configurations. Dispatch halted.")
        return

    df = pd.DataFrame(results_list)
    if df.empty: return
    
    bulls = df[df['Trend'] == 'BULLISH'].sort_values(by=['Net_Rank'], ascending=False).head(15)
    bears = df[df['Trend'] == 'BEARISH'].sort_values(by=['Net_Rank'], ascending=True).head(15)

    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"] = SENDER_EMAIL, RECIPIENT_EMAIL
    msg["Subject"] = f"Volume Physics Intraday Matrix - {datetime.now().strftime('%H:%M')}"

    html = f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; padding: 20px;">
      
      <div style="text-align: center; margin-bottom: 30px;">
          <h2 style="color: #ffffff; margin-bottom: 5px;">Pure Volume Physics Engine (v15.0)</h2>
          <p style="color: #aaaaaa; margin-top: 0;"><b>Price Eliminated. Time/Volume Physics Active:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
      </div>
      
      <h3 style="color: #4caf50; border-bottom: 2px solid #4caf50; padding-bottom: 5px; display: inline-block;">🚀 Top 15 Institutional Inflows (Above VWAP)</h3>
      {generate_html_table(bulls, "BULLISH")}
      
      <br><br>
      
      <h3 style="color: #f44336; border-bottom: 2px solid #f44336; padding-bottom: 5px; display: inline-block;">🔻 Top 15 Institutional Outflows (Below VWAP)</h3>
      {generate_html_table(bears, "BEARISH")}
      
      <div style="margin-top: 40px; padding: 15px; background-color: #1e1e1e; border-radius: 8px; border-left: 4px solid #00e676;">
          <h4 style="color: #ffffff; margin-top: 0;">The 3 Pillars of Volume Physics (Ranked out of 15.0):</h4>
          <ul style="color: #cccccc; line-height: 1.6;">
            <li><b style="color:#64b5f6;">Mass Rank (Total Volume):</b> Tracks how fast the total volume aggregated today compared to the 6-month historical peak speed.</li>
            <li><b style="color:#ff9800;">Violence Rank (Vol Volatility):</b> Tracks how fast the volume fluctuated (bar-to-bar absolute change). High scores mean aggressive warfare.</li>
            <li><b style="color:#4caf50;">Accel Rank (Vol ROC):</b> Tracks the rate of volume acceleration. Only counts positive momentum thrusts.</li>
            <li><b style="color:#00e676;">Net Speed:</b> The absolute average of Mass, Violence, and Acceleration. A Net Score of 15.0 is mathematically the strongest institutional footprint seen in half a year.</li>
          </ul>
      </div>
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
        logger.info(f"[EMAIL] Volume Physics Matrix successfully dispatched.")
    except Exception as e:
        logger.error(f"[EMAIL] SMTP Transmission Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# SYSTEM START
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("[LAUNCH] ASIT v15.0 - PURE VOLUME PHYSICS")
    print("=" * 80)
    symbols = get_live_fno_symbols()
    if not symbols: 
        logger.error("[TERMINATE] No underlying options targets found.")
        sys.exit(1)
        
    results = []
    # Safeguarded multithreading (3 workers) specifically tuned to allow 180-day double fetch
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(scan_symbol, sym): sym for sym in symbols}
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
                if res: results.append(res)
                if idx % 20 == 0: logger.info(f"[SCAN] Computing Volume Physics: {idx}/{len(symbols)} complete...")
            except Exception:
                pass

    if results:
        send_email_report(results)
        print("=" * 80)
        print("[SUCCESS] Physics Engine Scan Concluded.")
        print("=" * 80)
    else:
        logger.error("[FATAL ERROR] Matrix calculation failed.")

if __name__ == "__main__":
    main()

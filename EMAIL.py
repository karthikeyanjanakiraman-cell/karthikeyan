#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI STRATEGY - PRODUCTION v13.0 - PURE KINETIC EDITION
ZERO-LAG SPEED FORMULA | VWAP DIRECTION | 10k VOLUME TIME COMPARISON
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
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def get_live_fno_symbols():
    try:
        logger.info("[INIT] Fetching Live F&O Universe from FYERS Master...")
        text = requests.get(FYERS_FO_MASTER_URL, timeout=10).text
        raw = pd.read_csv(StringIO(text), header=None)
        underlying = raw[13].astype(str).str.strip().str.upper().dropna().unique()
        symbols = [f"NSE:{sym}-EQ" for sym in underlying if len(sym) >= 2 and sym not in {"", "SYMBOL"} and "NIFTY" not in sym and "SENSEX" not in sym]
        return sorted(set(symbols))
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch F&O master list: {e}")
        return []

def get_15m_history(symbol):
    try:
        now_dt = pd.Timestamp.now(tz="Asia/Kolkata")
        date_from = (now_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        
        payload = {
            "symbol": symbol, "resolution": "15", "date_format": 1,
            "range_from": date_from, "range_to": now_dt.strftime("%Y-%m-%d"), "cont_flag": 1
        }
        res = call_with_retries(fyers.history, data=payload)
        if not res or not isinstance(res, dict): return None
            
        candles = res.get('candles', [])
        if not candles: return None

        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# PURE KINETIC MATH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def process_pure_kinetic(df):
    if df is None or len(df) < 5: return None
    df = df.copy()
    
    # 1. VWAP Calculation (Resets Daily)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
    df['vol_price'] = df['volume'] * df['typical_price']
    cum_vol = df.groupby(df['timestamp'].dt.date)['volume'].cumsum()
    cum_vol_price = df.groupby(df['timestamp'].dt.date)['vol_price'].cumsum()
    df['vwap'] = cum_vol_price / np.where(cum_vol == 0, 1, cum_vol)
    
    # 2. Kinetic Speed (Seconds to clear 10,000 shares on a 15-min / 900s bar)
    safe_vol = np.where(df['volume'] == 0, 1, df['volume'])
    df['cur_time_10k'] = (10000 / safe_vol) * 900
    
    # 3. Peak Velocity Tracker (Lowest time achieved TODAY)
    df['lowest_time_today'] = df.groupby(df['timestamp'].dt.date)['cur_time_10k'].cummin()
    
    # 4. Pure Kinetic Rank Calculation (Out of 15)
    # Ratio = Lowest Time / Current Time (e.g., 45s / 90s = 0.5 ratio)
    df['kinetic_ratio'] = df['lowest_time_today'] / df['cur_time_10k']
    df['kinetic_rank'] = df['kinetic_ratio'] * 15.0
    
    # 5. Directional Engine: Positive for Long (Above VWAP), Negative for Short (Below VWAP)
    df['final_rank'] = np.where(df['close'] >= df['vwap'], df['kinetic_rank'], -df['kinetic_rank'])
    
    return df

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# SCANNER ROUTINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def format_speed(seconds):
    if pd.isna(seconds) or seconds == float('inf'): return "N/A"
    if seconds < 60: return f"{int(seconds)}s"
    return f"{seconds/60:.1f}m"

def get_pct_change(df):
    # Find the close price of the previous trading day
    today_date = df['timestamp'].iloc[-1].date()
    prev_days = df[df['timestamp'].dt.date < today_date]
    if not prev_days.empty:
        prev_close = prev_days.iloc[-1]['close']
        cur_close = df.iloc[-1]['close']
        return ((cur_close - prev_close) / prev_close) * 100
    return 0.0

def scan_symbol(symbol):
    df = get_15m_history(symbol)
    df_calc = process_pure_kinetic(df)
    
    if df_calc is None or df_calc.empty: return None
    
    latest = df_calc.iloc[-1]
    
    pct_change = get_pct_change(df_calc)
    rank = latest['final_rank']
    ratio = latest['kinetic_ratio']
    
    # Determine Status
    if ratio >= 0.90: status = "🔥 Peak Velocity"
    elif ratio >= 0.70: status = "🎯 High Momentum"
    elif ratio >= 0.50: status = "⚖️ Steady Flow"
    else: status = "📉 Fading Speed"
    
    return {
        'Symbol': symbol,
        'Rank': round(rank, 2),
        'Trend': 'BULLISH' if rank > 0 else 'BEARISH',
        'Status': status,
        'Peak_10k': format_speed(latest['lowest_time_today']),
        'Cur_10k': format_speed(latest['cur_time_10k']),
        'Ratio': ratio,
        'LTP': round(latest['close'], 2),
        '% Change': round(pct_change, 2)
    }

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# HTML & EMAIL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def get_status_colors(status):
    if "Peak Velocity" in status: return "#9c27b0", "#fff"
    if "High Momentum" in status: return "#d4af37", "#000"
    if "Steady Flow" in status: return "#2196f3", "#fff"
    return "#555555", "#fff"

def generate_html_table(df, side):
    if df.empty:
        return f"<p style='color:#aaaaaa; font-size: 13px;'>No active {side.lower()} flow detected.</p>"
        
    html = """
    <table style="width:100%; border-collapse: collapse; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #ffffff; text-align: center; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.5);">
        <thead>
            <tr style="background-color: #2d2d30; border-bottom: 2px solid #3e3e42;">
                <th style="padding: 12px 8px;">Symbol</th>
                <th style="padding: 12px 8px;">Kinetic State</th>
                <th style="padding: 12px 8px;">Peak 10k Time</th>
                <th style="padding: 12px 8px;">Cur 10k Time</th>
                <th style="padding: 12px 8px;">LTP</th>
                <th style="padding: 12px 8px;">% Chg</th>
                <th style="padding: 12px 8px; color: #00e676;">Pure Rank</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for idx, row in df.reset_index(drop=True).iterrows():
        bg_color = "#1e1e1e" if idx % 2 == 0 else "#252526"
        pct = row['% Change']
        pct_color = "#4caf50" if pct > 0 else "#f44336" if pct < 0 else "#ffffff"
        
        bg, fg = get_status_colors(row['Status'])
        badge_html = f"<span style='background-color:{bg}; color:{fg}; padding:4px 8px; border-radius:12px; font-size:12px; font-weight:bold; white-space:nowrap;'>{row['Status']}</span>"
        
        # Highlight if current speed matches peak speed
        cur_color = "#4caf50" if row['Cur_10k'] == row['Peak_10k'] else "#ff9800"
        
        html += f"""
            <tr style="background-color: {bg_color}; border-bottom: 1px solid #333333;">
                <td style="padding: 10px 8px; font-weight: bold; color: #64b5f6;">{row['Symbol'].replace('NSE:', '').replace('-EQ', '')}</td>
                <td style="padding: 10px 8px;">{badge_html}</td>
                <td style="padding: 10px 8px; color: #b388ff; font-weight: bold;">{row['Peak_10k']}</td>
                <td style="padding: 10px 8px; color: {cur_color}; font-weight: bold;">{row['Cur_10k']}</td>
                <td style="padding: 10px 8px; font-weight: 600;">{row['LTP']:.2f}</td>
                <td style="padding: 10px 8px; color: {pct_color}; font-weight:bold;">{pct:+.2f}%</td>
                <td style="padding: 10px 8px; font-weight:bold; font-size: 15px; color: #00e676;">{abs(row['Rank']):.2f}</td>
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
    
    # Filter out the fading stocks, we only want setups operating at 50% peak efficiency or higher
    df_elite = df[df['Ratio'] >= 0.50].copy()

    # Sort purely by the mathematical Rank
    bulls = df_elite[df_elite['Trend'] == 'BULLISH'].sort_values(by=['Rank'], ascending=False).head(15)
    bears = df_elite[df_elite['Trend'] == 'BEARISH'].sort_values(by=['Rank'], ascending=True).head(15)

    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"] = SENDER_EMAIL, RECIPIENT_EMAIL
    msg["Subject"] = f"Pure Kinetic Execution Matrix - {datetime.now().strftime('%H:%M')}"

    html = f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; padding: 20px;">
      
      <div style="text-align: center; margin-bottom: 30px;">
          <h2 style="color: #ffffff; margin-bottom: 5px;">Pure Kinetic Execution Matrix (v13.0)</h2>
          <p style="color: #aaaaaa; margin-top: 0;"><b>Zero-Lag Speed Engine Active:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
      </div>
      
      <h3 style="color: #4caf50; border-bottom: 2px solid #4caf50; padding-bottom: 5px; display: inline-block;">🚀 Top 15 Velocity Longs (Above VWAP)</h3>
      {generate_html_table(bulls, "BULLISH")}
      
      <br><br>
      
      <h3 style="color: #f44336; border-bottom: 2px solid #f44336; padding-bottom: 5px; display: inline-block;">🔻 Top 15 Velocity Shorts (Below VWAP)</h3>
      {generate_html_table(bears, "BEARISH")}
      
      <div style="margin-top: 40px; padding: 15px; background-color: #1e1e1e; border-radius: 8px; border-left: 4px solid #00e676;">
          <h4 style="color: #ffffff; margin-top: 0;">Math Blueprint:</h4>
          <ul style="color: #cccccc; line-height: 1.6;">
            <li><b>Formula:</b> <code>(Lowest Time Achieved Today / Current Time) * 15</code></li>
            <li><b>Pure Rank:</b> A Rank of <b>15.00</b> means the stock is currently matching its absolute fastest execution speed of the day.</li>
            <li><b>Direction:</b> Determined strictly by Price vs Daily VWAP.</li>
            <li><b>Efficiency:</b> Any stock where current speed drops below 50% of the peak speed is instantly killed from this list.</li>
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
        logger.info(f"[EMAIL] Pure Kinetic Matrix successfully dispatched.")
    except Exception as e:
        logger.error(f"[EMAIL] SMTP Transmission Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# SYSTEM START
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("[LAUNCH] ASIT v13.0 - PURE KINETIC EDITION")
    print("=" * 80)
    symbols = get_live_fno_symbols()
    if not symbols: 
        logger.error("[TERMINATE] No underlying options targets found.")
        sys.exit(1)
        
    results = []
    # With multi-timeframe fetching removed, we can safely bump workers to 3 for ultra-fast scans
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(scan_symbol, sym): sym for sym in symbols}
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
                if res: results.append(res)
                if idx % 20 == 0: logger.info(f"[SCAN] Processing Pure Velocity: {idx}/{len(symbols)} complete...")
            except Exception:
                pass

    if results:
        send_email_report(results)
        print("=" * 80)
        print("[SUCCESS] Pure Kinetic Scan Concluded.")
        print("=" * 80)
    else:
        logger.error("[FATAL ERROR] Matrix calculation failed.")

if __name__ == "__main__":
    main()

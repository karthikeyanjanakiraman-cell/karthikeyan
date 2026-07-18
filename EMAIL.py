#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI TMV ENGINE - PRODUCTION v19.0 (BULLETPROOF EDITION)
FIXES APPLIED: Pre-market exclusion, Exact Historical Max split, Safe Kinetic Tiers, API Limits
═══════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import time
import logging
import smtplib
from io import StringIO
from datetime import datetime, timedelta, time as dt_time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from fyers_apiv3 import fyersModel

# ===== LOGGING SETUP =====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== CREDENTIALS =====
CLIENT_ID = os.environ.get("CLIENT_ID", "YOUR_CLIENT_ID")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "YOUR_TOKEN")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "youremail@gmail.com")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "your_app_password")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "recipient@gmail.com")

FYERS_FO_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False, log_path="")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 1. DYNAMIC F&O UNIVERSE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def fetch_fo_universe():
    logger.info("Fetching Master F&O Universe...")
    try:
        response = requests.get(FYERS_FO_MASTER_URL)
        df = pd.read_csv(StringIO(response.text), header=None)
        raw_symbols = df[1].astype(str).tolist()
        
        base_symbols = set()
        for s in raw_symbols:
            if s.startswith('NSE:'):
                match = re.search(r'NSE:([A-Z&]+)\d+', s)
                if match: base_symbols.add(match.group(1))
        
        ignore_list = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'}
        base_symbols = base_symbols - ignore_list
        
        return [f"NSE:{sym}-EQ" for sym in base_symbols]
    except Exception as e:
        logger.error(f"Failed to fetch Universe: {e}")
        return []

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 2. BULLETPROOF KINETIC ACCELERATION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def calculate_kinetic_acceleration(df_today, target_vol):
    """Calculates time intervals only for the volume tiers actually reached today."""
    if target_vol <= 0 or df_today.empty: return 1.0
    
    cum_vol = df_today['volume'].cumsum().values
    
    def get_time_to_reach(pct):
        target = target_vol * pct
        idx = np.searchsorted(cum_vol, target)
        if idx < len(cum_vol): return (idx + 1) * 15 # Minutes elapsed
        return None # Target not reached yet today

    t1 = get_time_to_reach(0.5)   # Time to 50%
    t2 = get_time_to_reach(0.75)  # Time to 75%
    t3 = get_time_to_reach(0.875) # Time to 87.5%
    t4 = get_time_to_reach(1.0)   # Time to 100%
    
    # Measure deltas (time spent inside each tier)
    if t1 and t2 and t3:
        delta_1 = t2 - t1
        delta_2 = t3 - t2
        
        if t4: # Reached 100%
            delta_3 = t4 - t3
            if delta_3 < delta_2 < delta_1: return 2.0 # Maximum Acceleration
        elif delta_2 < delta_1:
            return 1.5 # Accelerating up to 87.5%
            
    return 1.0

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 3. CORE PHYSICS ENGINE (Fixed time boundaries & historical splitting)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def extract_raw_physics(symbol):
    try:
        time.sleep(0.1) # Prevent API Rate Limit (429 Error)
        now_dt = pd.Timestamp.now(tz="Asia/Kolkata")
        
        # Max 90 days for 15m resolution to avoid Fyers API rejection
        payload = {
            "symbol": symbol, "resolution": "15", "date_format": 1,
            "range_from": (now_dt - timedelta(days=90)).strftime("%Y-%m-%d"),
            "range_to": now_dt.strftime("%Y-%m-%d"), "cont_flag": 1
        }
        res = fyers.history(payload)
        
        if not res or 'candles' not in res or len(res['candles']) == 0: return None
        
        df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata")
        
        # FIX: Strict Time Boundaries (9:15 AM to Current Time)
        market_open = dt_time(9, 15)
        current_time = now_dt.time()
        
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        
        # Only keep candles during market hours, up to the current time of day
        df = df[(df['time'] >= market_open) & (df['time'] <= current_time)]
        if df.empty: return None
        
        # FIX: Separate History from Today
        today_date = now_dt.date()
        history_df = df[df['date'] < today_date]
        today_df = df[df['date'] == today_date]
        
        if history_df.empty or today_df.empty: return None
        
        # Calculate Historical Ceilings
        daily_groups = history_df.groupby('date').agg({'volume': 'sum', 'high': 'max', 'low': 'min'})
        daily_groups['range'] = daily_groups['high'] - daily_groups['low']
        
        max_vol = daily_groups['volume'].max()
        max_range = daily_groups['range'].max()
        if max_vol == 0 or max_range == 0: return None
        
        # Today's Action
        curr_vol = today_df['volume'].sum()
        curr_range = (today_df['high'].max() - today_df['low'].min())
        
        vol_ratio = curr_vol / max_vol
        volatility_ratio = curr_range / max_range
        
        # Trend Anchor (OBV EMA)
        df['price_dir'] = np.where(df['close'] > df['close'].shift(1), 1, 
                          np.where(df['close'] < df['close'].shift(1), -1, 0))
        df['obv'] = (df['price_dir'] * df['volume']).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=10, min_periods=1).mean()
        
        trend = 'BULLISH' if df['obv'].iloc[-1] > df['obv_ema'].iloc[-1] else 'BEARISH'
        accel = calculate_kinetic_acceleration(today_df, max_vol)
        
        return {
            'Symbol': symbol.replace('NSE:', '').replace('-EQ', ''),
            'Vol_Ratio': vol_ratio,
            'Volat_Ratio': volatility_ratio,
            'Accel': accel,
            'Trend': trend,
            'LTP': df['close'].iloc[-1]
        }
    except Exception as e:
        return None

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 4. EXECUTION & EMAIL
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def send_html_email(bullish_df, bearish_df):
    logger.info("Formatting and dispatching HTML Email...")
    html = f"""
    <html>
      <body style="font-family: Arial; padding: 20px;">
        <h2 style="color: #2e7d32;">🔥 TOP BULLISH BREAKOUTS</h2>
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
          <tr style="background-color: #2e7d32; color: white;">
            <th>Symbol</th><th>LTP</th><th>Vol Ratio</th><th>Volat Exp</th><th>TMV Score</th>
          </tr>
          {"".join(f"<tr><td style='border-bottom: 1px solid #ddd; padding: 8px;'>{row['Symbol']}</td><td style='border-bottom: 1px solid #ddd;'>{row['LTP']}</td><td style='border-bottom: 1px solid #ddd;'>{row['Vol_Ratio']}x</td><td style='border-bottom: 1px solid #ddd;'>{row['Volat_Ratio']}x</td><td style='border-bottom: 1px solid #ddd;'>{row['TMV_Score']}</td></tr>" for _, row in bullish_df.iterrows())}
        </table>

        <h2 style="color: #c62828; margin-top: 40px;">🩸 TOP BEARISH BREAKDOWNS</h2>
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
          <tr style="background-color: #c62828; color: white;">
            <th>Symbol</th><th>LTP</th><th>Vol Ratio</th><th>Volat Exp</th><th>TMV Score</th>
          </tr>
          {"".join(f"<tr><td style='border-bottom: 1px solid #ddd; padding: 8px;'>{row['Symbol']}</td><td style='border-bottom: 1px solid #ddd;'>{row['LTP']}</td><td style='border-bottom: 1px solid #ddd;'>{row['Vol_Ratio']}x</td><td style='border-bottom: 1px solid #ddd;'>{row['Volat_Ratio']}x</td><td style='border-bottom: 1px solid #ddd;'>{row['TMV_Score']}</td></tr>" for _, row in bearish_df.iterrows())}
        </table>
      </body>
    </html>
    """
    
    msg = MIMEMultipart("alternative")
    msg['Subject'] = f"TMV Engine Alert - {datetime.now().strftime('%I:%M %p')}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg.attach(MIMEText(html, "html"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logger.info("Email dispatched successfully.")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def main():
    symbols = fetch_fo_universe()
    if not symbols: return
        
    results = []
    # FIX: max_workers lowered to 4 to prevent API blocking
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_raw_physics, sym): sym for sym in symbols}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    df_results = pd.DataFrame(results)
    if df_results.empty: 
        logger.error("No valid data processed. Exiting.")
        return
        
    df_results['Vol_Rank'] = df_results['Vol_Ratio'].rank(pct=True)
    df_results['Volat_Rank'] = df_results['Volat_Ratio'].rank(pct=True)
    df_results['Accel_Rank'] = df_results['Accel'].rank(pct=True)
    
    df_results['TMV_Score'] = (
        (df_results['Vol_Rank'] * 0.5) + 
        (df_results['Volat_Rank'] * 0.3) + 
        (df_results['Accel_Rank'] * 0.2)
    ) * 100
    
    df_results = df_results.round(2)
    
    bullish_df = df_results[df_results['Trend'] == 'BULLISH'].sort_values('TMV_Score', ascending=False).head(15)
    bearish_df = df_results[df_results['Trend'] == 'BEARISH'].sort_values('TMV_Score', ascending=False).head(15)
    
    print("\n[+] Calculation Complete. Dispatching results to Email...")
    send_html_email(bullish_df, bearish_df)
    
if __name__ == "__main__":
    main()

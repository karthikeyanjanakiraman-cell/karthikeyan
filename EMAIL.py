#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI TMV ENGINE - PRODUCTION v22.0 (VOLUME SLICING BUILD)
FEATURES: 
- Dynamic F&O Universe (Column Hunting & Regex)
- Smart Time Routing (24/7 Execution Safety: Handles Weekends & Pre-Market)
- Time-Sliced Historical Ceilings (True Apples-to-Apples)
- Anchored Volatility Range (Tied strictly to the Max Volume Day)
- NEW: Intraday Volume Slicing (Vol Pace - Last 45 mins vs Baseline)
- Kinetic Acceleration Column Integration (Unmerged)
- Volatility Expansion Primary Sorting 
- Clean Responsive HTML Email Matrix
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

# ==========================================
# 1. SYSTEM & CREDENTIAL SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLIENT_ID = os.environ.get("CLIENT_ID", "YOUR_CLIENT_ID")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "YOUR_TOKEN")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "youremail@gmail.com")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "your_app_password")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "recipient@gmail.com")

FYERS_FO_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False, log_path="")

# ==========================================
# 2. DYNAMIC UNIVERSE BUILDER
# ==========================================
def fetch_fo_universe():
    """Scrapes Fyers for active F&O Equities, avoiding hardcoded lists."""
    logger.info("Fetching Master F&O Universe...")
    try:
        response = requests.get(FYERS_FO_MASTER_URL, timeout=15)
        df = pd.read_csv(StringIO(response.text), header=None)
        
        symbol_col = None
        for col in df.columns:
            if df[col].astype(str).str.startswith('NSE:').any():
                symbol_col = col
                break
                
        if symbol_col is None:
            logger.error("CRITICAL: Could not locate the Symbol column in Fyers CSV.")
            return []
            
        raw_symbols = df[symbol_col].astype(str).tolist()
        base_symbols = set()
        
        for s in raw_symbols:
            # Handles text, ampersands (M&M), and hyphens (BAJAJ-AUTO)
            match = re.search(r'NSE:([A-Z&\-]+)\d+', s)
            if match: 
                base_symbols.add(match.group(1))
        
        ignore_list = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'}
        base_symbols = base_symbols - ignore_list
        
        universe = [f"NSE:{sym}-EQ" for sym in base_symbols]
        logger.info(f"Successfully extracted {len(universe)} F&O equities.")
        return universe
        
    except Exception as e:
        logger.error(f"Failed to fetch Universe: {e}")
        return []

# ==========================================
# 3. KINETIC ACCELERATION MATH
# ==========================================
# ==========================================
# 3. UPDATED KINETIC ACCELERATION (Intraday Sensitive)
# ==========================================
def calculate_kinetic_acceleration(df_today):
    """Calculates if buying speed is accelerating relative to today's own volume accumulation."""
    if len(df_today) < 4: return 1.0 # Need at least 4 candles (1 hour) to measure speed
    
    cum_vol = df_today['volume'].cumsum().values
    total_vol_today = cum_vol[-1]
    
    if total_vol_today == 0: return 1.0
    
    # Calculate time taken to reach 25%, 50%, and 75% of TODAY'S cumulative volume
    def get_time_to_reach(pct):
        target = total_vol_today * pct
        idx = np.searchsorted(cum_vol, target)
        return idx # Index is proportional to time in 15m intervals

    t1 = get_time_to_reach(0.25)
    t2 = get_time_to_reach(0.50)
    t3 = get_time_to_reach(0.75)
    
    # Delta 1: Time to move from 25% to 50%
    # Delta 2: Time to move from 50% to 75%
    delta_1 = t2 - t1
    delta_2 = t3 - t2
    
    # If the time taken to move the next 25% of volume is shorter than the last, 
    # it means speed is INCREASING.
    if delta_2 > 0 and delta_1 > delta_2:
        return 1.5 # Accelerating
    elif delta_2 > 0 and delta_1 > delta_2 + 1: # Significant gap
        return 2.0 # Supernova
            
    return 1.0

# ==========================================
# 4. CORE PHYSICS ENGINE
# ==========================================
def extract_raw_physics(symbol):
    try:
        time.sleep(0.12) # Strict API rate limit protection
        now_dt = pd.Timestamp.now(tz="Asia/Kolkata")
        
        payload = {
            "symbol": symbol, "resolution": "15", "date_format": 1,
            "range_from": (now_dt - timedelta(days=90)).strftime("%Y-%m-%d"),
            "range_to": now_dt.strftime("%Y-%m-%d"), "cont_flag": 1
        }
        res = fyers.history(payload)
        
        if not res or 'candles' not in res or len(res['candles']) == 0: return None
        
        df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata")
        
        market_open = dt_time(9, 15)
        current_time = now_dt.time()
        
        # SMART TIME ROUTING: If run before market opens, analyze previous EOD profile
        if current_time < market_open:
            current_time = dt_time(15, 30)
            
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        
        df = df[(df['time'] >= market_open) & (df['time'] <= current_time)]
        if df.empty: return None
        
        # WEEKEND PATCH: Identify the last active trading session dynamically
        today_date = df['date'].max()
        
        history_df = df[df['date'] < today_date]
        today_df = df[df['date'] == today_date]
        
        if history_df.empty or today_df.empty: return None
        
        # HISTORICAL BENCHMARKS
        daily_groups = history_df.groupby('date').agg({'volume': 'sum', 'high': 'max', 'low': 'min'})
        daily_groups['range'] = daily_groups['high'] - daily_groups['low']

        # FIXED ANCHOR: Specific day with the highest volume
        max_vol_date = daily_groups['volume'].idxmax()
        
        # Extract the volume AND the price range from that exact same day
        max_vol = daily_groups.loc[max_vol_date, 'volume']
        max_range = daily_groups.loc[max_vol_date, 'range'] 
        
        if max_vol == 0 or max_range == 0: return None
        
        # TODAY'S LIVE ACTION
        curr_vol = today_df['volume'].sum()
        curr_range = (today_df['high'].max() - today_df['low'].min())
        
        vol_ratio = curr_vol / max_vol
        volatility_ratio = curr_range / max_range
        
        # --- NEW: INTRADAY VOLUME SLICING (Vol Pace) ---
        # Compares the average volume of the last 45 mins (3 candles) to the rest of the day
        if len(today_df) >= 5:
            recent_vol_avg = today_df['volume'].tail(3).mean()
            baseline_vol_avg = today_df['volume'].iloc[:-3].mean()
            vol_pace = (recent_vol_avg / baseline_vol_avg) if baseline_vol_avg > 0 else 1.0
        else:
            vol_pace = 1.0 # Not enough time has passed in the trading day to slice
        # -----------------------------------------------

        # TREND ANCHOR (OBV 10 EMA)
        df['price_dir'] = np.where(df['close'] > df['close'].shift(1), 1, 
                          np.where(df['close'] < df['close'].shift(1), -1, 0))
        df['obv'] = (df['price_dir'] * df['volume']).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=10, min_periods=1).mean()
        
        trend = 'BULLISH' if df['obv'].iloc[-1] > df['obv_ema'].iloc[-1] else 'BEARISH'
        accel = calculate_kinetic_acceleration(today_df)
        
        return {
            'Symbol': symbol.replace('NSE:', '').replace('-EQ', ''),
            'Vol_Ratio': vol_ratio,
            'Volat_Ratio': volatility_ratio,
            'Vol_Pace': vol_pace,
            'Accel': accel,
            'Trend': trend,
            'LTP': df['close'].iloc[-1]
        }
    except Exception as e:
        return None

# ==========================================
# 5. HTML REPORT GENERATOR & DISPATCH
# ==========================================
def send_html_email(bullish_df, bearish_df):
    logger.info("Formatting HTML matrix and dispatching...")
    
    html = f"""
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <style>
          body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f7f9fc; padding: 20px; color: #333; }}
          h2 {{ margin-bottom: 10px; font-weight: 600; }}
          table {{ width: 100%; border-collapse: collapse; margin-bottom: 40px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); background-color: #fff; }}
          th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
          th {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; }}
          .bullish th {{ background-color: #1b5e20; color: white; }}
          .bearish th {{ background-color: #b71c1c; color: white; }}
          tr:hover {{ background-color: #f5f5f5; }}
          .symbol {{ font-weight: bold; color: #1a73e8; }}
          .highlight {{ font-weight: bold; color: #333; font-size: 14px; }}
          .pace {{ font-weight: bold; color: #d84315; }}
        </style>
      </head>
      <body>
        <h2 style="color: #1b5e20;">🚀 TOP BULLISH BREAKOUTS (SORTED BY VOLAT EXP)</h2>
        <table class="bullish">
          <tr><th>Symbol</th><th>LTP</th><th>Vol Ratio</th><th>Volat Exp</th><th>Vol Pace</th><th>Accel</th></tr>
          {"".join(f"<tr><td class='symbol'>{row['Symbol']}</td><td>₹{row['LTP']:.2f}</td><td>{row['Vol_Ratio']:.2f}x</td><td class='highlight'>{row['Volat_Ratio']:.2f}x</td><td class='pace'>{row['Vol_Pace']:.2f}x</td><td>{row['Accel']:.2f}x</td></tr>" for _, row in bullish_df.iterrows())}
        </table>

        <h2 style="color: #b71c1c;">🩸 TOP BEARISH BREAKDOWNS (SORTED BY VOLAT EXP)</h2>
        <table class="bearish">
          <tr><th>Symbol</th><th>LTP</th><th>Vol Ratio</th><th>Volat Exp</th><th>Vol Pace</th><th>Accel</th></tr>
          {"".join(f"<tr><td class='symbol'>{row['Symbol']}</td><td>₹{row['LTP']:.2f}</td><td>{row['Vol_Ratio']:.2f}x</td><td class='highlight'>{row['Volat_Ratio']:.2f}x</td><td class='pace'>{row['Vol_Pace']:.2f}x</td><td>{row['Accel']:.2f}x</td></tr>" for _, row in bearish_df.iterrows())}
        </table>
        <p style="font-size: 12px; color: #777; text-align: center;">Asit Baran Pati TMV Engine v22.0 • Generated at {datetime.now().strftime('%I:%M %p')}</p>
      </body>
    </html>
    """
    
    msg = MIMEMultipart("alternative")
    msg['Subject'] = f"TMV Flow Scan: {datetime.now().strftime('%d %b - %I:%M %p')}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg.attach(MIMEText(html, "html"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logger.info("Email dispatched successfully to secure channel.")
    except Exception as e:
        logger.error(f"Failed to send email. Check credentials: {e}")

# ==========================================
# 6. MASTER EXECUTION THREAD
# ==========================================
def main():
    logger.info("Initializing TMV Engine v22.0...")
    symbols = fetch_fo_universe()
    
    if not symbols: 
        logger.error("Universe is completely empty. Shutting down engine.")
        return
        
    results = []
    logger.info(f"Commencing Time-Sliced Scan on {len(symbols)} symbols. This will take ~90 seconds...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_raw_physics, sym): sym for sym in symbols}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    df_results = pd.DataFrame(results)
    if df_results.empty: 
        logger.error("No valid data processed. Market data missing or API disconnected. Exiting.")
        return
        
    df_results = df_results.round(2)
    
    # ISOLATED CROSS-SECTION REFERENCE: Primary Sort strictly via Volat_Ratio (Volatility Expansion)
    bullish_df = df_results[df_results['Trend'] == 'BULLISH'].sort_values('Volat_Ratio', ascending=False).head(15)
    bearish_df = df_results[df_results['Trend'] == 'BEARISH'].sort_values('Volat_Ratio', ascending=False).head(15)
    
    send_html_email(bullish_df, bearish_df)
    logger.info("System execution completed successfully. Standing by.")
    
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI TMV ENGINE - PRODUCTION v24.2 (5-MIN CANDLE + 5-MIN RANK VELOCITY)
FEATURES: 
- Dynamic F&O Universe (Column Hunting & Regex)
- 5-Minute Candle High-Frequency Resolution Engine
- Smart Time Routing (24/7 Execution Safety: Handles Weekends & Pre-Market)
- Time Machine Backtester (Forces historical anchor for math & slicing)
- Time-Sliced Historical Ceilings (True Apples-to-Apples)
- Anchored Volatility Range (Tied strictly to the Max Volume Day)
- Intraday Volume Slicing (Vol Pace - Last 45 mins vs Baseline)
- Kinetic Acceleration (Accelerating vs Supernova)
- In-Memory Snapshot Momentum Tracker (T-5 Mins Lookback = 1 Candle)
- RESTORED: Primary Sorting Configured Strictly by Volat_Ratio & Vol_Pace
- Clean Responsive HTML Email Matrix (7-Column Extended CSS Architecture)
═══════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import time
import logging
import argparse
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
def calculate_kinetic_acceleration(df_today):
    """Calculates if buying speed is accelerating relative to today's own volume accumulation."""
    if len(df_today) < 4: return 1.0 
    
    cum_vol = df_today['volume'].cumsum().values
    total_vol_today = cum_vol[-1]
    
    if total_vol_today == 0: return 1.0
    
    def get_time_to_reach(pct):
        target = total_vol_today * pct
        idx = np.searchsorted(cum_vol, target)
        return idx 

    t1 = get_time_to_reach(0.25)
    t2 = get_time_to_reach(0.50)
    t3 = get_time_to_reach(0.75)
    
    delta_1 = t2 - t1
    delta_2 = t3 - t2
    
    if delta_2 > 0 and delta_1 > delta_2:
        return 1.5 # Accelerating
    elif delta_2 > 0 and delta_1 > delta_2 + 1:
        return 2.0 # Supernova
            
    return 1.0

# ==========================================
# 4. CORE PHYSICS ENGINE (5-MIN HIGH FREQUENCY CONFIG)
# ==========================================
def extract_raw_physics(symbol, target_dt=None):
    try:
        time.sleep(0.12) # Strict API rate limit protection
        
        # TIME MACHINE ANCHOR: Use target_dt if provided, else use current time
        now_dt = target_dt if target_dt else pd.Timestamp.now(tz="Asia/Kolkata")
        
        payload = {
            "symbol": symbol, "resolution": "5", "date_format": 1,
            "range_from": (now_dt - timedelta(days=90)).strftime("%Y-%m-%d"),
            "range_to": now_dt.strftime("%Y-%m-%d"), "cont_flag": 1
        }
        res = fyers.history(payload)
        
        if not res or 'candles' not in res or len(res['candles']) == 0: return None
        
        df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata")
        
        # TIME MACHINE FILTER: Squelch future candles during simulation
        df = df[df['timestamp'] <= now_dt]
        
        market_open = dt_time(9, 15)
        current_time = now_dt.time()
        
        # SMART TIME ROUTING
        if current_time < market_open:
            current_time = dt_time(15, 30)
            
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        
        df = df[(df['time'] >= market_open) & (df['time'] <= current_time)]
        if df.empty: return None
        
        today_date = df['date'].max()
        
        history_df = df[df['date'] < today_date]
        today_df = df[df['date'] == today_date]
        
        if history_df.empty or today_df.empty: return None
        
        # HISTORICAL BENCHMARKS
        daily_groups = history_df.groupby('date').agg({'volume': 'sum', 'high': 'max', 'low': 'min'})
        daily_groups['range'] = daily_groups['high'] - daily_groups['low']

        max_vol_date = daily_groups['volume'].idxmax()
        
        max_vol = daily_groups.loc[max_vol_date, 'volume']
        max_range = daily_groups.loc[max_vol_date, 'range'] 
        
        if max_vol == 0 or max_range == 0: return None
        
        # TODAY'S LIVE ACTION
        curr_vol = today_df['volume'].sum()
        curr_range = (today_df['high'].max() - today_df['low'].min())
        
        vol_ratio = curr_vol / max_vol
        volatility_ratio = curr_range / max_range
        
        # INTRADAY VOLUME SLICING (Vol Pace Configured for 5-Min Candles)
        if len(today_df) >= 5:
            recent_vol_avg = today_df['volume'].tail(3).mean()
            baseline_vol_avg = today_df['volume'].iloc[:-3].mean()
            vol_pace = (recent_vol_avg / baseline_vol_avg) if baseline_vol_avg > 0 else 1.0
        else:
            vol_pace = 1.0 

        # --- IN-MEMORY HISTORICAL SNAPSHOT GENERATOR (T-5 Mins Lookback = 1 Candle Back) ---
        past_volat_ratio = 0.0
        past_vol_pace = 1.0
        
        if len(today_df) > 1: # Require at least 2 candles to look back 1 candle
            past_today_df = today_df.iloc[:-1] # Hides EXACTLY the last 5 minutes (1 candle)
            curr_range_past = (past_today_df['high'].max() - past_today_df['low'].min())
            past_volat_ratio = curr_range_past / max_range if max_range > 0 else 0
            
            if len(past_today_df) >= 5:
                recent_vol_avg_past = past_today_df['volume'].tail(3).mean()
                baseline_vol_avg_past = past_today_df['volume'].iloc[:-3].mean()
                past_vol_pace = (recent_vol_avg_past / baseline_vol_avg_past) if baseline_vol_avg_past > 0 else 1.0
        # ----------------------------------------------------------------------------------

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
            'Past_Volat_Ratio': past_volat_ratio,
            'Past_Vol_Pace': past_vol_pace,
            'Accel': accel,
            'Trend': trend,
            'LTP': df['close'].iloc[-1]
        }
    except Exception as e:
        return None

# ==========================================
# 5. HTML REPORT GENERATOR & DISPATCH
# ==========================================
def send_html_email(bullish_df, bearish_df, sort_cols):
    criteria_str = ", ".join([c.replace('_', ' ') for c in sort_cols]).upper()
    logger.info(f"Formatting HTML matrix (Sorted by {criteria_str}) and dispatching...")
    
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
        <h2 style="color: #1b5e20;">🚀 TOP BULLISH BREAKOUTS (SORTED BY {criteria_str})</h2>
        <table class="bullish">
          <tr><th>Symbol</th><th>LTP</th><th>Vol Ratio</th><th>Volat Exp</th><th>Vol Pace</th><th>Accel</th><th>Rank Vel (5m)</th></tr>
          {"".join(f"<tr><td class='symbol'>{row['Symbol']}</td><td>₹{row['LTP']:.2f}</td><td>{row['Vol_Ratio']:.2f}x</td><td class='highlight'>{row['Volat_Ratio']:.2f}x</td><td class='pace'>{row['Vol_Pace']:.2f}x</td><td>{row['Accel']:.2f}x</td><td>{row['Rank_Vel_Str']}</td></tr>" for _, row in bullish_df.iterrows())}
        </table>

        <h2 style="color: #b71c1c;">🩸 TOP BEARISH BREAKDOWNS (SORTED BY {criteria_str})</h2>
        <table class="bearish">
          <tr><th>Symbol</th><th>LTP</th><th>Vol Ratio</th><th>Volat Exp</th><th>Vol Pace</th><th>Accel</th><th>Rank Vel (5m)</th></tr>
          {"".join(f"<tr><td class='symbol'>{row['Symbol']}</td><td>₹{row['LTP']:.2f}</td><td>{row['Vol_Ratio']:.2f}x</td><td class='highlight'>{row['Volat_Ratio']:.2f}x</td><td class='pace'>{row['Vol_Pace']:.2f}x</td><td>{row['Accel']:.2f}x</td><td>{row['Rank_Vel_Str']}</td></tr>" for _, row in bearish_df.iterrows())}
        </table>
        <p style="font-size: 12px; color: #777; text-align: center;">Asit Baran Pati TMV Engine v24.2 • Generated at {datetime.now().strftime('%I:%M %p')}</p>
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
# 6. MASTER EXECUTION THREAD & SORTER
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Input format: YYYY-MM-DD HH:MM")
    args = parser.parse_args()
    
    target_dt = None
    if args.date:
        try:
            date_str = args.date.replace('.', ':')
            target_dt = pd.to_datetime(date_str, dayfirst=True).tz_localize("Asia/Kolkata")
            logger.info(f"--- BACKTEST MODE: Simulating {target_dt} ---")
        except Exception as e:
            logger.error(f"Invalid Date/Time Format. Use YYYY-MM-DD HH:MM (24-hour). Error: {e}")
            return
    else:
        logger.info("--- LIVE MODE: Running real-time analysis ---")

    symbols = fetch_fo_universe()
    if not symbols: 
        logger.error("Universe is empty. Shutting down.")
        return
        
    results = []
    logger.info(f"Commencing High-Frequency Scan on {len(symbols)} symbols...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_raw_physics, sym, target_dt): sym for sym in symbols}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    df_results = pd.DataFrame(results)
    if df_results.empty: 
        logger.error("No valid data processed. Exiting.")
        return
        
    df_results = df_results.round(2)
    
    # --- RANK VELOCITY METRIC CALCULATION (5-Minute Speedometer) ---
    def calculate_rank_velocity(trend_df):
        if trend_df.empty: return trend_df
        
        # 1. Backtrack baseline: Rank positions based on original Volat_Ratio & Vol_Pace as of T-5 mins ago (1 candle)
        trend_df['Past_Rank'] = trend_df[['Past_Volat_Ratio', 'Past_Vol_Pace']].apply(tuple, axis=1).rank(method='min', ascending=False)
        
        # 2. Modern baseline: Rank positions right now using the identical logic
        trend_df['Curr_Rank'] = trend_df[['Volat_Ratio', 'Vol_Pace']].apply(tuple, axis=1).rank(method='min', ascending=False)
        
        # 3. Compute Rank Velocity Delta (Past - Current)
        trend_df['Rank_Delta'] = trend_df['Past_Rank'] - trend_df['Curr_Rank']
        
        # 4. Generate visual indicators for the Rank Vel column
        def format_velocity(row):
            if row['Past_Volat_Ratio'] == 0: 
                return "<span style='color:gray;'>➖ N/A</span>"
            if row['Rank_Delta'] > 0:
                return f"<span style='color:#1b5e20; font-weight:bold;'>🚀 +{int(row['Rank_Delta'])}</span>"
            elif row['Rank_Delta'] < 0:
                return f"<span style='color:#b71c1c; font-weight:bold;'>🔻 {int(row['Rank_Delta'])}</span>"
            else:
                return "<span style='color:gray;'>➖ 0</span>"
                
        trend_df['Rank_Vel_Str'] = trend_df.apply(format_velocity, axis=1)
        
        # 5. RESTORED ORIGINAL SCAN SORTING CRITERIA: Sorted primarily by Volat_Ratio, then Vol_Pace
        SORT_CRITERIA = ['Volat_Ratio', 'Vol_Pace']
        return trend_df.sort_values(SORT_CRITERIA, ascending=[False, False]).head(15)

    bullish_df = calculate_rank_velocity(df_results[df_results['Trend'] == 'BULLISH'].copy())
    bearish_df = calculate_rank_velocity(df_results[df_results['Trend'] == 'BEARISH'].copy())
    
    # Pass original matrix naming to email headers
    send_html_email(bullish_df, bearish_df, ['Volat_Ratio', 'Vol_Pace'])
    logger.info("System execution completed successfully.")
 
if __name__ == "__main__":
    main()

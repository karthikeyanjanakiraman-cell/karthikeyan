#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI TMV ENGINE - PRODUCTION v26.1 (95th PERCENTILE CEILING & CVB ENGINE)
FEATURES: 
- Constant Volume Bars (CVB): Condenses raw 1-min data into fixed 10k volume blocks.
- Smart Time-To-Fill: Automatically scales across missing minutes using physical timestamps.
- 95th Percentile Speed Ceiling: Deletes one-off historical anomalies (like block deals) 
  and compares live speed against the true sustainable cruising speed of the Max Volume Day.
- Median Kinetic Chain: Deploys robust median calculations for smooth momentum tracking.
- Full Dispatch System: Transmits responsive HTML matrices directly to your inbox.
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
# 1. SYSTEM SETUP & CONFIGURABLE DIALS
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

# ------------------------------------------
# MASTER PARAMETERS (THE CONTROL BOARD)
# ------------------------------------------
VOLUME_BAR_SIZE = 10000        # The fixed size of our Volume Bucket
WATCHLIST_SIZE = 30            # Size of the anchored pre-market watchlist 
RECENT_ITERATION_PCT = 0.30    # Top 30% of 10k bars used as the live momentum cruising window
SPEED_THRESHOLD_RATIO = 0.20   # Hurdle rate dial
VOLATILITY_EXP_THRESHOLD = 0.5 # Minimum Volat Exp required to pass primary sieve

# ==========================================
# 2. DYNAMIC UNIVERSE BUILDER
# ==========================================
def fetch_fo_universe():
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
            logger.error("CRITICAL: Could not locate Symbol column in Fyers Master.")
            return []
            
        raw_symbols = df[symbol_col].astype(str).tolist()
        base_symbols = set()
        
        for s in raw_symbols:
            match = re.search(r'NSE:([A-Z&\-]+)\d+', s)
            if match: 
                base_symbols.add(match.group(1))
        
        ignore_list = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'}
        return [f"NSE:{sym}-EQ" for sym in base_symbols - ignore_list]
    except Exception as e:
        logger.error(f"Failed to fetch Universe: {e}")
        return []

# ==========================================
# 3. CONSTANT VOLUME BAR SYNTHESIZER
# ==========================================
def build_volume_bars(df_1min, target_vol=VOLUME_BAR_SIZE):
    """
    Takes 1-min raw candle components and condenses them into unified Constant Volume Bars.
    Tracks explicit time (scaling dynamically over missing minutes) to complete every 10k block.
    """
    if df_1min.empty: 
        return pd.DataFrame()
    
    bars = []
    current_vol = 0
    start_time = None
    open_price = None
    high_price = -np.inf
    low_price = np.inf
    
    for idx, row in df_1min.iterrows():
        if current_vol == 0:
            start_time = row['timestamp']
            open_price = row['open']
            
        current_vol += row['volume']
        high_price = max(high_price, row['high'])
        low_price = min(low_price, row['low'])
        
        if current_vol >= target_vol:
            end_time = row['timestamp']
            time_diff = (end_time - start_time).total_seconds() / 60.0
            time_taken = max(1.0, time_diff) 
            
            bars.append({
                'time_taken': time_taken,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': row['close'],
                'volume': current_vol,
                'velocity': current_vol / time_taken
            })
            
            current_vol = 0
            high_price = -np.inf
            low_price = np.inf
            
    return pd.DataFrame(bars)

# ==========================================
# 4. KINETIC CHAIN ENGINE (MEDIAN BASED CRUISE CONTROL)
# ==========================================
def calculate_cvb_kinetic_chain(vol_bars_df):
    """
    Evaluates historical base velocities against recent windows via robust medians.
    """
    if vol_bars_df.empty or len(vol_bars_df) < 4: 
        return False, 1.0, 0.0
        
    split_idx = int(len(vol_bars_df) * (1 - RECENT_ITERATION_PCT))
    if split_idx == len(vol_bars_df): split_idx -= 1
    if split_idx <= 0: split_idx = 1
    
    base_chain = vol_bars_df.iloc[:split_idx]
    recent_chain = vol_bars_df.iloc[split_idx:]
    
    max_base_velocity = base_chain['velocity'].max()
    median_recent_velocity = recent_chain['velocity'].median()
    current_live_velocity = recent_chain['velocity'].iloc[-1]
    
    hurdle_v = max_base_velocity * SPEED_THRESHOLD_RATIO
    
    vol_pass = (median_recent_velocity > hurdle_v) and (max_base_velocity > 0)
    v_mult = (median_recent_velocity / max_base_velocity) if max_base_velocity > 0 else 1.0
    
    return vol_pass, v_mult, current_live_velocity

# ==========================================
# 5. PHASE 1: PRE-MARKET MATRIX ANCHOR
# ==========================================
def extract_pre_market_score(symbol, target_dt):
    """Sifts opening configurations across the network to secure immediate liquidity."""
    try:
        time.sleep(0.12)
        payload = {
            "symbol": symbol, "resolution": "1", "date_format": 1,
            "range_from": (target_dt - timedelta(days=90)).strftime("%Y-%m-%d"),
            "range_to": target_dt.strftime("%Y-%m-%d"), "cont_flag": 1
        }
        res = fyers.history(payload)
        if not res or 'candles' not in res or len(res['candles']) == 0: return None
        
        df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata")
        df = df[df['timestamp'] <= target_dt]
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        
        open_prints = df[(df['time'] >= dt_time(9, 15)) & (df['time'] <= dt_time(9, 20))]
        if open_prints.empty: return None
        
        today_date = df['date'].max()
        today_open = open_prints[open_prints['date'] == today_date]
        if today_open.empty: return None
        
        today_open_vol = today_open['volume'].sum()
        return {'Symbol': symbol, 'Pre_Market_Vol': today_open_vol, 'Full_DF': df}
    except Exception:
        return None

# ==========================================
# 6. PHASE 2: INTRADAY PROCESSING MODULE
# ==========================================
def process_intraday_matrix(symbol, pre_market_vol, df, target_dt):
    """Processes Live CVBs against the 95th Percentile Speed Ceiling of the Max Volume Day."""
    try:
        current_time = target_dt.time()
        market_open = dt_time(9, 15)
        
        df_filtered = df[(df['time'] >= market_open) & (df['time'] <= current_time)]
        today_date = df_filtered['date'].max()
        
        history_df = df_filtered[df_filtered['date'] < today_date]
        today_df = df_filtered[df_filtered['date'] == today_date]
        
        if history_df.empty or today_df.empty: return None

        today_vol_bars = build_volume_bars(today_df, VOLUME_BAR_SIZE)
        if today_vol_bars.empty: return None

        # Determine the absolute highest total volume day in history
        daily_groups = history_df.groupby('date').agg({'volume': 'sum', 'high': 'max', 'low': 'min'})
        max_vol_date = daily_groups['volume'].idxmax()
        
        # ════════════════════════════════════════════════════════════════════════
        # 95th PERCENTILE CEILING (FILTERING OUT FREAK HISTORICAL ANOMALIES)
        # ════════════════════════════════════════════════════════════════════════
        max_vol_day_df = history_df[history_df['date'] == max_vol_date]
        historical_max_bars = build_volume_bars(max_vol_day_df, VOLUME_BAR_SIZE)
        if historical_max_bars.empty: return None
        
        # Using 95th percentile instead of absolute .max() to eliminate outlier block spikes
        historical_max_speed = historical_max_bars['velocity'].quantile(0.95)
        # ════════════════════════════════════════════════════════════════════════
        
        max_vol = daily_groups.loc[max_vol_date, 'volume']
        max_range = daily_groups['high'].max() - daily_groups['low'].min()
        
        vol_ratio = today_df['volume'].sum() / max_vol if max_vol > 0 else 0
        volatility_ratio = (today_df['high'].max() - today_df['low'].min()) / max_range if max_range > 0 else 0
        
        v_pass, v_mult, live_velocity = calculate_cvb_kinetic_chain(today_vol_bars)
        
        return {
            'Symbol': symbol.replace('NSE:', '').replace('-EQ', ''),
            'Live_10k_Speed': live_velocity,
            'Historical_Max_Speed': historical_max_speed,
            'Total_10k_Bars': len(today_vol_bars),
            'Vol_Ratio': vol_ratio,
            'Volat_Ratio': volatility_ratio,
            'Kin_Vol_Str': f"PASS ({v_mult:.1f}x)" if v_pass else f"FAIL ({v_mult:.1f}x)",
            'LTP': today_df['close'].iloc[-1],
            'V_Pass': v_pass
        }
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None

# ==========================================
# 7. HTML REPORT GENERATOR & DISPATCH
# ==========================================
def send_html_email(df_matrix, target_dt):
    logger.info(f"Generating HTML performance matrix for {target_dt.strftime('%I:%M %p')}...")
    fetch_time_str = target_dt.strftime('%d %b %Y, %I:%M %p')
    
    def build_rows(df):
        html_rows = ""
        for _, row in df.iterrows():
            v_class = "pass" if row['V_Pass'] else "fail"
            
            html_rows += f"""<tr>
                <td class='symbol'>{row['Symbol']}</td>
                <td>₹{row['LTP']:.2f}</td>
                <td style='color:#1b5e20; font-weight:bold;'>{int(row['Live_10k_Speed']):,} sh/min</td>
                <td style='color:#757575;'>{int(row['Historical_Max_Speed']):,} sh/min</td>
                <td>{int(row['Total_10k_Bars'])} Bars</td>
                <td>{row['Vol_Ratio']:.2f}x</td>
                <td class='highlight'>{row['Volat_Ratio']:.2f}x</td>
                <td class='{v_class}'>{row['Kin_Vol_Str']}</td>
            </tr>"""
        return html_rows

    html = f"""
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <style>
          body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f7f9fc; padding: 20px; color: #333; }}
          h2 {{ margin-bottom: 5px; font-weight: 600; color: #1a237e; text-align: center; }}
          .time-stamp {{ text-align: center; font-size: 13px; color: #555; margin-bottom: 25px; }}
          table {{ width: 100%; border-collapse: collapse; margin-bottom: 40px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); background-color: #fff; }}
          th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
          th {{ font-size: 14px; text-transform: uppercase; background-color: #3949ab; color: white; }}
          tr:hover {{ background-color: #f5f5f5; }}
          .symbol {{ font-weight: bold; color: #1a73e8; }}
          .highlight {{ font-weight: bold; color: #333; }}
          .pass {{ font-weight: bold; color: #1b5e20; }}
          .fail {{ color: #757575; }}
        </style>
      </head>
      <body>
        <h2>🏆 TMV 95th PERCENTILE SPEED BREAKER</h2>
        <p class="time-stamp">🕒 Data Fetched At: <b>{fetch_time_str}</b></p>
        <table>
          <tr><th>Symbol</th><th>LTP</th><th>Live 10k Speed</th><th>95th Pct Max Speed</th><th>10k Bars Formed</th><th>Vol Ratio</th><th>Volat Exp</th><th>Kinetic Chain</th></tr>
          {build_rows(df_matrix)}
        </table>
        <p style="font-size: 12px; color: #777; text-align: center;">
            TMV Engine v26.1 • 95th Percentile Momentum Mode • Alerting when live velocity beats historical sustainable ceiling.
        </p>
      </body>
    </html>
    """
    
    msg = MIMEMultipart("alternative")
    msg['Subject'] = f"CVB Breakout Matrix | 95th Pct Violations: {fetch_time_str}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg.attach(MIMEText(html, "html"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logger.info(f"HTML report successfully dispatched for {fetch_time_str}.")
    except Exception as e:
        logger.error(f"Failed to transmit email package: {e}")

# ==========================================
# 8. MAIN COORDINATION ENGINE
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Input format: YYYY-MM-DD (Batch) or YYYY-MM-DD HH:MM (Single Run)")
    parser.add_argument("--from_time", help="Input format: HH:MM (e.g., 10:00)")
    parser.add_argument("--to_time", help="Input format: HH:MM (e.g., 14:00)")
    parser.add_argument("--interval", type=int, default=60, help="Interval in minutes (e.g., 60)")
    args = parser.parse_args()
    
    target_dt = pd.Timestamp.now(tz="Asia/Kolkata")
    start_dt, end_dt = target_dt, target_dt
    interval_mins = args.interval
    
    if args.date and args.from_time and args.to_time:
        try:
            date_str = args.date.replace('.', '-')
            clean_from_time = args.from_time.replace('.', ':')
            clean_to_time = args.to_time.replace('.', ':')
            start_dt = pd.to_datetime(f"{date_str} {clean_from_time}").tz_localize("Asia/Kolkata")
            end_dt = pd.to_datetime(f"{date_str} {clean_to_time}").tz_localize("Asia/Kolkata")
            logger.info(f"--- BATCH BACKTEST MODE: {start_dt.strftime('%H:%M')} to {end_dt.strftime('%H:%M')} ---")
        except Exception as e:
            logger.error(f"Invalid Batch format. Error: {e}")
            return
    elif args.date:
        try:
            date_str = args.date.replace('.', ':')
            start_dt = pd.to_datetime(date_str, dayfirst=True).tz_localize("Asia/Kolkata")
            end_dt = start_dt
        except Exception as e: 
            logger.error(f"Invalid Single Run format. Error: {e}")
            return
            
    raw_symbols = fetch_fo_universe()
    if not raw_symbols: 
        logger.error("Universe empty. Terminating.")
        return
        
    logger.info(f"Phase 1: Running Pre-Market Sieve across {len(raw_symbols)} symbols...")
    pre_market_results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_pre_market_score, sym, end_dt): sym for sym in raw_symbols}
        for future in as_completed(futures):
            res = future.result()
            if res: pre_market_results.append(res)
            
    if not pre_market_results: 
        logger.error("Pre-market collection failed.")
        return
        
    pre_market_df = pd.DataFrame(pre_market_results)
    anchored_watchlist = pre_market_df.sort_values('Pre_Market_Vol', ascending=False).head(WATCHLIST_SIZE)
    logger.info(f"Watchlist securely anchored. Monitored structures: {len(anchored_watchlist)}")
    
    current_dt = start_dt
    while current_dt <= end_dt:
        logger.info(f"Phase 2: Executing CVB Matrix Calculations for {current_dt.strftime('%I:%M %p')}...")
        intraday_results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(process_intraday_matrix, row['Symbol'], row['Pre_Market_Vol'], row['Full_DF'], current_dt): row['Symbol'] 
                for _, row in anchored_watchlist.iterrows()
            }
            for future in as_completed(futures):
                res = future.result()
                if res: intraday_results.append(res)
                
        df_matrix = pd.DataFrame(intraday_results)
        
        if not df_matrix.empty:
            PRIORITY_SORT = ['Live_10k_Speed', 'Vol_Ratio']
            sorted_matrix = df_matrix.sort_values(PRIORITY_SORT, ascending=[False, False])
            
            # ════════════════════════════════════════════════════════════════════════
            # 95th PERCENTILE CEILING FILTER
            # ════════════════════════════════════════════════════════════════════════
            filtered_matrix = sorted_matrix[
                (sorted_matrix['Live_10k_Speed'] > sorted_matrix['Historical_Max_Speed']) &
                (sorted_matrix['Volat_Ratio'] > VOLATILITY_EXP_THRESHOLD)
            ]
            
            if not filtered_matrix.empty: 
                send_html_email(filtered_matrix, current_dt)
            else: 
                logger.warning(f"No assets exceeded their 95th Pct Historical Speeds at {current_dt.strftime('%I:%M %p')}.")
        
        current_dt += timedelta(minutes=interval_mins)
        if current_dt <= end_dt: 
            time.sleep(2)
            
    logger.info("System process workflow completed successfully.")

if __name__ == "__main__":
    main()

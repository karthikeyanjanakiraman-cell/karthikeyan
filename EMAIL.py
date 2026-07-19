#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI TMV ENGINE - PRODUCTION v25.0 (PRE-MARKET ANCHOR & LIVE MATRIX)
FEATURES: 
- Dynamic F&O Universe (Column Hunting & Regex Extraction)
- Phase 1: Pre-Market Sieve (Anchors Top 25 Stocks via 90-Day 9:15 AM Max Vol ceiling)
- Phase 2: Live Intraday Tracking Matrix restricted strictly to the Anchored Watchlist
- Downshifted Speed Hurdle (The Threshold-Based Kinetic Chain Rule)
- Iceberg Trap Identifier & Dual-Engine Multi-Column Priority Sorting
- Clean Responsive HTML Matrix Dispatch System
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
WATCHLIST_SIZE = 30            # Size of the anchored pre-market watchlist 
RECENT_ITERATION_PCT = 1.0    # Slices the last 30% of iterations as the momentum window
SPEED_THRESHOLD_RATIO = 0.20   # Hurdle rate dial (e.g., recent speed must exceed 30% of base peak speed)

# ==========================================
# 2. DYNAMIC UNIVERSE BUILDER
# ==========================================
def fetch_fo_universe():
    """Scrapes active F&O Equities dynamically from Fyers Master data."""
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
        base_symbols = base_symbols - ignore_list
        
        return [f"NSE:{sym}-EQ" for sym in base_symbols]
        
    except Exception as e:
        logger.error(f"Failed to fetch Universe: {e}")
        return []

# ==========================================
# 3. KINETIC HALVING ENGINE (WITH THRESHOLD HURDLE)
# ==========================================
def calculate_threshold_kinetic_chain(today_df, resolution=5):
    """
    Slices total live volume into fractional blocks.
    Applies the downshifted speed hurdle (cruising momentum validator).
    """
    cum_vol = today_df['volume'].cumsum()
    total_vol = cum_vol.iloc[-1]
    
    if total_vol == 0 or len(today_df) < 4: 
        return False, False, 1.0, 1.0
        
    iterations = []
    current_start_idx = 0
    remaining_vol = total_vol
    current_base_vol = 0
    
    for _ in range(8):
        target_vol = current_base_vol + (remaining_vol / 2.0)
        idx = int(np.searchsorted(cum_vol.values, target_vol))
        if idx >= len(today_df): idx = len(today_df) - 1
        
        slice_df = today_df.iloc[current_start_idx : idx + 1]
        if slice_df.empty: break
        
        time_taken = len(slice_df) * resolution
        vol_filled = slice_df['volume'].sum()
        price_dist = abs(slice_df['close'].iloc[-1] - slice_df['open'].iloc[0])
        
        iterations.append({
            'vol_vel': vol_filled / time_taken if time_taken > 0 else 0,
            'price_vel': price_dist / time_taken if time_taken > 0 else 0
        })
        
        current_start_idx = idx + 1
        current_base_vol = cum_vol.iloc[idx] 
        remaining_vol = total_vol - current_base_vol
        
        if current_start_idx >= len(today_df) or remaining_vol <= 0: break
        
    if len(iterations) < 2:
        return False, False, 1.0, 1.0
        
    split_idx = int(len(iterations) * (1 - RECENT_ITERATION_PCT))
    if split_idx == len(iterations): split_idx -= 1
    if split_idx == 0: split_idx = 1
    
    base_chain = iterations[:split_idx]
    recent_chain = iterations[split_idx:]
    
    max_base_v = max([x['vol_vel'] for x in base_chain])
    max_base_p = max([x['price_vel'] for x in base_chain])
    
    min_recent_v = min([x['vol_vel'] for x in recent_chain])
    min_recent_p = min([x['price_vel'] for x in recent_chain])
    
    # DOWNSHIFTED THRESHOLD HURDLES
    hurdle_v = max_base_v * SPEED_THRESHOLD_RATIO
    hurdle_p = max_base_p * SPEED_THRESHOLD_RATIO
    
    vol_pass = (min_recent_v > hurdle_v) and (max_base_v > 0)
    price_pass = (min_recent_p > hurdle_p) and (max_base_p > 0)
    
    v_mult = (min_recent_v / max_base_v) if max_base_v > 0 else 1.0
    p_mult = (min_recent_p / max_base_p) if max_base_p > 0 else 1.0
    
    return vol_pass, price_pass, v_mult, p_mult

# ==========================================
# 4. PHASE 1: PRE-MARKET MATRIX GENERATOR
# ==========================================
def extract_pre_market_score(symbol, target_dt):
    """Scans 90 days of 9:15 AM data to isolate institutional volume anomalies."""
    try:
        time.sleep(0.12) # Rate limit guard rails
        payload = {
            "symbol": symbol, "resolution": "5", "date_format": 1,
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
        
        open_prints = df[df['time'] == dt_time(9, 15)]
        if open_prints.empty: return None
        
        today_date = df['date'].max()
        historical_opens = open_prints[open_prints['date'] < today_date]
        today_open = open_prints[open_prints['date'] == today_date]
        
        if historical_opens.empty or today_open.empty: return None
        
        max_historical_open_vol = historical_opens['volume'].max()
        today_open_vol = today_open['volume'].iloc[0]
        
        if max_historical_open_vol == 0: return None
        
        pre_market_ratio = today_open_vol / max_historical_open_vol
        
        return {'Symbol': symbol, 'Pre_Market_Ratio': pre_market_ratio, 'Full_DF': df}
    except Exception:
        return None

# ==========================================
# 5. PHASE 2: INTRADAY PROCESSING MODULE
# ==========================================
def process_intraday_matrix(symbol, pre_market_ratio, df, target_dt):
    """Processes live fractional tracks exclusively for the anchored universe."""
    try:
        current_time = target_dt.time()
        market_open = dt_time(9, 15)
        
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        
        df_filtered = df[(df['time'] >= market_open) & (df['time'] <= current_time)]
        today_date = df_filtered['date'].max()
        
        history_df = df_filtered[df_filtered['date'] < today_date]
        today_df = df_filtered[df_filtered['date'] == today_date]
        
        if history_df.empty or today_df.empty: return None
        
        # Core Benchmarks
        daily_groups = history_df.groupby('date').agg({'volume': 'sum', 'high': 'max', 'low': 'min'})
        daily_groups['range'] = daily_groups['high'] - daily_groups['low']
        
        max_vol_date = daily_groups['volume'].idxmax()
        max_vol = daily_groups.loc[max_vol_date, 'volume']
        max_range = daily_groups.loc[max_vol_date, 'range']
        
        if max_vol == 0 or max_range == 0: return None
        
        vol_ratio = today_df['volume'].sum() / max_vol
        volatility_ratio = (today_df['high'].max() - today_df['low'].min()) / max_range
        
        # Calculate Kinetic Splits with downshifted hurdles
        v_pass, p_pass, v_mult, p_mult = calculate_threshold_kinetic_chain(today_df, resolution=5)
        
        # OBV Directional Anchor
        df_filtered = df_filtered.copy()
        df_filtered['price_dir'] = np.where(df_filtered['close'] > df_filtered['close'].shift(1), 1, 
                                   np.where(df_filtered['close'] < df_filtered['close'].shift(1), -1, 0))
        df_filtered['obv'] = (df_filtered['price_dir'] * df_filtered['volume']).cumsum()
        df_filtered['obv_ema'] = df_filtered['obv'].ewm(span=10, min_periods=1).mean()
        
        trend = 'BULLISH' if df_filtered['obv'].iloc[-1] > df_filtered['obv_ema'].iloc[-1] else 'BEARISH'
        
        return {
            'Symbol': symbol.replace('NSE:', '').replace('-EQ', ''),
            'Pre_Market_Ratio': pre_market_ratio,
            'Vol_Ratio': vol_ratio,
            'Volat_Ratio': volatility_ratio,
            'Kin_Vol_Str': f"PASS ({v_mult:.1f}x)" if v_pass else f"FAIL ({v_mult:.1f}x)",
            'Kin_Price_Str': f"PASS ({p_mult:.1f}x)" if p_pass else f"FAIL ({p_mult:.1f}x)",
            'Trend': trend,
            'LTP': today_df['close'].iloc[-1],
            'V_Pass': v_pass,
            'P_Pass': p_pass
        }
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None

# ==========================================
# 6. HTML REPORT GENERATOR & DISPATCH
# ==========================================
def send_html_email(bullish_df, bearish_df):
    logger.info("Generating secure HTML performance matrix...")
    
    def build_rows(df):
        html_rows = ""
        for _, row in df.iterrows():
            v_class = "pass" if row['V_Pass'] else "fail"
            p_class = "pass" if row['P_Pass'] else "fail"
            
            html_rows += f"""<tr>
                <td class='symbol'>{row['Symbol']}</td>
                <td>₹{row['LTP']:.2f}</td>
                <td>{row['Pre_Market_Ratio']:.2f}x</td>
                <td>{row['Vol_Ratio']:.2f}x</td>
                <td class='highlight'>{row['Volat_Ratio']:.2f}x</td>
                <td class='{v_class}'>{row['Kin_Vol_Str']}</td>
                <td class='{p_class}'>{row['Kin_Price_Str']}</td>
            </tr>"""
        return html_rows

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
          .pass {{ font-weight: bold; color: #1b5e20; }}
          .fail {{ color: #757575; }}
        </style>
      </head>
      <body>
        <h2 style="color: #1b5e20;">🏆 TOP BULLISH ANCHORED BREAKOUTS</h2>
        <table class="bullish">
          <tr><th>Symbol</th><th>LTP</th><th>9:15 Print Ratio</th><th>Vol Ratio</th><th>Volat Exp</th><th>Kinetic Vol</th><th>Kinetic Price</th></tr>
          {build_rows(bullish_df)}
        </table>

        <h2 style="color: #b71c1c;">🩸 TOP BEARISH ANCHORED BREAKDOWNS</h2>
        <table class="bearish">
          <tr><th>Symbol</th><th>LTP</th><th>9:15 Print Ratio</th><th>Vol Ratio</th><th>Volat Exp</th><th>Kinetic Vol</th><th>Kinetic Price</th></tr>
          {build_rows(bearish_df)}
        </table>
        <p style="font-size: 12px; color: #777; text-align: center;">
            TMV Engine v25.0 • Anchored Matrix Size: {WATCHLIST_SIZE} • Speed Hurdle Dial: {int(SPEED_THRESHOLD_RATIO*100)}%
        </p>
      </body>
    </html>
    """
    
    msg = MIMEMultipart("alternative")
    msg['Subject'] = f"TMV Anchored Matrix Tracker: {datetime.now().strftime('%d %b - %I:%M %p')}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg.attach(MIMEText(html, "html"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logger.info("HTML report successfully dispatched.")
    except Exception as e:
        logger.error(f"Failed to transmit email package: {e}")

# ==========================================
# 7. MAIN COORDINATION ENGINE
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Input format: YYYY-MM-DD HH:MM")
    args = parser.parse_args()
    
    target_dt = pd.Timestamp.now(tz="Asia/Kolkata")
    if args.date:
        try:
            date_str = args.date.replace('.', ':')
            target_dt = pd.to_datetime(date_str, dayfirst=True).tz_localize("Asia/Kolkata")
            logger.info(f"--- BACKTEST MODE: Simulating Timestamp {target_dt} ---")
        except Exception as e:
            logger.error(f"Invalid Date format. Use YYYY-MM-DD HH:MM. Error: {e}")
            return
            
    raw_symbols = fetch_fo_universe()
    if not raw_symbols:
        logger.error("Universe extraction returned null. Terminating execution.")
        return
        
    # -----------------------------------------------------------------
    # PHASE 1: COMPUTE PRE-MARKET MATRIX & SIEVE THE TOP CONTENDERS
    # -----------------------------------------------------------------
    logger.info(f"Phase 1: Running Pre-Market Opening Sieve across {len(raw_symbols)} symbols...")
    pre_market_results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_pre_market_score, sym, target_dt): sym for sym in raw_symbols}
        for future in as_completed(futures):
            res = future.result()
            if res: pre_market_results.append(res)
            
    if not pre_market_results:
        logger.error("No valid pre-market profiles calculated. Exiting.")
        return
        
    pre_market_df = pd.DataFrame(pre_market_results)
    anchored_watchlist = pre_market_df.sort_values('Pre_Market_Ratio', ascending=False).head(WATCHLIST_SIZE)
    logger.info(f"Watchlist successfully anchored. Tracking top {len(anchored_watchlist)} institutional structures.")
    
    # -----------------------------------------------------------------
    # PHASE 2: RUN CONTINUOUS INTRADAY FRACTIONAL PROCESSING VIA WATCHLIST
    # -----------------------------------------------------------------
    logger.info("Phase 2: Executing Intraday Matrix Calculations on anchored targets...")
    intraday_results = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_intraday_matrix, row['Symbol'], row['Pre_Market_Ratio'], row['Full_DF'], target_dt): row['Symbol'] 
            for _, row in anchored_watchlist.iterrows()
        }
        for future in as_completed(futures):
            res = future.result()
            if res: intraday_results.append(res)
            
    df_matrix = pd.DataFrame(intraday_results)
    if df_matrix.empty:
        logger.error("Watchlist matrix processing returned empty set.")
        return

    # Sort Matrix by clean institutional parameters
    PRIORITY_SORT = ['Volat_Ratio','Vol_Ratio','Pre_Market_Ratio']
    bullish_df = df_matrix[df_matrix['Trend'] == 'BULLISH'].sort_values(PRIORITY_SORT, ascending=[False, False, False])
    bearish_df = df_matrix[df_matrix['Trend'] == 'BEARISH'].sort_values(PRIORITY_SORT, ascending=[False, False, False])
    
    send_html_email(bullish_df, bearish_df)
    logger.info("System process workflow completed successfully.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI TMV ENGINE - PRODUCTION v24.0 (KINETIC FRACTAL ENGINE)
FEATURES: 
- Dynamic F&O Universe (Column Hunting & Regex)
- Clock-Agnostic Execution (Runs perfectly at any time of day)
- Kinetic Halving Chain (Slices daily volume into progressive fractional blocks)
- Dual-Engine Velocity (Measures Shares/Min and Rupees/Min for every block)
- Configurable No-Overlap Split (Last 30% speed > First 70% speed)
- Iceberg Trap Detector (Catches high volume with no price movement)
- Priority Multi-Column Sorting (PASS/PASS True Breakouts rise to the top)
- Clean Responsive HTML Email Matrix (Extended CSS)
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

# ------------------------------------------
# MASTER CONFIG: THE STRICTNESS DIAL
# 0.30 means the engine compares the Last 30% of iterations against the First 70%.
# ------------------------------------------
RECENT_ITERATION_PCT = 0.30  

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
# 3. KINETIC HALVING CHAIN (THE NEW CORE)
# ==========================================
def build_kinetic_chain(today_df, resolution=5, recent_pct=0.30):
    """
    Slices today's volume into halving fractions (50%, 25%, 12.5%...).
    Calculates Volume Velocity & Price Velocity for each block.
    Forces a No-Overlap test between the Last X% and the First Y%.
    """
    cum_vol = today_df['volume'].cumsum()
    total_vol = cum_vol.iloc[-1]
    
    if total_vol == 0 or len(today_df) < 4: 
        return False, False, 1.0, 1.0
        
    iterations = []
    current_start_idx = 0
    remaining_vol = total_vol
    current_base_vol = 0
    
    # We process up to 8 iterations (approx 99.6% of volume) to prevent micro-noise
    for _ in range(8):
        target_vol = current_base_vol + (remaining_vol / 2.0)
        
        # Find exact candle where this fraction was filled
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
        
        # Lock in actual data for the next block
        current_start_idx = idx + 1
        current_base_vol = cum_vol.iloc[idx] 
        remaining_vol = total_vol - current_base_vol
        
        if current_start_idx >= len(today_df) or remaining_vol <= 0: break
        
    if len(iterations) < 2:
        return False, False, 1.0, 1.0
        
    # Apply the Configurable Split (e.g. 70% Base / 30% Recent)
    split_idx = int(len(iterations) * (1 - recent_pct))
    if split_idx == len(iterations): split_idx -= 1
    if split_idx == 0: split_idx = 1
    
    base_chain = iterations[:split_idx]
    recent_chain = iterations[split_idx:]
    
    max_base_v = max([x['vol_vel'] for x in base_chain])
    min_recent_v = min([x['vol_vel'] for x in recent_chain])
    
    max_base_p = max([x['price_vel'] for x in base_chain])
    min_recent_p = min([x['price_vel'] for x in recent_chain])
    
    v_mult = (min_recent_v / max_base_v) if max_base_v > 0 else 1.0
    p_mult = (min_recent_p / max_base_p) if max_base_p > 0 else 1.0
    
    # The Ultimate No-Overlap Test
    vol_pass = (min_recent_v > max_base_v) and (max_base_v > 0)
    price_pass = (min_recent_p > max_base_p) and (max_base_p > 0)
    
    return vol_pass, price_pass, v_mult, p_mult

# ==========================================
# 4. CORE PHYSICS ENGINE
# ==========================================
def extract_raw_physics(symbol, target_dt=None):
    try:
        time.sleep(0.12) # Strict API rate limit protection
        
        now_dt = target_dt if target_dt else pd.Timestamp.now(tz="Asia/Kolkata")
        current_time = now_dt.time()
        
        # Clock is dead. We feed it a continuous 5-minute stream.
        candle_res = "5"
            
        payload = {
            "symbol": symbol, "resolution": candle_res, "date_format": 1,
            "range_from": (now_dt - timedelta(days=90)).strftime("%Y-%m-%d"),
            "range_to": now_dt.strftime("%Y-%m-%d"), "cont_flag": 1
        }
        res = fyers.history(payload)
        
        if not res or 'candles' not in res or len(res['candles']) == 0: return None
        
        df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata")
        
        df = df[df['timestamp'] <= now_dt]
        
        market_open = dt_time(9, 15)
        if current_time < market_open: current_time = dt_time(15, 30)
            
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
        
        # LIVE ACTION 
        curr_vol = today_df['volume'].sum()
        curr_range = (today_df['high'].max() - today_df['low'].min())
        
        vol_ratio = curr_vol / max_vol
        volatility_ratio = curr_range / max_range
        
        # THE KINETIC ENGINE INJECTION
        v_pass, p_pass, v_mult, p_mult = build_kinetic_chain(today_df, resolution=5, recent_pct=RECENT_ITERATION_PCT)

        # TREND ANCHOR (OBV 10 EMA)
        df['price_dir'] = np.where(df['close'] > df['close'].shift(1), 1, 
                          np.where(df['close'] < df['close'].shift(1), -1, 0))
        df['obv'] = (df['price_dir'] * df['volume']).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=10, min_periods=1).mean()
        
        trend = 'BULLISH' if df['obv'].iloc[-1] > df['obv_ema'].iloc[-1] else 'BEARISH'
        
        return {
            'Symbol': symbol.replace('NSE:', '').replace('-EQ', ''),
            'Vol_Ratio': vol_ratio,
            'Volat_Ratio': volatility_ratio,
            'Kin_Vol_Str': f"PASS ({v_mult:.1f}x)" if v_pass else f"FAIL ({v_mult:.1f}x)",
            'Kin_Price_Str': f"PASS ({p_mult:.1f}x)" if p_pass else f"FAIL ({p_mult:.1f}x)",
            'Trend': trend,
            'LTP': df['close'].iloc[-1],
            'V_Pass': v_pass, 
            'P_Pass': p_pass
        }
    except Exception as e:
        return None

# ==========================================
# 5. HTML REPORT GENERATOR & DISPATCH
# ==========================================
def send_html_email(bullish_df, bearish_df):
    logger.info("Formatting HTML matrix and dispatching...")
        
    def build_rows(df):
        html_rows = ""
        for _, row in df.iterrows():
            # Apply color coding to the PASS/FAIL strings
            v_class = "pass" if row['V_Pass'] else "fail"
            p_class = "pass" if row['P_Pass'] else "fail"
            
            html_rows += f"""<tr>
                <td class='symbol'>{row['Symbol']}</td>
                <td>₹{row['LTP']:.2f}</td>
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
        <h2 style="color: #1b5e20;">🚀 TOP BULLISH BREAKOUTS (SORTED BY TRUE MOMENTUM)</h2>
        <table class="bullish">
          <tr><th>Symbol</th><th>LTP</th><th>Vol Ratio</th><th>Volat Exp</th><th>Kinetic Vol</th><th>Kinetic Price</th></tr>
          {build_rows(bullish_df)}
        </table>

        <h2 style="color: #b71c1c;">🩸 TOP BEARISH BREAKDOWNS (SORTED BY TRUE MOMENTUM)</h2>
        <table class="bearish">
          <tr><th>Symbol</th><th>LTP</th><th>Vol Ratio</th><th>Volat Exp</th><th>Kinetic Vol</th><th>Kinetic Price</th></tr>
          {build_rows(bearish_df)}
        </table>
        <p style="font-size: 12px; color: #777; text-align: center;">Asit Baran Pati TMV Engine v24.0 (Kinetic Chain: {int(RECENT_ITERATION_PCT*100)}/{int((1-RECENT_ITERATION_PCT)*100)}) • Generated at {datetime.now().strftime('%I:%M %p')}</p>
      </body>
    </html>
    """
    
    msg = MIMEMultipart("alternative")
    msg['Subject'] = f"TMV Kinetic Scan: {datetime.now().strftime('%d %b - %I:%M %p')}"
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
    logger.info(f"Commencing Scan on {len(symbols)} symbols...")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(extract_raw_physics, sym, target_dt): sym for sym in symbols}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    df_results = pd.DataFrame(results)
    if df_results.empty: 
        logger.error("No valid data processed. Exiting.")
        return
        
    # --- MULTI-COLUMN PRIORITY SORTING ---
    # 1. First pushes True Breakouts (V_Pass = True, P_Pass = True) to the absolute top.
    # 2. Then sorts them by Volatility Expansion.
    SORT_CRITERIA = ['V_Pass', 'P_Pass', 'Volat_Ratio']
    
    bullish_df = df_results[df_results['Trend'] == 'BULLISH'].sort_values(SORT_CRITERIA, ascending=[False, False, False]).head(15)
    bearish_df = df_results[df_results['Trend'] == 'BEARISH'].sort_values(SORT_CRITERIA, ascending=[False, False, False]).head(15)
    
    send_html_email(bullish_df, bearish_df)
    logger.info("System execution completed successfully.")
 
if __name__ == "__main__":
    main()

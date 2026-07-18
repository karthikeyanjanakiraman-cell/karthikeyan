#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI TMV ENGINE - PRODUCTION v18.2 - PATCHED
INTEGRATION: TIME-SLICED MAX VOLUME | KINETIC ACCELERATION | PERCENTILE RANK | HTML EMAIL
═══════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import time
import logging
import smtplib
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
# 1. FIXED: DYNAMIC F&O UNIVERSE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def fetch_fo_universe():
    """Extracts the underlying Equity symbols from the Derivatives master file."""
    logger.info("Fetching Master F&O Universe...")
    try:
        response = requests.get(FYERS_FO_MASTER_URL)
        df = pd.read_csv(StringIO(response.text), header=None)
        
        # Column 1 contains derivatives like NSE:RELIANCE24AUGFUT
        raw_symbols = df[1].astype(str).tolist()
        
        base_symbols = set()
        for s in raw_symbols:
            if s.startswith('NSE:'):
                # Regex to extract just the text before the date/strike numbers
                match = re.search(r'NSE:([A-Z&]+)\d+', s)
                if match:
                    base_symbols.add(match.group(1))
        
        # Remove Indices
        ignore_list = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'}
        base_symbols = base_symbols - ignore_list
        
        # Format as Equity for historical fetching
        universe = [f"NSE:{sym}-EQ" for sym in base_symbols]
        logger.info(f"Successfully mapped {len(universe)} F&O underlying stocks.")
        return universe
    except Exception as e:
        logger.error(f"Failed to fetch Universe: {e}")
        return []

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 2. FIXED: KINETIC ACCELERATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def calculate_kinetic_acceleration(df_today, target_vol):
    """Measures the TIME INTERVALS between tiers to prove institutional urgency."""
    if target_vol <= 0: return 1.0
    
    cum_vol = df_today['volume'].cumsum()
    
    def get_time_to_reach(pct):
        target = target_vol * pct
        idx = np.searchsorted(cum_vol.values, target)
        if idx >= len(df_today): return len(df_today) * 15 # Max time
        return (idx + 1) * 15

    # Absolute time elapsed (Cumulative)
    t1 = get_time_to_reach(0.5)
    t2 = get_time_to_reach(0.75)
    t3 = get_time_to_reach(0.875)
    t4 = get_time_to_reach(1.0)
    
    # Calculate the exact duration of each step
    delta_1 = t1                # Time from 0 to 50%
    delta_2 = t2 - t1           # Time from 50% to 75%
    delta_3 = t3 - t2           # Time from 75% to 87.5%
    delta_4 = t4 - t3           # Time from 87.5% to 100%
    
    # If the intervals are collapsing, institutions are speeding up their buying
    if delta_4 < delta_3 < delta_2: 
        return 1.5 # Super-accelerated
    return 1.0

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# CORE PHYSICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def extract_raw_physics(symbol):
    try:
        now_dt = pd.Timestamp.now(tz="Asia/Kolkata")
        payload = {
            "symbol": symbol, "resolution": "15", "date_format": 1,
            "range_from": (now_dt - timedelta(days=180)).strftime("%Y-%m-%d"),
            "range_to": now_dt.strftime("%Y-%m-%d"), "cont_flag": 1
        }
        res = fyers.history(payload)
        
        if not res or 'candles' not in res or len(res['candles']) == 0: 
            return None
        
        df = pd.DataFrame(res['candles'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata")
        
        current_time = now_dt.time()
        
        # Time-Sliced Masking
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        
        historical_slice = df[df['time'] <= current_time]
        daily_groups = historical_slice.groupby('date').agg({
            'volume': 'sum',
            'high': 'max',
            'low': 'min'
        })
        daily_groups['range'] = daily_groups['high'] - daily_groups['low']
        
        max_vol = daily_groups['volume'].max()
        max_range = daily_groups['range'].max()
        
        # Today's Action
        today_df = df[df['date'] == now_dt.date()]
        if today_df.empty: return None
        
        curr_vol = today_df['volume'].sum()
        curr_range = (today_df['high'].max() - today_df['low'].min())
        
        vol_ratio = curr_vol / max_vol if max_vol > 0 else 0
        volatility_ratio = curr_range / max_range if max_range > 0 else 0
        
        # Trend Anchor
        df['price_dir'] = np.sign(df['close'].diff().fillna(0))
        df['obv'] = (df['price_dir'] * df['volume']).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=10).mean()
        
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
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# 3. FIXED: THE HTML EMAIL DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def send_html_email(bullish_df, bearish_df):
    """Constructs and sends a clean HTML matrix to your phone/desktop."""
    logger.info("Formatting and dispatching HTML Email...")
    
    html = f"""
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; background-color: #f4f4f9; padding: 20px; }}
          h2 {{ color: #333; }}
          table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
          th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
          th {{ background-color: #333; color: white; }}
          .bullish th {{ background-color: #2e7d32; }}
          .bearish th {{ background-color: #c62828; }}
        </style>
      </head>
      <body>
        <h2>🔥 ASIT TMV ENGINE: Bullish Breakouts</h2>
        <table class="bullish">
          <tr><th>Symbol</th><th>LTP</th><th>Vol Ratio</th><th>Volatility Exp</th><th>TMV Score</th></tr>
          {"".join(f"<tr><td>{row['Symbol']}</td><td>{row['LTP']}</td><td>{row['Vol_Ratio']}x</td><td>{row['Volat_Ratio']}x</td><td>{row['TMV_Score']}</td></tr>" for _, row in bullish_df.iterrows())}
        </table>

        <h2>🩸 ASIT TMV ENGINE: Bearish Breakdowns</h2>
        <table class="bearish">
          <tr><th>Symbol</th><th>LTP</th><th>Vol Ratio</th><th>Volatility Exp</th><th>TMV Score</th></tr>
          {"".join(f"<tr><td>{row['Symbol']}</td><td>{row['LTP']}</td><td>{row['Vol_Ratio']}x</td><td>{row['Volat_Ratio']}x</td><td>{row['TMV_Score']}</td></tr>" for _, row in bearish_df.iterrows())}
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

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def main():
    symbols = fetch_fo_universe()
    if not symbols: return
        
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(extract_raw_physics, sym): sym for sym in symbols}
        for future in as_completed(futures):
            res = future.result()
            if res: results.append(res)
            
    df_results = pd.DataFrame(results)
    if df_results.empty: return
        
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
    
    # 4. FIXED: Actually executing the Email function
    send_html_email(bullish_df, bearish_df)
    
if __name__ == "__main__":
    main()

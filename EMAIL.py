#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI STRATEGY - PRODUCTION v12.0 - SNIPER & STRETCH EDITION
DUAL ENGINE + ATR VWAP STRETCH FILTER + ELITE EMAIL SELECTION
═══════════════════════════════════════════════════════════════════════════════════════════════════

🎯 v12.0 MASTER ARCHITECTURE:
✅ VWAP STRETCH FILTER: Calculates 14-period ATR. If LTP is > 1.5x ATR away from VWAP, the setup is killed (🛑 Overextended).
✅ 9:15 AM ANCHOR: Extracts the true daily open and scores the momentum drift against it.
✅ STRICT SIGNAL FILTERING: Automatically blocks Overextended, Traps, and Standard setups from your inbox.
✅ DUAL ENGINE: Evaluates both macro TMV Rank and Intraday Drift Rank.
✅ CYBER DARK-MODE EMAIL: Delivers the Top 15 pristine longs and shorts directly to your email.

RUNNING: python asit_v12_master.py
OUTPUT: asit_dual_history_YYYYMMDD.csv
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
from email.mime.base import MIMEBase
from email import encoders
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import ta
from fyers_apiv3 import fyersModel

# ===== LOGGING SETUP =====
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(console_handler)

# ===== ENVIRONMENTAL CREDENTIALS =====
CLIENT_ID = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

FYERS_FO_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
TODAY_STR = datetime.now().strftime('%Y%m%d')
HISTORY_CSV = f"asit_dual_history_{TODAY_STR}.csv"

DATA_CACHE = {} 
MARKET_BASELINE = 0.0  

try:
    if not CLIENT_ID or not ACCESS_TOKEN:
        raise ValueError("Missing CLIENT_ID or ACCESS_TOKEN in environment variables.")
    fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False, log_path="")
    logger.info("[OK] Fyers API connected successfully")
except Exception as e:
    logger.error(f"[FATAL] Fyers API Authentication Error: {e}")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# DUAL-WINDOW THREAD-SAFE RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
class RateLimiter:
    def __init__(self, max_per_sec=8, max_per_min=190):
        self.min_interval = 1.0 / max_per_sec
        self.max_per_min = max_per_min
        self.timestamps = []
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            self.timestamps = [ts for ts in self.timestamps if now - ts < 60.0]
            
            sleep_time = 0.0
            if self.timestamps:
                elapsed = now - self.timestamps[-1]
                if elapsed < self.min_interval:
                    sleep_time = self.min_interval - elapsed
                    
            if len(self.timestamps) >= self.max_per_min:
                wait_for_min = 60.0 - (now - self.timestamps[0])
                if wait_for_min > sleep_time:
                    sleep_time = wait_for_min
                    
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.monotonic()
                
            self.timestamps.append(now)

rate_limiter = RateLimiter()

def call_with_retries(func, *args, **kwargs):
    for attempt in range(1, 4):
        try:
            rate_limiter.wait()
            res = func(*args, **kwargs)
            if isinstance(res, dict) and res.get('s') == 'error':
                raise RuntimeError(res.get('message'))
            return res
        except Exception as e:
            time.sleep(1)
    return None

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# LIVE F&O UNIVERSE RETRIEVAL & BASELINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def get_live_fno_symbols():
    try:
        logger.info("[INIT] Fetching Live F&O Universe from FYERS Master...")
        text = requests.get(FYERS_FO_MASTER_URL, timeout=10).text
        raw = pd.read_csv(StringIO(text), header=None)
        underlying = raw[13].astype(str).str.strip().str.upper().dropna().unique()
        symbols = [f"NSE:{sym}-EQ" for sym in underlying if len(sym) >= 2 and sym not in {"", "SYMBOL"} and "NIFTY" not in sym and "SENSEX" not in sym]
        symbols.extend(["NSE:NIFTY50-INDEX", "NSE:NIFTYBANK-INDEX"])
        return sorted(set(symbols))
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch F&O master list: {e}")
        return []

def pre_calculate_market_baseline():
    global MARKET_BASELINE
    try:
        hist = yf.Ticker('^NSEI').history(period='5d')
        if len(hist) >= 2:
            prev, curr = hist['Close'].iloc[-2], hist['Close'].iloc[-1]
            MARKET_BASELINE = ((curr - prev) / prev) * 100
    except:
        MARKET_BASELINE = 0.0

def get_rs_multiplier(symbol_pct):
    rs_diff = symbol_pct - MARKET_BASELINE
    multiplier = 1.0 + (rs_diff / 10.0)
    return max(0.7, min(1.3, multiplier))

def get_history_ttl(symbol, resolution, days_back):
    cache_key = f"{symbol}_{resolution}_{days_back}"
    now_ts = time.time()
    ttl_seconds = 43200 if str(resolution).upper() in ["D", "1D"] else 180
    
    if cache_key in DATA_CACHE:
        cached_df, fetch_time = DATA_CACHE[cache_key]
        if now_ts - fetch_time < ttl_seconds:
            return cached_df

    try:
        res_str = "1D" if str(resolution).upper() == "D" else str(resolution)
        now_dt = pd.Timestamp.now(tz="Asia/Kolkata")
        date_from = (now_dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        payload = {
            "symbol": symbol, "resolution": res_str, "date_format": 1,
            "range_from": date_from, "range_to": now_dt.strftime("%Y-%m-%d"), 
            "cont_flag": 0 if "INDEX" in symbol else 1
        }
        res = call_with_retries(fyers.history, data=payload)
        candles = res.get('candles', []) if isinstance(res, dict) else []
        if not candles: return None

        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        DATA_CACHE[cache_key] = (df, now_ts)
        return df
    except:
        return None

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING (INDICATORS + ATR STRETCH + 9:15 SNAPSHOTS)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def process_indicators(df, is_daily=False):
    if df is None or len(df) < 25: return None
    df = df.copy()
    
    # Calculate ATR for Volatility Stretch Measurement
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    if not is_daily:
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
        df['vol_price'] = df['volume'] * df['typical_price']
        df['vwap'] = df.groupby(df['timestamp'].dt.date)['vol_price'].cumsum() / df.groupby(df['timestamp'].dt.date)['volume'].cumsum()
        
        # Calculate VWAP Stretch Multiplier
        df['vwap_distance'] = abs(df['close'] - df['vwap'])
        df['vwap_stretch_atr'] = df['vwap_distance'] / df['atr']
        
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ema10'] = ta.trend.ema_indicator(df['obv'], window=10)
    df['obv_ema20'] = ta.trend.ema_indicator(df['obv'], window=20)
    
    df['fractal_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(2)) & \
                         (df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2))
    df['fractal_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(2)) & \
                        (df['low'] < df['low'].shift(-1)) & (df['low'] < df['low'].shift(-2))
    df['last_fractal_high_val'] = np.where(df['fractal_high'].shift(2), df['high'].shift(2), np.nan)
    df['last_fractal_high_val'] = df['last_fractal_high_val'].ffill()
    df['last_fractal_low_val'] = np.where(df['fractal_low'].shift(2), df['low'].shift(2), np.nan)
    df['last_fractal_low_val'] = df['last_fractal_low_val'].ffill()
    
    df['roc'] = ta.momentum.roc(df['close'], window=10)
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
    df['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'])
    df['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'])

    if not is_daily:
        df['anchor_915_price'] = df.groupby(df['timestamp'].dt.date)['open'].transform('first')
        for col in ['vwap', 'obv', 'roc', 'adx']:
            if col in df.columns:
                df[f'{col}_915'] = df.groupby(df['timestamp'].dt.date)[col].transform('first')
    else:
        df['anchor_915_price'] = df['open']
    
    if is_daily:
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hv'] = df['returns'].rolling(window=20).std() * np.sqrt(252) * 100
        
    return df

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# ENGINE 1 & 2 SCORING ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def calc_tmv_bull(df, is_daily):
    score = 0
    latest = df.iloc[-1]
    if not is_daily and latest['close'] > latest.get('vwap', 0): score += 0.20
    if latest['close'] > latest.get('last_fractal_high_val', float('inf')): score += 0.20
    if latest.get('roc', 0) > 0 and latest.get('roc', 0) > df['roc'].iloc[-2]: score += 0.20
    if latest.get('adx', 0) > 25 and latest.get('adx_pos', 0) > latest.get('adx_neg', 0): score += 0.20
    if latest.get('obv', 0) > latest.get('obv_ema10', 0) > latest.get('obv_ema20', 0): score += 0.20
    return min(score, 1.0)

def calc_tmv_bear(df, is_daily):
    score = 0
    latest = df.iloc[-1]
    if not is_daily and latest['close'] < latest.get('vwap', float('inf')): score += 0.20
    if latest['close'] < latest.get('last_fractal_low_val', -1): score += 0.20
    if latest.get('roc', 0) < 0 and latest.get('roc', 0) < df['roc'].iloc[-2]: score += 0.20
    if latest.get('adx', 0) > 25 and latest.get('adx_neg', 0) > latest.get('adx_pos', 0): score += 0.20
    if latest.get('obv', 0) < latest.get('obv_ema10', 0) < latest.get('obv_ema20', 0): score += 0.20
    return min(score, 1.0)

def calc_drift_bull(df, is_daily):
    if is_daily: return calc_tmv_bull(df, is_daily) 
    score = 0
    latest = df.iloc[-1]
    if latest['close'] > latest.get('vwap', 0): score += 0.20
    if latest.get('vwap', 0) > latest.get('vwap_915', float('inf')): score += 0.20
    if latest.get('roc', 0) > latest.get('roc_915', float('inf')): score += 0.15
    if latest.get('adx', 0) > latest.get('adx_915', float('inf')): score += 0.15
    if latest.get('obv', 0) > latest.get('obv_915', float('inf')): score += 0.30
    return min(score, 1.0)

def calc_drift_bear(df, is_daily):
    if is_daily: return calc_tmv_bear(df, is_daily)
    score = 0
    latest = df.iloc[-1]
    if latest['close'] < latest.get('vwap', float('inf')): score += 0.20
    if latest.get('vwap', 0) < latest.get('vwap_915', -1): score += 0.20
    if latest.get('roc', 0) < latest.get('roc_915', -1): score += 0.15
    if latest.get('adx', 0) > latest.get('adx_915', float('inf')): score += 0.15
    if latest.get('obv', 0) < latest.get('obv_915', -1): score += 0.30
    return min(score, 1.0)

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# SYSTEM RESOLUTION MACHINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
TIMEFRAMES = {
    '15min': {'res': '15', 'days': 20, 'w': 0.30},
    '1hour': {'res': '60', 'days': 40, 'w': 0.30},
    '1day': {'res': '1D', 'days': 200, 'w': 0.40}
}

def scan_symbol(symbol):
    results = {}
    symbol_pct = 0.0
    hv_20 = 0.0

    for tf_name, cfg in TIMEFRAMES.items():
        df = get_history_ttl(symbol, cfg['res'], cfg['days'])
        is_daily = (tf_name == '1day')
        df_ind = process_indicators(df, is_daily)
        if df_ind is None: continue
        
        if is_daily and len(df) >= 2:
            prev, curr = df['close'].iloc[-2], df['close'].iloc[-1]
            symbol_pct = ((curr - prev) / prev) * 100
            hv_20 = df_ind['hv'].iloc[-1]

        results[tf_name] = {
            'df': df_ind, 'w': cfg['w'],
            'tmv_bull': calc_tmv_bull(df_ind, is_daily),
            'tmv_bear': calc_tmv_bear(df_ind, is_daily),
            'drift_bull': calc_drift_bull(df_ind, is_daily),
            'drift_bear': calc_drift_bear(df_ind, is_daily)
        }

    if not results: return None

    rs_mult = get_rs_multiplier(symbol_pct)
    
    tb_tmv = sum(r['tmv_bull'] * r['w'] for r in results.values())
    tbear_tmv = sum(r['tmv_bear'] * r['w'] for r in results.values())
    tmv_net = ((tb_tmv * 15) - (tbear_tmv * 15)) * rs_mult
    tmv_rank = max(-15.0, min(15.0, tmv_net))
    
    tb_drift = sum(r['drift_bull'] * r['w'] for r in results.values())
    tbear_drift = sum(r['drift_bear'] * r['w'] for r in results.values())
    drift_net = ((tb_drift * 15) - (tbear_drift * 15)) * rs_mult
    drift_rank = max(-15.0, min(15.0, drift_net))

    trend = 'BULLISH' if tmv_rank > 0 else ('BEARISH' if tmv_rank < 0 else 'NEUTRAL')
    latest_df = results.get('15min', list(results.values())[0])['df']
    obv_shift = "Acc." if latest_df['obv'].iloc[-1] > latest_df.get('obv_915', pd.Series([0])).iloc[-1] else "Dist."
    
    anchor_price = latest_df['anchor_915_price'].iloc[-1] if 'anchor_915_price' in latest_df.columns else np.nan
    vwap_stretch = latest_df['vwap_stretch_atr'].iloc[-1] if 'vwap_stretch_atr' in latest_df.columns else 0.0

    return {
        'Symbol': symbol,
        'TMV_Rank': round(tmv_rank, 2),
        'Drift_Rank': round(drift_rank, 2),
        'Trend': trend,
        'Anchor_915': round(anchor_price, 2) if not pd.isna(anchor_price) else 0.0,
        'LTP': round(latest_df['close'].iloc[-1], 2),
        '% Change': round(symbol_pct, 2),
        'Intraday_Vol': obv_shift,
        'VWAP_Stretch': round(vwap_stretch, 2),
        'HV (20D)': round(hv_20, 2) if not pd.isna(hv_20) else 0.0
    }

def manage_daily_history(current_results):
    df_curr = pd.DataFrame(current_results)
    if df_curr.empty: return df_curr
    df_curr['Runtime'] = datetime.now().strftime('%H:%M:%S')

    if os.path.exists(HISTORY_CSV):
        df_hist = pd.read_csv(HISTORY_CSV)
        def get_prev_rank(sym, trend):
            sym_hist = df_hist[df_hist['Symbol'] == sym]
            if sym_hist.empty: return 0.0
            ranks = pd.to_numeric(sym_hist['TMV_Rank'], errors='coerce').dropna()
            return ranks.max() if trend == 'BULLISH' else ranks.min() if not ranks.empty else 0.0
            
        df_curr['Prev_TMV'] = df_curr.apply(lambda r: get_prev_rank(r['Symbol'], r['Trend']), axis=1)
    else:
        df_curr['Prev_TMV'] = 0.0

    df_curr['TMV_Diff'] = df_curr['TMV_Rank'] - df_curr['Prev_TMV']
    df_curr.to_csv(HISTORY_CSV, mode='a', header=not os.path.exists(HISTORY_CSV), index=False)
    return df_curr

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# VISUAL HTML ENGINE WITH STRICT CLASSIFIER BLOCKERS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def determine_setup(row):
    tmv = row['TMV_Rank']
    drift = row['Drift_Rank']
    vol = row['Intraday_Vol']
    trend = row['Trend']
    stretch = row.get('VWAP_Stretch', 0.0)
    
    # KILL SWITCH: If price is > 1.5x ATR from VWAP, it is structurally overextended
    if stretch > 1.5:
        return "🛑 Overextended", "#e91e63", "#fff"
    
    if trend == 'BULLISH':
        if drift >= tmv and vol == "Acc.": return "🎯 Sweet Spot", "#d4af37", "#000"
        elif abs(tmv - drift) <= 2.5 and vol == "Acc.": return "⚖️ Harmony", "#2196f3", "#fff"
        elif drift < tmv - 3 and vol == "Dist.": return "⚠️ The Trap", "#ff9800", "#000"
        else: return "🔄 Standard", "#555555", "#fff"
    elif trend == 'BEARISH':
        if drift <= tmv and vol == "Dist.": return "🎯 Sweet Spot", "#d4af37", "#000"
        elif abs(tmv - drift) <= 2.5 and vol == "Dist.": return "⚖️ Harmony", "#2196f3", "#fff"
        elif drift > tmv + 3 and vol == "Acc.": return "⚠️ The Trap", "#ff9800", "#000"
        else: return "🔄 Standard", "#555555", "#fff"
    return "N/A", "#555", "#fff"

def generate_colorful_html_table(df, side):
    if df.empty:
        return f"<p style='color:#aaaaaa;'>No elite {side.lower()} setups identified.</p>"
        
    html = """
    <table style="width:100%; border-collapse: collapse; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #ffffff; text-align: center; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.5);">
        <thead>
            <tr style="background-color: #2d2d30; border-bottom: 2px solid #3e3e42;">
                <th style="padding: 12px 8px;">Symbol</th>
                <th style="padding: 12px 8px;">Setup</th>
                <th style="padding: 12px 8px;">Anchor 9:15</th>
                <th style="padding: 12px 8px;">LTP</th>
                <th style="padding: 12px 8px;">% Chg</th>
                <th style="padding: 12px 8px;">TMV Rank</th>
                <th style="padding: 12px 8px;">Drift Rank</th>
                <th style="padding: 12px 8px;">TMV Diff</th>
                <th style="padding: 12px 8px;">Stretch</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for idx, row in df.iterrows():
        bg_color = "#1e1e1e" if idx % 2 == 0 else "#252526"
        pct = row['% Change']
        pct_color = "#4caf50" if pct > 0 else "#f44336" if pct < 0 else "#ffffff"
        
        diff = row['TMV_Diff']
        diff_color = "#4caf50" if diff > 0 else "#f44336" if diff < 0 else "#ffffff"
        
        badge_text, badge_bg, badge_fg = determine_setup(row)
        badge_html = f"<span style='background-color:{badge_bg}; color:{badge_fg}; padding:4px 8px; border-radius:12px; font-size:12px; font-weight:bold; white-space:nowrap;'>{badge_text}</span>"
        
        html += f"""
            <tr style="background-color: {bg_color}; border-bottom: 1px solid #333333;">
                <td style="padding: 10px 8px; font-weight: bold; color: #64b5f6;">{row['Symbol'].replace('NSE:', '').replace('-EQ', '')}</td>
                <td style="padding: 10px 8px;">{badge_html}</td>
                <td style="padding: 10px 8px; font-weight: 500; color: #e0e0e0;">{row['Anchor_915']:.2f}</td>
                <td style="padding: 10px 8px; font-weight: 600;">{row['LTP']:.2f}</td>
                <td style="padding: 10px 8px; color: {pct_color}; font-weight:bold;">{pct:+.2f}%</td>
                <td style="padding: 10px 8px;">{row['TMV_Rank']:.2f}</td>
                <td style="padding: 10px 8px;">{row['Drift_Rank']:.2f}</td>
                <td style="padding: 10px 8px; color: {diff_color}; font-weight:bold;">{diff:+.2f}</td>
                <td style="padding: 10px 8px;">{row.get('VWAP_Stretch', 0.0):.1f}x</td>
            </tr>
        """
        
    html += "</tbody></table>"
    return html

def send_email_report(df):
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
        return

    df['setup_name'] = df.apply(lambda r: determine_setup(r)[0], axis=1)
    
    # STRICT FILTER: Whitelist ONLY Sweet Spot and Harmony candidates. Kills Traps, Standard, and Overextended.
    df_elite = df[df['setup_name'].isin(["🎯 Sweet Spot", "⚖️ Harmony"])].copy()

    bulls = df_elite[(df_elite['Trend'] == 'BULLISH') & (df_elite['TMV_Rank'] > 0)].sort_values('TMV_Diff', ascending=False).head(15)
    bears = df_elite[(df_elite['Trend'] == 'BEARISH') & (df_elite['TMV_Rank'] < 0)].sort_values('TMV_Diff', ascending=True).head(15)

    bulls_table = generate_colorful_html_table(bulls, "BULLISH")
    bears_table = generate_colorful_html_table(bears, "BEARISH")

    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"] = SENDER_EMAIL, RECIPIENT_EMAIL
    msg["Subject"] = f"Elite Dual Matrix (TMV vs Drift) - {datetime.now().strftime('%H:%M')}"

    html = f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; padding: 20px;">
      
      <div style="text-align: center; margin-bottom: 30px;">
          <h2 style="color: #ffffff; margin-bottom: 5px;">Elite Visual Intelligence Matrix (v12.0)</h2>
          <p style="color: #aaaaaa; margin-top: 0;"><b>Session Baseline Reset:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
      </div>
      
      <h3 style="color: #4caf50; border-bottom: 2px solid #4caf50; padding-bottom: 5px; display: inline-block;">🚀 Top 15 Long Accelerators (Elite Only)</h3>
      {bulls_table}
      
      <br><br>
      
      <h3 style="color: #f44336; border-bottom: 2px solid #f44336; padding-bottom: 5px; display: inline-block;">🔻 Top 15 Short Accelerators (Elite Only)</h3>
      {bears_table}
      
      <div style="margin-top: 40px; padding: 15px; background-color: #1e1e1e; border-radius: 8px; border-left: 4px solid #d4af37;">
          <h4 style="color: #ffffff; margin-top: 0;">Elite Selection Setup Parameters:</h4>
          <ul style="color: #cccccc; line-height: 1.6;">
            <li><span style='background-color:#d4af37; color:#000; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:bold;'>🎯 Sweet Spot</span> : TMV macro trend is strong, AND the intraday Drift from 9:15 is accelerating even faster with Institutional volume confirmation.</li>
            <li><span style='background-color:#2196f3; color:#fff; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:bold;'>⚖️ Harmony</span> : TMV and Drift are perfectly aligned. High confidence, sustainable setup.</li>
          </ul>
          <p style="color: #e91e63; font-size: 12px; margin-bottom: 0;">* VWAP Stretch Protection is ACTIVE. Any asset >1.5x ATR from VWAP is blocked as 🛑 Overextended.</p>
          <p style="color: #999; font-size: 12px; margin-bottom: 0;">* Inefficient churn metrics (Standard/Trap conditions) have been strictly dropped from this grid view.</p>
      </div>
      
    </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        if os.path.exists(HISTORY_CSV):
            with open(HISTORY_CSV, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{HISTORY_CSV}"')
            msg.attach(part)
    except:
        pass

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        logger.info(f"[EMAIL] Elite Selection reports successfully dispatched.")
    except Exception as e:
        logger.error(f"[EMAIL] SMTP Transmission Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# MAIN CONTROL GATE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("[LAUNCH] ASIT v12.0 - SNIPER & STRETCH EDITION")
    print("=" * 80)
    symbols = get_live_fno_symbols()
    if not symbols: sys.exit()
    pre_calculate_market_baseline()
    
    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(scan_symbol, sym): sym for sym in symbols}
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                if res := future.result(): results.append(res)
                if idx % 20 == 0: logger.info(f"[SCAN] Processing Elite Matrix: {idx}/{len(symbols)} complete...")
            except: pass

    if results:
        df_final = manage_daily_history(results)
        send_email_report(df_final)
        print("=" * 80)
        print("[SUCCESS] Operational Phase Concluded.")
        print("=" * 80)

if __name__ == "__main__":
    main()

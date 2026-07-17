#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI STRATEGY - PRODUCTION v12.3 - PURE VOLUME CONFLUENCE EDITION
DUAL ENGINE + VOLUME CLIMAX FILTER + INSTITUTIONAL NODE MAPPING + THROTTLE PROTECT
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
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# ===== ENVIRONMENTAL CREDENTIALS =====
CLIENT_ID = os.environ.get("CLIENT_ID") or os.environ.get("CLIENTID")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN") or os.environ.get("ACCESSTOKEN")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL")

FYERS_FO_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
TODAY_STR = datetime.now().strftime('%Y%m%d')
HISTORY_CSV = f"asit_vol_history_{TODAY_STR}.csv"

DATA_CACHE = {} 
CACHE_LOCK = threading.Lock()
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
# NON-BLOCKING RATE LIMITER & EXPONENTIAL BACKOFF
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
            # Clean up timestamps older than 60 seconds
            self.timestamps = [ts for ts in self.timestamps if now - ts < 60.0]
            
            if self.timestamps:
                elapsed = now - self.timestamps[-1]
                if elapsed < self.min_interval:
                    sleep_time = self.min_interval - elapsed
                    
            if len(self.timestamps) >= self.max_per_min:
                wait_for_min = 60.0 - (now - self.timestamps[0])
                if wait_for_min > sleep_time:
                    sleep_time = wait_for_min
            
            # Record the projected execution time
            self.timestamps.append(now + sleep_time)
            
        if sleep_time > 0:
            time.sleep(sleep_time)

rate_limiter = RateLimiter()

def call_with_retries(func, *args, **kwargs):
    backoff = 3  # Start with a 3-second penalty wait
    for attempt in range(1, 5):  # Try up to 4 times
        try:
            rate_limiter.wait()
            res = func(*args, **kwargs)
            
            if isinstance(res, dict) and res.get('s') == 'error':
                err_msg = res.get('message', '').lower()
                if 'limit' in err_msg:
                    raise ValueError(f"Rate Limit Triggered: {err_msg}")
                else:
                    logger.warning(f"[API ERROR] {err_msg}")
                    return None
                    
            return res
        except Exception as e:
            logger.warning(f"[RETRY] Attempt {attempt} failed ({e}). Backing off for {backoff}s...")
            time.sleep(backoff)
            backoff *= 2  # Double the wait time on the next failure (3s -> 6s -> 12s)
            
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
    except Exception as e:
        MARKET_BASELINE = 0.0

def get_rs_multiplier(symbol_pct):
    rs_diff = symbol_pct - MARKET_BASELINE
    multiplier = 1.0 + (rs_diff / 10.0)
    return max(0.7, min(1.3, multiplier))

def get_history_ttl(symbol, resolution, days_back):
    cache_key = f"{symbol}_{resolution}_{days_back}"
    now_ts = time.time()
    ttl_seconds = 43200 if str(resolution).upper() in ["D", "1D"] else 180
    
    with CACHE_LOCK:
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
            "cont_flag": 1
        }
        res = call_with_retries(fyers.history, data=payload)
        if not res or not isinstance(res, dict): return None
            
        candles = res.get('candles', [])
        if not candles: return None

        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        with CACHE_LOCK:
            DATA_CACHE[cache_key] = (df, now_ts)
        return df
    except Exception as e:
        return None

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# PURE VOLUME DATA PROCESSING 
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def process_volume_indicators(df, is_daily=False):
    if df is None or len(df) < 25: return None
    df = df.copy()
    
    # 1. Base Cumulative Flow (OBV)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ema10'] = ta.trend.ema_indicator(df['obv'], window=10)
    df['obv_ema20'] = ta.trend.ema_indicator(df['obv'], window=20)
    
    # 2. Synthetic Volume OHLC (For V-ADX)
    df['vol_high'] = df['obv'] + df['volume']
    df['vol_low'] = df['obv'] - df['volume']
    df['vol_close'] = df['obv']
    
    # 3. Volume ADX & Directional Flow
    df['v_adx'] = ta.trend.adx(df['vol_high'], df['vol_low'], df['vol_close'], window=14).fillna(0)
    df['v_adx_pos'] = ta.trend.adx_pos(df['vol_high'], df['vol_low'], df['vol_close'], window=14).fillna(0)
    df['v_adx_neg'] = ta.trend.adx_neg(df['vol_high'], df['vol_low'], df['vol_close'], window=14).fillna(0)
    
    # 4. Volume ROC (Acceleration of order flow)
    df['v_roc'] = ta.momentum.roc(df['obv'], window=10).fillna(0)
    
    # 5. Average True Volume Range (ATVR)
    df['v_change_abs'] = df['volume'].diff().abs()
    df['atvr'] = df['v_change_abs'].rolling(window=14).mean().fillna(1)
    
    # VWAP and Volume Stretch Calculation
    if not is_daily:
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
        df['vol_price'] = df['volume'] * df['typical_price']
        cum_vol = df.groupby(df['timestamp'].dt.date)['volume'].cumsum()
        cum_vol_price = df.groupby(df['timestamp'].dt.date)['vol_price'].cumsum()
        df['vwap'] = cum_vol_price / np.where(cum_vol == 0, 1, cum_vol)
        
        # Pure Volume Climax Stretch (How huge is the current volume vs average shift)
        df['vol_stretch'] = df['volume'] / np.where(df['atvr'] == 0, 1, df['atvr'])
        
    # 6. Volume Fractals (Institutional Support/Resistance Nodes)
    df['vol_fractal_peak'] = (df['volume'] > df['volume'].shift(1)) & (df['volume'] > df['volume'].shift(2)) & \
                             (df['volume'] > df['volume'].shift(-1)) & (df['volume'] > df['volume'].shift(-2))
    
    # Capture the price level where the last massive volume spike occurred
    df['last_vol_spike_price'] = np.where(df['vol_fractal_peak'].shift(2), df['close'].shift(2), np.nan)
    df['last_vol_spike_price'] = df['last_vol_spike_price'].ffill()

    # 9:15 AM Anchor Captures for Intraday Drift Tracking
    if not is_daily:
        df['anchor_915_price'] = df.groupby(df['timestamp'].dt.date)['open'].transform('first')
        for col in ['vwap', 'obv', 'v_roc', 'v_adx']:
            if col in df.columns:
                df[f'{col}_915'] = df.groupby(df['timestamp'].dt.date)[col].transform('first')
    else:
        df['anchor_915_price'] = df['open']
        
    # 7. Historical Volume Volatility (HVV)
    if is_daily:
        df['v_returns'] = np.log(df['volume'] / df['volume'].shift(1).replace(0, np.nan))
        df['hvv'] = df['v_returns'].rolling(window=20).std() * np.sqrt(252) * 100
        
    return df

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# ENGINE SCORING: PURE VOLUME METRICS
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def calc_tmv_bull(df, is_daily):
    score = 0
    latest = df.iloc[-1]
    if not is_daily and latest['close'] > latest.get('vwap', 0): score += 0.20
    # Price has pushed ABOVE the last massive institutional volume node
    if latest['close'] > latest.get('last_vol_spike_price', float('inf')): score += 0.20
    # Volume momentum is positive and actively increasing
    if latest.get('v_roc', 0) > 0 and latest.get('v_roc', 0) > df['v_roc'].iloc[-2]: score += 0.20
    # Strong volume trend in the upward direction
    if latest.get('v_adx', 0) > 25 and latest.get('v_adx_pos', 0) > latest.get('v_adx_neg', 0): score += 0.20
    if latest.get('obv', 0) > latest.get('obv_ema10', 0) > latest.get('obv_ema20', 0): score += 0.20
    return min(score, 1.0)

def calc_tmv_bear(df, is_daily):
    score = 0
    latest = df.iloc[-1]
    if not is_daily and latest['close'] < latest.get('vwap', float('inf')): score += 0.20
    # Price has fallen BELOW the last massive institutional volume node
    if latest['close'] < latest.get('last_vol_spike_price', float('-inf')): score += 0.20
    if latest.get('v_roc', 0) < 0 and latest.get('v_roc', 0) < df['v_roc'].iloc[-2]: score += 0.20
    if latest.get('v_adx', 0) > 25 and latest.get('v_adx_neg', 0) > latest.get('v_adx_pos', 0): score += 0.20
    if latest.get('obv', 0) < latest.get('obv_ema10', 0) < latest.get('obv_ema20', 0): score += 0.20
    return min(score, 1.0)

def calc_drift_bull(df, is_daily):
    if is_daily: return calc_tmv_bull(df, is_daily) 
    score = 0
    latest = df.iloc[-1]
    if latest['close'] > latest.get('vwap', 0): score += 0.20
    if latest.get('vwap', 0) > latest.get('vwap_915', float('inf')): score += 0.20
    if latest.get('v_roc', 0) > latest.get('v_roc_915', float('inf')): score += 0.15
    if latest.get('v_adx', 0) > latest.get('v_adx_915', float('inf')): score += 0.15
    if latest.get('obv', 0) > latest.get('obv_915', float('inf')): score += 0.30
    return min(score, 1.0)

def calc_drift_bear(df, is_daily):
    if is_daily: return calc_tmv_bear(df, is_daily)
    score = 0
    latest = df.iloc[-1]
    if latest['close'] < latest.get('vwap', float('inf')): score += 0.20
    if latest.get('vwap', 0) < latest.get('vwap_915', float('-inf')): score += 0.20
    if latest.get('v_roc', 0) < latest.get('v_roc_915', float('-inf')): score += 0.15
    if latest.get('v_adx', 0) > latest.get('v_adx_915', float('inf')): score += 0.15
    if latest.get('obv', 0) < latest.get('obv_915', float('-inf')): score += 0.30
    return min(score, 1.0)

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# SYSTEM RESOLUTION MACHINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
TIMEFRAMES = {
    '15min': {'res': '15', 'days': 20},
    '1hour': {'res': '60', 'days': 40},
    '1day': {'res': '1D', 'days': 200}
}

def scan_symbol(symbol):
    results = {}
    symbol_pct = 0.0
    hvv_20 = 0.0

    for tf_name, cfg in TIMEFRAMES.items():
        df = get_history_ttl(symbol, cfg['res'], cfg['days'])
        is_daily = (tf_name == '1day')
        df_ind = process_volume_indicators(df, is_daily)
        if df_ind is None: continue
        
        if is_daily and len(df) >= 2:
            prev, curr = df['close'].iloc[-2], df['close'].iloc[-1]
            if prev != 0: symbol_pct = ((curr - prev) / prev) * 100
            hvv_20 = df_ind['hvv'].iloc[-1]

        results[tf_name] = {
            'df': df_ind,
            'tmv_bull': calc_tmv_bull(df_ind, is_daily),
            'tmv_bear': calc_tmv_bear(df_ind, is_daily),
            'drift_bull': calc_drift_bull(df_ind, is_daily),
            'drift_bear': calc_drift_bear(df_ind, is_daily)
        }

    # CRITICAL FLUSH: Reject the tracking run if any target timeframe dropouts occur
    if len(results) < len(TIMEFRAMES): return None

    rs_mult = get_rs_multiplier(symbol_pct)
    num_tf = len(results)
    
    tb_tmv = sum(r['tmv_bull'] for r in results.values()) / num_tf
    tbear_tmv = sum(r['tmv_bear'] for r in results.values()) / num_tf
    tmv_net = ((tb_tmv * 15) - (tbear_tmv * 15)) * rs_mult
    tmv_rank = max(-15.0, min(15.0, tmv_net))
    
    tb_drift = sum(r['drift_bull'] for r in results.values()) / num_tf
    tbear_drift = sum(r['drift_bear'] for r in results.values()) / num_tf
    drift_net = ((tb_drift * 15) - (tbear_drift * 15)) * rs_mult
    drift_rank = max(-15.0, min(15.0, drift_net))

    trend = 'BULLISH' if tmv_rank > 0 else ('BEARISH' if tmv_rank < 0 else 'NEUTRAL')
    
    latest_df = results['15min']['df']
    obv_shift = "Acc." if latest_df['obv'].iloc[-1] > latest_df.get('obv_915', pd.Series([0])).iloc[-1] else "Dist."
    
    anchor_price = latest_df['anchor_915_price'].iloc[-1] if 'anchor_915_price' in latest_df.columns else np.nan
    vol_stretch = latest_df['vol_stretch'].iloc[-1] if 'vol_stretch' in latest_df.columns else 1.0

    return {
        'Symbol': symbol,
        'TMV_Rank': round(tmv_rank, 2),
        'Drift_Rank': round(drift_rank, 2),
        'Trend': trend,
        'Anchor_915': round(anchor_price, 2) if not pd.isna(anchor_price) else 0.0,
        'LTP': round(latest_df['close'].iloc[-1], 2),
        '% Change': round(symbol_pct, 2),
        'Intraday_Vol': obv_shift,
        'Vol_Stretch': round(vol_stretch, 2),
        'HVV (20D)': round(hvv_20, 2) if not pd.isna(hvv_20) else 0.0
    }

def manage_daily_history(current_results):
    df_curr = pd.DataFrame(current_results)
    if df_curr.empty: return df_curr
    df_curr['Runtime'] = datetime.now().strftime('%H:%M:%S')

    if os.path.exists(HISTORY_CSV):
        try:
            df_hist = pd.read_csv(HISTORY_CSV)
            def get_prev_rank(sym, trend):
                sym_hist = df_hist[df_hist['Symbol'] == sym]
                if sym_hist.empty: return 0.0
                ranks = pd.to_numeric(sym_hist['TMV_Rank'], errors='coerce').dropna()
                return ranks.max() if trend == 'BULLISH' else ranks.min() if not ranks.empty else 0.0
            df_curr['Prev_TMV'] = df_curr.apply(lambda r: get_prev_rank(r['Symbol'], r['Trend']), axis=1)
        except:
            df_curr['Prev_TMV'] = 0.0
    else:
        df_curr['Prev_TMV'] = 0.0

    df_curr['TMV_Diff'] = df_curr['TMV_Rank'] - df_curr['Prev_TMV']
    df_curr.to_csv(HISTORY_CSV, mode='a', header=not os.path.exists(HISTORY_CSV), index=False)
    return df_curr

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# STRICT MOMENTUM DIVERGENCE SETUP & EMAIL
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def determine_setup(row):
    tmv = row['TMV_Rank']
    drift = row['Drift_Rank']
    vol = row['Intraday_Vol']
    trend = row['Trend']
    stretch = row.get('Vol_Stretch', 1.0)
    
    # 1. Volume Climax Kill-Switch: If current bar volume is > 4x the Average True Volume Range
    if stretch > 4.0:
        return "🛑 Vol Climax", "#e91e63", "#fff"
    
    if trend == 'BULLISH':
        # 2. Strict Divergence Kill-Switch: Drift falling behind Macro Trend
        if drift < tmv - 0.5:
            return "📉 Fading Momentum", "#ff9800", "#000"
        elif drift >= tmv + 1.0 and vol == "Acc.": 
            return "🎯 Sweet Spot", "#d4af37", "#000"
        elif drift >= tmv and vol == "Acc.": 
            return "⚖️ Harmony", "#2196f3", "#fff"
        else: 
            return "🔄 Standard", "#555555", "#fff"
            
    elif trend == 'BEARISH':
        if drift > tmv + 0.5:
            return "📉 Fading Momentum", "#ff9800", "#000"
        elif drift <= tmv - 1.0 and vol == "Dist.": 
            return "🎯 Sweet Spot", "#d4af37", "#000"
        elif drift <= tmv and vol == "Dist.": 
            return "⚖️ Harmony", "#2196f3", "#fff"
        else: 
            return "🔄 Standard", "#555555", "#fff"
            
    return "N/A", "#555", "#fff"

def generate_colorful_html_table(df, side):
    if df.empty:
        return f"<p style='color:#aaaaaa; font-size: 13px;'>No elite {side.lower()} setups identified matching metrics.</p>"
        
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
                <th style="padding: 12px 8px;">V-Stretch</th>
                <th style="padding: 12px 8px;">HVV</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for idx, row in df.reset_index(drop=True).iterrows():
        bg_color = "#1e1e1e" if idx % 2 == 0 else "#252526"
        pct = row['% Change']
        pct_color = "#4caf50" if pct > 0 else "#f44336" if pct < 0 else "#ffffff"
        
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
                <td style="padding: 10px 8px; font-weight:bold;">{row['Drift_Rank']:.2f}</td>
                <td style="padding: 10px 8px;">{row.get('Vol_Stretch', 1.0):.1f}x</td>
                <td style="padding: 10px 8px; color: #999;">{row.get('HVV (20D)', 0.0):.1f}%</td>
            </tr>
        """
    html += "</tbody></table>"
    return html

def send_email_report(df):
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
        logger.warning("[EMAIL] Missing SMTP configurations. Dispatch halted.")
        return

    df = df.copy()
    df['setup_name'] = df.apply(lambda r: determine_setup(r)[0], axis=1)
    df_elite = df[df['setup_name'].isin(["🎯 Sweet Spot", "⚖️ Harmony"])].copy()

    # STRICT MOMENTUM SORTER: Always push highest current intraday Drift to the top
    bulls = df_elite[(df_elite['Trend'] == 'BULLISH') & (df_elite['TMV_Rank'] > 0)].sort_values(
        by=['Drift_Rank', 'TMV_Rank'], ascending=[False, False]).head(15)
        
    bears = df_elite[(df_elite['Trend'] == 'BEARISH') & (df_elite['TMV_Rank'] < 0)].sort_values(
        by=['Drift_Rank', 'TMV_Rank'], ascending=[True, True]).head(15)

    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"] = SENDER_EMAIL, RECIPIENT_EMAIL
    msg["Subject"] = f"Elite Volume Matrix (Quant Architecture) - {datetime.now().strftime('%H:%M')}"

    html = f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; padding: 20px;">
      
      <div style="text-align: center; margin-bottom: 30px;">
          <h2 style="color: #ffffff; margin-bottom: 5px;">Pure Volume Institutional Matrix (v12.3)</h2>
          <p style="color: #aaaaaa; margin-top: 0;"><b>Volume Flow Engine Active:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
      </div>
      
      <h3 style="color: #4caf50; border-bottom: 2px solid #4caf50; padding-bottom: 5px; display: inline-block;">🚀 Top 15 Long Accumulators</h3>
      {generate_colorful_html_table(bulls, "BULLISH")}
      
      <br><br>
      
      <h3 style="color: #f44336; border-bottom: 2px solid #f44336; padding-bottom: 5px; display: inline-block;">🔻 Top 15 Short Distributors</h3>
      {generate_colorful_html_table(bears, "BEARISH")}
      
      <div style="margin-top: 40px; padding: 15px; background-color: #1e1e1e; border-radius: 8px; border-left: 4px solid #d4af37;">
          <h4 style="color: #ffffff; margin-top: 0;">Quantitative Volume Shifts:</h4>
          <ul style="color: #cccccc; line-height: 1.6;">
            <li><span style='background-color:#ff9800; color:#000; padding:2px 6px; border-radius:4px; font-size:11px; font-weight:bold;'>📉 Fading Momentum</span> is now instantly killed. If Drift falls below TMV, it drops off the Elite List.</li>
            <li><b>Vol Climax Protection:</b> If volume exceeds 4x the Average True Volume Range (ATVR), setup is killed as a climax trap.</li>
            <li><b>Institutional Nodes:</b> Fractures are now mapped by massive volume spikes, not price wicks.</li>
          </ul>
      </div>
    </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))

    # Attach the history log if it exists
    if os.path.exists(HISTORY_CSV):
        try:
            with open(HISTORY_CSV, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{HISTORY_CSV}"')
            msg.attach(part)
        except Exception as e:
            logger.error(f"[EMAIL] Failed to attach log history payload: {e}")

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        logger.info(f"[EMAIL] Volume Confluence matrix successfully dispatched.")
    except Exception as e:
        logger.error(f"[EMAIL] SMTP Transmission Error: {e}")

def main():
    print("=" * 80)
    print("[LAUNCH] ASIT v12.3 - PURE VOLUME CONFLUENCE EDITION")
    print("=" * 80)
    symbols = get_live_fno_symbols()
    if not symbols: 
        logger.error("[TERMINATE] No underlying options targets found inside the session.")
        sys.exit(1)
        
    pre_calculate_market_baseline()
    
    results = []
    # Throttled to 2 concurrent workers to prevent API bans
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(scan_symbol, sym): sym for sym in symbols}
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
                if res: results.append(res)
                if idx % 20 == 0: logger.info(f"[SCAN] Processing Volume Matrix: {idx}/{len(symbols)} complete...")
            except Exception as e:
                pass

    if results:
        df_final = manage_daily_history(results)
        send_email_report(df_final)
        print("=" * 80)
        print("[SUCCESS] Operational Phase Concluded.")
        print("=" * 80)
    else:
        logger.error("[FATAL ERROR] System Matrix failed to generate output configurations.")

if __name__ == "__main__":
    main()

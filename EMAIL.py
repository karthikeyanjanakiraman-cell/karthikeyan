#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
ASIT BARAN PATI STRATEGY - PRODUCTION v12.5 - TRUE KINETIC QUANT EDITION
TIME-TO-VOLUME ANCHOR + UNIVERSE Z-SCORE + SQUEEZE DETECTOR + BLOCK FILTRATION
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
HISTORY_CSV = f"asit_kinetic_quant_{TODAY_STR}.csv"

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
    backoff = 3  
    for attempt in range(1, 5):  
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
# LIVE F&O UNIVERSE RETRIEVAL
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

def get_history_ttl(symbol, resolution, days_back):
    cache_key = f"{symbol}_{resolution}_{days_back}"
    now_ts = time.time()
    ttl_seconds = 43200 if str(resolution).upper() in ["D", "1D"] else 180
    
    with CACHE_LOCK:
        if cache_key in DATA_CACHE:
            cached_df, fetch_time = DATA_CACHE[cache_key]
            if now_ts - fetch_time < ttl_seconds: return cached_df

    try:
        res_str = "1D" if str(resolution).upper() == "D" else str(resolution)
        now_dt = pd.Timestamp.now(tz="Asia/Kolkata")
        date_from = (now_dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        payload = {
            "symbol": symbol, "resolution": res_str, "date_format": 1,
            "range_from": date_from, "range_to": now_dt.strftime("%Y-%m-%d"), "cont_flag": 1
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
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# CORE KINETIC INDICATOR PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def process_kinetic_quant_indicators(df, tf_name, is_daily=False):
    if df is None or len(df) < 30: return None
    df = df.copy()
    
    # Baseline Metrics
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14).fillna(0)
    df['v_change_abs'] = df['volume'].diff().abs()
    df['atvr'] = df['v_change_abs'].rolling(window=14).mean().fillna(1)
    
    # 1. Block Deal / Dark Pool Filtration
    df['price_spread'] = df['high'] - df['low']
    df['is_block_deal'] = (df['volume'] > 5 * df['atvr']) & (df['price_spread'] < 0.25 * df['atr'])
    
    # 2. Volatility Squeeze Check (BB inside KC)
    df['bb_up'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
    df['bb_low'] = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
    df['kc_up'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'], window=20, window_atr=1.5)
    df['kc_low'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'], window=20, window_atr=1.5)
    df['squeeze_on'] = (df['bb_up'] < df['kc_up']) & (df['bb_low'] > df['kc_low'])
    
    # 3. Base Flow Indicators
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ema10'] = ta.trend.ema_indicator(df['obv'], window=10)
    df['v_roc'] = ta.momentum.roc(df['obv'], window=10).fillna(0)
    
    # 4. RESTORED: Kinetic Volume Time Anchoring
    if tf_name == '15min': sec = 900
    elif tf_name == '1hour': sec = 3600
    else: sec = 22500  
    
    # Clean volume for velocity calculation: Ignore block deal anomalies
    clean_vol = np.where(df['is_block_deal'], df['volume'].rolling(14).mean().fillna(1), df['volume'])
    safe_vol = np.where(clean_vol <= 0, 1, clean_vol)
    
    df['cur_time_10k'] = (10000 / safe_vol) * sec
    
    if not is_daily:
        df['peak_time_10k'] = df.groupby(df['timestamp'].dt.date)['cur_time_10k'].cummin()
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
        df['vol_price'] = df['volume'] * df['typical_price']
        cum_vol = df.groupby(df['timestamp'].dt.date)['volume'].cumsum()
        cum_vol_price = df.groupby(df['timestamp'].dt.date)['vol_price'].cumsum()
        df['vwap'] = cum_vol_price / np.where(cum_vol == 0, 1, cum_vol)
        df['vol_stretch'] = df['volume'] / np.where(df['atvr'] == 0, 1, df['atvr'])
    else:
        df['peak_time_10k'] = df['cur_time_10k'].cummin()
        
    # Speed Efficiency Metric
    df['speed_efficiency'] = df['peak_time_10k'] / df['cur_time_10k']
    
    # 5. Volume Fractals mapped strictly onto clean execution speed
    df['vol_fractal_peak'] = (df['cur_time_10k'] < df['cur_time_10k'].shift(1)) & (df['cur_time_10k'] < df['cur_time_10k'].shift(2)) & \
                             (df['cur_time_10k'] < df['cur_time_10k'].shift(-1)) & (df['cur_time_10k'] < df['cur_time_10k'].shift(-2))
    df['last_vol_spike_price'] = np.where(df['vol_fractal_peak'].shift(2), df['close'].shift(2), np.nan)
    df['last_vol_spike_price'] = df['last_vol_spike_price'].ffill()

    return df

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# KINETIC SCORING ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def calc_tmv_bull(df, is_daily):
    score = 0
    latest = df.iloc[-1]
    if not is_daily and latest['close'] > latest.get('vwap', 0): score += 0.20
    if latest['close'] > latest.get('last_vol_spike_price', float('inf')): score += 0.20
    if latest.get('v_roc', 0) > 0 and latest.get('v_roc', 0) > df['v_roc'].iloc[-2]: score += 0.20
    if not latest.get('squeeze_on', True) and df['squeeze_on'].iloc[-5:-1].any(): score += 0.30
    if latest.get('obv', 0) > latest.get('obv_ema10', 0): score += 0.10
    return min(score, 1.0)

def calc_tmv_bear(df, is_daily):
    score = 0
    latest = df.iloc[-1]
    if not is_daily and latest['close'] < latest.get('vwap', float('inf')): score += 0.20
    if latest['close'] < latest.get('last_vol_spike_price', float('-inf')): score += 0.20
    if latest.get('v_roc', 0) < 0 and latest.get('v_roc', 0) < df['v_roc'].iloc[-2]: score += 0.20
    if not latest.get('squeeze_on', True) and df['squeeze_on'].iloc[-5:-1].any(): score += 0.30
    if latest.get('obv', 0) < latest.get('obv_ema10', 0): score += 0.10
    return min(score, 1.0)

def calc_drift_bull(df, is_daily):
    if is_daily: return calc_tmv_bull(df, is_daily) 
    score = 0
    latest = df.iloc[-1]
    speed_eff = latest.get('speed_efficiency', 0.0)
    
    if latest['close'] > latest.get('vwap', 0): score += 0.20
    if latest.get('obv', 0) > latest.get('obv_ema10', float('inf')): score += 0.20
    if latest.get('v_roc', 0) > 0: score += 0.15
    
    # Score based strictly on Kinetic Speed Efficiency
    if speed_eff >= 0.80: score += 0.45       
    elif speed_eff >= 0.50: score += 0.25     
    elif speed_eff >= 0.30: score += 0.10
    return min(score, 1.0)

def calc_drift_bear(df, is_daily):
    if is_daily: return calc_tmv_bear(df, is_daily)
    score = 0
    latest = df.iloc[-1]
    speed_eff = latest.get('speed_efficiency', 0.0)
    
    if latest['close'] < latest.get('vwap', float('inf')): score += 0.20
    if latest.get('obv', 0) < latest.get('obv_ema10', float('-inf')): score += 0.20
    if latest.get('v_roc', 0) < 0: score += 0.15
    
    if speed_eff >= 0.80: score += 0.45       
    elif speed_eff >= 0.50: score += 0.25     
    elif speed_eff >= 0.30: score += 0.10
    return min(score, 1.0)

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL CORE SCANNER
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
TIMEFRAMES = {
    '15min': {'res': '15', 'days': 20},
    '1hour': {'res': '60', 'days': 40},
    '1day': {'res': '1D', 'days': 200}
}

def format_speed(seconds):
    if pd.isna(seconds) or seconds == float('inf'): return "N/A"
    if seconds < 60: return f"{int(seconds)}s"
    return f"{seconds/60:.1f}m"

def scan_symbol(symbol):
    results = {}
    symbol_pct = 0.0

    for tf_name, cfg in TIMEFRAMES.items():
        df = get_history_ttl(symbol, cfg['res'], cfg['days'])
        is_daily = (tf_name == '1day')
        df_ind = process_kinetic_quant_indicators(df, tf_name, is_daily)
        if df_ind is None: continue
        
        if is_daily and len(df) >= 2:
            prev, curr = df['close'].iloc[-2], df['close'].iloc[-1]
            if prev != 0: symbol_pct = ((curr - prev) / prev) * 100

        results[tf_name] = {
            'df': df_ind,
            'tmv_bull': calc_tmv_bull(df_ind, is_daily),
            'tmv_bear': calc_tmv_bear(df_ind, is_daily),
            'drift_bull': calc_drift_bull(df_ind, is_daily),
            'drift_bear': calc_drift_bear(df_ind, is_daily)
        }

    if len(results) < len(TIMEFRAMES): return None

    num_tf = len(results)
    tb_tmv = sum(r['tmv_bull'] for r in results.values()) / num_tf
    tbear_tmv = sum(r['tmv_bear'] for r in results.values()) / num_tf
    tmv_net = (tb_tmv * 15) - (tbear_tmv * 15)
    
    tb_drift = sum(r['drift_bull'] for r in results.values()) / num_tf
    tbear_drift = sum(r['drift_bear'] for r in results.values()) / num_tf
    drift_net = (tb_drift * 15) - (tbear_drift * 15)

    trend = 'BULLISH' if tmv_net > 0 else ('BEARISH' if tmv_net < 0 else 'NEUTRAL')
    
    latest_df = results['15min']['df']
    macro_df = results['1day']['df']
    
    peak_speed = latest_df['peak_time_10k'].iloc[-1]
    cur_speed = latest_df['cur_time_10k'].iloc[-1]
    vol_stretch = latest_df['vol_stretch'].iloc[-1] if 'vol_stretch' in latest_df.columns else 1.0
    
    squeeze = "Yes" if not macro_df['squeeze_on'].iloc[-1] and macro_df['squeeze_on'].iloc[-5:-1].any() else "No"
    block_alert = "⚠️ Block Print" if latest_df['is_block_deal'].iloc[-1] else ""

    return {
        'Symbol': symbol,
        'TMV_Rank': round(tmv_net, 2),
        'Drift_Rank': round(drift_net, 2),
        'Trend': trend,
        'Peak_10k': format_speed(peak_speed),
        'Cur_10k': format_speed(cur_speed),
        'Squeeze': squeeze,
        'Block_Alert': block_alert,
        'Vol_Stretch': round(vol_stretch, 2),
        'LTP': round(latest_df['close'].iloc[-1], 2),
        '% Change': round(symbol_pct, 2)
    }

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# CROSS-SECTIONAL Z-SCORE SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def apply_universe_z_score(results_list):
    df = pd.DataFrame(results_list)
    if df.empty: return df
    
    drift_mean = df['Drift_Rank'].mean()
    drift_std = df['Drift_Rank'].std()
    
    if drift_std > 0:
        df['Z_Score'] = (df['Drift_Rank'] - drift_mean) / drift_std
    else:
        df['Z_Score'] = 0.0
        
    return df

def determine_setup(row):
    z = row['Z_Score']
    drift = row['Drift_Rank']
    tmv = row['TMV_Rank']
    trend = row['Trend']
    stretch = row.get('Vol_Stretch', 1.0)
    
    if row['Block_Alert'] != "":
        return "🛑 Dark Pool Print", "#e91e63", "#fff"
        
    if stretch > 4.0:
        return "🛑 Vol Climax", "#e91e63", "#fff"
    
    if trend == 'BULLISH':
        if drift < tmv - 0.5: return "📉 Fading Momentum", "#ff9800", "#000"
        elif z >= 2.0: return "🔥 Systemic Inflow", "#9c27b0", "#fff"
        elif z >= 1.0: return "🎯 Sweet Spot", "#d4af37", "#000"
        elif drift >= tmv: return "⚖️ Harmony", "#2196f3", "#fff"
        else: return "🔄 Standard", "#555555", "#fff"
            
    elif trend == 'BEARISH':
        if drift > tmv + 0.5: return "📉 Fading Momentum", "#ff9800", "#000"
        elif z <= -2.0: return "🔥 Systemic Outflow", "#9c27b0", "#fff"
        elif z <= -1.0: return "🎯 Sweet Spot", "#d4af37", "#000"
        elif drift <= tmv: return "⚖️ Harmony", "#2196f3", "#fff"
        else: return "🔄 Standard", "#555555", "#fff"
            
    return "N/A", "#555", "#fff"

def generate_colorful_html_table(df, side):
    if df.empty:
        return f"<p style='color:#aaaaaa; font-size: 13px;'>No anomalous kinetic {side.lower()} setups identified.</p>"
        
    html = """
    <table style="width:100%; border-collapse: collapse; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1e1e1e; color: #ffffff; text-align: center; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 8px rgba(0,0,0,0.5);">
        <thead>
            <tr style="background-color: #2d2d30; border-bottom: 2px solid #3e3e42;">
                <th style="padding: 12px 8px;">Symbol</th>
                <th style="padding: 12px 8px;">Kinetic Verdict</th>
                <th style="padding: 12px 8px;">Peak 10k Spd</th>
                <th style="padding: 12px 8px;">Cur 10k Spd</th>
                <th style="padding: 12px 8px;">Squeeze?</th>
                <th style="padding: 12px 8px;">LTP</th>
                <th style="padding: 12px 8px;">% Chg</th>
                <th style="padding: 12px 8px;">TMV</th>
                <th style="padding: 12px 8px;">Drift</th>
                <th style="padding: 12px 8px; color: #b388ff;">Universe Z-Score</th>
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
        
        sqz_color = "#e91e63" if row['Squeeze'] == "Yes" else "#555555"
        sqz_html = f"<span style='color:{sqz_color}; font-weight:bold;'>{row['Squeeze']}</span>"
        cur_color = "#e0e0e0" if row['Cur_10k'] == row['Peak_10k'] else "#ff9800"
        
        html += f"""
            <tr style="background-color: {bg_color}; border-bottom: 1px solid #333333;">
                <td style="padding: 10px 8px; font-weight: bold; color: #64b5f6;">{row['Symbol'].replace('NSE:', '').replace('-EQ', '')}</td>
                <td style="padding: 10px 8px;">{badge_html}</td>
                <td style="padding: 10px 8px; color: #4caf50; font-weight: bold;">{row['Peak_10k']}</td>
                <td style="padding: 10px 8px; color: {cur_color}; font-weight: bold;">{row['Cur_10k']}</td>
                <td style="padding: 10px 8px;">{sqz_html}</td>
                <td style="padding: 10px 8px; font-weight: 600;">{row['LTP']:.2f}</td>
                <td style="padding: 10px 8px; color: {pct_color}; font-weight:bold;">{pct:+.2f}%</td>
                <td style="padding: 10px 8px;">{row['TMV_Rank']:.2f}</td>
                <td style="padding: 10px 8px; font-weight:bold;">{row['Drift_Rank']:.2f}</td>
                <td style="padding: 10px 8px; font-weight:bold; color: #b388ff;">{row['Z_Score']:+.2f}σ</td>
            </tr>
        """
    html += "</tbody></table>"
    return html

def send_email_report(df_full):
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
        logger.warning("[EMAIL] Missing SMTP configurations. Dispatch halted.")
        return

    df = df_full.copy()
    df['setup_name'] = df.apply(lambda r: determine_setup(r)[0], axis=1)
    
    valid_setups = ["🔥 Systemic Inflow", "🔥 Systemic Outflow", "🎯 Sweet Spot", "⚖️ Harmony"]
    df_elite = df[df['setup_name'].isin(valid_setups)].copy()

    # Sort mathematically by Cross-Sectional Z-Score leaders
    bulls = df_elite[(df_elite['Trend'] == 'BULLISH')].sort_values(by=['Z_Score'], ascending=False).head(15)
    bears = df_elite[(df_elite['Trend'] == 'BEARISH')].sort_values(by=['Z_Score'], ascending=True).head(15)

    msg = MIMEMultipart("alternative")
    msg["From"], msg["To"] = SENDER_EMAIL, RECIPIENT_EMAIL
    msg["Subject"] = f"Elite Kinetic Quant Matrix - {datetime.now().strftime('%H:%M')}"

    html = f"""
    <html>
    <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #121212; color: #e0e0e0; padding: 20px;">
      
      <div style="text-align: center; margin-bottom: 30px;">
          <h2 style="color: #ffffff; margin-bottom: 5px;">Kinetic Quant Institutional Matrix (v12.5)</h2>
          <p style="color: #aaaaaa; margin-top: 0;"><b>Cross-Sectional Confluence Active:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
      </div>
      
      <h3 style="color: #4caf50; border-bottom: 2px solid #4caf50; padding-bottom: 5px; display: inline-block;">🚀 Top 15 Kinetic Long Accumulators</h3>
      {generate_colorful_html_table(bulls, "BULLISH")}
      
      <br><br>
      
      <h3 style="color: #f44336; border-bottom: 2px solid #f44336; padding-bottom: 5px; display: inline-block;">🔻 Top 15 Kinetic Short Distributors</h3>
      {generate_colorful_html_table(bears, "BEARISH")}
      
      <div style="margin-top: 40px; padding: 15px; background-color: #1e1e1e; border-radius: 8px; border-left: 4px solid #b388ff;">
          <h4 style="color: #ffffff; margin-top: 0;">Kinetic Multi-Timeframe System Specs:</h4>
          <ul style="color: #cccccc; line-height: 1.6;">
            <li><b>Peak 10k Spd vs Cur 10k Spd:</b> Raw metrics mapping institutional speed. Tracks execution time rather than generic bar counts.</li>
            <li><b>Universe Z-Score (σ):</b> Cross-sectional validation. Measures exactly how many standard deviations an asset's kinetic momentum is outperforming the F&O market average.</li>
            <li><b>Squeeze?:</b> Indicates structural expansion potential—triggered when historical daily compression coils open up dynamically.</li>
            <li><b>Filter Protections:</b> Any dark pool prints or block trades with low true high-low spread are flagged and neutralized instantly.</li>
          </ul>
      </div>
    </body>
    </html>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        df_full['Runtime'] = datetime.now().strftime('%H:%M:%S')
        df_full.to_csv(HISTORY_CSV, mode='a', header=not os.path.exists(HISTORY_CSV), index=False)
        with open(HISTORY_CSV, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{HISTORY_CSV}"')
        msg.attach(part)
    except: pass

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        logger.info(f"[EMAIL] Kinetic Quant Matrix successfully dispatched.")
    except Exception as e:
        logger.error(f"[EMAIL] SMTP Transmission Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════════════════════════
# MAIN SYSTEM CONTROL CORE
# ═══════════════════════════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("[LAUNCH] ASIT v12.5 - TRUE KINETIC QUANT EDITION")
    print("=" * 80)
    symbols = get_live_fno_symbols()
    if not symbols: 
        logger.error("[TERMINATE] No underlying options targets found.")
        sys.exit(1)
        
    results = []
    # Safeguarded at 2 concurrent workers to systematically bypass Fyers API rate throttling
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(scan_symbol, sym): sym for sym in symbols}
        for idx, future in enumerate(as_completed(futures), 1):
            try:
                res = future.result()
                if res: results.append(res)
                if idx % 20 == 0: logger.info(f"[SCAN] Processing Kinetic Quant Matrix: {idx}/{len(symbols)} complete...")
            except Exception as e:
                pass

    if results:
        # Step into cross-sectional verification
        df_scored = apply_universe_z_score(results)
        send_email_report(df_scored)
        print("=" * 80)
        print("[SUCCESS] Operational Phase Concluded Natively.")
        print("=" * 80)
    else:
        logger.error("[FATAL ERROR] Matrix calculation returned null configurations.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
SPATIAL MATRIX & F&O MULTI-CHANNEL 64D HYPER-TENSOR ENGINE (v16.5 QUANTITATIVE FIXED)
- Direct Canvas Labeling: Stamps clear text headers directly onto historical and live charts.
- Dual-Image Architecture: AI uses a 30-candle math blob, but emails a 32+ candle visual blob.
- Strict Visual Lockdown: Replaces masks with TM_CCOEFF_NORMED to strictly punish visual noise.
- Quantitative Win Rate: Tracks occurrences using a 15-day time-clustering cooldown filter.
- Dynamic ATR Anchoring: Canvas span scales to 2.5x of a 14-period True Range.
═══════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import time
import yaml
import sqlite3
import logging
import argparse
import smtplib
import asyncio
import aiohttp
import urllib.parse
from io import StringIO
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import requests

# =================================================================================================
# 1. ENVIRONMENT & STACK INITIALIZATION
# =================================================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "spatial_matrix_atlas.db"
DB_LOCK = asyncio.Lock()
UPSTOX_KEYS = {} 

try:
    with open("config.yml", "r") as f:
        _raw_cfg = yaml.safe_load(f)
        cfg = _raw_cfg.get("trading_engine", {})
except FileNotFoundError:
    logger.warning("config.yml not found. Using optimal production defaults.")
    cfg = {}

MACRO_WINDOW = cfg.get("macro_window_min", 30)
HIST_TRAVERSAL_LOOKBACK = cfg.get("historical_traversal_lookback", "1 year")
LIVE_LOOKBACK_DAYS = cfg.get("live_lookback_days", 30)
TRIGGER_THRESH = cfg.get("correlation", {}).get("initial_trigger_threshold", 0.80)
COMPRESSION_MAX = 0.06  
MATCH_MARGIN = 0.02 

UPSTOX_ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "")

MAX_CONCURRENT_API_CALLS = 6
API_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)
DATA_CACHE = {} 
CPU_EXECUTOR = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# =================================================================================================
# 2. ASYNC LOCAL DATA STORAGE MANAGEMENT
# =================================================================================================
async def initialize_spatial_database():
    def _init():
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA cache_size = -10000;") 
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spatial_blueprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, timeframe TEXT, direction TEXT, matrix_type TEXT,
                    match_blob BLOB, display_blob BLOB, 
                    hist_max_move_pct REAL, hist_linear_periods INTEGER,
                    detected_timestamp TEXT,
                    UNIQUE(symbol, timeframe, detected_timestamp, matrix_type)
                )
            """)
    await asyncio.get_running_loop().run_in_executor(CPU_EXECUTOR, _init)

def get_last_timestamp_from_db(symbol, timeframe):
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT MAX(detected_timestamp) FROM spatial_blueprints WHERE symbol=? AND timeframe=?", (symbol, timeframe))
        row = cur.fetchone()
        if row and row[0]:
            return pd.to_datetime(row[0]).tz_localize("Asia/Kolkata")
    return None

# =================================================================================================
# 3. CORE MULTI-CHANNEL VISUAL SPATIAL MATRIX ENGINE (LABELED)
# =================================================================================================
def generate_multichannel_spatial_matrix(p_open, p_high, p_low, p_close, volume, future_candles=0, label_text=""):
    if len(p_high) < MACRO_WINDOW: return None
        
    grid_h, grid_w = 1024, 1024 
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    actual_min, actual_max = p_low.min(), p_high.max()
    actual_span = actual_max - actual_min if actual_max > actual_min else 1.0
    
    shifted_close = np.roll(p_close, 1)
    shifted_close[0] = p_open[0] 
    tr = np.maximum(p_high - p_low, np.maximum(np.abs(p_high - shifted_close), np.abs(p_low - shifted_close)))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
    
    min_required_span = atr * 2.5 
    
    if actual_span < min_required_span:
        p_span = min_required_span
        center_price = (actual_max + actual_min) / 2.0
        p_max = center_price + (min_required_span / 2.0)
        p_min = center_price - (min_required_span / 2.0)
    else:
        p_span = actual_span
        p_max = actual_max
        p_min = actual_min
    
    v_min, v_max = volume.min(), volume.max()
    v_avg = np.mean(volume) if np.mean(volume) > 0 else 1.0
    
    typical_price = (p_high + p_low + p_close) / 3.0
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    ema = pd.Series(p_close).ewm(span=min(9, len(p_close)), adjust=False).mean().values
    
    num_candles = len(p_high)
    base_w = grid_w / num_candles
    
    if future_candles > 0:
        sep_idx = num_candles - future_candles
        sep_x = int(sep_idx * base_w)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (sep_x, 0), (grid_w, grid_h), (30, 0, 0), -1) 
        cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
        cv2.line(canvas, (sep_x, 0), (sep_x, grid_h), (255, 255, 255), 2, cv2.LINE_AA) 
    
    anchor_idx = max(1, int(num_candles * 0.8))
    ch_high = p_high[:anchor_idx].max()
    ch_low = p_low[:anchor_idx].min()
    
    ch_y = int(grid_h * (1.0 - (ch_high - p_min) / p_span))
    cl_y = int(grid_h * (1.0 - (ch_low - p_min) / p_span))
    
    if 0 <= ch_y < grid_h: cv2.line(canvas, (0, ch_y), (grid_w, ch_y), (40, 40, 40), 3, cv2.LINE_AA)
    if 0 <= cl_y < grid_h: cv2.line(canvas, (0, cl_y), (grid_w, cl_y), (40, 40, 40), 3, cv2.LINE_AA)
    
    prev_x = prev_vwap_y = prev_ema_y = None
    
    for idx in range(num_candles):
        x_center = int((idx + 0.5) * base_w)
        
        o_y = int(np.clip(grid_h * (1.0 - (p_open[idx] - p_min) / p_span), 0, grid_h - 1))
        h_y = int(np.clip(grid_h * (1.0 - (p_high[idx] - p_min) / p_span), 0, grid_h - 1))
        l_y = int(np.clip(grid_h * (1.0 - (p_low[idx] - p_min) / p_span), 0, grid_h - 1))
        c_y = int(np.clip(grid_h * (1.0 - (p_close[idx] - p_min) / p_span), 0, grid_h - 1))
        vwap_y = int(np.clip(grid_h * (1.0 - (vwap[idx] - p_min) / p_span), 0, grid_h - 1))
        ema_y = int(np.clip(grid_h * (1.0 - (ema[idx] - p_min) / p_span), 0, grid_h - 1))
        
        v_ratio = volume[idx] / v_avg
        dynamic_w = int(base_w * 0.45 * v_ratio)
        body_w = max(1, min(dynamic_w, int(base_w * 0.9))) 
        wick_w = max(3, int(base_w * 0.08)) 
            
        if prev_x is not None:
            cv2.line(canvas, (prev_x, prev_vwap_y), (x_center, vwap_y), (255, 0, 0), 5, cv2.LINE_AA)  
            cv2.line(canvas, (prev_x, prev_ema_y), (x_center, ema_y), (255, 0, 255), 5, cv2.LINE_AA)    
        prev_x, prev_vwap_y, prev_ema_y = x_center, vwap_y, ema_y

        body_len = max(1, abs(p_close[idx] - p_open[idx]))
        top_price, bot_price = max(p_close[idx], p_open[idx]), min(p_close[idx], p_open[idx])
        
        upper_wick_len = p_high[idx] - top_price
        lower_wick_len = bot_price - p_low[idx]
        
        is_bullish = p_close[idx] >= p_open[idx]
        candle_color = (0, 200, 0) if is_bullish else (0, 0, 200) 
        
        up_wick_color = (0, 255, 255) if (upper_wick_len > body_len * 2) else candle_color
        dn_wick_color = (0, 255, 255) if (lower_wick_len > body_len * 2) else candle_color
        
        top_y, bot_y = min(o_y, c_y), max(o_y, c_y)
        
        cv2.line(canvas, (x_center, h_y), (x_center, top_y), up_wick_color, wick_w, cv2.LINE_AA)
        cv2.line(canvas, (x_center, bot_y), (x_center, l_y), dn_wick_color, wick_w, cv2.LINE_AA)
        
        if top_y == bot_y: bot_y += 1 
        cv2.rectangle(canvas, (x_center - body_w, top_y), (x_center + body_w, bot_y), candle_color, -1)

        v_h = int(np.clip((volume[idx] - v_min) / (v_max - v_min) * 200, 1, 200)) if v_max > v_min else 1
        cv2.rectangle(canvas, (x_center - body_w, grid_h - v_h), (x_center + body_w, grid_h), (0, 100, 0), -1)

    if label_text:
        cv2.rectangle(canvas, (0, 0), (grid_w, 75), (20, 20, 20), -1)
        cv2.line(canvas, (0, 75), (grid_w, 75), (100, 100, 100), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 255) if "HISTORICAL" in label_text else (0, 255, 0)
        cv2.putText(canvas, label_text, (35, 48), font, 1.1, color, 3, cv2.LINE_AA)

    return canvas

# =================================================================================================
# 4. ASYNC HISTORICAL PROFILER 
# =================================================================================================
def parse_traversal_window(window_str):
    clean = window_str.lower().strip()
    digits = int(re.search(r'\d+', clean).group()) if re.search(r'\d+', clean) else 365
    if 'year' in clean: return digits * 365
    if 'month' in clean: return digits * 30
    if 'week' in clean: return digits * 7
    if 'day' in clean: return digits
    return 365

async def fetch_historical_raw_data_async(session, symbol, resolution, total_days_back, target_end_dt=None, context="HIST"):
    end_date = target_end_dt if target_end_dt else pd.Timestamp.now(tz="Asia/Kolkata")
    cache_key = f"{symbol}_{resolution}_{total_days_back}_{end_date.strftime('%Y-%m-%d_%H')}"
    if cache_key in DATA_CACHE: return DATA_CACHE[cache_key]
    
    instrument_key = UPSTOX_KEYS.get(symbol)
    if not instrument_key: return None

    encoded_key = urllib.parse.quote(instrument_key)
    all_candles = []
    days_fetched = 0
    chunk_size = 365
    
    headers = {
        'Accept': 'application/json',
        'Api-Version': '2.0',
        'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'
    }
    
    while days_fetched < total_days_back:
        start_date = end_date - timedelta(days=chunk_size)
        str_to = end_date.strftime("%Y-%m-%d")
        str_from = start_date.strftime("%Y-%m-%d")
        url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/{resolution}/{str_to}/{str_from}"
        
        success = False
        for attempt in range(4):
            async with API_SEMAPHORE: 
                try:
                    async with session.get(url, headers=headers) as response:
                        res_text = await response.text()
                        try: res = await response.json()
                        except Exception: res = {} 
                        
                        if response.status != 200 or res.get('status') == 'error':
                            if response.status in [429, 403]:
                                await asyncio.sleep(2.5 * (attempt + 1))
                                continue
                            err_msg = res.get('errors', [{}])[0].get('message', 'No detailed message')
                            logger.error(f"❌ UPSTOX BLOCK [{symbol}]: HTTP {response.status} | {err_msg} | Raw: {res_text[:60]}")
                            break 
                                
                        if 'data' in res and 'candles' in res['data'] and res['data']['candles']:
                            all_candles.extend(res['data']['candles'])
                            success = True
                            break
                        else: break 
                            
                except Exception as e:
                    logger.error(f"💥 NETWORK ERROR [{symbol}]: {e}")
                    await asyncio.sleep(1)
                
        if not success: break
        end_date = start_date - timedelta(days=1)
        days_fetched += chunk_size

    if not all_candles: 
        if context == "LIVE": logger.warning(f"[{symbol}-{resolution}] Scan skipped: Zero candles found.")
        return None
    
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp'])
    
    DATA_CACHE[cache_key] = df 
    return df

def _cpu_process_historical_data(symbol, res, df):
    if len(df) < (MACRO_WINDOW + 20): return []
    
    opens, closes, highs, lows, volumes = df['open'].values, df['close'].values, df['high'].values, df['low'].values, df['volume'].values
    timestamps = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').values
    db_records = []
    
    for i in range(MACRO_WINDOW, len(df) - 10):
        o_slice = opens[i-MACRO_WINDOW:i]
        c_slice = closes[i-MACRO_WINDOW:i]
        h_slice = highs[i-MACRO_WINDOW:i]
        l_slice = lows[i-MACRO_WINDOW:i]
        v_slice = volumes[i-MACRO_WINDOW:i]
        
        base_ltp = c_slice[-1]
        if base_ltp == 0 or ((h_slice.max() - l_slice.min()) / base_ltp) >= COMPRESSION_MAX: 
            continue
            
        f_close, f_high, f_low = closes[i:i+10], highs[i:i+10], lows[i:i+10]
        if len(f_close) < 2: continue
            
        trigger_price = f_close[0]
        
        local_max_high = h_slice[-20:].max()
        local_min_low = l_slice[-20:].min()
        
        direction = "UP" if trigger_price > local_max_high else ("DOWN" if trigger_price < local_min_low else None)
        if not direction: continue
            
        linear_periods = 0
        if direction == "UP":
            for j in range(len(f_close)):
                prev_c = base_ltp if j == 0 else f_close[j-1]
                if f_close[j] > prev_c and f_low[j] >= (prev_c * 0.988):
                    linear_periods += 1
                else: break 
            if linear_periods < 2: continue
            max_move_pct = ((f_high[:linear_periods].max() - base_ltp) / base_ltp) * 100.0
            
        else: 
            for j in range(len(f_close)):
                prev_c = base_ltp if j == 0 else f_close[j-1]
                if f_close[j] < prev_c and f_high[j] <= (prev_c * 1.012):
                    linear_periods += 1
                else: break
            if linear_periods < 2: continue
            max_move_pct = ((base_ltp - f_low[:linear_periods].min()) / base_ltp) * 100.0
                
        matrix_type = "SUCCESS" if max_move_pct >= 4.0 else "TRAP"
        
        match_mat = generate_multichannel_spatial_matrix(o_slice, h_slice, l_slice, c_slice, v_slice, future_candles=0, label_text="")
        if match_mat is None: continue
        
        disp_o = opens[i-MACRO_WINDOW : i+linear_periods]
        disp_h = highs[i-MACRO_WINDOW : i+linear_periods]
        disp_l = lows[i-MACRO_WINDOW : i+linear_periods]
        disp_c = closes[i-MACRO_WINDOW : i+linear_periods]
        disp_v = volumes[i-MACRO_WINDOW : i+linear_periods]
        display_mat = generate_multichannel_spatial_matrix(disp_o, disp_h, disp_l, disp_c, disp_v, future_candles=linear_periods, label_text="HISTORICAL BLUEPRINT (POST-BREAKOUT)")
        if display_mat is None: continue

        success_m, enc_match = cv2.imencode('.webp', match_mat, [cv2.IMWRITE_WEBP_QUALITY, 85])
        success_d, enc_disp = cv2.imencode('.webp', display_mat, [cv2.IMWRITE_WEBP_QUALITY, 85])
        if not success_m or not success_d: continue
        
        db_records.append((symbol, res, direction, matrix_type, enc_match.tobytes(), enc_disp.tobytes(), float(max_move_pct), int(linear_periods), timestamps[i-1]))
        
    return db_records

# =================================================================================================
# 5. LIVE EVALUATOR & SCANNER (QUANTITATIVE TRACKING)
# =================================================================================================
def _cpu_evaluate_live_market(symbol, live_canvas, res):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        blueprints = conn.execute("SELECT * FROM spatial_blueprints WHERE symbol=? AND timeframe=?", (symbol, res)).fetchall()
        
    if not blueprints: return None
        
    live_gray = cv2.cvtColor(live_canvas, cv2.COLOR_BGR2GRAY)
    live_gray_blur = cv2.GaussianBlur(live_gray, (5, 5), 0)
    
    valid_matches = []
        
    for bp in blueprints:
        try:
            bp_img = cv2.imdecode(np.frombuffer(bp['match_blob'], dtype=np.uint8), cv2.IMREAD_COLOR)
            bp_gray = cv2.cvtColor(bp_img, cv2.COLOR_BGR2GRAY)
            bp_gray_blur = cv2.GaussianBlur(bp_gray, (5, 5), 0)
            
            crop_margin = 64
            bp_core = bp_gray_blur[crop_margin:-crop_margin, crop_margin:-crop_margin]
            
            match_res = cv2.matchTemplate(live_gray_blur, bp_core, cv2.TM_CCOEFF_NORMED)
            _, shape_score, _, max_loc = cv2.minMaxLoc(match_res)
            shape_score = float(max(0.0, min(1.0, shape_score)))
            
            h, w = bp_core.shape
            live_color_crop = live_canvas[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w]
            bp_color_core = bp_img[crop_margin:-crop_margin, crop_margin:-crop_margin]
            
            color_res = cv2.matchTemplate(live_color_crop, bp_color_core, cv2.TM_CCOEFF_NORMED)
            _, color_score, _, _ = cv2.minMaxLoc(color_res)
            color_score = float(max(0.0, min(1.0, color_score)))
            
            final_score = (shape_score * 0.80) + (color_score * 0.20)
            
            if final_score >= TRIGGER_THRESH:
                valid_matches.append({
                    'score': final_score,
                    'type': bp['matrix_type'],
                    'timestamp': pd.to_datetime(bp['detected_timestamp']),
                    'bp': bp
                })
        except Exception: continue

    if not valid_matches: return None

    # Sort chronologically to filter overlaps accurately
    valid_matches.sort(key=lambda x: x['timestamp'])
    
    occurrence_count = 0
    success_count = 0
    last_counted_ts = None
    
    best_success_score = 0.0
    best_trap_score = 0.0
    matched_blueprint_row = None
    matched_display_blob_cache = None

    for m in valid_matches:
        ts = m['timestamp']
        if ts.tzinfo is not None: ts = ts.tz_localize(None)

        # TIME CLUSTERING: Minimum 15 days between separate occurrences
        if last_counted_ts is None or (ts - last_counted_ts).days > 15:
            occurrence_count += 1
            if m['type'] == 'SUCCESS':
                success_count += 1
            last_counted_ts = ts
            
        if m['type'] == 'SUCCESS':
            if m['score'] > best_success_score:
                best_success_score = m['score']
                matched_blueprint_row = m['bp']
                matched_display_blob_cache = m['bp']['display_blob']
        else:
            if m['score'] > best_trap_score:
                best_trap_score = m['score']

    if best_success_score < TRIGGER_THRESH: return None
    
    if best_success_score <= (best_trap_score + MATCH_MARGIN):
        logger.info(f"   -> FILTERED: {symbol} Trap ({best_trap_score:.3f}) neutralized Success ({best_success_score:.3f}).")
        return None

    # Calculate True Historical Win Rate
    historical_win_rate = (success_count / occurrence_count * 100.0) if occurrence_count > 0 else 0.0

    logger.info(f"🚀 [{symbol}-{res}] TARGET LOCKED! (Score: {best_success_score:.3f} | Win Rate: {historical_win_rate:.1f}%)")
    
    disp_img = cv2.imdecode(np.frombuffer(matched_display_blob_cache, dtype=np.uint8), cv2.IMREAD_COLOR)
    
    return {
        'Symbol': symbol,
        'Direction': matched_blueprint_row['direction'],
        'Match_Score': best_success_score,
        'Total_Occurrences': occurrence_count,
        'Success_Rate': historical_win_rate,
        'Hist_Max_Move_Pct': matched_blueprint_row['hist_max_move_pct'],
        'Hist_Linear_Periods': matched_blueprint_row['hist_linear_periods'],
        'Timeframe': matched_blueprint_row['timeframe'],
        'Live_Image_Bytes': cv2.imencode('.png', live_canvas)[1].tobytes(),
        'Blueprint_Image_Bytes': cv2.imencode('.png', disp_img)[1].tobytes()
    }

async def process_live_scanning_sequence_async(session, symbol, target_dt):
    resolutions = ['day']
    loop = asyncio.get_running_loop()
    
    for res in resolutions:
        df = await fetch_historical_raw_data_async(session, symbol, res, LIVE_LOOKBACK_DAYS, target_end_dt=target_dt, context="LIVE")
        if df is None or len(df) < MACRO_WINDOW + 20: continue
            
        last_known_time = get_last_timestamp_from_db(symbol, res)
        historical_df = df if last_known_time is None else df[df['timestamp'] > last_known_time]
        
        if len(historical_df) > (MACRO_WINDOW + 20):
            records = await loop.run_in_executor(CPU_EXECUTOR, _cpu_process_historical_data, symbol, res, historical_df)
            if records:
                async with DB_LOCK:
                    def _batch_insert():
                        with sqlite3.connect(DB_PATH) as conn:
                            conn.executemany("""
                                INSERT OR IGNORE INTO spatial_blueprints 
                                (symbol, timeframe, direction, matrix_type, match_blob, display_blob, hist_max_move_pct, hist_linear_periods, detected_timestamp)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, records)
                    await loop.run_in_executor(CPU_EXECUTOR, _batch_insert)
        
        r_slice = df.tail(MACRO_WINDOW)
        ltp = float(r_slice['close'].iloc[-1])
        
        live_canvas = await loop.run_in_executor(CPU_EXECUTOR, generate_multichannel_spatial_matrix, 
            r_slice['open'].values, r_slice['high'].values, r_slice['low'].values, r_slice['close'].values, r_slice['volume'].values, 0, "LIVE MARKET SCAN (CURRENT)")
            
        if live_canvas is None: continue
            
        match_result = await loop.run_in_executor(CPU_EXECUTOR, _cpu_evaluate_live_market, symbol, live_canvas, res)
        
        if match_result:
            match_result['LTP'] = ltp
            p_initial = float(r_slice['close'].iloc[-5]) if MACRO_WINDOW >= 5 else ltp
            achieved = max(0.0, abs(ltp - p_initial) / p_initial * 100.0) if p_initial > 0 else 0.0
                
            match_result['Achieved_Pct'] = round(achieved, 2)
            match_result['Pending_Pct'] = round(max(0.0, match_result['Hist_Max_Move_Pct'] - achieved), 2)
            return match_result
            
    return None

# =================================================================================================
# 6. EMAIL TRANSMISSION & MASTER PIPELINE (WITH WIN RATES)
# =================================================================================================
def dispatch_predictive_analysis_report(df_matrix, target_dt):
    if not SENDER_EMAIL or not RECIPIENT_EMAIL: return
        
    scan_time_str = target_dt.strftime('%d %b %Y, %I:%M %p')
    msg = MIMEMultipart('related')
    msg['Subject'] = f"🎯 SUCCESS MATRIX ALERT: F&O Spatial Breakout | {scan_time_str}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    
    html_rows = ""
    image_attachments = []
    
    for idx, row in df_matrix.iterrows():
        sym, dir_col = row['Symbol'], "#1b5e20" if row['Direction'] == 'UP' else "#b71c1c"
        dir_lbl = "📈 UPWARD" if row['Direction'] == 'UP' else "📉 DOWNWARD"
        live_cid, bp_cid = f"live_{sym}_{idx}", f"bp_{sym}_{idx}"
        
        wr_color = "#1b5e20" if row['Success_Rate'] >= 70.0 else ("#f57f17" if row['Success_Rate'] >= 50.0 else "#b71c1c")
        
        html_rows += f"""
        <tr style='border-bottom: 1px solid #ddd;'>
            <td style='padding: 12px; font-weight: bold; color: #1a73e8;'>{sym}</td>
            <td style='padding: 12px; font-weight: bold; color: {dir_col};'>{dir_lbl}</td>
            <td style='padding: 12px;'><b>{row['Match_Score']*100:.1f}%</b></td>
            <td style='padding: 12px; text-align: center; background-color: #f8f9fa;'>
                <b style='color: {wr_color}; font-size: 16px;'>{row['Success_Rate']:.1f}%</b><br/>
                <span style='font-size: 11px; color: #666;'>({row['Total_Occurrences']} Occurrences)</span>
            </td>
            <td style='padding: 12px;'>₹{row['LTP']:.2f}</td>
            <td style='padding: 12px; color: #2e7d32;'><b>{row['Hist_Max_Move_Pct']:.2f}%</b> ({row['Hist_Linear_Periods']} pd)</td>
            <td style='padding: 12px; color: #e65100;'><b>{row['Achieved_Pct']:.2f}%</b></td>
            <td style='padding: 12px; color: #1565c0; font-weight: bold;'>{row['Pending_Pct']:.2f}%</td>
        </tr>
        <tr>
            <td colspan='8' style='padding: 15px; text-align: center;'>
                <img src="cid:{live_cid}" width="400" height="400" style='border: 1px solid #ccc; margin-right: 10px;' />
                <img src="cid:{bp_cid}" width="400" height="400" style='border: 1px solid #ccc;' />
            </td>
        </tr>
        """
        image_attachments.extend([(live_cid, row['Live_Image_Bytes']), (bp_cid, row['Blueprint_Image_Bytes'])])

    html_body = f"<html><body style='font-family: Arial; padding: 20px;'><h2 style='color: #1a237e;'>🎯 HYPER-TENSOR TARGET DETECTOR</h2><table style='width: 100%; border-collapse: collapse;'><thead><tr style='background-color: #283593; color: white;'><th style='padding: 12px;'>Asset</th><th style='padding: 12px;'>Type</th><th style='padding: 12px;'>Match Score</th><th style='padding: 12px;'>Win Rate</th><th style='padding: 12px;'>LTP</th><th style='padding: 12px;'>Target</th><th style='padding: 12px;'>Achieved</th><th style='padding: 12px;'>Pending</th></tr></thead><tbody>{html_rows}</tbody></table></body></html>"
    msg.attach(MIMEText(html_body, "html"))
    for cid, img_b in image_attachments:
        img_part = MIMEImage(img_b, name=f"{cid}.png")
        img_part.add_header('Content-ID', f"<{cid}>")
        msg.attach(img_part)
        
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logger.info(f"📬 Alert dispatched (via 587 TLS) for {len(df_matrix)} assets.")
    except Exception as e: 
        logger.warning(f"TLS Email failed ({e}). Attempting SSL fallback...")
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
            logger.info(f"📬 Alert dispatched (via 465 SSL) for {len(df_matrix)} assets.")
        except Exception as ssl_e:
            logger.error(f"❌ EMAIL CRASH: Both TLS and SSL protocols failed. {ssl_e}")

def fetch_fo_universe():
    global UPSTOX_KEYS
    logger.info("Fetching F&O Base List & Mapping to Upstox ISINs...")
    try:
        res_fyers = requests.get("https://public.fyers.in/sym_details/NSE_FO.csv", timeout=15)
        df_fyers = pd.read_csv(StringIO(res_fyers.text), header=None)
        sym_col = next((col for col in df_fyers.columns if df_fyers[col].astype(str).str.startswith('NSE:').any()), None)
        
        if sym_col is None: return []
        
        base_symbols = {re.search(r'NSE:([A-Z&\-]+)\d+', s).group(1) for s in df_fyers[sym_col].astype(str) if re.search(r'NSE:([A-Z&\-]+)\d+', s)}
        fo_names = base_symbols - {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'}
        logger.info(f"Extracted {len(fo_names)} target F&O symbols. Downloading Upstox Master...")

        df_upstox = pd.read_csv("https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz")
        eq_df = df_upstox[df_upstox['instrument_key'].astype(str).str.startswith('NSE_EQ|')]
        
        for _, row in eq_df.iterrows():
            ts = str(row['tradingsymbol']).strip()
            if ts in fo_names: UPSTOX_KEYS[ts] = row['instrument_key']

        valid_symbols = sorted(list(UPSTOX_KEYS.keys()))
        logger.info(f"✅ Successfully mapped {len(valid_symbols)} F&O symbols to Upstox Instrument Keys.")
        return valid_symbols
        
    except Exception as e:
        logger.error(f"Failed to map Upstox F&O universe: {e}")
        return []

async def execute_engine_pass_async(session, target_dt, symbols):
    logger.info(f"⚡ Booting sweep for target window: {target_dt.strftime('%H:%M:%S')}")
    
    tasks = [process_live_scanning_sequence_async(session, sym, target_dt) for sym in symbols]
    results = await asyncio.gather(*tasks)
    
    live_signals = [res for res in results if res]
    
    if live_signals:
        df_matrix = pd.DataFrame(live_signals).sort_values(
            by=['Hist_Linear_Periods', 'Match_Score'], 
            ascending=[True, False]
        )
        dispatch_predictive_analysis_report(df_matrix, target_dt)
    else:
        logger.info(f"🔍 Scan complete: Zero setups met strict Success criteria.")

async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="")
    parser.add_argument("--from_time", default="")
    parser.add_argument("--to_time", default="")
    parser.add_argument("--interval", default="60")
    parser.add_argument("--seed_history", action="store_true", help="Force rebuild 1-year history")
    args = parser.parse_args()
    
    if not UPSTOX_ACCESS_TOKEN:
        logger.error("UPSTOX_ACCESS_TOKEN environment variable not set. Exiting script.")
        return

    if args.seed_history and os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        logger.info("🗑️ --seed_history passed. Wiped DB for fresh build.")
        
    await initialize_spatial_database()
    symbols = fetch_fo_universe()
    if not symbols: return
    
    async with aiohttp.ClientSession() as global_session:
        if args.seed_history:
            logger.info("⚙️ Initiating deep historical profiling...")
            loop = asyncio.get_running_loop()
            tasks = []
            for sym in symbols:
                for res in ['day']: 
                    df = await fetch_historical_raw_data_async(global_session, sym, res, parse_traversal_window(HIST_TRAVERSAL_LOOKBACK), context="HIST")
                    if df is not None:
                        tasks.append(loop.run_in_executor(CPU_EXECUTOR, _cpu_process_historical_data, sym, res, df))
            
            all_records = await asyncio.gather(*tasks)
            flat_records = [item for sublist in all_records for item in sublist if item]
            if flat_records:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.executemany("INSERT OR IGNORE INTO spatial_blueprints (symbol, timeframe, direction, matrix_type, match_blob, display_blob, hist_max_move_pct, hist_linear_periods, detected_timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", flat_records)
            
            with sqlite3.connect(DB_PATH) as conn:
                count = conn.execute("SELECT COUNT(*) FROM spatial_blueprints").fetchone()[0]
            logger.info(f"✅ Deep database generation finalized. Indexed {count} institutional blueprints.")
            
        if args.date and args.from_time and args.to_time:
            start_dt = pd.to_datetime(f"{args.date} {args.from_time}").tz_localize("Asia/Kolkata")
            end_dt = pd.to_datetime(f"{args.date} {args.to_time}").tz_localize("Asia/Kolkata")
            current_dt = start_dt
            while current_dt <= end_dt:
                await execute_engine_pass_async(global_session, current_dt, symbols)
                current_dt += timedelta(minutes=int(args.interval))
        elif args.date:
            await execute_engine_pass_async(global_session, pd.to_datetime(args.date).tz_localize("Asia/Kolkata"), symbols)
        else:
            await execute_engine_pass_async(global_session, pd.Timestamp.now(tz="Asia/Kolkata"), symbols)

if __name__ == "__main__":
    asyncio.run(async_main())

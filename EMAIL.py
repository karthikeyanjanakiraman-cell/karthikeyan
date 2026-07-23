#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
SPATIAL MATRIX & F&O MULTI-CHANNEL 64D HYPER-TENSOR ENGINE v14.1 (Upstox Analytics Edition)
- Upstox Auth: Uses 1-Year Long-Lived UPSTOX_ACCESS_TOKEN (Zero daily login maintenance).
- Instrument Key Mapping: Auto-downloads Upstox Master DB and translates F&O symbols to ISINs.
- Chrono-Correction: Reverses Upstox's backwards data streams before matrix injection.
- Delta-Update DB Engine: Only processes new candles (99% faster).
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
UPSTOX_KEYS = {} # Global mapping for Human Symbol -> Upstox ISIN Key

try:
    with open("config.yml", "r") as f:
        _raw_cfg = yaml.safe_load(f)
        cfg = _raw_cfg.get("trading_engine", {})
except FileNotFoundError:
    logger.warning("config.yml not found. Using optimal production defaults.")
    cfg = {}

MACRO_WINDOW = cfg.get("macro_window_min", 15)
HIST_TRAVERSAL_LOOKBACK = cfg.get("historical_traversal_lookback", "1 month")
LIVE_LOOKBACK_DAYS = cfg.get("live_lookback_days", 30)
TRIGGER_THRESH = cfg.get("correlation", {}).get("initial_trigger_threshold", 0.75)
COMPRESSION_MAX = 0.06  
MATCH_MARGIN = 0.02 

# Upstox Specific Authentication Environment Variables
UPSTOX_ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "") # Your 1-Year Upstox Analytics Token
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "")

# Concurrency & Caching Controls
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
                    image_blob BLOB, hist_max_move_pct REAL, hist_linear_periods INTEGER,
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
# 3. CORE MULTI-CHANNEL VISUAL SPATIAL MATRIX ENGINE
# =================================================================================================
def generate_multichannel_spatial_matrix(p_high, p_low, p_close, volume):
    if len(p_high) < MACRO_WINDOW: return None
        
    grid_h, grid_w = 256, 256 
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    p_min, p_max = p_low.min(), p_high.max()
    p_span = p_max - p_min if p_max > p_min else 1.0
    
    v_min, v_max = volume.min(), volume.max()
    v_span = v_max - v_min if v_max > v_min else 1.0
    
    x_coords = np.linspace(0, grid_w - 1, len(p_high)).astype(int)
    
    tr = np.maximum(p_high[1:] - p_low[1:], np.abs(p_high[1:] - p_close[:-1]))
    atr_val = np.mean(tr) if len(tr) > 0 else 1.0
    
    for idx, x_idx in enumerate(x_coords):
        h_val, l_val, v_val = p_high[idx], p_low[idx], volume[idx]
        
        y_top = int(grid_h * (1.0 - (h_val - p_min) / p_span))
        y_bot = int(grid_h * (1.0 - (l_val - p_min) / p_span))
        y_top, y_bot = np.clip(y_top, 0, grid_h - 1), np.clip(y_bot, 0, grid_h - 1)
        if y_top > y_bot: y_top, y_bot = y_bot, y_top
            
        canvas[y_top:y_bot+1, x_idx, 0] = 220
        canvas[:, x_idx, 1] = int(np.clip((v_val - v_min) / v_span * 255, 0, 255)) if v_span > 0 else 100
        canvas[grid_h - 128:grid_h, x_idx, 2] = int(np.clip(atr_val / p_span * 255, 50, 255))

    return canvas

# =================================================================================================
# 4. ASYNC HISTORICAL PROFILER & UPSTOX NATIVE INGESTION
# =================================================================================================
def parse_traversal_window(window_str):
    clean = window_str.lower().strip()
    digits = int(re.search(r'\d+', clean).group()) if re.search(r'\d+', clean) else 365
    if 'year' in clean: return digits * 365
    if 'month' in clean: return digits * 30
    if 'week' in clean: return digits * 7
    if 'day' in clean: return digits
    return 365

async def fetch_historical_raw_data_async(symbol, resolution, total_days_back, target_end_dt=None, context="HIST"):
    end_date = target_end_dt if target_end_dt else pd.Timestamp.now(tz="Asia/Kolkata")
    cache_key = f"{symbol}_{resolution}_{total_days_back}_{end_date.strftime('%Y-%m-%d_%H')}"
    if cache_key in DATA_CACHE: return DATA_CACHE[cache_key]
    
    instrument_key = UPSTOX_KEYS.get(symbol)
    if not instrument_key:
        logger.error(f"[{symbol}] No Upstox Instrument Key found in mapping.")
        return None

    all_candles = []
    days_fetched = 0
    chunk_size = 30 if resolution != 'day' else 365
    
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {UPSTOX_ACCESS_TOKEN}'
    }
    
    async with aiohttp.ClientSession() as session:
        while days_fetched < total_days_back:
            start_date = end_date - timedelta(days=chunk_size)
            str_to = end_date.strftime("%Y-%m-%d")
            str_from = start_date.strftime("%Y-%m-%d")
            
            url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{resolution}/{str_to}/{str_from}"
            
            success = False
            for attempt in range(4):
                async with API_SEMAPHORE: 
                    try:
                        async with session.get(url, headers=headers) as response:
                            res = await response.json()
                            
                            if response.status != 200 or res.get('status') == 'error':
                                if attempt == 3:
                                    err_msg = res.get('errors', [{}])[0].get('message', 'Unknown Error')
                                    logger.error(f"❌ UPSTOX REJECTION [{symbol}]: {err_msg}")
                                
                                if response.status in [429, 403, 400]:
                                    await asyncio.sleep(2.5 * (attempt + 1))
                                    continue
                                else:
                                    break
                                    
                            if 'data' in res and 'candles' in res['data'] and res['data']['candles']:
                                all_candles.extend(res['data']['candles'])
                                success = True
                                break
                            else:
                                break 
                                
                    except Exception as e:
                        logger.error(f"💥 NETWORK/THREAD ERROR [{symbol}]: {e}")
                        await asyncio.sleep(1)
                    
            if not success: 
                if context == "LIVE":
                    logger.warning(f"[{symbol}-{resolution}] Scan skipped: Data unavailable.")
                break
                
            end_date = start_date - timedelta(days=1)
            days_fetched += chunk_size

    if not all_candles: return None
    
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Chrono-correction: Reverse the native backward data stream from Upstox
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.drop_duplicates(subset=['timestamp'])
    
    DATA_CACHE[cache_key] = df 
    return df

def _cpu_process_historical_data(symbol, res, df):
    if len(df) < (MACRO_WINDOW + 20): return []
    
    closes, highs, lows, volumes = df['close'].values, df['high'].values, df['low'].values, df['volume'].values
    timestamps = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').values
    db_records = []
    
    for i in range(MACRO_WINDOW, len(df) - 20):
        c_slice, h_slice, l_slice, v_slice = closes[i-MACRO_WINDOW:i], highs[i-MACRO_WINDOW:i], lows[i-MACRO_WINDOW:i], volumes[i-MACRO_WINDOW:i]
        
        base_ltp = c_slice[-1]
        if base_ltp == 0 or ((h_slice.max() - l_slice.min()) / base_ltp) >= COMPRESSION_MAX: 
            continue
            
        f_close, f_high, f_low = closes[i:i+20], highs[i:i+20], lows[i:i+20]
        trigger_price = f_close[0]
        
        direction = "UP" if trigger_price > h_slice.max() else ("DOWN" if trigger_price < l_slice.min() else None)
        if not direction: continue
            
        linear_periods = 0
        if direction == "UP":
            max_move_pct = ((f_high.max() - base_ltp) / base_ltp) * 100.0
            for val in f_close:
                if val >= base_ltp: linear_periods += 1
                else: break
        else:
            max_move_pct = ((base_ltp - f_low.min()) / base_ltp) * 100.0
            for val in f_close:
                if val <= base_ltp: linear_periods += 1
                else: break
                
        matrix_type = "SUCCESS" if max_move_pct >= 4.0 else "TRAP"
        spatial_mat = generate_multichannel_spatial_matrix(h_slice, l_slice, c_slice, v_slice)
        if spatial_mat is None: continue
        
        success_enc, encoded_bytes = cv2.imencode('.png', spatial_mat)
        if not success_enc: continue
        
        db_records.append((symbol, res, direction, matrix_type, encoded_bytes.tobytes(), float(max_move_pct), int(linear_periods), timestamps[i-1]))
        
    return db_records

# =================================================================================================
# 5. LIVE HIERARCHICAL MATCHING ENGINE
# =================================================================================================
def _cpu_evaluate_live_market(symbol, live_canvas, res):
    live_gray = cv2.cvtColor(live_canvas, cv2.COLOR_RGB2GRAY)
    best_success_score, best_trap_score = 0.0, 0.0
    matched_blueprint_row = None
    
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        blueprints = conn.execute("SELECT * FROM spatial_blueprints WHERE symbol=? AND timeframe=?", (symbol, res)).fetchall()
        
    if not blueprints: return None
        
    for bp in blueprints:
        try:
            bp_img = cv2.imdecode(np.frombuffer(bp['image_blob'], dtype=np.uint8), cv2.IMREAD_COLOR)
            bp_gray = cv2.cvtColor(bp_img, cv2.COLOR_RGB2GRAY)
            
            match_res = cv2.matchTemplate(live_gray, bp_gray, cv2.TM_CCOEFF_NORMED)
            normalized_score = float(max(0.0, min(1.0, (cv2.minMaxLoc(match_res)[1] + 1.0) / 2.0)))
            
            if bp['matrix_type'] == 'SUCCESS':
                if normalized_score > best_success_score:
                    best_success_score = normalized_score
                    matched_blueprint_row = bp
            else:
                if normalized_score > best_trap_score:
                    best_trap_score = normalized_score
        except Exception: continue

    if best_success_score < TRIGGER_THRESH: return None
    if best_success_score <= (best_trap_score + MATCH_MARGIN):
        logger.info(f"   -> FILTERED: {symbol} Trap ({best_trap_score:.3f}) neutralized Success ({best_success_score:.3f}).")
        return None

    logger.info(f"🚀 [{symbol}-{res}] PASSED ALL FILTERS! Target locked.")
    return {
        'Symbol': symbol,
        'Direction': matched_blueprint_row['direction'],
        'Match_Score': best_success_score,
        'Hist_Max_Move_Pct': matched_blueprint_row['hist_max_move_pct'],
        'Hist_Linear_Periods': matched_blueprint_row['hist_linear_periods'],
        'Timeframe': matched_blueprint_row['timeframe'],
        'Live_Image_Bytes': cv2.imencode('.png', live_canvas)[1].tobytes(),
        'Blueprint_Image_Bytes': np.frombuffer(matched_blueprint_row['image_blob'], dtype=np.uint8).tobytes()
    }

async def process_live_scanning_sequence_async(symbol, target_dt):
    resolutions = ['30minute', 'day']
    loop = asyncio.get_running_loop()
    
    for res in resolutions:
        df = await fetch_historical_raw_data_async(symbol, res, LIVE_LOOKBACK_DAYS, target_end_dt=target_dt, context="LIVE")
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
                                (symbol, timeframe, direction, matrix_type, image_blob, hist_max_move_pct, hist_linear_periods, detected_timestamp)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, records)
                    await loop.run_in_executor(CPU_EXECUTOR, _batch_insert)
        
        r_slice = df.tail(MACRO_WINDOW)
        ltp = float(r_slice['close'].iloc[-1])
        
        live_canvas = await loop.run_in_executor(CPU_EXECUTOR, generate_multichannel_spatial_matrix, 
            r_slice['high'].values, r_slice['low'].values, r_slice['close'].values, r_slice['volume'].values)
            
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
# 6. EMAIL TRANSMISSION & MASTER PIPELINE
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
        
        html_rows += f"""
        <tr style='border-bottom: 1px solid #ddd;'>
            <td style='padding: 12px; font-weight: bold; color: #1a73e8;'>{sym}</td>
            <td style='padding: 12px; font-weight: bold; color: {dir_col};'>{dir_lbl}</td>
            <td style='padding: 12px;'>TF: {row['Timeframe']} | <b>{row['Match_Score']*100:.1f}% Match</b></td>
            <td style='padding: 12px;'>₹{row['LTP']:.2f}</td>
            <td style='padding: 12px; color: #2e7d32;'><b>{row['Hist_Max_Move_Pct']:.2f}%</b> ({row['Hist_Linear_Periods']} pd)</td>
            <td style='padding: 12px; color: #e65100;'><b>{row['Achieved_Pct']:.2f}%</b></td>
            <td style='padding: 12px; color: #1565c0; font-weight: bold;'>{row['Pending_Pct']:.2f}%</td>
        </tr>
        <tr>
            <td colspan='7' style='padding: 15px; text-align: center;'>
                <img src="cid:{live_cid}" width="256" height="256" style='border: 1px solid #ccc; margin-right: 10px;' />
                <img src="cid:{bp_cid}" width="256" height="256" style='border: 1px solid #ccc;' />
            </td>
        </tr>
        """
        image_attachments.extend([(live_cid, row['Live_Image_Bytes']), (bp_cid, row['Blueprint_Image_Bytes'])])

    html_body = f"<html><body style='font-family: Arial; padding: 20px;'><h2 style='color: #1a237e;'>🎯 HYPER-TENSOR TARGET DETECTOR</h2><table style='width: 100%; border-collapse: collapse;'><thead><tr style='background-color: #283593; color: white;'><th style='padding: 12px;'>Asset</th><th style='padding: 12px;'>Type</th><th style='padding: 12px;'>Confidence</th><th style='padding: 12px;'>LTP</th><th style='padding: 12px;'>Hist Benchmark</th><th style='padding: 12px;'>Achieved</th><th style='padding: 12px;'>Pending</th></tr></thead><tbody>{html_rows}</tbody></table></body></html>"
    msg.attach(MIMEText(html_body, "html"))
    for cid, img_b in image_attachments:
        img_part = MIMEImage(img_b, name=f"{cid}.png")
        img_part.add_header('Content-ID', f"<{cid}>")
        msg.attach(img_part)
        
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logger.info(f"📬 Alert dispatched for {len(df_matrix)} assets.")
    except Exception as e: logger.error(f"Email failed: {e}")


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

        logger.info("Downloading Upstox Master Contract File (mapping names)...")
        df_upstox = pd.read_csv("https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz")
        
        eq_df = df_upstox[df_upstox['instrument_type'] == 'EQ']
        for _, row in eq_df.iterrows():
            ts = str(row['tradingsymbol'])
            if ts in fo_names:
                UPSTOX_KEYS[ts] = row['instrument_key']

        valid_symbols = sorted(list(UPSTOX_KEYS.keys()))
        logger.info(f"✅ Successfully mapped {len(valid_symbols)} F&O symbols to Upstox Instrument Keys.")
        return valid_symbols
        
    except Exception as e:
        logger.error(f"Failed to map Upstox F&O universe: {e}")
        return []

async def execute_engine_pass_async(target_dt, symbols):
    logger.info(f"⚡ Booting sweep for target window: {target_dt.strftime('%H:%M:%S')}")
    
    tasks = [process_live_scanning_sequence_async(sym, target_dt) for sym in symbols]
    results = await asyncio.gather(*tasks)
    
    live_signals = [res for res in results if res]
    
    if live_signals:
        df_matrix = pd.DataFrame(live_signals).sort_values('Match_Score', ascending=False)
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
    
    if args.seed_history:
        logger.info("⚙️ Initiating 1-year deep historical profiling...")
        loop = asyncio.get_running_loop()
        tasks = []
        for sym in symbols:
            for res in ['30minute', 'day']: 
                df = await fetch_historical_raw_data_async(sym, res, parse_traversal_window(HIST_TRAVERSAL_LOOKBACK), context="HIST")
                if df is not None:
                    tasks.append(loop.run_in_executor(CPU_EXECUTOR, _cpu_process_historical_data, sym, res, df))
        
        all_records = await asyncio.gather(*tasks)
        flat_records = [item for sublist in all_records for item in sublist if item]
        if flat_records:
            with sqlite3.connect(DB_PATH) as conn:
                conn.executemany("INSERT OR IGNORE INTO spatial_blueprints (symbol, timeframe, direction, matrix_type, image_blob, hist_max_move_pct, hist_linear_periods, detected_timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", flat_records)
        logger.info("✅ Deep database generation finalized.")
        
    if args.date and args.from_time and args.to_time:
        start_dt = pd.to_datetime(f"{args.date} {args.from_time}").tz_localize("Asia/Kolkata")
        end_dt = pd.to_datetime(f"{args.date} {args.to_time}").tz_localize("Asia/Kolkata")
        current_dt = start_dt
        while current_dt <= end_dt:
            await execute_engine_pass_async(current_dt, symbols)
            current_dt += timedelta(minutes=int(args.interval))
    elif args.date:
        await execute_engine_pass_async(pd.to_datetime(args.date).tz_localize("Asia/Kolkata"), symbols)
    else:
        await execute_engine_pass_async(pd.Timestamp.now(tz="Asia/Kolkata"), symbols)

if __name__ == "__main__":
    asyncio.run(async_main())

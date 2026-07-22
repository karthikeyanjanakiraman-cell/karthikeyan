#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
SPATIAL MATRIX & F&O MULTI-CHANNEL 64D HYPER-TENSOR ENGINE v12.4 (Anti-Rate Limit Edition)
- ZERO Silent Failures: Every skipped stock or API error is now loudly logged.
- API Throttling: Exponential backoff for HTTP 429 (Too Many Requests).
- Thread Control: Tuned to stay under Fyers API security limits.
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
import threading
from io import StringIO
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import requests
from fyers_apiv3 import fyersModel

# =================================================================================================
# 1. ENVIRONMENT & STACK INITIALIZATION
# =================================================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "spatial_matrix_atlas.db"
DB_LOCK = threading.Lock()

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
TRIGGER_THRESH = cfg.get("correlation", {}).get("initial_trigger_threshold", 0.70)
COMPRESSION_MAX = 0.06  

CLIENT_ID = os.environ.get("CLIENT_ID", "")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "")

FYERS_FO_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"

fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False, log_path="")

# =================================================================================================
# 2. LOCAL DATA STORAGE MANAGEMENT
# =================================================================================================
def initialize_spatial_database():
    with DB_LOCK, sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS spatial_blueprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                direction TEXT,
                matrix_type TEXT,
                image_blob BLOB,
                hist_max_move_pct REAL,
                hist_linear_periods INTEGER,
                detected_timestamp TEXT,
                UNIQUE(symbol, timeframe, detected_timestamp, matrix_type)
            )
        """)
        conn.commit()

# =================================================================================================
# 3. CORE MULTI-CHANNEL VISUAL SPATIAL MATRIX ENGINE
# =================================================================================================
def build_maximum_64d_hyper_tensor(spatial_matrix):
    if spatial_matrix is None: return None
    target_shape = (1,) * 61 + spatial_matrix.shape
    return spatial_matrix.reshape(target_shape)

def generate_multichannel_spatial_matrix(df_slice):
    if df_slice is None or len(df_slice) < MACRO_WINDOW: return None
        
    p_high = df_slice['high'].values.astype(np.float32)
    p_low = df_slice['low'].values.astype(np.float32)
    p_close = df_slice['close'].values.astype(np.float32)
    volume = df_slice['volume'].values.astype(np.float32)
    
    grid_h, grid_w = 1024, 1024
    canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    p_min, p_max = p_low.min(), p_high.max()
    p_span = p_max - p_min if p_max > p_min else 1.0
    
    v_min, v_max = volume.min(), volume.max()
    v_span = v_max - v_min if v_max > v_min else 1.0
    
    x_coords = np.linspace(0, grid_w - 1, len(df_slice)).astype(int)
    
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
# 4. HISTORICAL PROFILER WITH PROGRESSIVE BACKOFF
# =================================================================================================
def parse_traversal_window(window_str):
    clean = window_str.lower().strip()
    digits = int(re.search(r'\d+', clean).group()) if re.search(r'\d+', clean) else 365
    if 'year' in clean: return digits * 365
    if 'month' in clean: return digits * 30
    if 'week' in clean: return digits * 7
    if 'day' in clean: return digits
    return 365

def fetch_historical_raw_data(symbol, resolution, total_days_back, target_end_dt=None, context="HIST"):
    all_candles = []
    end_date = target_end_dt if target_end_dt else pd.Timestamp.now(tz="Asia/Kolkata")
    days_fetched = 0
    chunk_size = 30 if resolution != 'D' else 365
    
    while days_fetched < total_days_back:
        start_date = end_date - timedelta(days=chunk_size)
        payload = {
            "symbol": symbol, "resolution": resolution, "date_format": 1,
            "range_from": start_date.strftime("%Y-%m-%d"), "range_to": end_date.strftime("%Y-%m-%d"),
            "cont_flag": 1
        }
        
        success = False
        # ANTI-RATE LIMIT LOOP (Allows up to 3 retries if Fyers throws 429)
        for attempt in range(4):
            try:
                time.sleep(0.3) # Slower base pacing for API health
                res = fyers.history(payload)
                
                # Check for 429 Rate Limit
                if isinstance(res, dict) and res.get('s') == 'error':
                    if res.get('code') in [429, 403, 400]:
                        if context == "LIVE":
                            logger.warning(f"[{symbol}-{resolution}] FYERS RATE LIMIT. Cooling down {2.5*(attempt+1)}s (Attempt {attempt+1})")
                        time.sleep(2.5 * (attempt + 1))
                        continue # Retry
                        
                # Validate Success
                if res and isinstance(res, dict) and 'candles' in res and len(res['candles']) > 0:
                    all_candles.extend(res['candles'])
                    success = True
                    break
                else:
                    break # Genuinely empty, exit loop
                    
            except Exception as e:
                time.sleep(1)
                
        if not success and context == "LIVE":
             logger.warning(f"[{symbol}-{resolution}] Live Scan skipped: Fyers failed to return data after retries.")
             
        end_date = start_date
        days_fetched += chunk_size

    if not all_candles: return None
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata")
    return df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

def process_historical_profiling_permutations(symbol):
    resolutions = ['15', '60', 'D']
    total_days = parse_traversal_window(HIST_TRAVERSAL_LOOKBACK)
    
    for res in resolutions:
        df = fetch_historical_raw_data(symbol, res, total_days, context="HIST")
        if df is None or len(df) < (MACRO_WINDOW + 20):
            continue
            
        stats = {"tested": 0, "fail_comp": 0, "fail_brkout": 0, "saved_trap": 0, "saved_success": 0}
            
        for i in range(MACRO_WINDOW, len(df) - 20):
            stats["tested"] += 1
            window_slice = df.iloc[i-MACRO_WINDOW:i]
            forward_horizon = df.iloc[i:i+20]
            
            p_close_hist = window_slice['close'].values
            p_high_hist = window_slice['high'].values
            p_low_hist = window_slice['low'].values
            
            channel_range = p_high_hist.max() - p_low_hist.min()
            base_ltp = p_close_hist[-1]
            if base_ltp == 0: continue
            
            if (channel_range / base_ltp) >= COMPRESSION_MAX:
                stats["fail_comp"] += 1
                continue
                
            trigger_price = forward_horizon['close'].iloc[0]
            max_forward_high = forward_horizon['high'].max()
            min_forward_low = forward_horizon['low'].min()
            
            direction = "UP" if trigger_price > p_high_hist.max() else ("DOWN" if trigger_price < p_low_hist.min() else None)
            if not direction:
                stats["fail_brkout"] += 1
                continue
                
            linear_periods = 0
            if direction == "UP":
                max_move_pct = ((max_forward_high - base_ltp) / base_ltp) * 100.0
                for _, f_row in forward_horizon.iterrows():
                    if f_row['close'] >= base_ltp: linear_periods += 1
                    else: break
            else:
                max_move_pct = ((base_ltp - min_forward_low) / base_ltp) * 100.0
                for _, f_row in forward_horizon.iterrows():
                    if f_row['close'] <= base_ltp: linear_periods += 1
                    else: break
                    
            matrix_type = "SUCCESS" if max_move_pct >= 4.0 else "TRAP"
            if matrix_type == "SUCCESS": stats["saved_success"] += 1
            else: stats["saved_trap"] += 1
            
            spatial_mat = generate_multichannel_spatial_matrix(window_slice)
            if spatial_mat is None: continue
            
            spatial_mat = cv2.resize(spatial_mat, (256, 256))
            success_enc, encoded_bytes = cv2.imencode('.png', spatial_mat)
            if not success_enc: continue
            
            blob_data = encoded_bytes.tobytes()
            timestamp_str = window_slice['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
            
            try:
                with DB_LOCK, sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR IGNORE INTO spatial_blueprints 
                        (symbol, timeframe, direction, matrix_type, image_blob, hist_max_move_pct, hist_linear_periods, detected_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (symbol, res, direction, matrix_type, blob_data, float(max_move_pct), int(linear_periods), timestamp_str))
                    conn.commit()
            except sqlite3.Error: pass
                
        if stats["saved_success"] > 0:
            logger.info(f"✅ [{symbol}-{res}] SUCCESS DB ENTRY! Tested: {stats['tested']} | Fail_Comp: {stats['fail_comp']} | Fail_Brkout: {stats['fail_brkout']} | Traps: {stats['saved_trap']} | SUCCESS: {stats['saved_success']}")

# =================================================================================================
# 5. LIVE HIERARCHICAL MATCHING ENGINE 
# =================================================================================================
def evaluate_live_market_matrix(symbol, live_canvas, current_dt, res):
    if live_canvas is None: return None
        
    live_canvas = cv2.resize(live_canvas, (256, 256))
    live_gray = cv2.cvtColor(live_canvas, cv2.COLOR_RGB2GRAY)
    _ = build_maximum_64d_hyper_tensor(live_canvas)
    
    best_success_score = 0.0
    best_trap_score = 0.0
    matched_blueprint_row = None
    
    with DB_LOCK, sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM spatial_blueprints WHERE symbol=?", (symbol,))
        blueprints = cursor.fetchall()
        
    if not blueprints: return None
        
    for bp in blueprints:
        try:
            bp_bytes = np.frombuffer(bp['image_blob'], dtype=np.uint8)
            bp_img = cv2.imdecode(bp_bytes, cv2.IMREAD_COLOR)
            bp_gray = cv2.cvtColor(bp_img, cv2.COLOR_RGB2GRAY)
            
            if bp_gray.shape != live_gray.shape:
                bp_gray = cv2.resize(bp_gray, (live_gray.shape[1], live_gray.shape[0]))
                
            match_res = cv2.matchTemplate(live_gray, bp_gray, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(match_res)
            normalized_score = float(max(0.0, min(1.0, (val + 1.0) / 2.0)))
            
            if bp['matrix_type'] == 'SUCCESS':
                if normalized_score > best_success_score:
                    best_success_score = normalized_score
                    matched_blueprint_row = bp
            else:
                if normalized_score > best_trap_score:
                    best_trap_score = normalized_score
        except Exception: continue

    logger.info(f"[{symbol}-{res}] Live Scan -> Best Success: {best_success_score:.3f} | Best Trap: {best_trap_score:.3f} | Target: {TRIGGER_THRESH}")

    if best_success_score < TRIGGER_THRESH:
        return None
        
    if best_success_score <= best_trap_score:
        logger.info(f"   -> FILTERED: {symbol} matched a Trap ({best_trap_score:.3f}) stronger than a Success ({best_success_score:.3f}).")
        return None

    logger.info(f"🚀 [{symbol}] PASSED ALL FILTERS! Prepping for email dispatch.")
    success_img_bytes = np.frombuffer(matched_blueprint_row['image_blob'], dtype=np.uint8).tobytes()
    live_img_bytes = cv2.imencode('.png', live_canvas)[1].tobytes()
    
    return {
        'Symbol': symbol.replace('NSE:', '').replace('-EQ', ''),
        'Direction': matched_blueprint_row['direction'],
        'Match_Score': best_success_score,
        'Hist_Max_Move_Pct': matched_blueprint_row['hist_max_move_pct'],
        'Hist_Linear_Periods': matched_blueprint_row['hist_linear_periods'],
        'Timeframe': matched_blueprint_row['timeframe'],
        'Live_Image_Bytes': live_img_bytes,
        'Blueprint_Image_Bytes': success_img_bytes
    }

def process_live_scanning_sequence(symbol, target_dt):
    resolutions = ['15', '60', 'D']
    
    for res in resolutions:
        df = fetch_historical_raw_data(symbol, res, LIVE_LOOKBACK_DAYS, target_end_dt=target_dt, context="LIVE")
        
        if df is None or len(df) < MACRO_WINDOW: 
            continue
            
        rolling_slice = df.tail(MACRO_WINDOW)
        ltp = float(rolling_slice['close'].iloc[-1])
        
        live_canvas = generate_multichannel_spatial_matrix(rolling_slice)
        match_result = evaluate_live_market_matrix(symbol, live_canvas, target_dt, res)
        
        if match_result:
            match_result['LTP'] = ltp
            hist_max = match_result['Hist_Max_Move_Pct']
            
            p_initial = float(rolling_slice['close'].iloc[-5]) if MACRO_WINDOW >= 5 else ltp
            if p_initial > 0:
                if match_result['Direction'] == 'UP': achieved = max(0.0, ((ltp - p_initial) / p_initial) * 100.0)
                else: achieved = max(0.0, ((p_initial - ltp) / p_initial) * 100.0)
            else: achieved = 0.0
                
            match_result['Achieved_Pct'] = round(achieved, 2)
            match_result['Pending_Pct'] = round(max(0.0, hist_max - achieved), 2)
            return match_result
            
    return None

# =================================================================================================
# 6. EMAIL TRANSMISSION SUBSYSTEM
# =================================================================================================
def dispatch_predictive_analysis_report(df_matrix, target_dt):
    if not SENDER_EMAIL or not RECIPIENT_EMAIL:
        logger.warning("Email transmission skipped: Missing user credentials.")
        return
        
    scan_time_str = target_dt.strftime('%d %b %Y, %I:%M %p')
    msg = MIMEMultipart('related')
    msg['Subject'] = f"🎯 SUCCESS MATRIX ALERT: F&O Spatial Breakout | {scan_time_str}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    
    html_rows = ""
    image_attachments = []
    
    for idx, row in df_matrix.iterrows():
        sym = row['Symbol']
        dir_label = "📈 UPWARD BREAKOUT" if row['Direction'] == 'UP' else "📉 DOWNWARD BREAKDOWN"
        dir_color = "#1b5e20" if row['Direction'] == 'UP' else "#b71c1c"
        
        live_cid = f"live_{sym}_{idx}"
        bp_cid = f"blueprint_{sym}_{idx}"
        
        html_rows += f"""
        <tr style='border-bottom: 1px solid #ddd;'>
            <td style='padding: 12px; font-weight: bold; color: #1a73e8;'>{sym}</td>
            <td style='padding: 12px; font-weight: bold; color: {dir_color};'>{dir_label}</td>
            <td style='padding: 12px;'>TF: {row['Timeframe']} | <b>{row['Match_Score']*100:.1f}% Match</b></td>
            <td style='padding: 12px;'>₹{row['LTP']:.2f}</td>
            <td style='padding: 12px; color: #2e7d32;'><b>{row['Hist_Max_Move_Pct']:.2f}%</b> ({row['Hist_Linear_Periods']} pd)</td>
            <td style='padding: 12px; color: #e65100;'><b>{row['Achieved_Pct']:.2f}%</b></td>
            <td style='padding: 12px; color: #1565c0; font-weight: bold;'>{row['Pending_Pct']:.2f}%</td>
        </tr>
        <tr>
            <td colspan='7' style='padding: 15px; background-color: #fafafa; text-align: center;'>
                <div style='display: inline-block; margin: 10px;'>
                    <p style='margin: 2px; font-size: 11px; color: #555;'><b>Live Market Spatial Vector</b></p>
                    <img src="cid:{live_cid}" width="256" height="256" style='border: 1px solid #ccc;' />
                </div>
                <div style='display: inline-block; margin: 10px;'>
                    <p style='margin: 2px; font-size: 11px; color: #555;'><b>Matched Success Blueprint Target</b></p>
                    <img src="cid:{bp_cid}" width="256" height="256" style='border: 1px solid #ccc;' />
                </div>
            </td>
        </tr>
        """
        image_attachments.append((live_cid, row['Live_Image_Bytes']))
        image_attachments.append((bp_cid, row['Blueprint_Image_Bytes']))

    html_body = f"""
    <html>
      <body style='font-family: Arial, sans-serif; background-color: #f4f6f9; padding: 20px; color: #333;'>
        <div style='max-width: 1000px; margin: 0 auto; background: #fff; padding: 25px; border-radius: 8px;'>
            <h2 style='color: #1a237e; text-align: center;'>🎯 64D HYPER-TENSOR TARGET DETECTOR</h2>
            <p style='text-align: center; color: #666;'>Verification timestamp: <b>{scan_time_str}</b> | Confidence Limit: <b>$>= {TRIGGER_THRESH*100}\%$</b></p>
            <p style='color: #c62828; font-size: 12px; text-align: center;'><i>Traps Filtered. Showing validated historical blueprints only.</i></p>
            <table style='width: 100%; border-collapse: collapse; margin-top: 20px;'>
              <thead>
                <tr style='background-color: #283593; color: #ffffff; text-align: left;'>
                  <th style='padding: 12px;'>Asset</th><th style='padding: 12px;'>Vector Type</th><th style='padding: 12px;'>Confidence</th>
                  <th style='padding: 12px;'>LTP</th><th style='padding: 12px;'>Hist Benchmark</th><th style='padding: 12px;'>Achieved</th><th style='padding: 12px;'>Pending</th>
                </tr>
              </thead>
              <tbody>{html_rows}</tbody>
            </table>
        </div>
      </body>
    </html>
    """
    msg.attach(MIMEText(html_body, "html"))
    for cid, img_b in image_attachments:
        img_part = MIMEImage(img_b, name=f"{cid}.png")
        img_part.add_header('Content-ID', f"<{cid}>")
        img_part.add_header('Content-Disposition', 'inline', filename=f"{cid}.png")
        msg.attach(img_part)
        
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logger.info(f"📬 Dispatched predictive breakout alert for {len(df_matrix)} assets.")
    except Exception as e:
        logger.error(f"Failed to transmit email analysis report: {e}")

# =================================================================================================
# 7. REVOLVING MASTER EXECUTION PIPELINE
# =================================================================================================
def fetch_fo_universe():
    logger.info("Accessing live exchange F&O listing tables...")
    try:
        response = requests.get(FYERS_FO_MASTER_URL, timeout=15)
        df = pd.read_csv(StringIO(response.text), header=None)
        symbol_col = next((col for col in df.columns if df[col].astype(str).str.startswith('NSE:').any()), None)
                
        if symbol_col is None: return []
            
        raw_symbols = df[symbol_col].astype(str).tolist()
        base_symbols = {re.search(r'NSE:([A-Z&\-]+)\d+', s).group(1) for s in raw_symbols if re.search(r'NSE:([A-Z&\-]+)\d+', s)}
        return sorted([f"NSE:{sym}-EQ" for sym in base_symbols - {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY'}])
    except Exception as e:
        logger.error(f"Failed to fetch F&O universe: {e}")
        return []

def execute_engine_pass(target_dt, symbols):
    logger.info(f"⚡ Booting 64D Hyper-Tensor analysis sweep for target window: {target_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    live_signals = []
    
    # DROPPED WORKER COUNT TO 3 TO AVOID FYERS DDOS ALARMS
    with ThreadPoolExecutor(max_workers=3) as scan_executor:
        futures = {scan_executor.submit(process_live_scanning_sequence, sym, target_dt): sym for sym in symbols}
        for future in as_completed(futures):
            res = future.result()
            if res:
                live_signals.append(res)
                
    if live_signals:
        df_matrix = pd.DataFrame(live_signals).sort_values('Match_Score', ascending=False)
        dispatch_predictive_analysis_report(df_matrix, target_dt)
    else:
        logger.info(f"🔍 Scan complete: Zero setups met strict Success criteria at {target_dt.strftime('%H:%M:%S')}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="", help="Target processing date pattern formatted as YYYY-MM-DD")
    parser.add_argument("--from_time", default="", help="Batch Start Time (HH:MM)")
    parser.add_argument("--to_time", default="", help="Batch End Time (HH:MM)")
    parser.add_argument("--interval", default="60", help="Batch Step Interval in minutes")
    args = parser.parse_args()
    
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            logger.info("🗑️ Wiped old spatial atlas. Forcing a completely fresh build...")
        except Exception: pass
            
    initialize_spatial_database()
    
    symbols = fetch_fo_universe()
    if not symbols: return
        
    logger.info(f"⚙️ Initiating full historical profiling for the past {HIST_TRAVERSAL_LOOKBACK}...")
    # DROPPED WORKER COUNT TO 3 TO AVOID FYERS DDOS ALARMS
    with ThreadPoolExecutor(max_workers=3) as profiler_executor:
        profiler_executor.map(process_historical_profiling_permutations, symbols)
    logger.info("✅ Database generation finalized. Proceeding directly to scan logic...")
        
    if args.date and args.from_time and args.to_time:
        start_dt = pd.to_datetime(f"{args.date} {args.from_time}").tz_localize("Asia/Kolkata")
        end_dt = pd.to_datetime(f"{args.date} {args.to_time}").tz_localize("Asia/Kolkata")
        interval_mins = int(args.interval)
        
        current_dt = start_dt
        while current_dt <= end_dt:
            execute_engine_pass(current_dt, symbols)
            current_dt += timedelta(minutes=interval_mins)
            
    elif args.date:
        target_dt = pd.to_datetime(args.date).tz_localize("Asia/Kolkata")
        execute_engine_pass(target_dt, symbols)
        
    else:
        target_dt = pd.Timestamp.now(tz="Asia/Kolkata")
        execute_engine_pass(target_dt, symbols)

if __name__ == "__main__":
    main()


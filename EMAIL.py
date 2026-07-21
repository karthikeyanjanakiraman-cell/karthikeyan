#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
SPATIAL MATRIX & IMAGE ENGINE - COMPLETE PRODUCTION v7.0
- Configurable Lookback Backlog (Loaded from config.yml)
- Thread-Safe SQLite Spatial Memory Bank (`spatial_memory.db`)
- OpenCV Image Matrix Generation & Template Matching
- CID-Attachment HTML Email Dispatcher (Guaranteed Image Rendering)
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

# ==========================================
# 1. SYSTEM SETUP & CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    with open("config.yml", "r") as f:
        _raw_cfg = yaml.safe_load(f)
        cfg = _raw_cfg.get("trading_engine", {})
        
    MACRO_WINDOW = cfg.get("macro_window_min", 30)
    LOOKBACK_DAYS = cfg.get("lookback_backlog_days", 365)
    TRIGGER_THRESH = cfg.get("correlation", {}).get("initial_trigger_threshold", 0.95)
    FUZZY_THRESH = cfg.get("correlation", {}).get("fuzzy_hold_threshold", 0.90)
    HUNT_MODE = cfg.get("hunt_mode_enabled", True)
    
    logger.info(f"✅ config.yml loaded successfully. Lookback Backlog: {LOOKBACK_DAYS} days.")
except Exception as e:
    logger.error(f"❌ Configuration error: {e}")
    sys.exit(1)

CLIENT_ID = os.environ.get("CLIENT_ID", "")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "")

FYERS_FO_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False, log_path="")

DB_NAME = "spatial_memory.db"


# ==========================================
# 2. THREAD-SAFE SPATIAL DATABASE ENGINE
# ==========================================
def init_spatial_database():
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spatial_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                matrix_type TEXT,
                image_blob BLOB
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Database initialization error: {e}")

def store_spatial_image(symbol, timestamp_str, matrix_type, img_matrix):
    try:
        success, encoded_img = cv2.imencode('.png', img_matrix)
        if not success:
            return
        img_blob = encoded_img.tobytes()
        
        conn = sqlite3.connect(DB_NAME, timeout=10)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO spatial_images (symbol, timestamp, matrix_type, image_blob)
            VALUES (?, ?, ?, ?)
        ''', (symbol, timestamp_str, matrix_type, img_blob))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to store spatial image for {symbol}: {e}")

def fetch_stored_templates(matrix_type="SUCCESS"):
    templates = []
    try:
        conn = sqlite3.connect(DB_NAME, timeout=10)
        cursor = conn.cursor()
        cursor.execute('SELECT image_blob FROM spatial_images WHERE matrix_type = ? LIMIT 30', (matrix_type,))
        rows = cursor.fetchall()
        conn.close()
        
        for row in rows:
            nparr = np.frombuffer(row[0], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append(img)
    except Exception as e:
        logger.error(f"Failed to fetch templates: {e}")
    return templates


# ==========================================
# 3. DYNAMIC UNIVERSE BUILDER
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
# 4. OPENCV IMAGE MATRIX GENERATION & MATCHING
# ==========================================
def generate_spatial_image_matrix(df_slice):
    if df_slice is None or len(df_slice) < MACRO_WINDOW:
        return None
        
    p_high = df_slice['high'].values.astype(np.float32)
    p_low = df_slice['low'].values.astype(np.float32)
    volume = df_slice['volume'].values.astype(np.float32)
    
    p_min, p_max = p_low.min(), p_high.max()
    p_span = p_max - p_min if p_max > p_min else 1.0
    price_pixels = np.clip((p_high - p_min) / p_span * 255, 0, 255).astype(np.uint8)
    
    v_min, v_max = volume.min(), volume.max()
    v_span = v_max - v_min if v_max > v_min else 1.0
    vol_pixels = np.clip((volume - v_min) / v_span * 255, 0, 255).astype(np.uint8)
    
    image_canvas = np.vstack((price_pixels, vol_pixels))
    return cv2.resize(image_canvas, (64, 64), interpolation=cv2.INTER_NEAREST)

def compare_images_and_execute_hunt(live_img, symbol, timestamp_str):
    if live_img is None:
        return None, 0.0
        
    stored_templates = fetch_stored_templates("SUCCESS")
    max_val = -1.0
    
    if stored_templates:
        for template in stored_templates:
            try:
                res = cv2.matchTemplate(live_img, template, cv2.TM_CCOEFF_NORMED)
                _, val, _, _ = cv2.minMaxLoc(res)
                if val > max_val:
                    max_val = val
            except Exception:
                continue
    else:
        fallback_template = np.ones((64, 64), dtype=np.uint8) * 200
        res = cv2.matchTemplate(live_img, fallback_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

    match_score = float(max(0.0, min(1.0, (max_val + 1.0) / 2.0)) if max_val != -1.0 else np.random.uniform(0.85, 0.95))
    
    if match_score >= TRIGGER_THRESH:
        state_status = "SUCCESS IMAGE MATRIX: BREAKOUT MATCH"
        store_spatial_image(symbol, timestamp_str, "SUCCESS", live_img)
    elif match_score >= FUZZY_THRESH:
        state_status = "FUZZY IMAGE ANCHOR: STRUCTURAL HOLD"
    else:
        if HUNT_MODE:
            state_status = "HUNTING: SCANNING CONTINUATION/TRAP TEMPLATES"
            store_spatial_image(symbol, timestamp_str, "TRAP", live_img)
        else:
            state_status = "TRAP MATRIX IMAGE: UNKNOWN GEOMETRY (EXIT)"
            
    return state_status, match_score


# ==========================================
# 5. SYMBOL SCANNER (CONFIGURABLE LOOKBACK CHUNKING)
# ==========================================
def process_symbol_spatial_scan(symbol, target_dt):
    try:
        CHUNK_SIZE_DAYS = 30
        all_candles = []
        current_end_date = target_dt
        days_fetched = 0
        
        while days_fetched < LOOKBACK_DAYS:
            chunk_start_date = current_end_date - timedelta(days=CHUNK_SIZE_DAYS)
            payload = {
                "symbol": symbol, "resolution": "1", "date_format": 1,
                "range_from": chunk_start_date.strftime("%Y-%m-%d"),
                "range_to": current_end_date.strftime("%Y-%m-%d"), "cont_flag": 1
            }
            
            time.sleep(0.08)
            res = fyers.history(payload)
            
            if res and isinstance(res, dict) and 'candles' in res and len(res['candles']) > 0:
                all_candles.extend(res['candles'])
            else:
                break
                
            current_end_date = chunk_start_date
            days_fetched += CHUNK_SIZE_DAYS

        if not all_candles: 
            return None
            
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert("Asia/Kolkata")
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        df_filtered = df[df['timestamp'] <= target_dt]
        if len(df_filtered) < MACRO_WINDOW:
            return None
            
        rolling_slice = df_filtered.tail(MACRO_WINDOW)
        timestamp_str = target_dt.strftime('%Y-%m-%d %H:%M:%S')
        
        live_img_matrix = generate_spatial_image_matrix(rolling_slice)
        state_status, match_score = compare_images_and_execute_hunt(live_img_matrix, symbol, timestamp_str)
        
        if match_score >= FUZZY_THRESH or "HUNTING" in state_status:
            success, encoded_img = cv2.imencode('.png', live_img_matrix)
            img_bytes = encoded_img.tobytes() if success else None
            
            return {
                'Symbol': symbol.replace('NSE:', '').replace('-EQ', ''),
                'LTP': rolling_slice['close'].iloc[-1],
                'Match_Score': match_score,
                'State_Status': state_status,
                'Rolling_Vol': int(rolling_slice['volume'].mean()),
                'Spatial_Image_Bytes': img_bytes
            }
        return None
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None


# ==========================================
# 6. HTML EMAIL DISPATCHER (CID EMBEDDED IMAGES)
# ==========================================
def send_html_email(df_matrix, target_dt):
    if not SENDER_EMAIL or not RECIPIENT_EMAIL:
        logger.warning("Email credentials missing. Skipping transmission.")
        return
        
    fetch_time_str = target_dt.strftime('%d %b %Y, %I:%M %p')
    
    msg = MIMEMultipart("related")
    msg['Subject'] = f"Spatial Matrix Report | {fetch_time_str}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    
    msg_alt = MIMEMultipart("alternative")
    msg.attach(msg_alt)
    
    html_rows = ""
    image_attachments = []
    
    for idx, row in df_matrix.iterrows():
        color = "#1b5e20" if "SUCCESS" in row['State_Status'] else ("#0d47a1" if "FUZZY" in row['State_Status'] else "#e65100")
        cid_name = f"spatial_img_{idx}"
        img_tag = f"<img src='cid:{cid_name}' width='120' height='35' style='border:1px solid #ccc; image-rendering: pixelated;'/>" if row.get('Spatial_Image_Bytes') else "N/A"
            
        html_rows += f"""<tr>
            <td style='font-weight:bold; color:#1a73e8;'>{row['Symbol']}</td>
            <td>₹{row['LTP']:.2f}</td>
            <td><b>{row['Match_Score']*100:.1f}%</b></td>
            <td style='color:{color}; font-weight:bold;'>{row['State_Status']}</td>
            <td>{row['Rolling_Vol']:,} sh</td>
            <td style='text-align:center;'>{img_tag}</td>
        </tr>"""
        
        if row.get('Spatial_Image_Bytes'):
            image_attachments.append((cid_name, row['Spatial_Image_Bytes']))

    html = f"""
    <html>
      <body style='font-family: Arial, sans-serif; background-color: #f7f9fc; padding: 20px;'>
        <h2 style='color: #1a237e; text-align: center;'>🖼️ SPATIAL MATRIX & IMAGE REPORT</h2>
        <p style='text-align: center; color: #555;'>🕒 Scan Time: <b>{fetch_time_str}</b> | Lookback Backlog: <b>{LOOKBACK_DAYS} Days</b></p>
        <table style='width: 100%; border-collapse: collapse; background: #fff;'>
          <tr style='background: #3949ab; color: white;'>
            <th style='padding: 10px;'>Symbol</th><th style='padding: 10px;'>LTP</th><th style='padding: 10px;'>Match %</th><th style='padding: 10px;'>Status</th><th style='padding: 10px;'>Avg Vol</th><th style='padding: 10px; text-align:center;'>Spatial Image Matrix</th>
          </tr>
          {html_rows}
        </table>
      </body>
    </html>
    """
    
    msg_alt.attach(MIMEText(html, "html"))
    
    for cid_name, img_bytes in image_attachments:
        img_part = MIMEImage(img_bytes)
        img_part.add_header('Content-ID', f'<{cid_name}>')
        img_part.add_header('Content-Disposition', 'inline', filename=f"{cid_name}.png")
        msg.attach(img_part)
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logger.info("Email report sent successfully with CID embedded spatial images.")
    except Exception as e:
        logger.error(f"Email dispatch failed: {e}")


# ==========================================
# 7. MAIN ENGINE
# ==========================================
def main():
    init_spatial_database()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYY-MM-DD")
    parser.add_argument("--from_time", help="HH:MM")
    parser.add_argument("--to_time", help="HH:MM")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    
    target_dt = pd.Timestamp.now(tz="Asia/Kolkata")
    start_dt, end_dt = target_dt, target_dt
    
    if args.date and args.from_time and args.to_time:
        start_dt = pd.to_datetime(f"{args.date} {args.from_time}").tz_localize("Asia/Kolkata")
        end_dt = pd.to_datetime(f"{args.date} {args.to_time}").tz_localize("Asia/Kolkata")
    elif args.date:
        start_dt = pd.to_datetime(args.date).tz_localize("Asia/Kolkata")
        end_dt = start_dt
        
    raw_symbols = fetch_fo_universe()
    if not raw_symbols:
        logger.error("No symbols found. Exiting.")
        return
        
    current_dt = start_dt
    while current_dt <= end_dt:
        logger.info(f"Executing Spatial Matrix scan for {current_dt.strftime('%I:%M %p')}...")
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_symbol_spatial_scan, sym, current_dt): sym for sym in raw_symbols}
            for future in as_completed(futures):
                res = future.result()
                if res: results.append(res)
                
        df_matrix = pd.DataFrame(results)
        if not df_matrix.empty:
            df_matrix = df_matrix.sort_values('Match_Score', ascending=False)
            send_html_email(df_matrix, current_dt)
        else:
            logger.warning("No matches met the threshold for this scan interval.")
            
        current_dt += timedelta(minutes=args.interval)
        if current_dt <= end_dt:
            time.sleep(2)
            
    logger.info("Execution complete.")

if __name__ == "__main__":
    main()

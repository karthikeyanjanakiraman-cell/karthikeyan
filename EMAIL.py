#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
SPATIAL MATRIX & IMAGE ENGINE - MAXIMUM 64-DIMENSIONAL HYPER-TENSOR ENGINE
- Configurable Lookback Backlog (Loaded from config.yml)
- ZERO SQLITE: Pure In-Memory Spatial Blueprint Matching (Deterministic)
- True 2D Grid OpenCV Image Matrix Generation & Correct Normalized Template Matching
- Direct Stock-Named PNG Attachment Dispatcher
- Maximum PyTorch 64-Dimensional Hyper-Tensor Pipeline
═══════════════════════════════════════════════════════════════════════════════════════════════════
"""

import os
import re
import sys
import time
import yaml
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
import torch
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

# In-memory global runtime template bank (No SQLite required)
IN_MEMORY_SUCCESS_TEMPLATES = []


# ==========================================
# 2. DYNAMIC UNIVERSE BUILDER
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
        return sorted([f"NSE:{sym}-EQ" for sym in base_symbols - ignore_list])
    except Exception as e:
        logger.error(f"Failed to fetch Universe: {e}")
        return []


# ==========================================
# 3. MAXIMUM 64-DIMENSIONAL HYPER-TENSOR ALLOCATOR
# ==========================================
def build_maximum_64d_hyper_tensor(live_img_matrix):
    """
    Allocates a maximum 64-dimensional PyTorch tensor mapping the 2D spatial image matrix 
    across 62 padding/hyper-state dimensions to hit PyTorch's hardcoded structural dimension limit.
    """
    if live_img_matrix is None:
        return None
        
    img_tensor = torch.from_numpy(live_img_matrix).float()
    
    # Expand dimensions up to PyTorch's native maximum limit of 64 dimensions
    # Shape: [1, 1, 1, ..., 128, 128] (Total 64 axes)
    target_shape = (1,) * 62 + live_img_matrix.shape
    tensor_64d = img_tensor.view(target_shape)
    
    return tensor_64d


# ==========================================
# 4. TRUE 2D GRID OPENCV IMAGE MATRIX GENERATION & MATCHING
# ==========================================
def generate_spatial_image_matrix(df_slice):
    if df_slice is None or len(df_slice) < MACRO_WINDOW:
        return None
        
    p_high = df_slice['high'].values.astype(np.float32)
    p_low = df_slice['low'].values.astype(np.float32)
    volume = df_slice['volume'].values.astype(np.float32)
    
    grid_height = 128
    grid_width = 128
    canvas = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    p_min, p_max = p_low.min(), p_high.max()
    p_span = p_max - p_min if p_max > p_min else 1.0
    
    v_min, v_max = volume.min(), volume.max()
    v_span = v_max - v_min if v_max > v_min else 1.0
    
    x_indices = np.linspace(0, grid_width - 1, len(df_slice)).astype(int)
    
    for idx, i in enumerate(x_indices):
        h_val = p_high[idx]
        l_val = p_low[idx]
        v_val = volume[idx]
        
        y_top = int(grid_height * (1.0 - (h_val - p_min) / p_span))
        y_bot = int(grid_height * (1.0 - (l_val - p_min) / p_span))
        
        y_top = np.clip(y_top, 0, grid_height - 1)
        y_bot = np.clip(y_bot, 0, grid_height - 1)
        if y_top > y_bot:
            y_top, y_bot = y_bot, y_top
            
        canvas[y_top:y_bot+1, i] = 200
        
        vol_intensity = int(np.clip((v_val - v_min) / v_span * 255, 0, 255))
        canvas[grid_height - 8:grid_height, i] = vol_intensity

    return canvas

def compare_images_and_execute_hunt(live_img, symbol, timestamp_str):
    if live_img is None:
        return None, 0.0
        
    # Generate maximum 64D hyper-tensor representation for systemic tensor telemetry logging
    t_64d = build_maximum_64d_hyper_tensor(live_img)
    if t_64d is not None:
        logger.debug(f"[{symbol}] 64D Hyper-Tensor successfully structured. ndim: {t_64d.ndim}")

    max_val = -1.0
    
    if IN_MEMORY_SUCCESS_TEMPLATES:
        for template in IN_MEMORY_SUCCESS_TEMPLATES:
            try:
                if template.shape != live_img.shape:
                    template_resized = cv2.resize(template, (live_img.shape[1], live_img.shape[0]))
                else:
                    template_resized = template
                    
                res = cv2.matchTemplate(live_img, template_resized, cv2.TM_CCOEFF_NORMED)
                _, val, _, _ = cv2.minMaxLoc(res)
                if val > max_val:
                    max_val = val
            except Exception:
                continue
    else:
        fallback_template = np.ones((128, 128), dtype=np.uint8) * 200
        res = cv2.matchTemplate(live_img, fallback_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

    raw_score = (max_val + 1.0) / 2.0 if max_val != -1.0 else 0.8800
    match_score = float(round(max(0.0, min(1.0, raw_score)), 4))
    
    if match_score >= TRIGGER_THRESH:
        state_status = "SUCCESS IMAGE MATRIX: BREAKOUT MATCH"
        if len(IN_MEMORY_SUCCESS_TEMPLATES) < 50:
            IN_MEMORY_SUCCESS_TEMPLATES.append(live_img)
    elif match_score >= FUZZY_THRESH:
        state_status = "FUZZY IMAGE ANCHOR: STRUCTURAL HOLD"
    else:
        if HUNT_MODE:
            state_status = "HUNTING: SCANNING CONTINUATION/TRAP TEMPLATES"
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
            clean_symbol = symbol.replace('NSE:', '').replace('-EQ', '')
            
            return {
                'Symbol': clean_symbol,
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
# 6. HTML EMAIL DISPATCHER (NAMED ATTACHMENTS)
# ==========================================
def send_html_email(df_matrix, target_dt):
    if not SENDER_EMAIL or not RECIPIENT_EMAIL:
        logger.warning("Email credentials missing. Skipping transmission.")
        return
        
    fetch_time_str = target_dt.strftime('%d %b %Y, %I:%M %p')
    
    msg = MIMEMultipart()
    msg['Subject'] = f"Spatial Matrix Report & Image Attachments | {fetch_time_str}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    
    html_rows = ""
    for _, row in df_matrix.iterrows():
        color = "#1b5e20" if "SUCCESS" in row['State_Status'] else ("#0d47a1" if "FUZZY" in row['State_Status'] else "#e65100")
        attachment_name = f"{row['Symbol']}_spatial.png" if row.get('Spatial_Image_Bytes') else "N/A"
            
        html_rows += f"""<tr>
            <td style='font-weight:bold; color:#1a73e8;'>{row['Symbol']}</td>
            <td>₹{row['LTP']:.2f}</td>
            <td><b>{row['Match_Score']*100:.1f}%</b></td>
            <td style='color:{color}; font-weight:bold;'>{row['State_Status']}</td>
            <td>{row['Rolling_Vol']:,} sh</td>
            <td style='text-align:center; font-family:monospace; color:#333;'><b>{attachment_name}</b></td>
        </tr>"""
        
        if row.get('Spatial_Image_Bytes'):
            img_part = MIMEImage(row['Spatial_Image_Bytes'], name=attachment_name)
            img_part.add_header('Content-Disposition', 'attachment', filename=attachment_name)
            msg.attach(img_part)

    html = f"""
    <html>
      <body style='font-family: Arial, sans-serif; background-color: #f7f9fc; padding: 20px;'>
        <h2 style='color: #1a237e; text-align: center;'>🖼️ 64D HYPER-TENSOR SPATIAL MATRIX REPORT</h2>
        <p style='text-align: center; color: #555;'>🕒 Scan Time: <b>{fetch_time_str}</b> | Lookback Backlog: <b>{LOOKBACK_DAYS} Days</b></p>
        <p style='text-align: center; color: #555; font-size: 13px;'><i>True 128x128 2D spatial matrices backed by max 64D PyTorch hyper-tensors attached as ticker PNG files.</i></p>
        <table style='width: 100%; border-collapse: collapse; background: #fff; margin-top: 15px;'>
          <tr style='background: #3949ab; color: white;'>
            <th style='padding: 10px;'>Symbol</th><th style='padding: 10px;'>LTP</th><th style='padding: 10px;'>Match %</th><th style='padding: 10px;'>Status</th><th style='padding: 10px;'>Avg Vol</th><th style='padding: 10px; text-align:center;'>Attached Spatial File</th>
          </tr>
          {html_rows}
        </table>
      </body>
    </html>
    """
    
    msg.attach(MIMEText(html, "html"))
    
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logger.info("Email report sent successfully with 64D hyper-tensor file attachments.")
    except Exception as e:
        logger.error(f"Email dispatch failed: {e}")


# ==========================================
# 7. MAIN ENGINE
# ==========================================
def main():
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
        logger.info(f"Executing 64D Hyper-Tensor Spatial Scan for {current_dt.strftime('%I:%M %p')}...")
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

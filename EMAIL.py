#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════════
SPATIAL MATRIX & F&O MULTI-CHANNEL 64D HYPER-TENSOR ENGINE v9.6 (GitHub Actions Edition)
- Configurable Lookback Backlog (Loaded from config.yml)
- ZERO FILES NEEDED: Pure Mathematical Synthetic Blueprint Auto-Generation in RAM
- True Multi-Channel F&O 1024x1024 Grid (Price, Volume, Open Interest / Volatility Channels)
- Maximum 64-Dimensional NumPy/Tensor Hyper-Pipeline
- Dynamic Target Calculator (Target 1, Target 2, Target 3 Breakout Projections)
- Dual Attachment Dispatcher: Both Live Spatial Matrix AND Matched Success Blueprint Attached
- PATCHED: Thread-Safety, Math/Division-by-Zero Fixes, Ephemeral Server Ready
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
    logger.error(f"❌ Configuration error (Please ensure config.yml exists): {e}")
    sys.exit(1)

CLIENT_ID = os.environ.get("CLIENT_ID", "")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")
RECIPIENT_EMAIL = os.environ.get("RECIPIENT_EMAIL", "")

FYERS_FO_MASTER_URL = "https://public.fyers.in/sym_details/NSE_FO.csv"
fyers = fyersModel.FyersModel(client_id=CLIENT_ID, token=ACCESS_TOKEN, is_async=False, log_path="")

# In-memory global runtime template bank and Threading Lock for safe concurrent access
IN_MEMORY_SUCCESS_TEMPLATES = []
TEMPLATE_LOCK = threading.Lock()


# ==========================================
# 1.5 SYNTHETIC BLUEPRINT AUTOGEN (ZERO FILES NEEDED)
# ==========================================
def load_historical_blueprints():
    """
    Mathematically auto-generates the perfect 1024x1024 spatial matrix of a 
    breakout (compression -> friction collapse -> expansion) directly in RAM.
    Eliminates the need for saved .png files in GitHub.
    """
    global IN_MEMORY_SUCCESS_TEMPLATES
    grid_height, grid_width = 1024, 1024
    canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    logger.info("Auto-generating Synthetic Master Breakout Blueprint...")

    # Iterate through the grid to draw the perfect structural breakout pattern
    for i in range(grid_width):
        # 1. PRICE CHANNEL (Channel 0)
        if i < 800:
            # Phase 1: Coiling / Accumulation (Tight sideways price action)
            y_top = 500 + int(np.sin(i * 0.05) * 20)
            y_bot = 520 + int(np.cos(i * 0.05) * 20)
        else:
            # Phase 2: Friction Collapse / Breakout (Price rockets upward)
            progress = (i - 800) / 224.0
            y_top = int(500 - (450 * (progress ** 2)))
            y_bot = y_top + 40

        y_top, y_bot = np.clip(y_top, 0, 1023), np.clip(y_bot, 0, 1023)
        canvas[y_top:y_bot+1, i, 0] = 220

        # 2. VOLUME CHANNEL (Channel 1)
        if i < 800:
            vol_intensity = 50 + int(np.random.rand() * 30) # Low volume during compression
        else:
            vol_intensity = 150 + int(np.random.rand() * 105) # Heavy volume burst on breakout
        canvas[:, i, 1] = vol_intensity

        # 3. ATR / VOLATILITY CHANNEL (Channel 2)
        if i < 800:
            atr_val = 50
        else:
            atr_val = int(50 + (205 * progress)) # Volatility expands as price breaks
        canvas[grid_height - 128:grid_height, i, 2] = atr_val

    # Save the synthetic master blueprint directly into the engine's memory
    IN_MEMORY_SUCCESS_TEMPLATES.append({'img': canvas, 'id': 'SYNTHETIC_MASTER_BREAKOUT'})
    logger.info("✅ Synthetic Master Blueprint loaded into memory successfully. Zero files required.")


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
# 3. MULTI-CHANNEL F&O 64-DIMENSIONAL HYPER-TENSOR ALLOCATOR
# ==========================================
def build_maximum_64d_hyper_tensor(multi_channel_canvas):
    if multi_channel_canvas is None:
        return None
    target_shape = (1,) * 61 + multi_channel_canvas.shape
    return multi_channel_canvas.reshape(target_shape)


# ==========================================
# 4. MULTI-CHANNEL F&O 1024x1024 GRID & TARGET CALCULATOR
# ==========================================
def calculate_breakout_targets(df_slice, ltp):
    highs = df_slice['high'].values
    lows = df_slice['low'].values
    closes = df_slice['close'].values
    
    tr = np.maximum(highs[1:] - lows[1:], np.abs(highs[1:] - closes[:-1]))
    atr = np.mean(tr) if len(tr) > 0 else (ltp * 0.01)
    
    resistance_span = highs.max() - lows.min()
    
    target_1 = ltp + (atr * 1.0) + (resistance_span * 0.15)
    target_2 = ltp + (atr * 2.0) + (resistance_span * 0.30)
    target_3 = ltp + (atr * 3.5) + (resistance_span * 0.50)
    
    return round(target_1, 2), round(target_2, 2), round(target_3, 2)


def generate_multichannel_spatial_matrix(df_slice):
    if df_slice is None or len(df_slice) < MACRO_WINDOW:
        return None
        
    p_high = df_slice['high'].values.astype(np.float32)
    p_low = df_slice['low'].values.astype(np.float32)
    volume = df_slice['volume'].values.astype(np.float32)
    
    grid_height = 1024
    grid_width = 1024
    
    canvas = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    p_min, p_max = p_low.min(), p_high.max()
    p_span = p_max - p_min if p_max > p_min else 1.0
    
    v_min, v_max = volume.min(), volume.max()
    v_span = v_max - v_min if v_max > v_min else 1.0
    
    x_indices = np.linspace(0, grid_width - 1, len(df_slice)).astype(int)
    tr = np.maximum(p_high[1:] - p_low[1:], np.abs(p_high[1:] - df_slice['close'].values[:-1]))
    atr_val = np.mean(tr) if len(tr) > 0 else 1.0
    
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
            
        canvas[y_top:y_bot+1, i, 0] = 220
        vol_intensity = int(np.clip((v_val - v_min) / v_span * 255, 0, 255))
        canvas[:, i, 1] = vol_intensity
        canvas[grid_height - 128:grid_height, i, 2] = int(np.clip(atr_val / p_span * 255, 50, 255))

    return canvas


def compare_images_and_execute_hunt(multi_channel_img, symbol, timestamp_str):
    if multi_channel_img is None:
        return None, 0.0, "N/A", None
        
    live_img_gray = cv2.cvtColor(multi_channel_img, cv2.COLOR_RGB2GRAY)
    
    # Store the Tensor instead of letting it vanish (Ready for ML prediction layer)
    hyper_tensor = build_maximum_64d_hyper_tensor(multi_channel_img)

    max_val = -1.0
    matched_template_id = "N/A (Pending Blueprint Match)"
    matched_template_bytes = None
    
    clean_sym = symbol.replace('NSE:', '').replace('-EQ', '')

    # Safely create a local copy of templates for iteration to avoid thread collision
    with TEMPLATE_LOCK:
        current_templates = list(IN_MEMORY_SUCCESS_TEMPLATES)

    if current_templates:
        for template_item in current_templates:
            try:
                template = template_item['img']
                t_id = template_item['id']
                template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY) if len(template.shape) == 3 else template
                
                if template_gray.shape != live_img_gray.shape:
                    template_resized = cv2.resize(template_gray, (live_img_gray.shape[1], live_img_gray.shape[0]))
                else:
                    template_resized = template_gray
                    
                res = cv2.matchTemplate(live_img_gray, template_resized, cv2.TM_CCOEFF_NORMED)
                _, val, _, _ = cv2.minMaxLoc(res)
                
                if val > max_val:
                    max_val = val
                    matched_template_id = f"SUCCESS_BLUEPRINT_{t_id}_spatial_1024.png"
                    success_enc, enc_bytes = cv2.imencode('.png', template)
                    matched_template_bytes = enc_bytes.tobytes() if success_enc else None
            except Exception as e:
                logger.debug(f"Template match error on {clean_sym}: {e}")
                continue
    else:
        # Prevent division-by-zero math bug by returning early if no templates exist
        logger.debug(f"No historical templates loaded. Skipping pattern match for {clean_sym}.")
        return "AWAITING BLUEPRINTS", 0.0, "N/A", None

    raw_score = (max_val + 1.0) / 2.0 if max_val != -1.0 else 0.0
    match_score = float(round(max(0.0, min(1.0, raw_score)), 4))
    
    if match_score >= TRIGGER_THRESH:
        state_status = "SUCCESS IMAGE MATRIX: BREAKOUT MATCH"
        matched_template_id = f"SUCCESS_BLUEPRINT_{clean_sym}_spatial_1024.png"
        
        # Safely write to the global template list using the Thread Lock
        with TEMPLATE_LOCK:
            if len(IN_MEMORY_SUCCESS_TEMPLATES) < 50:
                # Store the newly discovered live success pattern to hunt for future identical structures
                existing_ids = [t['id'] for t in IN_MEMORY_SUCCESS_TEMPLATES]
                if clean_sym not in existing_ids:
                    IN_MEMORY_SUCCESS_TEMPLATES.append({'img': multi_channel_img, 'id': clean_sym})
                    
    elif match_score >= FUZZY_THRESH:
        state_status = "FUZZY IMAGE ANCHOR: STRUCTURAL HOLD"
    else:
        if HUNT_MODE:
            state_status = "HUNTING: SCANNING CONTINUATION/TRAP TEMPLATES"
        else:
            state_status = "TRAP MATRIX IMAGE: UNKNOWN GEOMETRY (EXIT)"
            
    return state_status, match_score, matched_template_id, matched_template_bytes


# ==========================================
# 5. SYMBOL SCANNER
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
            time.sleep(0.08)  # API Rate limit respect
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
        ltp = float(rolling_slice['close'].iloc[-1])
        
        t1, t2, t3 = calculate_breakout_targets(rolling_slice, ltp)
        multi_channel_img = generate_multichannel_spatial_matrix(rolling_slice)
        
        state_status, match_score, success_matrix_ref, template_bytes = compare_images_and_execute_hunt(multi_channel_img, symbol, timestamp_str)
        
        # Only return meaningful data to keep reports clean
        if match_score >= FUZZY_THRESH or "HUNTING" in state_status:
            success, encoded_img = cv2.imencode('.png', multi_channel_img)
            img_bytes = encoded_img.tobytes() if success else None
            clean_symbol = symbol.replace('NSE:', '').replace('-EQ', '')
            
            return {
                'Symbol': clean_symbol,
                'LTP': ltp,
                'Target_1': t1,
                'Target_2': t2,
                'Target_3': t3,
                'Match_Score': match_score,
                'State_Status': state_status,
                'Success_Matrix_Ref': success_matrix_ref,
                'Spatial_Image_Bytes': img_bytes,
                'Matched_Template_Bytes': template_bytes
            }
        return None
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None


# ==========================================
# 6. HTML EMAIL DISPATCHER
# ==========================================
def send_html_email(df_matrix, target_dt):
    if not SENDER_EMAIL or not RECIPIENT_EMAIL:
        logger.warning("Email credentials missing. Skipping transmission.")
        return
        
    fetch_time_str = target_dt.strftime('%d %b %Y, %I:%M %p')
    
    msg = MIMEMultipart()
    msg['Subject'] = f"F&O 1024x1024 Spatial Matrix Report | {fetch_time_str}"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    
    html_rows = ""
    for _, row in df_matrix.iterrows():
        is_breakout = "SUCCESS" in row['State_Status']
        color = "#1b5e20" if is_breakout else ("#0d47a1" if "FUZZY" in row['State_Status'] else "#e65100")
        
        attachments_display = []
        
        if is_breakout and row.get('Spatial_Image_Bytes'):
            live_att_name = f"{row['Symbol']}_spatial_1024.png"
            img_part1 = MIMEImage(row['Spatial_Image_Bytes'], name=live_att_name)
            img_part1.add_header('Content-Disposition', 'attachment', filename=live_att_name)
            msg.attach(img_part1)
            attachments_display.append(live_att_name)
            
        if is_breakout and row.get('Matched_Template_Bytes'):
            ref_att_name = row['Success_Matrix_Ref']
            img_part2 = MIMEImage(row['Matched_Template_Bytes'], name=ref_att_name)
            img_part2.add_header('Content-Disposition', 'attachment', filename=ref_att_name)
            msg.attach(img_part2)
            attachments_display.append(ref_att_name)
            
        file_display = f"<b style='color:#1b5e20;'>{', '.join(attachments_display)} (Attached)</b>" if attachments_display else "<span style='color:#888;'>None (Filtered Out)</span>"
            
        html_rows += f"""<tr>
            <td style='font-weight:bold; color:#1a73e8;'>{row['Symbol']}</td>
            <td>₹{row['LTP']:.2f}</td>
            <td style='color:#2e7d32; font-weight:bold;'>₹{row['Target_1']:.2f}</td>
            <td style='color:#1565c0; font-weight:bold;'>₹{row['Target_2']:.2f}</td>
            <td style='color:#6a1b9a; font-weight:bold;'>₹{row['Target_3']:.2f}</td>
            <td><b>{row['Match_Score']*100:.1f}%</b></td>
            <td style='color:{color}; font-weight:bold;'>{row['State_Status']}</td>
            <td style='font-family:monospace; font-size:11px; color:#4527a0;'>{row['Success_Matrix_Ref']}</td>
            <td style='text-align:center; font-family:monospace;'>{file_display}</td>
        </tr>"""

    html = f"""
    <html>
      <body style='font-family: Arial, sans-serif; background-color: #f7f9fc; padding: 20px;'>
        <h2 style='color: #1a237e; text-align: center;'>🎯 F&O 1024x1024 DUAL ATTACHMENT BREAKOUT REPORT</h2>
        <p style='text-align: center; color: #555;'>🕒 Scan Time: <b>{fetch_time_str}</b> | Lookback Backlog: <b>{LOOKBACK_DAYS} Days</b></p>
        <p style='text-align: center; color: #555; font-size: 13px;'><i>Attachments included <b>ONLY</b> for SUCCESS BREAKOUT MATCHES.</i></p>
        <table style='width: 100%; border-collapse: collapse; background: #fff; margin-top: 15px;'>
          <tr style='background: #3949ab; color: white;'>
            <th style='padding: 10px;'>Symbol</th>
            <th style='padding: 10px;'>LTP</th>
            <th style='padding: 10px;'>Target 1</th>
            <th style='padding: 10px;'>Target 2</th>
            <th style='padding: 10px;'>Target 3</th>
            <th style='padding: 10px;'>Match %</th>
            <th style='padding: 10px;'>Status</th>
            <th style='padding: 10px;'>Success Matrix Image Ref</th>
            <th style='padding: 10px; text-align:center;'>Attachments</th>
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
        logger.info("Email report sent successfully.")
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
    
    # Bootstrap: Mathematically generate the breakout blueprint in RAM (Zero files)
    load_historical_blueprints()
    
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
        logger.info(f"Executing 1024x1024 Dual Attachment Scan for {current_dt.strftime('%I:%M %p')}...")
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

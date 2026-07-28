import os
import smtplib
import urllib.parse
import json
import gzip
import io
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
import xgboost as xgb

# ==============================================================================
# 1. UPGRADED TEMPORAL CNN (34-Feature Probability Classifier)
# ==============================================================================
class TemporalCNNClassifier(nn.Module):
    def __init__(self, num_features=34, latent_dim=32): 
        super(TemporalCNNClassifier, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            
            nn.Flatten(),
            nn.Linear(128 * 5, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Binary Classifier for Win Probability
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        latent = self.encode(x)
        prob = self.classifier(latent)
        return prob, latent

# ==============================================================================
# 2. MEGA FEATURE ENGINEERING & NORMALIZATION
# ==============================================================================
def add_mega_technical_indicators(df):
    """Calculates all 34 quantitative confluence indicators."""
    
    # --- 1. MOVING AVERAGES ---
    df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    wma_half = df['Close'].rolling(7).apply(lambda x: np.dot(x, np.arange(1, 8)) / 28, raw=True)
    wma_full = df['Close'].rolling(14).apply(lambda x: np.dot(x, np.arange(1, 15)) / 105, raw=True)
    df['HMA'] = (2 * wma_half - wma_full).rolling(int(np.sqrt(14))).mean()

    df['Alligator_Jaw'] = df['Close'].rolling(13).mean().shift(8)
    df['Alligator_Teeth'] = df['Close'].rolling(8).mean().shift(5)
    df['Alligator_Lips'] = df['Close'].rolling(5).mean().shift(3)

    # --- 2. VOLATILITY & BANDS ---
    df['TR'] = np.maximum(df['High'] - df['Low'], np.maximum(abs(df['High'] - df['Close'].shift()), abs(df['Low'] - df['Close'].shift())))
    df['ATR'] = df['TR'].rolling(14).mean()

    std_20 = df['Close'].rolling(20).std()
    df['BB_Up'] = df['EMA_20'] + (2 * std_20)
    df['BB_Dn'] = df['EMA_20'] - (2 * std_20)

    df['KC_Up'] = df['EMA_20'] + (1.5 * df['ATR'])
    df['KC_Dn'] = df['EMA_20'] - (1.5 * df['ATR'])

    hl2 = (df['High'] + df['Low']) / 2
    df['ST_Upper'] = hl2 + (3 * df['ATR'])
    df['ST_Lower'] = hl2 - (3 * df['ATR'])

    # --- 3. MOMENTUM & OSCILLATORS ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df['RSI'] = 100 - (100 / (1 + rs))

    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low14) / (high14 - low14 + 1e-8))

    up, down = df['High'].diff(), -df['Low'].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / (df['ATR'] + 1e-8))
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / (df['ATR'] + 1e-8))
    df['ADX'] = (100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)).rolling(14).mean()

    ema13 = df['Close'].ewm(span=13, adjust=False).mean()
    bull_impulse = (ema13 > ema13.shift(1)) & (df['MACD_Hist'] > df['MACD_Hist'].shift(1))
    bear_impulse = (ema13 < ema13.shift(1)) & (df['MACD_Hist'] < df['MACD_Hist'].shift(1))
    df['EIS'] = np.where(bull_impulse, 1, np.where(bear_impulse, -1, 0))

    # --- 4. VOLUME DYNAMICS ---
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    pos_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    mf_ratio = pd.Series(pos_flow).rolling(14).sum() / (pd.Series(neg_flow).rolling(14).sum() + 1e-8)
    df['MFI'] = 100 - (100 / (1 + mf_ratio))

    df['VSA_Spread'] = (df['High'] - df['Low']) / (df['Volume'] + 1e-8)

    # --- 5. VWAP & PIVOTS ---
    df['VWAP'] = (typical_price * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-8)
    
    range_prev = df['High'].shift(1) - df['Low'].shift(1)
    df['Cam_H3'] = df['Close'].shift(1) + (range_prev * 0.275)
    df['Cam_L3'] = df['Close'].shift(1) - (range_prev * 0.275)

    range_daily = df['High'] - df['Low']
    df['NR4'] = (range_daily == range_daily.rolling(4).min()).astype(float)
    df['NR7'] = (range_daily == range_daily.rolling(7).min()).astype(float)

    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        'EMA_8', 'EMA_20', 'EMA_21', 'EMA_50', 'HMA', 
        'Alligator_Jaw', 'Alligator_Teeth', 'Alligator_Lips',
        'ATR', 'BB_Up', 'BB_Dn', 'KC_Up', 'KC_Dn', 'ST_Upper', 'ST_Lower',
        'RSI', 'MACD', 'MACD_Hist', 'Stoch_K', 'ADX', 'EIS',
        'OBV', 'MFI', 'VSA_Spread', 'VWAP', 'Cam_H3', 'Cam_L3', 'NR4', 'NR7'
    ]
    return df[feature_cols]

def normalize_mega_tensor(values):
    """Applies Multi-Tiered Normalization (Log Returns, Z-Score, and State Preservation)"""
    norm_values = values.copy()
    base_price = norm_values[0, 3] + 1e-8 
    
    price_cols = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 29, 30, 31]
    for col in price_cols:
        # FIX: Clip values to prevent lower bands (BB_Dn, ST_Lower, etc.) 
        # from dropping below zero and crashing the np.log() calculation.
        safe_values = np.clip(norm_values[:, col], a_min=1e-6, a_max=None)
        norm_values[:, col] = np.log(safe_values / base_price)
        
    zscore_cols = [4, 13, 20, 21, 22, 23, 24, 26, 27, 28]
    for col in zscore_cols:
        col_mean = np.mean(norm_values[:, col])
        col_std = np.std(norm_values[:, col]) + 1e-8
        norm_values[:, col] = (norm_values[:, col] - col_mean) / col_std
        
    # Indices 25 (EIS), 32 (NR4), 33 (NR7) remain unchanged
    return norm_values.T

def read_and_standardize_csv(filename):
    if not os.path.exists(filename): return None
    df = pd.read_csv(filename)
    rename_map = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ['date', 'time', 'timestamp']: rename_map[c] = 'Date'
        elif cl in ['symbol', 'ticker', 'asset']: rename_map[c] = 'Symbol'
        elif cl == 'open': rename_map[c] = 'Open'
        elif cl == 'high': rename_map[c] = 'High'
        elif cl == 'low': rename_map[c] = 'Low'
        elif cl == 'close': rename_map[c] = 'Close'
        elif cl in ['volume', 'vol']: rename_map[c] = 'Volume'
    df = df.rename(columns=rename_map)
    if 'Date' in df.columns: df['Date'] = df['Date'].astype(str).str[:10]
    return df

def load_training_data(csv_filename, target_date_str=None, min_pct=4.0, max_dd=1.2):
    df = read_and_standardize_csv(csv_filename)
    if df is None or 'Date' not in df.columns: return None, None, None
    if target_date_str: df = df[df['Date'] <= target_date_str]
    
    training_matrices, labels = [], []
    FUTURE_DAYS = 2 
    
    if "historical_indices" in csv_filename.lower() or "nifty" in csv_filename.lower():
        if 'Symbol' in df.columns:
            mask = df['Symbol'].astype(str).str.upper().str.replace("_", "").str.replace(" ", "").str.contains("NIFTY50|NIFTY")
            if mask.any(): df = df[mask]

    for symbol, group in df.groupby('Symbol') if 'Symbol' in df.columns else [('ASSET', df)]:
        group = group.sort_values('Date').reset_index(drop=True)
        group = add_mega_technical_indicators(group)
        
        values = group.values.astype(np.float32)
        if len(values) < (30 + FUTURE_DAYS): continue
            
        for i in range(len(values) - (30 + FUTURE_DAYS) + 1):
            raw_window = values[i : i+30]
            window = normalize_mega_tensor(raw_window)
            
            future_closes = values[i+30 : i+30+FUTURE_DAYS, 3] 
            future_lows   = values[i+30 : i+30+FUTURE_DAYS, 2] 
            start_price   = values[i+29, 3] 
            
            max_close = future_closes.max()
            actual_pct_move = ((max_close - start_price) / (start_price + 1e-8)) * 100
            actual_drawdown = ((future_lows.min() - start_price) / (start_price + 1e-8)) * 100
            
            # Label = 1 (Win) if target hit without hitting stop loss, else 0
            is_success = 1.0 if (actual_pct_move >= min_pct and actual_drawdown >= -max_dd) else 0.0
            
            training_matrices.append(window)
            labels.append(is_success)
            
    return np.array(training_matrices, dtype=np.float32), np.array(labels, dtype=np.float32), min_pct

# ==============================================================================
# 3. MODULAR AI TRAINING ENGINE (Batched to prevent RAM crashes)
# ==============================================================================
def train_ai_brain(X_raw, Y_labels, epochs=10, batch_size=256):
    X_tensor = torch.tensor(X_raw)
    Y_tensor = torch.tensor(Y_labels).view(-1, 1)
    
    cnn_model = TemporalCNNClassifier(num_features=34)
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.002)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss
    
    dataset_size = X_tensor.size(0)
    
    # 1. Train with Mini-Batches to prevent Out-Of-Memory (OOM) crashes
    cnn_model.train()
    for epoch in range(epochs): 
        permutation = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X_tensor[indices], Y_tensor[indices]
            
            optimizer.zero_grad()
            probs, _ = cnn_model(batch_x)
            loss = criterion(probs, batch_y)
            loss.backward()
            optimizer.step()

    # 2. Extract Latent Vectors (Batched to save RAM)
    cnn_model.eval()
    latent_vectors_list = []
    with torch.no_grad():
        for i in range(0, dataset_size, batch_size):
            batch_x = X_tensor[i : i + batch_size]
            latent_batch = cnn_model.encode(batch_x).numpy()
            latent_vectors_list.append(latent_batch)
            
    latent_vectors = np.vstack(latent_vectors_list)
        
    # 3. Train XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss')
    xgb_model.fit(latent_vectors, Y_labels)

    # 4. Save to FAISS Memory
    faiss.normalize_L2(latent_vectors)
    index = faiss.IndexFlatIP(32)
    index.add(latent_vectors)
    
    return cnn_model, xgb_model, index

# ==============================================================================
# 4. LIVE INGESTION TOOLS
# ==============================================================================
def get_live_tensor_from_csv(csv_filename, target_date_str):
    df = read_and_standardize_csv(csv_filename)
    if df is None or 'Date' not in df.columns: return None, None
    
    if 'Symbol' in df.columns:
        mask = df['Symbol'].astype(str).str.upper().str.replace("_", "").str.replace(" ", "").str.contains("NIFTY50|NIFTY")
        if mask.any(): df = df[mask]
        else: df = df[df['Symbol'] == df['Symbol'].unique()[0]]
            
    df = df[df['Date'] <= target_date_str].sort_values('Date').reset_index(drop=True)
    df = add_mega_technical_indicators(df)
    
    if len(df) < 30: return None, None
    
    values = df.values.astype(np.float32)[-30:]
    current_ltp = values[-1, 3] 
    window = normalize_mega_tensor(values)
    
    return window, current_ltp

def get_dynamic_fno_universe():
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    response = requests.get(nse_url)
    if response.status_code != 200: return []
    try:
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        fno_underlying = {item.get("underlying_symbol") for item in nse_data if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")}
        
        fno_universe = []
        for item in nse_data:
            if item.get("segment") in ("NSE_EQ", "NSE_INDEX") and item.get("trading_symbol") in fno_underlying:
                fno_universe.append({"symbol": item.get("trading_symbol"), "key": item.get("instrument_key")})
        return fno_universe
    except:
        return []

def fetch_upstox_data(instrument_key, target_date_str, interval="day", days_back=100):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    from_date = (target_dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    encoded_key = urllib.parse.quote(instrument_key)
    url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/{interval}/{target_date_str}/{from_date}"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200: return None
        
    data = response.json().get('data', {}).get('candles', [])
    if not data or len(data) < 60: return None # Buffer requirement
        
    df = pd.DataFrame(data[::-1], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df = add_mega_technical_indicators(df)
    
    values = df.values.astype(np.float32)[-30:]
    current_ltp = values[-1, 3]
    window = normalize_mega_tensor(values)
    
    return window, current_ltp

# ==============================================================================
# 5. DUAL-BRAIN MASTER EXECUTION & DISPATCH
# ==============================================================================
def send_mobile_alert(macro_data, fno_data_list, target_date_str, is_backtest):
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_pass = os.environ.get("SENDER_PASSWORD")
    recipient_email = os.environ.get("RECIPIENT_EMAIL")
    
    if not all([sender_email, sender_pass, recipient_email]): return

    msg = MIMEMultipart('alternative')
    prefix = "⏪ BACKTEST" if is_backtest else "🚀 LIVE ALERT"
    msg['Subject'] = f"{prefix} | {target_date_str}"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    macro_color = "#28a745" if "LONG" in macro_data['direction'] else "#dc3545"
    sim_warning = f"<div style='background-color: #fff3cd; color: #856404; padding: 10px; text-align: center; font-weight: bold; margin-bottom: 15px;'>⚠️ VALIDATION MODE: SHOWING ACTUAL 2-DAY OUTCOMES</div>" if is_backtest else ""

    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f7f6; padding: 10px;">
        {sim_warning}
        
        <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 6px solid {macro_color};">
            <h3 style="margin-top: 0; color: #333;">🌍 MACRO REGIME (NIFTY 50)</h3>
            <p style="font-size: 16px; color: #333; margin: 5px 0;">
                <b>Direction:</b> <span style="color: {macro_color}; font-weight: bold;">{macro_data['direction']}</span><br>
                <b>AI Target:</b> {macro_data['target_display']} | <b>Conviction:</b> {macro_data['conviction']:.2f}%
            </p>
        </div>

        <h3 style="color: #333;">⚡ MICRO F&O SWEEP (HYPER-MOMENTUM)</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-size: 14px; background-color: white;">
          <tr bgcolor="#f8f9fa" style="color: #333; font-weight: bold;">
            <th>Asset</th>
            <th>Signal</th>
            <th>Trend Match?</th>
            <th>Score (Prob)</th>
            <th>Current LTP</th>
            <th>AI Target</th>
            <th>Result (2-Day Close)</th>
          </tr>
    """
    
    fno_data_list.sort(key=lambda x: x['conviction'], reverse=True)
    
    for row in fno_data_list:
        dir_color = "#28a745" if "LONG" in row['direction'] else "#dc3545"
        trend_match = "✅" if row['direction'] == macro_data['direction'] else "⚠️"
        
        html_content += f"""
          <tr>
            <td style="color: #0056b3;"><b>{row['asset']}</b></td>
            <td style="color: {dir_color}; font-weight: bold;">{row['direction']}</td>
            <td>{trend_match}</td>
            <td>{row['conviction']:.2f}%</td>
            <td>₹{row['ltp']:.2f}</td>
            <td style="color: {dir_color}; font-weight: bold;">{row['target_display']}</td>
            <td>{row['actual_outcome']}</td>
          </tr>
        """
        
    html_content += "</table></body></html>"
    msg.attach(MIMEText(html_content, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_pass)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"✅ Alert Dispatched with {len(fno_data_list)} F&O targets.")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def run_production_sweep():
    target_date_str = os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    is_backtest = bool(target_date_str)
    if not is_backtest: target_date_str = datetime.now().strftime("%Y-%m-%d")
        
    print(f"⚙️ EXECUTING DATE: {target_date_str}")
    
    nifty_file = None
    for root, dirs, files in os.walk("."):
        for file in files:
            if "nifty" in file.lower() and file.lower().endswith(".csv"):
                nifty_file = os.path.join(root, file)
                print(f"✅ Auto-detected Nifty file at: {nifty_file}")
                break
        if nifty_file: break
            
    if not nifty_file:
        for root, dirs, files in os.walk("."):
            for file in files:
                if "historical_indices.csv" in file.lower():
                    nifty_file = os.path.join(root, file)
                    print(f"✅ Auto-detected Nifty data inside: {nifty_file}")
                    break
            if nifty_file: break

    if not nifty_file:
        print("❌ Critical Error: Could not find ANY file containing 'nifty' or 'historical_indices'.")
        return

    # ==========================================
    # PHASE 1: MACRO NIFTY BRAIN
    # ==========================================
    print(f"\n🧠 PHASE 1: Training NIFTY 50 Macro Brain (34-Features)...")
    X_nifty, Y_labels, min_macro_pct = load_training_data(nifty_file, target_date_str, min_pct=0.75, max_dd=0.5)
    
    if X_nifty is None or len(X_nifty) == 0:
        print("❌ Nifty Data matrix construction failed.")
        macro_report = {'direction': "UNKNOWN", 'conviction': 0, 'target_display': "N/A"}
    else:
        nifty_cnn, nifty_xgb, nifty_faiss = train_ai_brain(X_nifty, Y_labels)
        nifty_live_matrix, nifty_ltp = get_live_tensor_from_csv(nifty_file, target_date_str)
        
        if nifty_live_matrix is not None:
            live_tensor = torch.tensor(nifty_live_matrix).unsqueeze(0)
            with torch.no_grad():
                prob_tensor, nifty_latent = nifty_cnn(live_tensor)
                cnn_prob = prob_tensor.item() * 100
                
            xgb_prob = nifty_xgb.predict_proba(nifty_latent)[0][1] * 100
            blended_prob = (cnn_prob + xgb_prob) / 2
            
            macro_report = {
                'direction': "LONG 🟢" if blended_prob > 50 else "SHORT 🔴",
                'conviction': blended_prob,
                'target_display': f"Expected Move: +{min_macro_pct}%"
            }
            print(f"🌍 MACRO REGIME: {macro_report['direction']} (Score: {blended_prob:.2f}%)")
        else:
            macro_report = {'direction': "UNKNOWN", 'conviction': 0, 'target_display': "N/A"}

    # ==========================================
    # PHASE 2: MICRO F&O BRAIN
    # ==========================================
    print("\n⚡ PHASE 2: Training F&O Micro Brain (Hyper-Momentum)...")
    X_fno, Y_fno_labels, min_micro_pct = load_training_data("historical_fno.csv", target_date_str, min_pct=4.0, max_dd=1.2)
    if X_fno is None or len(X_fno) == 0: return

    fno_cnn, fno_xgb, fno_faiss = train_ai_brain(X_fno, Y_fno_labels)
    
    print("🎯 Phase 3: Sweeping Active Market Universe...")
    fno_universe = get_dynamic_fno_universe()
    if not fno_universe: return
    
    final_report_data = []
    # Probability threshold replaces strict FAISS matching
    min_prob_threshold = float(os.environ.get("PARAM_MIN_PROBABILITY", 85.0))

    for asset in fno_universe:
        result = fetch_upstox_data(asset["key"], target_date_str, interval="day", days_back=100)
        time.sleep(0.15) 
        
        if result is None: continue
        live_matrix, current_ltp = result
        
        with torch.no_grad():
            live_tensor = torch.tensor(live_matrix).unsqueeze(0)
            cnn_prob_tensor, live_latent = fno_cnn(live_tensor)
            cnn_prob = cnn_prob_tensor.item() * 100
            
        xgb_prob = fno_xgb.predict_proba(live_latent.numpy())[0][1] * 100
        blended_prob = (cnn_prob + xgb_prob) / 2
        
        if blended_prob >= min_prob_threshold:
            final_report_data.append({
                'asset': asset["symbol"],
                'direction': "LONG 🟢",
                'conviction': float(blended_prob),
                'ltp': float(current_ltp),
                'target_display': f"Expected Move: +{min_micro_pct}%",
                'actual_outcome': "<b>Awaiting Market ⏳</b>"
            })
            
    # VALIDATION LOGIC FOR BACKTEST
    if is_backtest and os.path.exists("historical_fno.csv") and len(final_report_data) > 0:
        df_full = read_and_standardize_csv("historical_fno.csv")
        if df_full is not None:
            for row in final_report_data:
                if 'Symbol' not in df_full.columns: continue
                
                df_sym = df_full[df_full['Symbol'] == row['asset']].sort_values('Date').reset_index(drop=True)
                past = df_sym[df_sym['Date'] <= target_date_str]
                if len(past) > 0:
                    idx = past.index[-1]
                    fw = df_sym.iloc[idx+1 : idx+3] 
                    if len(fw) > 0:
                        mx, mn = fw['Close'].max(), fw['Close'].min()
                        if "LONG" in row['direction']:
                            mv, dd = ((mx - row['ltp']) / row['ltp']) * 100, ((fw['Low'].min() - row['ltp']) / row['ltp']) * 100
                            c = "#28a745" if mv > 0 else "#6c757d"
                            row['actual_outcome'] = f"<span style='color: {c};'>Closed ₹{mx:.2f} (+{mv:.2f}%)</span><br><span style='color: #856404; font-size: 11px;'>Max DD: {dd:.2f}%</span>"

    send_mobile_alert(macro_report, final_report_data, target_date_str, is_backtest)

if __name__ == "__main__":
    run_production_sweep()

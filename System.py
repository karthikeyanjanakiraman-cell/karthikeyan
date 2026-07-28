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
# 1. UPGRADED TEMPORAL CNN (Multi-Class 3-State Classifier)
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
        
        # Output strictly maps to 3 classes: 0=Neutral, 1=Long, 2=Short
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3) 
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        latent = self.encode(x)
        logits = self.classifier(latent)
        return logits, latent

# ==============================================================================
# 2. MEGA FEATURE ENGINEERING & NORMALIZATION
# ==============================================================================
def add_mega_technical_indicators(df):
    """Calculates all 34 indicators. Lookahead bias eliminated by dropping NaNs."""
    
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

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    pos_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    mf_ratio = pd.Series(pos_flow).rolling(14).sum() / (pd.Series(neg_flow).rolling(14).sum() + 1e-8)
    df['MFI'] = 100 - (100 / (1 + mf_ratio))

    df['VSA_Spread'] = (df['High'] - df['Low']) / (df['Volume'] + 1e-8)

    df['VWAP'] = (typical_price * df['Volume']).cumsum() / (df['Volume'].cumsum() + 1e-8)
    
    range_prev = df['High'].shift(1) - df['Low'].shift(1)
    df['Cam_H3'] = df['Close'].shift(1) + (range_prev * 0.275)
    df['Cam_L3'] = df['Close'].shift(1) - (range_prev * 0.275)

    range_daily = df['High'] - df['Low']
    df['NR4'] = (range_daily == range_daily.rolling(4).min()).astype(float)
    df['NR7'] = (range_daily == range_daily.rolling(7).min()).astype(float)

    # NO BACKFILL. Drop rows with NaNs to completely kill lookahead bias.
    df.dropna(inplace=True)
    
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
    """Upgraded to Percentage Deviation instead of Log Returns to avoid math crashes on negative bands."""
    norm_values = values.copy()
    base_price = norm_values[-1, 3] + 1e-8 # Scale relative to the Day 30 Close
    
    price_cols = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 29, 30, 31]
    for col in price_cols:
        norm_values[:, col] = (norm_values[:, col] - base_price) / base_price
        
    zscore_cols = [4, 13, 20, 21, 22, 23, 24, 26, 27, 28]
    for col in zscore_cols:
        col_mean = np.mean(norm_values[:, col])
        col_std = np.std(norm_values[:, col]) + 1e-8
        norm_values[:, col] = (norm_values[:, col] - col_mean) / col_std
        
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
        group = add_mega_technical_indicators(group) # NaNs drop here
        
        values = group.values.astype(np.float32)
        if len(values) < (30 + FUTURE_DAYS): continue
            
        for i in range(len(values) - (30 + FUTURE_DAYS) + 1):
            raw_window = values[i : i+30]
            window = normalize_mega_tensor(raw_window)
            
            future_slice = values[i+30 : i+30+FUTURE_DAYS] 
            exec_price = future_slice[0, 0] # Real-world logic: Enter at Day 1 Open
            
            long_tp = exec_price * (1 + (min_pct / 100))
            long_sl = exec_price * (1 - (max_dd / 100))
            short_tp = exec_price * (1 - (min_pct / 100))
            short_sl = exec_price * (1 + (max_dd / 100))
            
            label = 0 # Default Neutral
            
            # Chronological Execution Loop
            for day_idx in range(FUTURE_DAYS):
                day_high = future_slice[day_idx, 1]
                day_low  = future_slice[day_idx, 2]
                
                # Check for Whipsaws first (Pessimistic labeling)
                if (day_high >= long_tp) and (day_low <= long_sl):
                    label = 0
                    break
                if (day_low <= short_tp) and (day_high >= short_sl):
                    label = 0
                    break
                    
                # Clean Long Win
                if day_high >= long_tp and day_low > long_sl:
                    label = 1
                    break
                    
                # Clean Short Win
                if day_low <= short_tp and day_high < short_sl:
                    label = 2
                    break
                    
                # Stop Loss Hit
                if day_low <= long_sl or day_high >= short_sl:
                    label = 0
                    break
            
            training_matrices.append(window)
            labels.append(label)
            
    return np.array(training_matrices, dtype=np.float32), np.array(labels, dtype=np.int64), min_pct

# ==============================================================================
# 3. MODULAR AI TRAINING ENGINE (Batched Multi-Class)
# ==============================================================================
def train_ai_brain(X_raw, Y_labels, epochs=10, batch_size=256):
    X_tensor = torch.tensor(X_raw)
    Y_tensor = torch.tensor(Y_labels, dtype=torch.long)
    
    cnn_model = TemporalCNNClassifier(num_features=34)
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()
    
    dataset_size = X_tensor.size(0)
    
    cnn_model.train()
    for epoch in range(epochs): 
        permutation = torch.randperm(dataset_size)
        
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X_tensor[indices], Y_tensor[indices]
            
            optimizer.zero_grad()
            logits, _ = cnn_model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    cnn_model.eval()
    latent_vectors_list = []
    with torch.no_grad():
        for i in range(0, dataset_size, batch_size):
            batch_x = X_tensor[i : i + batch_size]
            latent_batch = cnn_model.encode(batch_x).numpy()
            latent_vectors_list.append(latent_batch)
            
    latent_vectors = np.vstack(latent_vectors_list)
        
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=4, 
        objective='multi:softprob', 
        num_class=3, 
        eval_metric='mlogloss'
    )
    xgb_model.fit(latent_vectors, Y_labels)
    
    return cnn_model, xgb_model

def extract_xgb_probs(xgb_model, latent_array):
    """Safely extracts probabilities avoiding crashes if a class was missing from training data."""
    raw_probs = xgb_model.predict_proba(latent_array)[0]
    classes = xgb_model.classes_
    prob_dict = {0: 0.0, 1: 0.0, 2: 0.0}
    
    for cls_val, prob in zip(classes, raw_probs):
        prob_dict[cls_val] = prob
        
    return prob_dict[0], prob_dict[1], prob_dict[2]

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

def fetch_upstox_data(instrument_key, target_date_str, interval="day", days_back=150): # Increased to survive dropna buffer
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    from_date = (target_dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    encoded_key = urllib.parse.quote(instrument_key)
    url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/{interval}/{target_date_str}/{from_date}"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200: return None
        
    data = response.json().get('data', {}).get('candles', [])
    if not data or len(data) < 80: return None 
        
    df = pd.DataFrame(data[::-1], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    df = add_mega_technical_indicators(df)
    
    if len(df) < 30: return None
    
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

    # HTML Color logic updated to handle NEUTRAL visually
    if "LONG" in macro_data['direction']: macro_color = "#28a745"
    elif "SHORT" in macro_data['direction']: macro_color = "#dc3545"
    else: macro_color = "#6c757d" # Gray for Neutral

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

        <h3 style="color: #333;">⚡ MICRO F&O SWEEP (LONG & SHORT MULTI-CLASS)</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-size: 14px; background-color: white;">
          <tr bgcolor="#f8f9fa" style="color: #333; font-weight: bold;">
            <th>Asset</th>
            <th>Signal</th>
            <th>Trend Match?</th>
            <th>Score (Prob)</th>
            <th>Current LTP</th>
            <th>AI Target</th>
            <th>Result (Next Open -> 2-Day Outcome)</th>
          </tr>
    """
    
    fno_data_list.sort(key=lambda x: x['conviction'], reverse=True)
    
    for row in fno_data_list:
        dir_color = "#28a745" if "LONG" in row['direction'] else "#dc3545"
        trend_match = "✅" if macro_data['direction'] in row['direction'] else ("⚠️" if "NEUTRAL" in macro_data['direction'] else "❌")
        
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
    print(f"\n🧠 PHASE 1: Training NIFTY 50 Macro Brain (Multi-Class)...")
    # Reduced Nifty target expectations since it's an index, not a stock
    X_nifty, Y_labels, min_macro_pct = load_training_data(nifty_file, target_date_str, min_pct=0.75, max_dd=0.5)
    
    if X_nifty is None or len(X_nifty) == 0:
        print("❌ Nifty Data matrix construction failed.")
        macro_report = {'direction': "UNKNOWN", 'conviction': 0, 'target_display': "N/A"}
    else:
        nifty_cnn, nifty_xgb = train_ai_brain(X_nifty, Y_labels)
        nifty_live_matrix, nifty_ltp = get_live_tensor_from_csv(nifty_file, target_date_str)
        
        if nifty_live_matrix is not None:
            live_tensor = torch.tensor(nifty_live_matrix).unsqueeze(0)
            with torch.no_grad():
                logits, nifty_latent = nifty_cnn(live_tensor)
                cnn_probs = torch.softmax(logits, dim=1).numpy()[0]
                
            xgb_n0, xgb_n1, xgb_n2 = extract_xgb_probs(nifty_xgb, nifty_latent.numpy())
            
            blended_neutral = ((cnn_probs[0] + xgb_n0) / 2) * 100
            blended_long = ((cnn_probs[1] + xgb_n1) / 2) * 100
            blended_short = ((cnn_probs[2] + xgb_n2) / 2) * 100
            
            probs_array = [blended_neutral, blended_long, blended_short]
            winner_idx = np.argmax(probs_array)
            best_macro_prob = probs_array[winner_idx]
            
            if winner_idx == 1:
                macro_dir, macro_tgt = "LONG 🟢", f"Expected Move: +{min_macro_pct}%"
            elif winner_idx == 2:
                macro_dir, macro_tgt = "SHORT 🔴", f"Expected Drop: -{min_macro_pct}%"
            else:
                macro_dir, macro_tgt = "NEUTRAL ⚪", "Choppy / Sideways"
                
            macro_report = {
                'direction': macro_dir,
                'conviction': best_macro_prob,
                'target_display': macro_tgt
            }
            print(f"🌍 MACRO REGIME: {macro_report['direction']} (Score: {best_macro_prob:.2f}%)")
        else:
            macro_report = {'direction': "UNKNOWN", 'conviction': 0, 'target_display': "N/A"}

    # ==========================================
    # PHASE 2: MICRO F&O BRAIN
    # ==========================================
    print("\n⚡ PHASE 2: Training F&O Micro Brain (Multi-Class)...")
    X_fno, Y_fno_labels, min_micro_pct = load_training_data("historical_fno.csv", target_date_str, min_pct=4.0, max_dd=1.2)
    if X_fno is None or len(X_fno) == 0: return

    fno_cnn, fno_xgb = train_ai_brain(X_fno, Y_fno_labels)
    
    print("🎯 Phase 3: Sweeping Active Market Universe...")
    fno_universe = get_dynamic_fno_universe()
    if not fno_universe: return
    
    final_report_data = []
    min_prob_threshold = float(os.environ.get("PARAM_MIN_PROBABILITY", 50.0))

    for asset in fno_universe:
        result = fetch_upstox_data(asset["key"], target_date_str, interval="day", days_back=150)
        time.sleep(0.15) 
        
        if result is None: continue
        live_matrix, current_ltp = result
        
        with torch.no_grad():
            live_tensor = torch.tensor(live_matrix).unsqueeze(0)
            logits, live_latent = fno_cnn(live_tensor)
            cnn_probs = torch.softmax(logits, dim=1).numpy()[0]
            
        xgb_f0, xgb_f1, xgb_f2 = extract_xgb_probs(fno_xgb, live_latent.numpy())
        
        blended_neutral = ((cnn_probs[0] + xgb_f0) / 2) * 100
        blended_long = ((cnn_probs[1] + xgb_f1) / 2) * 100
        blended_short = ((cnn_probs[2] + xgb_f2) / 2) * 100
        
        probs_array = [blended_neutral, blended_long, blended_short]
        winner_idx = np.argmax(probs_array)
        best_prob = probs_array[winner_idx]
        
        # Only report if Long (1) or Short (2) is the strict winner AND crosses threshold
        if winner_idx == 1 and best_prob >= min_prob_threshold:
            final_report_data.append({
                'asset': asset["symbol"],
                'direction': "LONG 🟢",
                'conviction': float(best_prob),
                'ltp': float(current_ltp),
                'target_display': f"Expected Move: +{min_micro_pct}%",
                'actual_outcome': "<b>Awaiting Market ⏳</b>"
            })
        elif winner_idx == 2 and best_prob >= min_prob_threshold:
            final_report_data.append({
                'asset': asset["symbol"],
                'direction': "SHORT 🔴",
                'conviction': float(best_prob),
                'ltp': float(current_ltp),
                'target_display': f"Expected Drop: -{min_micro_pct}%",
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
                        exec_price = fw.iloc[0]['Open'] # Simulates entering at Next Day's Open
                        mx, mn = fw['High'].max(), fw['Low'].min()
                        
                        if "LONG" in row['direction']:
                            mv, dd = ((mx - exec_price) / exec_price) * 100, ((mn - exec_price) / exec_price) * 100
                            c = "#28a745" if mv > 0 else "#dc3545"
                            row['actual_outcome'] = f"<span style='color: {c};'>High ₹{mx:.2f} (+{mv:.2f}%)</span><br><span style='color: #856404; font-size: 11px;'>Max DD: {dd:.2f}%</span>"
                        else:
                            mv, dd = ((mn - exec_price) / exec_price) * 100, ((mx - exec_price) / exec_price) * 100
                            c = "#28a745" if mv < 0 else "#dc3545"
                            row['actual_outcome'] = f"<span style='color: {c};'>Low ₹{mn:.2f} ({mv:.2f}%)</span><br><span style='color: #856404; font-size: 11px;'>Max DD: +{dd:.2f}%</span>"

    send_mobile_alert(macro_report, final_report_data, target_date_str, is_backtest)

if __name__ == "__main__":
    run_production_sweep()

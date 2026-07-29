import os
import smtplib
import urllib.parse
import json
import gzip
import io
import time
import random
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
# 0. DETERMINISTIC ENVIRONMENT LOCK
# ==============================================================================
def set_deterministic_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# ==============================================================================
# 1. TEMPORAL AUTOENCODER (The 1D CNN Brain)
# ==============================================================================
class TemporalAutoencoder(nn.Module):
    def __init__(self, num_features=5, latent_dim=12):
        super(TemporalAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            
            nn.Flatten(),
            nn.Linear(32 * 5, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 5),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (32, 5)),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=3, output_padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, num_features, kernel_size=2, stride=2, output_padding=0),
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        latent = self.encode(x)
        return self.decoder(latent), latent

# ==============================================================================
# 2. DYNAMIC TRAINING LOADER & STANDARDIZER (Strict 9:15 AM Open Anchor)
# ==============================================================================
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
    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str).str[:10]
    return df

def load_training_data(csv_filename, target_date_str=None, min_pct=4.0, max_pct=50.0, max_dd=1.2, wick_ratio=0.40):
    df = read_and_standardize_csv(csv_filename)
    if df is None or 'Date' not in df.columns: return None, None, None
        
    if target_date_str:
        df = df[df['Date'] <= target_date_str]
    
    training_matrices, price_targets, time_targets = [], [], []
    FUTURE_DAYS = 2 
    
    if "historical_indices" in csv_filename.lower() or "nifty" in csv_filename.lower():
        if 'Symbol' in df.columns:
            mask = df['Symbol'].astype(str).str.upper().str.replace("_", "").str.replace(" ", "").str.contains("NIFTY50|NIFTY")
            if mask.any(): df = df[mask]

    for symbol, group in df.groupby('Symbol') if 'Symbol' in df.columns else [('ASSET', df)]:
        group = group.sort_values('Date').reset_index(drop=True)
        values = group[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
        
        if len(values) < (30 + FUTURE_DAYS): continue
            
        for i in range(len(values) - (30 + FUTURE_DAYS) + 1):
            raw_window = values[i : i+30]
            w_min, w_max = raw_window.min(axis=0), raw_window.max(axis=0)
            norm_window = (raw_window - w_min) / (w_max - w_min + 1e-8)
            window = norm_window.T 
            
            # 🎯 STRICT 9:15 AM ENTRY (Day T+1 Open)
            entry_price = values[i+30, 0] 
            if entry_price <= 0: continue
            
            future_closes = values[i+30 : i+30+FUTURE_DAYS, 3]
            future_highs  = values[i+30 : i+30+FUTURE_DAYS, 1]
            future_lows   = values[i+30 : i+30+FUTURE_DAYS, 2]
            
            max_close, min_close = future_closes.max(), future_closes.min()
            
            # Calculate actual moves directly from the 9:15 AM Open Anchor
            if (max_close - entry_price) > (entry_price - min_close):
                is_long = True
                actual_pct_move = ((max_close - entry_price) / entry_price) * 100
                actual_drawdown = ((future_lows.min() - entry_price) / entry_price) * 100
                rejection_wick = ((future_highs.max() - max_close) / entry_price) * 100
            else:
                is_long = False
                actual_pct_move = ((min_close - entry_price) / entry_price) * 100
                actual_drawdown = ((future_highs.max() - entry_price) / entry_price) * 100
                rejection_wick = ((min_close - future_lows.min()) / entry_price) * 100
                
            if abs(actual_pct_move) < min_pct or abs(actual_pct_move) > max_pct or entry_price < 10.0:
                continue
                
            if is_long:
                if actual_drawdown < -max_dd: continue 
                if rejection_wick > (actual_pct_move * wick_ratio): continue 
            else:
                if actual_drawdown > max_dd: continue 
                if rejection_wick > (abs(actual_pct_move) * wick_ratio): continue 

            days_to_target = float(np.argmax(np.abs(future_closes - entry_price)) + 1)
            training_matrices.append(window)
            price_targets.append(actual_pct_move)
            time_targets.append(days_to_target)
            
    return np.array(training_matrices, dtype=np.float32), np.array(price_targets, dtype=np.float32), np.array(time_targets, dtype=np.float32)

# ==============================================================================
# 3. MODULAR AI TRAINING ENGINE
# ==============================================================================
def train_ai_brain(X_raw, Y_price, Y_time, epochs=15):
    X_tensor = torch.tensor(X_raw)
    
    cnn_model = TemporalAutoencoder()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    cnn_model.train()
    for _ in range(epochs): 
        optimizer.zero_grad()
        reconstructed, _ = cnn_model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()

    cnn_model.eval()
    with torch.no_grad():
        latent_vectors = cnn_model.encode(X_tensor).numpy()
        
    xgb_price = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4).fit(latent_vectors, Y_price)
    xgb_time = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4).fit(latent_vectors, Y_time)

    faiss.normalize_L2(latent_vectors)
    index = faiss.IndexFlatIP(12) 
    index.add(latent_vectors)
    
    return cnn_model, xgb_price, xgb_time, index

# ==============================================================================
# 4. LIVE INGESTION TOOLS (API & 9:15 Anchors)
# ==============================================================================
def fetch_915_open_from_upstox(instrument_key, target_date_str):
    """Fetches exact 9:15 AM Open price via Upstox intraday API."""
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token or not instrument_key: return None
    
    encoded_key = urllib.parse.quote(instrument_key)
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_key}/1minute"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            candles = response.json().get('data', {}).get('candles', [])
            if candles:
                for c in candles:
                    if target_date_str in str(c[0]) and "09:15" in str(c[0]): 
                        return float(c[1]) # Index 1 is Open price
                return float(candles[-1][1]) # Fallback to earliest available open
    except Exception as e:
        pass
    return None

def fetch_upstox_data_with_915_anchor(instrument_key, target_date_str, days_back=60):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token: return None, None
    
    # 1. Fetch historical data STRICTLY prior to target date for clean 30-day matrix
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    to_date = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    from_date = (target_dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    encoded_key = urllib.parse.quote(instrument_key)
    url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/day/{to_date}/{from_date}"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200: return None, None
        
    data = response.json().get('data', {}).get('candles', [])
    if not data or len(data) < 30: return None, None
        
    ohlcv = np.array([candle[1:6] for candle in data], dtype=np.float32)[::-1] 
    
    ohlcv_30 = ohlcv[-30:]
    ohlcv_min = ohlcv_30.min(axis=0)
    ohlcv_max = ohlcv_30.max(axis=0)
    normalized_ohlcv = (ohlcv_30 - ohlcv_min) / (ohlcv_max - ohlcv_min + 1e-8)
    
    # 2. Fetch the true live execution anchor for the target date
    entry_price = fetch_915_open_from_upstox(instrument_key, target_date_str)
    if entry_price is None:
        entry_price = float(ohlcv_30[-1][3]) # Failsafe to yesterday close if API drops
        
    return normalized_ohlcv.T, entry_price

def get_fno_live_tensor(asset_symbol, asset_key, target_date_str, is_backtest, df_full=None):
    """Dynamically routes between local fast-CSV for backtests and Upstox API for live runs."""
    if is_backtest and df_full is not None:
        df_sym = df_full[df_full['Symbol'] == asset_symbol].sort_values('Date').reset_index(drop=True)
        df_history = df_sym[df_sym['Date'] < target_date_str]
        if len(df_history) < 30: return None, None
        
        values = df_history[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-30:]
        
        df_future = df_sym[df_sym['Date'] >= target_date_str]
        if df_future.empty: return None, None
        
        entry_price = float(df_future.iloc[0]['Open']) # True CSV Open Anchor
        w_min, w_max = values.min(axis=0), values.max(axis=0)
        norm_window = (values - w_min) / (w_max - w_min + 1e-8)
        return norm_window.T, entry_price
    else:
        # Live Production Route
        return fetch_upstox_data_with_915_anchor(asset_key, target_date_str)

def get_live_nifty_tensor_from_csv(csv_filename, target_date_str, instrument_key=None, is_backtest=False):
    df = read_and_standardize_csv(csv_filename)
    if df is None or 'Date' not in df.columns: return None, None
    
    if 'Symbol' in df.columns:
        mask = df['Symbol'].astype(str).str.upper().str.replace("_", "").str.replace(" ", "").str.contains("NIFTY50|NIFTY")
        if mask.any(): df = df[mask]
        else: df = df[df['Symbol'] == df['Symbol'].unique()[0]]
            
    df_history = df[df['Date'] < target_date_str].sort_values('Date').reset_index(drop=True)
    if len(df_history) < 30: return None, None
    
    values = df_history[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-30:]
    close_t = values[-1, 3] 
    
    entry_price = None
    df_future = df[df['Date'] >= target_date_str].sort_values('Date').reset_index(drop=True)
    
    if not is_backtest and instrument_key:
        entry_price = fetch_915_open_from_upstox(instrument_key, target_date_str)
        
    if entry_price is None and not df_future.empty:
        entry_price = float(df_future.iloc[0]['Open'])
        
    if entry_price is None:
        entry_price = close_t # Failsafe
        
    w_min, w_max = values.min(axis=0), values.max(axis=0)
    norm_window = (values - w_min) / (w_max - w_min + 1e-8)
    return norm_window.T, entry_price

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
            <th>Score</th>
            <th>Entry (9:15)</th>
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
            <td>₹{row['entry']:.2f}</td>
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
    print(f"\n🧠 PHASE 1: Training NIFTY 50 Macro Brain using {nifty_file}...")
    X_nifty, Y_np, Y_nt = load_training_data(nifty_file, target_date_str, min_pct=0.75, max_pct=5.0, max_dd=0.5, wick_ratio=0.5)
    
    if X_nifty is None or len(X_nifty) == 0:
        print("❌ Nifty Data matrix construction failed.")
        return

    nifty_cnn, nifty_xgb_p, nifty_xgb_t, nifty_faiss = train_ai_brain(X_nifty, Y_np, Y_nt)
    
    fno_universe = get_dynamic_fno_universe()
    nifty_key = next((item["key"] for item in fno_universe if item["symbol"] in ["NIFTY 50", "NIFTY"]), None)
    
    nifty_live_matrix, nifty_entry = get_live_nifty_tensor_from_csv(nifty_file, target_date_str, nifty_key, is_backtest)
    
    if nifty_live_matrix is not None:
        live_nifty_tensor = torch.tensor(nifty_live_matrix).unsqueeze(0)
        with torch.no_grad():
            nifty_latent = nifty_cnn.encode(live_nifty_tensor).numpy()
            
        n_pct = nifty_xgb_p.predict(nifty_latent)[0]
        faiss.normalize_L2(nifty_latent)
        n_score, _ = nifty_faiss.search(nifty_latent, k=5)
        n_conviction = n_score[0][0] * 100
        
        macro_report = {
            'direction': "LONG 🟢" if n_pct > 0 else "SHORT 🔴",
            'conviction': float(n_conviction),
            'target_display': f"₹{nifty_entry * (1 + (n_pct / 100)):.2f} ({'+' if n_pct>0 else ''}{n_pct:.2f}%)"
        }
        print(f"🌍 MACRO REGIME: {macro_report['direction']} (Score: {n_conviction:.2f}%)")
    else:
        macro_report = {'direction': "UNKNOWN", 'conviction': 0, 'target_display': "N/A"}

    # ==========================================
    # PHASE 2: MICRO F&O BRAIN
    # ==========================================
    print("\n⚡ PHASE 2: Training F&O Micro Brain (Hyper-Momentum)...")
    X_fno, Y_fp, Y_ft = load_training_data("historical_fno.csv", target_date_str, min_pct=4.0, max_pct=50.0, max_dd=1.2, wick_ratio=0.4)
    if X_fno is None or len(X_fno) == 0: return

    fno_cnn, fno_xgb_p, fno_xgb_t, fno_faiss = train_ai_brain(X_fno, Y_fp, Y_ft)
    
    print("🎯 Phase 3: Sweeping Active Market Universe...")
    if not fno_universe: return
    
    final_report_data = []
    min_conviction = float(os.environ.get("PARAM_MIN_CONVICTION", 99.92))
    
    fno_df_full = read_and_standardize_csv("historical_fno.csv") if is_backtest else None

    for asset in fno_universe:
        live_matrix, entry_price = get_fno_live_tensor(asset["symbol"], asset["key"], target_date_str, is_backtest, fno_df_full)
        
        if not is_backtest: time.sleep(0.15) # Protect API Rate Limits
        if live_matrix is None or entry_price is None: continue
        
        with torch.no_grad():
            live_vector = fno_cnn.encode(torch.tensor(live_matrix).unsqueeze(0)).numpy()
        
        pred_pct = fno_xgb_p.predict(live_vector)[0]
        faiss.normalize_L2(live_vector)
        score, _ = fno_faiss.search(live_vector, k=5)
        conviction = score[0][0] * 100
        
        if conviction >= min_conviction:
            final_report_data.append({
                'asset': asset["symbol"],
                'direction': "LONG 🟢" if pred_pct > 0 else "SHORT 🔴",
                'conviction': float(conviction),
                'entry': float(entry_price),
                'target_display': f"₹{entry_price * (1 + (pred_pct / 100)):.2f} ({'+' if pred_pct>0 else ''}{pred_pct:.2f}%)",
                'actual_outcome': "<b>Awaiting Market ⏳</b>"
            })
            
    # VALIDATION LOGIC FOR BACKTEST
    if is_backtest and fno_df_full is not None and len(final_report_data) > 0:
        for row in final_report_data:
            if 'Symbol' not in fno_df_full.columns: continue
            
            df_sym = fno_df_full[fno_df_full['Symbol'] == row['asset']].sort_values('Date').reset_index(drop=True)
            df_future = df_sym[df_sym['Date'] >= target_date_str]
            
            if len(df_future) >= 2:
                fw = df_future.iloc[:2] 
                mx, mn = fw['Close'].max(), fw['Close'].min()
                if "LONG" in row['direction']:
                    mv, dd = ((mx - row['entry']) / row['entry']) * 100, ((fw['Low'].min() - row['entry']) / row['entry']) * 100
                    c = "#28a745" if mv > 0 else "#6c757d"
                    row['actual_outcome'] = f"<span style='color: {c};'>Closed ₹{mx:.2f} (+{mv:.2f}%)</span><br><span style='color: #856404; font-size: 11px;'>Max DD: {dd:.2f}%</span>"
                else:
                    mv, dd = ((mn - row['entry']) / row['entry']) * 100, ((fw['High'].max() - row['entry']) / row['entry']) * 100
                    c = "#28a745" if mv < 0 else "#6c757d"
                    row['actual_outcome'] = f"<span style='color: {c};'>Closed ₹{mn:.2f} ({mv:.2f}%)</span><br><span style='color: #856404; font-size: 11px;'>Max DD: +{dd:.2f}%</span>"

    send_mobile_alert(macro_report, final_report_data, target_date_str, is_backtest)

if __name__ == "__main__":
    run_production_sweep()

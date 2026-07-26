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
# 2. DYNAMIC TRAINING LOADER & STANDARDIZER
# ==============================================================================
def read_and_standardize_csv(filename):
    """Universally maps Fyers, Upstox, or Yahoo CSV formats into standard ML inputs."""
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
        # Strip timestamps (e.g. "2010-01-01 15:30") to purely "YYYY-MM-DD" for safe string comparisons
        df['Date'] = df['Date'].astype(str).str[:10]
        
    return df

def load_training_data(csv_filename, target_date_str=None, min_pct=4.0, max_pct=50.0, max_dd=1.2, wick_ratio=0.40):
    df = read_and_standardize_csv(csv_filename)
    if df is None or 'Date' not in df.columns:
        print(f"⚠️ Warning: Missing or invalid '{csv_filename}'")
        return None, None, None
        
    # 🛑 PREVENT DATA LEAKAGE
    if target_date_str:
        df = df[df['Date'] <= target_date_str]
    
    training_matrices = []
    price_targets = []
    time_targets = []
    FUTURE_DAYS = 2 
    
    # Anti-Interleaving for Index files
    if "historical_indices" in csv_filename.lower() or "nifty" in csv_filename.lower():
        if 'Symbol' in df.columns:
            mask = df['Symbol'].astype(str).str.upper().str.replace("_", "").str.replace(" ", "").str.contains("NIFTY50|NIFTY")
            if mask.any():
                df = df[mask]

    for symbol, group in df.groupby('Symbol') if 'Symbol' in df.columns else [('ASSET', df)]:
        group = group.sort_values('Date').reset_index(drop=True)
        values = group[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
        
        if len(values) < (30 + FUTURE_DAYS): 
            continue
            
        for i in range(len(values) - (30 + FUTURE_DAYS) + 1):
            raw_window = values[i : i+30]
            w_min = raw_window.min(axis=0)
            w_max = raw_window.max(axis=0)
            norm_window = (raw_window - w_min) / (w_max - w_min + 1e-8)
            window = norm_window.T 
            
            future_closes = values[i+30 : i+30+FUTURE_DAYS, 3]
            future_highs  = values[i+30 : i+30+FUTURE_DAYS, 1]
            future_lows   = values[i+30 : i+30+FUTURE_DAYS, 2]
            start_price   = values[i+29, 3] 
            
            max_close = future_closes.max()
            min_close = future_closes.min()
            
            if (max_close - start_price) > (start_price - min_close):
                is_long = True
                actual_pct_move = ((max_close - start_price) / (start_price + 1e-8)) * 100
                actual_drawdown = ((future_lows.min() - start_price) / (start_price + 1e-8)) * 100
                rejection_wick = ((future_highs.max() - max_close) / (start_price + 1e-8)) * 100
            else:
                is_long = False
                actual_pct_move = ((min_close - start_price) / (start_price + 1e-8)) * 100
                actual_drawdown = ((future_highs.max() - start_price) / (start_price + 1e-8)) * 100
                rejection_wick = ((min_close - future_lows.min()) / (start_price + 1e-8)) * 100
                
            if abs(actual_pct_move) < min_pct or abs(actual_pct_move) > max_pct or start_price < 10.0:
                continue
                
            if is_long:
                if actual_drawdown < -max_dd: continue 
                if rejection_wick > (actual_pct_move * wick_ratio): continue 
            else:
                if actual_drawdown > max_dd: continue 
                if rejection_wick > (abs(actual_pct_move) * wick_ratio): continue 

            days_to_target = float(np.argmax(np.abs(future_closes - start_price)) + 1)
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
# 4. LIVE INGESTION TOOLS
# ==============================================================================
def get_live_tensor_from_csv(csv_filename, target_date_str):
    df = read_and_standardize_csv(csv_filename)
    if df is None or 'Date' not in df.columns: return None, None
    
    if 'Symbol' in df.columns:
        mask = df['Symbol'].astype(str).str.upper().str.replace("_", "").str.replace(" ", "").str.contains("NIFTY50|NIFTY")
        if mask.any():
            df = df[mask]
        else:
            df = df[df['Symbol'] == df['Symbol'].unique()[0]]
            
    df = df[df['Date'] <= target_date_str].sort_values('Date').reset_index(drop=True)
    
    if len(df) < 30: return None, None
    
    values = df[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-30:]
    current_ltp = values[-1, 3] 
    
    w_min = values.min(axis=0)
    w_max = values.max(axis=0)
    norm_window = (values - w_min) / (w_max - w_min + 1e-8)
    
    return norm_window.T, current_ltp

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

def fetch_upstox_data(instrument_key, target_date_str, interval="day", days_back=60):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    to_date = target_date_str
    from_date = (target_dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    encoded_key = urllib.parse.quote(instrument_key)
    url = f"https://api.upstox.com/v2/historical-candle/{encoded_key}/{interval}/{to_date}/{from_date}"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200: return None
        
    data = response.json().get('data', {}).get('candles', [])
    if not data or len(data) < 30: return None
        
    current_ltp = float(data[0][4])
    ohlcv = np.array([candle[1:6] for candle in data], dtype=np.float32)[::-1] 
    
    ohlcv_30 = ohlcv[-30:]
    ohlcv_min = ohlcv_30.min(axis=0)
    ohlcv_max = ohlcv_30.max(axis=0)
    normalized_ohlcv = (ohlcv_30 - ohlcv_min) / (ohlcv_max - ohlcv_min + 1e-8)
    
    return normalized_ohlcv.T, current_ltp

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
    
    # Path Resolver Strategy
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
        print("📁 Here are the files the runner actually sees:")
        for root, dirs, files in os.walk("."):
            for file in files:
                print(os.path.join(root, file))
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
    
    nifty_live_matrix, nifty_ltp = get_live_tensor_from_csv(nifty_file, target_date_str)
    
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
            'target_display': f"₹{nifty_ltp * (1 + (n_pct / 100)):.2f} ({'+' if n_pct>0 else ''}{n_pct:.2f}%)"
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
    fno_universe = get_dynamic_fno_universe()
    if not fno_universe: return
    
    final_report_data = []
    min_conviction = float(os.environ.get("PARAM_MIN_CONVICTION", 85.0))

    for asset in fno_universe:
        result = fetch_upstox_data(asset["key"], target_date_str, interval="day", days_back=60)
        time.sleep(0.15) 
        
        if result is None: continue
        live_matrix, current_ltp = result
        
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
                'ltp': float(current_ltp),
                'target_display': f"₹{current_ltp * (1 + (pred_pct / 100)):.2f} ({'+' if pred_pct>0 else ''}{pred_pct:.2f}%)",
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
                        else:
                            mv, dd = ((mn - row['ltp']) / row['ltp']) * 100, ((fw['High'].max() - row['ltp']) / row['ltp']) * 100
                            c = "#28a745" if mv < 0 else "#6c757d"
                            row['actual_outcome'] = f"<span style='color: {c};'>Closed ₹{mn:.2f} ({mv:.2f}%)</span><br><span style='color: #856404; font-size: 11px;'>Max DD: +{dd:.2f}%</span>"

    send_mobile_alert(macro_report, final_report_data, target_date_str, is_backtest)

if __name__ == "__main__":
    run_production_sweep()

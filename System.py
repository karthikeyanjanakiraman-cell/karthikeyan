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
# 2. HIGH-OCTANE HYPER-MOMENTUM TRAINING LOADER
# ==============================================================================
def load_real_training_data(csv_filename="historical_fno.csv", target_date_str=None):
    if not os.path.exists(csv_filename):
        raise FileNotFoundError(f"Missing '{csv_filename}' in repository! Run the data generator workflow first.")
        
    df = pd.read_csv(csv_filename)
    
    # 🛑 PREVENT DATA LEAKAGE: Delete all future data from AI's memory
    if target_date_str:
        df = df[df['Date'] <= target_date_str]
    
    training_matrices = []
    price_targets = []
    time_targets = []
    FUTURE_DAYS = 2 
    
    for symbol, group in df.groupby('Symbol'):
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
            raw_future_window = values[i+30 : i+30+FUTURE_DAYS, 3] 
            start_price = values[i+29, 3] 
            
            max_price = raw_future_window.max()
            min_price = raw_future_window.min()
            
            if (max_price - start_price) > (start_price - min_price):
                max_pct_move = ((max_price - start_price) / (start_price + 1e-8)) * 100
            else:
                max_pct_move = ((min_price - start_price) / (start_price + 1e-8)) * 100
                
            if abs(max_pct_move) < 4.0 or abs(max_pct_move) > 50.0 or start_price < 20.0:
                continue
                
            days_to_target = float(np.argmax(np.abs(raw_future_window - start_price)) + 1)
            training_matrices.append(window)
            price_targets.append(max_pct_move)
            time_targets.append(days_to_target)
            
    return np.array(training_matrices, dtype=np.float32), np.array(price_targets, dtype=np.float32), np.array(time_targets, dtype=np.float32)

# ==============================================================================
# 3. DYNAMIC UNIVERSE INGESTION & LIVE FETCHING
# ==============================================================================
def get_dynamic_fno_universe():
    print("🌐 Downloading Live Upstox NSE Master Contract...")
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    response = requests.get(nse_url)
    if response.status_code != 200: 
        return []

    try:
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        fno_underlying = {item.get("underlying_symbol") for item in nse_data if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")}
        
        fno_universe = []
        for item in nse_data:
            if item.get("segment") in ("NSE_EQ", "NSE_INDEX") and item.get("trading_symbol") in fno_underlying:
                fno_universe.append({"symbol": item.get("trading_symbol"), "key": item.get("instrument_key")})
        return fno_universe
    except Exception as e:
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
    if response.status_code != 200: 
        return None
        
    data = response.json().get('data', {}).get('candles', [])
    if not data or len(data) < 30: 
        return None
        
    current_ltp = float(data[0][4])
    ohlcv = np.array([candle[1:6] for candle in data], dtype=np.float32)
    ohlcv = ohlcv[::-1] 
    
    ohlcv_30 = ohlcv[-30:]
    ohlcv_min = ohlcv_30.min(axis=0)
    ohlcv_max = ohlcv_30.max(axis=0)
    normalized_ohlcv = (ohlcv_30 - ohlcv_min) / (ohlcv_max - ohlcv_min + 1e-8)
    
    return normalized_ohlcv.T, current_ltp

# ==============================================================================
# 4. DISPATCH ENGINE (With 2-Day Actual Validation Column)
# ==============================================================================
def send_mobile_alert(report_data_list, target_date_str, is_backtest):
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_pass = os.environ.get("SENDER_PASSWORD")
    recipient_email = os.environ.get("RECIPIENT_EMAIL")
    
    if not all([sender_email, sender_pass, recipient_email]) or len(report_data_list) == 0:
        print("Missing Email credentials or No Targets Found. Skipping dispatch.")
        return

    msg = MIMEMultipart('alternative')
    prefix = "⏪ BACKTEST" if is_backtest else "🚀 LIVE ALERT"
    msg['Subject'] = f"{prefix} | {target_date_str}"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    top_trade = max(report_data_list, key=lambda x: x['conviction'])
    top_color = "#28a745" if top_trade['direction'] == "LONG 🟢" else "#dc3545"
    top_bg = "#e8f5e9" if top_trade['direction'] == "LONG 🟢" else "#f8d7da"

    sim_warning = f"<div style='background-color: #fff3cd; color: #856404; padding: 10px; text-align: center; font-weight: bold; margin-bottom: 15px;'>⚠️ VALIDATION MODE: SHOWING ACTUAL PERFORMANCE AFTER {target_date_str}</div>" if is_backtest else ""

    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f7f6; padding: 10px;">
        {sim_warning}
        <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #333; border-bottom: 1px solid #ccc; padding-bottom: 5px;">👑 HIGHEST PROBABILITY TRADE</h3>
            <div style="background-color: {top_bg}; border-left: 6px solid {top_color}; padding: 15px; border-radius: 4px;">
                <h1 style="margin: 0; color: {top_color}; font-size: 24px;">{top_trade['asset']} : {top_trade['direction']}</h1>
                <p style="font-size: 16px; color: #333; margin-top: 10px; line-height: 1.6;">
                    <b>Match Score:</b> <span style="font-size: 18px; font-weight: bold;">{top_trade['conviction']:.2f}%</span><br>
                    <b>AI Target Price:</b> {top_trade['target_display']}<br>
                    <b>Actual Result:</b> <b>{top_trade['actual_outcome']}</b>
                </p>
            </div>
        </div>

        <h3 style="color: #333;">⚡ ALL HYPER-MOMENTUM SETUPS</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-size: 14px; background-color: white;">
          <tr bgcolor="#f8f9fa" style="color: #333; font-weight: bold;">
            <th>Asset</th>
            <th>Direction</th>
            <th>Score</th>
            <th>Current LTP</th>
            <th>AI Target Price</th>
            <th>Actual Result (2-Day)</th>
          </tr>
    """
    
    report_data_list.sort(key=lambda x: x['sort_days'])
    
    for row in report_data_list:
        dir_color = "#28a745" if row['direction'] == "LONG 🟢" else "#dc3545"
        
        html_content += f"""
          <tr>
            <td style="color: #0056b3;"><b>{row['asset']}</b></td>
            <td style="color: {dir_color}; font-weight: bold;">{row['direction']}</td>
            <td>{row['conviction']:.2f}%</td>
            <td>₹{row['ltp']:.2f}</td>
            <td style="color: {dir_color}; font-weight: bold;">{row['target_display']}</td>
            <td><b>{row['actual_outcome']}</b></td>
          </tr>
        """
        
    html_content += """
        </table>
      </body>
    </html>
    """
    
    msg.attach(MIMEText(html_content, 'html'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_pass)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print(f"✅ Alert Dispatched with {len(report_data_list)} targets.")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

# ==============================================================================
# 5. MASTER PRODUCTION SWEEP EXECUTION
# ==============================================================================
def run_production_sweep():
    target_date_str = os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    is_backtest = bool(target_date_str)
    
    if not is_backtest:
        target_date_str = datetime.now().strftime("%Y-%m-%d")
        
    print(f"⚙️ Target Date: {target_date_str} {'(BACKTEST MODE)' if is_backtest else '(LIVE MODE)'}")
    print("📥 Loading high-octane historical training data from CSV...")
    try:
        X_train_raw, Y_price_targets, Y_time_targets = load_real_training_data("historical_fno.csv", target_date_str)
        if len(X_train_raw) == 0:
            print("❌ Training dataset empty after applying momentum filter.")
            return
    except Exception as e:
        print(f"❌ Critical failure reading CSV: {e}")
        return
        
    X_tensor = torch.tensor(X_train_raw)
    
    print("🧠 Phase 1: Training PyTorch 1D CNN Autoencoder...")
    cnn_model = TemporalAutoencoder()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    cnn_model.train()
    for epoch in range(15): 
        optimizer.zero_grad()
        reconstructed, _ = cnn_model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()

    print("⚡ Phase 2: Compressing Space Vectors into Latent Coordinates...")
    cnn_model.eval()
    with torch.no_grad():
        latent_vectors = cnn_model.encode(X_tensor).numpy()
        
    print("🌲 Phase 3: Synchronizing Dual XGBoost Predictors...")
    xgb_price_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4).fit(latent_vectors, Y_price_targets)
    xgb_time_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4).fit(latent_vectors, Y_time_targets)

    print("🔍 Phase 4: Constructing FAISS Spatial Similarity Matrix...")
    faiss.normalize_L2(latent_vectors)
    index = faiss.IndexFlatIP(12) 
    index.add(latent_vectors)
    
    print("🎯 Phase 5: Sweeping Active Market Universe...")
    fno_universe = get_dynamic_fno_universe()
    if not fno_universe:
        return
    
    final_report_data = []
    min_conviction = float(os.environ.get("PARAM_MIN_CONVICTION", 85.0))

    for asset in fno_universe:
        result = fetch_upstox_data(asset["key"], target_date_str, interval="day", days_back=60)
        time.sleep(0.15) 
        
        if result is None: 
            continue
            
        live_matrix, current_ltp = result
        live_tensor = torch.tensor(live_matrix).unsqueeze(0) 
        
        with torch.no_grad():
            live_vector = cnn_model.encode(live_tensor).numpy()
        
        predicted_target_pct = xgb_price_model.predict(live_vector)[0]
        predicted_target_days = max(1, min(2, int(round(xgb_time_model.predict(live_vector)[0])))) 
        
        faiss.normalize_L2(live_vector)
        cosine_similarity_score, _ = index.search(live_vector, k=5)
        conviction_percentage = cosine_similarity_score[0][0] * 100
        
        if conviction_percentage >= min_conviction:
            target_price_rupee = current_ltp * (1 + (predicted_target_pct / 100))
            direction_tag = "LONG 🟢" if predicted_target_pct > 0 else "SHORT 🔴"
            target_sign = "+" if predicted_target_pct > 0 else ""
            target_display = f"₹{target_price_rupee:.2f} ({target_sign}{predicted_target_pct:.2f}%)"
            days_display = "IMMEDIATE (TODAY) ⚡" if predicted_target_days == 1 else f"{predicted_target_days} Days"
            
            final_report_data.append({
                'asset': asset["symbol"],
                'direction': direction_tag,
                'conviction': float(conviction_percentage),
                'sort_days': predicted_target_days,
                'days_display': days_display,
                'ltp': float(current_ltp),
                'target_display': target_display,
                'actual_outcome': "Awaiting Market ⏳" # Default for live mode
            })
            
    # 🔮 THE TIME MACHINE: Calculate actual future performance if Backtesting
    if is_backtest and os.path.exists("historical_fno.csv") and len(final_report_data) > 0:
        print("🔮 Validating predictions against actual future data...")
        df_full = pd.read_csv("historical_fno.csv")
        
        for row in final_report_data:
            symbol = row['asset']
            ltp = row['ltp']
            direction = row['direction']
            
            df_sym = df_full[df_full['Symbol'] == symbol].sort_values('Date').reset_index(drop=True)
            past_dates = df_sym[df_sym['Date'] <= target_date_str]
            
            if len(past_dates) > 0:
                idx = past_dates.index[-1]
                future_window = df_sym.iloc[idx+1 : idx+3] # Look exactly 2 days ahead
                
                if len(future_window) > 0:
                    max_high = future_window['High'].max()
                    min_low = future_window['Low'].min()
                    
                    if "LONG" in direction:
                        actual_move = ((max_high - ltp) / ltp) * 100
                        color = "#28a745" if actual_move > 0 else "#6c757d"
                        row['actual_outcome'] = f"<span style='color: {color};'>Hit ₹{max_high:.2f} (+{actual_move:.2f}%)</span>"
                    else:
                        actual_move = ((min_low - ltp) / ltp) * 100
                        color = "#28a745" if actual_move < 0 else "#6c757d"
                        row['actual_outcome'] = f"<span style='color: {color};'>Hit ₹{min_low:.2f} ({actual_move:.2f}%)</span>"
                else:
                    row['actual_outcome'] = "<span style='color: #6c757d;'>Data ended</span>"
                    
    send_mobile_alert(final_report_data, target_date_str, is_backtest)

if __name__ == "__main__":
    run_production_sweep()
                             

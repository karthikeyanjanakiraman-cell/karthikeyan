import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timedelta

import requests
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
# 2. DATA INGESTION (Upstox API)
# ==============================================================================
def fetch_upstox_data(instrument_key, interval="day", days_back=60):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    url = f"https://api.upstox.com/v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
        
    data = response.json().get('data', {}).get('candles', [])
    if not data or len(data) < 30:
        return None
        
    ohlcv = np.array([candle[1:6] for candle in data], dtype=np.float32)
    ohlcv = ohlcv[::-1] 
    
    ohlcv_min = ohlcv.min(axis=0)
    ohlcv_max = ohlcv.max(axis=0)
    normalized_ohlcv = (ohlcv - ohlcv_min) / (ohlcv_max - ohlcv_min + 1e-8)
    
    return normalized_ohlcv[-30:].T, data[0][4]

# ==============================================================================
# 3. DISPATCH (Email Delivery Engine)
# ==============================================================================
def send_mobile_alert(report_data_list):
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_pass = os.environ.get("SENDER_PASSWORD")
    recipient_email = os.environ.get("RECIPIENT_EMAIL")
    
    if not all([sender_email, sender_pass, recipient_email]) or len(report_data_list) == 0:
        print("Missing Email credentials or No Targets Found. Skipping dispatch.")
        return

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"🎯 QUANT ENGINE: F&O Master Report | {datetime.now().strftime('%d %b')}"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    html_content = """
    <html>
      <body style="font-family: Arial, sans-serif;">
        <h3 style="color: #333;">🎯 OMNIDIRECTIONAL TARGET DETECTOR</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-size: 14px;">
          <tr bgcolor="#f8f9fa" style="color: #333; font-weight: bold;">
            <th>Asset</th>
            <th>Direction</th>
            <th>Match Score</th>
            <th>Expected Time</th>
            <th>Current LTP</th>
            <th>Remaining Target</th>
          </tr>
    """
    
    for row in report_data_list:
        dir_color = "#28a745" if row['direction'] == "LONG 🟢" else "#dc3545"
        
        html_content += f"""
          <tr>
            <td style="color: #0056b3;"><b>{row['asset']}</b></td>
            <td style="color: {dir_color}; font-weight: bold;">{row['direction']}</td>
            <td>{row['conviction']:.2f}%</td>
            <td><b>{row['days']} Days</b></td>
            <td>₹{row['ltp']:.2f}</td>
            <td style="color: {dir_color}; font-weight: bold;">{row['remaining']}%</td>
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
        print(f"✅ Live Quant Alert Dispatched with {len(report_data_list)} targets.")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

# ==============================================================================
# 4. MASTER EXECUTION LOOP
# ==============================================================================
def run_production_sweep():
    print("📥 Ingesting simulated historical matrix...")
    X_train_raw = np.random.rand(1000, 5, 30).astype(np.float32) 
    
    # 1. Target Data for PRICE MOVE (-10% to +10%)
    Y_price_targets = (np.random.rand(1000).astype(np.float32) * 20.0) - 10.0 
    
    # 2. Target Data for TIME DURATION (1 to 14 days)
    Y_time_targets = (np.random.rand(1000).astype(np.float32) * 13.0) + 1.0 
    
    X_tensor = torch.tensor(X_train_raw)
    
    print("🧠 Phase 1: Training PyTorch 1D CNN Timeline Extractor...")
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

    print("⚡ Phase 2: Compressing Timeline into Latent Coordinates...")
    cnn_model.eval()
    with torch.no_grad():
        latent_vectors = cnn_model.encode(X_tensor).numpy()
        
    print("🌲 Phase 3: Training DUAL XGBoost Predictors (Price & Time)...")
    
    # Train the Price Predictor
    xgb_price_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4)
    xgb_price_model.fit(latent_vectors, Y_price_targets)

    # Train the Time Predictor
    xgb_time_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4)
    xgb_time_model.fit(latent_vectors, Y_time_targets)

    print("🔍 Phase 4: Mapping FAISS Multi-Dimensional Similarity Grid...")
    faiss.normalize_L2(latent_vectors)
    index = faiss.IndexFlatIP(12) 
    index.add(latent_vectors)
    
    print("🎯 Phase 5: Scanning Live Market Data...")
    
    fno_universe = [
        {"symbol": "RELIANCE", "key": "NSE_EQ|INE002A01018"},
        {"symbol": "HDFCBANK", "key": "NSE_EQ|INE040A01034"},
        {"symbol": "TCS",      "key": "NSE_EQ|INE467B01029"},
        {"symbol": "INFY",     "key": "NSE_EQ|INE009A01021"}
    ]
    
    final_report_data = []

    for asset in fno_universe:
        result = fetch_upstox_data(asset["key"], interval="day", days_back=60)
        if result is None: continue
            
        live_matrix, current_ltp = result
        live_tensor = torch.tensor(live_matrix).unsqueeze(0) 
        
        with torch.no_grad():
            live_vector = cnn_model.encode(live_tensor).numpy()
        
        # Dual Predictions from the XGBoost Trees
        predicted_target_price = xgb_price_model.predict(live_vector)[0]
        predicted_target_days = max(1, int(round(xgb_time_model.predict(live_vector)[0]))) # Prevents zero or negative days
        
        faiss.normalize_L2(live_vector)
        cosine_similarity_score, _ = index.search(live_vector, k=5)
        conviction_percentage = cosine_similarity_score[0][0] * 100
        
        if conviction_percentage > 85.0:
            
            direction_tag = "LONG 🟢" if predicted_target_price > 0 else "SHORT 🔴"
            target_sign = "+" if predicted_target_price > 0 else ""
            
            final_report_data.append({
                'asset': asset["symbol"],
                'direction': direction_tag,
                'conviction': conviction_percentage,
                'days': predicted_target_days,
                'ltp': current_ltp,
                'remaining': f"{target_sign}{predicted_target_price:.2f}"
            })
            
    send_mobile_alert(final_report_data)

if __name__ == "__main__":
    run_production_sweep()
        

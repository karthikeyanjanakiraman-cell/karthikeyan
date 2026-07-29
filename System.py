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
# 1. INDICATOR ENGINE (The Holy Trinity)
# ==============================================================================
def add_holy_trinity(df):
    """Calculates RSI(7), ADX(14), and Supertrend(7,2) without external libraries."""
    df = df.copy()
    
    # 1. RSI (7)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/7, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/7, adjust=False).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_7'] = 100 - (100 / (1 + rs))

    # 2. ADX (14)
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr1 = df['High'] - df['Low']
    tr2 = np.abs(df['High'] - df['Close'].shift(1))
    tr3 = np.abs(df['Low'] - df['Close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr14)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr14)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df['ADX_14'] = dx.ewm(alpha=1/14, adjust=False).mean()

    # 3. Supertrend (7, 2)
    atr7 = tr.ewm(alpha=1/7, adjust=False).mean()
    hl2 = (df['High'] + df['Low']) / 2
    upperband = hl2 + (2 * atr7)
    lowerband = hl2 - (2 * atr7)
    
    st = [0.0] * len(df)
    st_dir = [1] * len(df)
    for i in range(1, len(df)):
        st_dir[i] = st_dir[i-1]
        if df['Close'].iloc[i] > upperband.iloc[i-1]:
            st_dir[i] = 1
        elif df['Close'].iloc[i] < lowerband.iloc[i-1]:
            st_dir[i] = -1
        
        if st_dir[i] == 1:
            st[i] = max(lowerband.iloc[i], st[i-1] if st_dir[i-1]==1 else 0)
        else:
            st[i] = min(upperband.iloc[i], st[i-1] if st_dir[i-1]==-1 else float('inf'))
            
    df['ST_7_2'] = st
    df['ST_Dist'] = (df['Close'] - df['ST_7_2']) / (df['Close'] + 1e-8)
    
    df.fillna(method='bfill', inplace=True)
    return df

# ==============================================================================
# 2. TEMPORAL AUTOENCODER (Upgraded to 8 Features)
# ==============================================================================
class TemporalAutoencoder(nn.Module):
    def __init__(self, num_features=8, latent_dim=12):
        super(TemporalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(3),
            nn.Flatten(), nn.Linear(32 * 5, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 5), nn.ReLU(inplace=True), nn.Unflatten(1, (32, 5)),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=3, output_padding=0),
            nn.BatchNorm1d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, num_features, kernel_size=2, stride=2, output_padding=0),
            nn.Sigmoid()
        )

    def encode(self, x): return self.encoder(x)
    def forward(self, x): latent = self.encode(x); return self.decoder(latent), latent

# ==============================================================================
# 3. DYNAMIC DATA LOADER
# ==============================================================================
def load_training_data(csv_filename, target_date_str=None):
    if not os.path.exists(csv_filename): return None, None, None
    df = pd.read_csv(csv_filename)
    df.columns = [str(c).title().strip() for c in df.columns]
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True).dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=['Date'])
    if target_date_str: df = df[df['Date'] <= target_date_str]
    
    training_matrices, price_targets = [], []
    
    for symbol, group in df.groupby('Symbol') if 'Symbol' in df.columns else [('ASSET', df)]:
        group = group.sort_values('Date').reset_index(drop=True)
        group = add_holy_trinity(group)
        values = group[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_7', 'ADX_14', 'ST_Dist']].values.astype(np.float32)
        
        if len(values) < 32: continue
        for i in range(len(values) - 32 + 1):
            raw_window = values[i : i+30]
            w_min, w_max = raw_window.min(axis=0), raw_window.max(axis=0)
            norm_window = (raw_window - w_min) / (w_max - w_min + 1e-8)
            
            future_closes = values[i+30 : i+32, 3]
            start_price = values[i+29, 3]
            actual_pct_move = ((future_closes.max() if future_closes.max() - start_price > start_price - future_closes.min() else future_closes.min()) - start_price) / start_price * 100
            
            if abs(actual_pct_move) < 2.0: continue
            
            training_matrices.append(norm_window.T)
            price_targets.append(actual_pct_move)
            
    return np.array(training_matrices, dtype=np.float32), np.array(price_targets, dtype=np.float32), None

# ==============================================================================
# 4. ALWAYS-TRAIN AI BRAIN (No Saving, No Loading)
# ==============================================================================
def get_ai_brain(X_raw, Y_price, prefix="fno"):
    print(f"⚙️ Initiating Deep Training (50 Epochs) for [{prefix}]...")
    if X_raw is None or len(X_raw) == 0: return None, None, None, None
    
    cnn_model = TemporalAutoencoder()
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Always train for 50 epochs
    cnn_model.train()
    for e in range(50):
        optimizer.zero_grad()
        reconstructed, _ = cnn_model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()

    cnn_model.eval()
    
    with torch.no_grad(): 
        latent_vectors = cnn_model.encode(X_tensor).numpy()
    latent_vectors = np.ascontiguousarray(latent_vectors, dtype=np.float32)
        
    # Always fit XGBoost
    xgb_price = xgb.XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42).fit(latent_vectors, Y_price)
    
    # Always build Faiss index
    faiss.normalize_L2(latent_vectors)
    index = faiss.IndexFlatIP(12) 
    index.add(latent_vectors)
    
    return cnn_model, xgb_price, index, Y_price

# ==============================================================================
# 5. LIVE INGESTION
# ==============================================================================
def fetch_live_data_and_indicators(csv_filename, target_date_str, is_nifty=False):
    df = pd.read_csv(csv_filename)
    df.columns = [str(c).title().strip() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True).dt.strftime('%Y-%m-%d')
    if is_nifty and 'Symbol' in df.columns:
        df = df[df['Symbol'].astype(str).str.contains("NIFTY")]
    df = df[df['Date'] <= target_date_str].sort_values('Date').reset_index(drop=True)
    if len(df) < 30: return None, None, None
    
    df = add_holy_trinity(df)
    values = df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_7', 'ADX_14', 'ST_Dist']].values.astype(np.float32)[-30:]
    current_ltp = values[-1, 3]
    current_sl = df['ST_7_2'].iloc[-1] 
    
    w_min, w_max = values.min(axis=0), values.max(axis=0)
    norm_window = (values - w_min) / (w_max - w_min + 1e-8)
    return norm_window.T, current_ltp, current_sl

# ==============================================================================
# 6. MASTER EXECUTION & DISPATCH
# ==============================================================================
def run_production_sweep():
    set_deterministic_seeds(42)
    raw_date_str = os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    target_date_str = pd.to_datetime(raw_date_str, dayfirst=True).strftime("%Y-%m-%d") if raw_date_str else datetime.now().strftime("%Y-%m-%d")
    print(f"⚙️ EXECUTING DATE: {target_date_str}")
    
    # 1. Macro Brain
    X_nifty, Y_np, _ = load_training_data("historical_indices.csv", target_date_str)
    nifty_cnn, nifty_xgb_p, nifty_faiss, nifty_hist_y = get_ai_brain(X_nifty, Y_np, prefix="macro")
    
    if not nifty_cnn: return 
    
    n_mat, n_ltp, n_sl = fetch_live_data_and_indicators("historical_indices.csv", target_date_str, is_nifty=True)
    
    macro_direction = "UNKNOWN"
    if n_mat is not None:
        live_t = torch.tensor(n_mat, dtype=torch.float32).unsqueeze(0)
        n_lat = np.ascontiguousarray(nifty_cnn.encode(live_t).detach().numpy(), dtype=np.float32)
        n_pct = nifty_xgb_p.predict(n_lat)[0]
        macro_direction = "LONG 🟢" if n_pct > 0 else "SHORT 🔴"

    # 2. Micro F&O Brain
    X_fno, Y_fp, _ = load_training_data("historical_fno.csv", target_date_str)
    fno_cnn, fno_xgb_p, fno_faiss, fno_hist_y = get_ai_brain(X_fno, Y_fp, prefix="micro")
    
    if not fno_cnn: return 
    
    final_report = []
    fno_df = pd.read_csv("historical_fno.csv")
    fno_df.columns = [str(c).title().strip() for c in fno_df.columns]
    symbols = fno_df['Symbol'].unique() if 'Symbol' in fno_df.columns else []

    print("🎯 Phase 3: Sweeping Universe with Consensus Logic...")
    for sym in symbols:
        sym_file = f"temp_{sym}.csv"
        fno_df[fno_df['Symbol'] == sym].to_csv(sym_file, index=False)
        mat, ltp, sl = fetch_live_data_and_indicators(sym_file, target_date_str)
        os.remove(sym_file)
        if mat is None: continue
        
        live_t = torch.tensor(mat, dtype=torch.float32).unsqueeze(0)
        lat = np.ascontiguousarray(fno_cnn.encode(live_t).detach().numpy(), dtype=np.float32)
        pred_pct = fno_xgb_p.predict(lat)[0]
        direction = "LONG 🟢" if pred_pct > 0 else "SHORT 🔴"
        
        faiss.normalize_L2(lat)
        scores, indices = fno_faiss.search(lat, k=5)
        raw_conviction = scores[0][0] * 100
        
        historical_outcomes = fno_hist_y[indices[0]]
        if pred_pct > 0:
            consensus = sum(1 for y in historical_outcomes if y > 0) / 5.0
        else:
            consensus = sum(1 for y in historical_outcomes if y < 0) / 5.0
            
        if consensus < 0.6: continue 
        
        final_conviction = raw_conviction * consensus
        
        if final_conviction >= 50.0:
            outcome_text = "<b>Awaiting Market ⏳</b>"
            if raw_date_str:
                df_sym = fno_df[(fno_df['Symbol'] == sym) & (fno_df['Date'] > target_date_str)].sort_values('Date')
                if len(df_sym) >= 2:
                    fw = df_sym.iloc[:2]
                    mx, mn, c2 = fw['High'].max(), fw['Low'].min(), fw['Close'].iloc[1]
                    
                    if direction == "LONG 🟢":
                        if mn < sl:
                            outcome_text = f"<span style='color: #dc3545;'>❌ STOP HIT (₹{sl:.2f})</span>"
                        else:
                            outcome_text = f"<span style='color: #28a745;'>✅ Closed ₹{mx:.2f}</span>"
                    else:
                        if mx > sl:
                            outcome_text = f"<span style='color: #dc3545;'>❌ STOP HIT (₹{sl:.2f})</span>"
                        else:
                            outcome_text = f"<span style='color: #28a745;'>✅ Closed ₹{mn:.2f}</span>"
            
            final_report.append({
                'asset': sym, 'direction': direction, 'conviction': final_conviction,
                'ltp': float(ltp), 'sl': float(sl), 
                'target': f"₹{ltp * (1 + (pred_pct / 100)):.2f}", 'actual': outcome_text
            })

    # 3. Email Dispatch
    html = f"""<html><body style="font-family: Arial; padding: 10px;">
        <h3>🌍 MACRO REGIME: {macro_direction}</h3>
        <table border="1" cellpadding="8" cellspacing="0" width="100%">
          <tr bgcolor="#f8f9fa"><th>Asset</th><th>Signal</th><th>Conviction</th><th>LTP</th><th>Invalidation (SL)</th><th>AI Target</th><th>Result</th></tr>"""
    for r in sorted(final_report, key=lambda x: x['conviction'], reverse=True):
        html += f"<tr><td><b>{r['asset']}</b></td><td>{r['direction']}</td><td>{r['conviction']:.1f}%</td><td>₹{r['ltp']:.2f}</td><td style='color:#dc3545;'>₹{r['sl']:.2f}</td><td>{r['target']}</td><td>{r['actual']}</td></tr>"
    html += "</table></body></html>"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"🚀 AI SWEEP | {target_date_str}"
    msg['From'], msg['To'] = os.environ.get("SENDER_EMAIL"), os.environ.get("RECIPIENT_EMAIL")
    if msg['From']:
        msg.attach(MIMEText(html, 'html'))
        server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls()
        server.login(msg['From'], os.environ.get("SENDER_PASSWORD"))
        server.sendmail(msg['From'], msg['To'], msg.as_string()); server.quit()
        print("✅ Alert Dispatched!")

if __name__ == "__main__":
    run_production_sweep()

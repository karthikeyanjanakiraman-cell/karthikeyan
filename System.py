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
import faiss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. DETERMINISTIC ENVIRONMENT
# ==============================================================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ==============================================================================
# 1. INSTITUTIONAL DEEP LEARNING: TIME2VEC + CONV-TRANSFORMER
# ==============================================================================
class Time2Vec(nn.Module):
    def __init__(self, seq_len):
        super(Time2Vec, self).__init__()
        self.seq_len = seq_len
        self.w0 = nn.parameter.Parameter(torch.randn(seq_len, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(seq_len, 1))
        self.w = nn.parameter.Parameter(torch.randn(seq_len, 3)) 
        self.b = nn.parameter.Parameter(torch.randn(seq_len, 3))
        
    def forward(self, x):
        time_linear = self.w0 * x[:, :, 0:1] + self.b0
        time_periodic = torch.sin(x[:, :, 0:1] * self.w + self.b)
        t2v = torch.cat([time_linear, time_periodic], dim=-1) 
        return torch.cat([x, t2v], dim=-1) 

class AdvancedQuantBrain(nn.Module):
    def __init__(self, num_features=5, d_model=32, nhead=4, num_layers=2, latent_dim=16):
        super(AdvancedQuantBrain, self).__init__()
        self.t2v = Time2Vec(seq_len=120)
        t2v_out_dim = num_features + 4
        
        self.conv1 = nn.Conv1d(in_channels=t2v_out_dim, out_channels=d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.attention_pool = nn.Linear(d_model, 1)
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, latent_dim)
        )
        
        self.reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 120 * num_features),
            nn.Unflatten(1, (120, num_features))
        )
        
        self.return_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def encode(self, x):
        x = self.t2v(x) 
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2) 
        
        x = self.transformer(x)
        attn_weights = torch.softmax(self.attention_pool(x), dim=1)
        x_pooled = torch.sum(x * attn_weights, dim=1) 
        
        latent = self.latent_proj(x_pooled) 
        return latent

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.reconstruction_head(latent)
        pred_return = self.return_head(latent)
        return reconstructed, pred_return, latent

# ==============================================================================
# 2. MACRO STATISTICAL ANCHORS
# ==============================================================================
def extract_macro_statistics(ohlcv_window):
    opens, highs, lows, closes, volumes = ohlcv_window[:, 0], ohlcv_window[:, 1], ohlcv_window[:, 2], ohlcv_window[:, 3], ohlcv_window[:, 4]
    macro_min, macro_max = lows.min(), highs.max()
    pos_in_macro = (closes[-1] - macro_min) / (macro_max - macro_min + 1e-8)
    ret_120d = (closes[-1] - closes[0]) / (closes[0] + 1e-8)
    
    daily_returns = np.diff(closes) / (closes[:-1] + 1e-8)
    vol_long = np.std(daily_returns) if len(daily_returns) > 0 else 1e-8
    vol_short = np.std(daily_returns[-10:]) if len(daily_returns) >= 10 else vol_long
    vol_ratio = vol_short / (vol_long + 1e-8)
    
    return np.array([pos_in_macro, ret_120d, vol_ratio, closes[-1]/(opens[-1]+1e-8)], dtype=np.float32)

# ==============================================================================
# 3. DATA PROCESSING COMPILER (WITH X-RAY LOGS & INDIAN DATE FIX)
# ==============================================================================
def read_standard_csv(filename):
    if not filename or not os.path.exists(filename): 
        print(f"   ❌ ERROR: Could not locate file: {filename}")
        return None
        
    df = pd.read_csv(filename)
    df.rename(columns=lambda x: str(x).lower().strip(), inplace=True)
    col_map = {'date':'Date', 'timestamp':'Date', 'symbol':'Symbol', 'ticker':'Symbol', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close', 'volume':'Volume'}
    df.rename(columns=col_map, inplace=True)
    
    if 'Date' in df.columns: 
        # FIX: Added format='mixed' to handle DD-MM-YYYY vs YYYY-MM-DD 
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed').dt.strftime('%Y-%m-%d')
        dropped = df['Date'].isna().sum()
        if dropped > 0:
            print(f"   ⚠️ WARNING: Dropped {dropped} rows due to unparseable dates.")
        df.dropna(subset=['Date'], inplace=True)
        
    return df

def build_deep_training_tensors(csv_filename, target_date_str=None, min_pct=0.75):
    print(f"   -> Loading memory from: {csv_filename}")
    df = read_standard_csv(csv_filename)
    if df is None: return None, None, None, None
    if 'Date' not in df.columns:
        print(f"   ❌ ERROR: No 'Date' column found in {csv_filename}")
        return None, None, None, None
        
    if target_date_str: 
        df = df[df['Date'] <= target_date_str]
        
    print(f"   -> Valid rows available up to {target_date_str}: {len(df)}")
    
    X_seq, X_macro, Y_price, Y_risk = [], [], [], []
    LOOKBACK, FUTURE = 120, 2 

    if "nifty" in csv_filename.lower() and 'Symbol' in df.columns:
        df = df[df['Symbol'].astype(str).str.upper().str.contains("NIFTY50|NIFTY")]

    total_extracted = 0
    for symbol, group in df.groupby('Symbol') if 'Symbol' in df.columns else [('ASSET', df)]:
        group = group.sort_values('Date').reset_index(drop=True)
        vals = group[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
        if len(vals) < (LOOKBACK + FUTURE): continue
            
        for i in range(len(vals) - (LOOKBACK + FUTURE) + 1):
            window = vals[i : i+LOOKBACK]
            entry_price = vals[i+LOOKBACK, 0] 
            if entry_price <= 0: continue
            
            future_closes = vals[i+LOOKBACK : i+LOOKBACK+FUTURE, 3]
            future_highs  = vals[i+LOOKBACK : i+LOOKBACK+FUTURE, 1]
            future_lows   = vals[i+LOOKBACK : i+LOOKBACK+FUTURE, 2]
            
            mx, mn = future_closes.max(), future_closes.min()
            
            if (mx - entry_price) > (entry_price - mn):
                actual_pct = ((mx - entry_price) / entry_price) * 100.0
                adv_exc = abs((future_lows.min() - entry_price) / entry_price) * 100.0
            else:
                actual_pct = ((mn - entry_price) / entry_price) * 100.0
                adv_exc = abs((future_highs.max() - entry_price) / entry_price) * 100.0
                
            if abs(actual_pct) < min_pct or entry_price < 10.0: continue
                
            w_min, w_max = window.min(axis=0), window.max(axis=0)
            norm_seq = (window - w_min) / (w_max - w_min + 1e-8)
            
            X_seq.append(norm_seq)
            X_macro.append(extract_macro_statistics(window))
            Y_price.append(actual_pct)
            Y_risk.append(adv_exc)
            total_extracted += 1
            
    if total_extracted == 0:
        print(f"   ❌ ERROR: No valid training tensors could be created. Is min_pct ({min_pct}%) too high?")
        return None, None, None, None
        
    print(f"   ✅ Successfully built {total_extracted} Deep Learning Tensors.")
    return np.array(X_seq, dtype=np.float32), np.array(X_macro, dtype=np.float32), np.array(Y_price, dtype=np.float32), np.array(Y_risk, dtype=np.float32)

# ==============================================================================
# 4. MULTI-TASK NETWORK TRAINING (Supervised Contrastive Influence)
# ==============================================================================
def train_high_end_brain(X_seq, X_macro, Y_price, Y_risk, epochs=30):
    print(f"   [AI] Fusing Time2Vec & Transformer over {len(X_seq)} parameters...")
    model = AdvancedQuantBrain(latent_dim=16)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    criterion_recon = nn.MSELoss()
    criterion_return = nn.HuberLoss() 
    
    tensor_x = torch.tensor(X_seq)
    tensor_y = torch.tensor(Y_price).unsqueeze(1)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model.train()
    for e in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            recon, pred_y, _ = model(batch_x)
            
            loss_r = criterion_recon(recon, batch_x)
            loss_y = criterion_return(pred_y, batch_y)
            total_loss = (0.7 * loss_r) + (0.3 * loss_y)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    with torch.no_grad():
        latent_vectors = model.encode(tensor_x).numpy()
        
    fused_features = np.hstack((latent_vectors, X_macro))
    fused_contig = np.ascontiguousarray(fused_features, dtype=np.float32)
    
    faiss.normalize_L2(fused_contig)
    index = faiss.IndexFlatIP(fused_contig.shape[1]) 
    index.add(fused_contig)
    
    return model, index, Y_price, Y_risk

# ==============================================================================
# 5. LIVE LIVE INGESTION
# ==============================================================================
def fetch_915_open(key, dt_str):
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not token or not key: return None
    try:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(key)}/1minute"
        resp = requests.get(url, headers={'Accept': 'application/json', 'Authorization': f'Bearer {token}'}, timeout=5)
        if resp.status_code == 200:
            for c in resp.json().get('data', {}).get('candles', []):
                if dt_str in str(c[0]) and "09:15" in str(c[0]): return float(c[1])
            return float(resp.json().get('data', {}).get('candles', [])[-1][1])
    except: pass
    return None

def get_live_tensors(symbol, key, dt_str, is_backtest, df_full=None):
    if is_backtest and df_full is not None:
        df_sym = df_full[df_full['Symbol'] == symbol].sort_values('Date').reset_index(drop=True)
        hist = df_sym[df_sym['Date'] < dt_str]
        if len(hist) < 120: return None, None, None
        
        vals = hist[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-120:]
        fut = df_sym[df_sym['Date'] >= dt_str]
        entry = float(fut.iloc[0]['Open']) if not fut.empty else float(vals[-1, 3])
        
        w_min, w_max = vals.min(axis=0), vals.max(axis=0)
        norm_seq = (vals - w_min) / (w_max - w_min + 1e-8)
        return norm_seq, extract_macro_statistics(vals), entry
    else:
        token = os.environ.get("UPSTOX_ACCESS_TOKEN")
        if not token: return None, None, None
        dt = datetime.strptime(dt_str, "%Y-%m-%d")
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(key)}/day/{(dt-timedelta(days=1)).strftime('%Y-%m-%d')}/{(dt-timedelta(days=200)).strftime('%Y-%m-%d')}"
        resp = requests.get(url, headers={'Accept': 'application/json', 'Authorization': f'Bearer {token}'})
        if resp.status_code != 200: return None, None, None
        data = resp.json().get('data', {}).get('candles', [])
        if not data or len(data) < 120: return None, None, None
        
        vals = np.array([c[1:6] for c in data], dtype=np.float32)[::-1][-120:]
        entry = fetch_915_open(key, dt_str)
        if entry is None: entry = float(vals[-1][3])
        
        w_min, w_max = vals.min(axis=0), vals.max(axis=0)
        norm_seq = (vals - w_min) / (w_max - w_min + 1e-8)
        return norm_seq, extract_macro_statistics(vals), entry

def get_fno_universe():
    try:
        resp = requests.get("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", timeout=10)
        if resp.status_code != 200: return []
        data = json.load(gzip.GzipFile(fileobj=io.BytesIO(resp.content)))
        und = {i.get("underlying_symbol") for i in data if i.get("segment") == "NSE_FO" and i.get("underlying_symbol")}
        return [{"symbol": i.get("trading_symbol"), "key": i.get("instrument_key")} for i in data if i.get("segment") in ("NSE_EQ", "NSE_INDEX") and i.get("trading_symbol") in und]
    except: return []

# ==============================================================================
# 6. MASTER EXECUTION ENGINE
# ==============================================================================
def send_mobile_alert(macro_data, fno_data_list, target_date_str, is_backtest):
    sender_email, sender_pass, recipient_email = os.environ.get("SENDER_EMAIL"), os.environ.get("SENDER_PASSWORD"), os.environ.get("RECIPIENT_EMAIL")
    if not all([sender_email, sender_pass, recipient_email]): return

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"{'⏪ BACKTEST' if is_backtest else '🚀 LIVE TRANSFORMER AI ALERT'} | {target_date_str}"
    msg['From'], msg['To'] = sender_email, recipient_email

    macro_color = "#28a745" if "LONG" in macro_data['direction'] else "#dc3545" if "SHORT" in macro_data['direction'] else "#ffc107"
    sim_warning = f"<div style='background-color: #fff3cd; color: #856404; padding: 10px; text-align: center; font-weight: bold; margin-bottom: 15px;'>⚠️ VALIDATION MODE: SHOWING ACTUAL OUTCOMES</div>" if is_backtest else ""

    html = f"""
    <html><body style="font-family: Arial, sans-serif; background-color: #f4f7f6; padding: 10px;">{sim_warning}
        <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 6px solid {macro_color};">
            <h3 style="margin-top: 0; color: #333;">🌍 MACRO REGIME (Deep Transformer Embeddings)</h3>
            <p style="font-size: 16px; color: #333; margin: 5px 0;">
                <b>Direction:</b> <span style="color: {macro_color}; font-weight: bold;">{macro_data['direction']}</span><br>
                <b>Expected Target:</b> {macro_data['target_display']} | <b>Expected Max Pain:</b> {macro_data['risk_pct']:.2f}%<br>
                <b>Structural Confidence:</b> {macro_data['conviction']:.2f}%
            </p>
        </div>
        <h3 style="color: #333;">⚡ MICRO F&O SWEEP (3/5 Dynamic Consensus)</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-size: 14px; background-color: white;">
          <tr bgcolor="#f8f9fa" style="color: #333; font-weight: bold;">
            <th>Asset</th><th>Consensus</th><th>AI Similarity</th><th>Execution Price</th><th>Max Hist. Pain (SL)</th><th>Expected Target</th><th>Result</th>
          </tr>"""
    
    for row in sorted(fno_data_list, key=lambda x: x['conviction'], reverse=True):
        dc = "#28a745" if "LONG" in row['direction'] else "#dc3545"
        html += f"<tr><td style='color: #0056b3;'><b>{row['asset']}</b></td><td style='color: {dc}; font-weight: bold;'>{row['direction']}</td><td>{row['conviction']:.2f}%</td><td>₹{row['entry']:.2f}</td><td style='color: #dc3545;'>₹{row['ai_stop']:.2f} (-{row['risk_pct']:.2f}%)</td><td style='color: {dc}; font-weight: bold;'>{row['target_display']}</td><td>{row['actual_outcome']}</td></tr>"
        
    html += "</table></body></html>"
    msg.attach(MIMEText(html, 'html'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls(); server.login(sender_email, sender_pass)
        server.sendmail(sender_email, recipient_email, msg.as_string()); server.quit()
        print("✅ Elite Quant Report Dispatched.")
    except Exception as e: print(f"Failed to send email: {str(e)}")

def run_production_sweep():
    set_seeds(42)
    dt_str = os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    is_bt = bool(dt_str)
    if not is_bt: dt_str = datetime.now().strftime("%Y-%m-%d")
        
    print(f"⚙️ EXECUTING DEEP TIME2VEC-TRANSFORMER ENGINE | DATE: {dt_str}")
    
    nifty_file = None
    for root, dirs, files in os.walk("."):
        for f in files:
            if "nifty" in f.lower() or "historical_indices.csv" in f.lower():
                nifty_file = os.path.join(root, f)
                break
        if nifty_file: break
                
    if not nifty_file:
        print("❌ FATAL: Could not locate a NIFTY CSV file in the directory.")
        return

    # ---------------------------------------------------------
    # PHASE 1: NIFTY MACRO TRANSFORMER
    # ---------------------------------------------------------
    print("\n🧠 PHASE 1: Building Nifty 50 Multi-Task Neural Memory...")
    Xs_n, Xm_n, Yp_n, Yr_n = build_deep_training_tensors(nifty_file, dt_str, min_pct=0.25)
    
    if Xs_n is None:
        print("❌ FATAL: Phase 1 (Nifty) data pipeline failed. Cannot proceed.")
        return
    
    n_model, n_faiss, n_yp, n_yr = train_high_end_brain(Xs_n, Xm_n, Yp_n, Yr_n)
    
    universe = get_fno_universe()
    n_key = next((i["key"] for i in universe if i["symbol"] in ["NIFTY 50", "NIFTY"]), None)
    n_seq, n_mac, n_entry = get_live_tensors("NIFTY 50", n_key, dt_str, is_bt, read_standard_csv(nifty_file))
    
    mac_rep = {'direction': "CHAOTIC 🟡", 'conviction': 0, 'risk_pct': 0, 'target_display': "N/A"}
    if n_seq is not None:
        n_model.eval()
        with torch.no_grad(): lat = n_model.encode(torch.tensor(n_seq).unsqueeze(0)).numpy()[0]
        
        fused = np.ascontiguousarray(np.hstack((lat, n_mac)).reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(fused)
        scores, idxs = n_faiss.search(fused, k=5)
        
        conv = (max(0.0, scores[0][0]) ** 0.5) * 100.0
        p_ret, p_rsk = n_yp[idxs[0]], n_yr[idxs[0]]
        pos, neg = sum(1 for r in p_ret if r > 0), sum(1 for r in p_ret if r < 0)
        
        if pos >= 3:
            pct, rsk = np.mean([r for r in p_ret if r > 0]), np.mean([r for r, pr in zip(p_rsk, p_ret) if pr > 0])
            mac_rep = {'direction': "LONG 🟢", 'conviction': conv, 'risk_pct': rsk, 'target_display': f"₹{n_entry * (1 + (pct / 100)):.2f} (+{pct:.2f}%)"}
        elif neg >= 3:
            pct, rsk = np.mean([r for r in p_ret if r < 0]), np.mean([r for r, pr in zip(p_rsk, p_ret) if pr < 0])
            mac_rep = {'direction': "SHORT 🔴", 'conviction': conv, 'risk_pct': rsk, 'target_display': f"₹{n_entry * (1 + (pct / 100)):.2f} ({pct:.2f}%)"}
            
    # ---------------------------------------------------------
    # PHASE 2: F&O MASTER TRANSFORMER (Global Setup Alignment)
    # ---------------------------------------------------------
    print("\n⚡ PHASE 2: Compiling Global F&O Deep Latent Space...")
    Xs_f, Xm_f, Yp_f, Yr_f = build_deep_training_tensors("historical_fno.csv", dt_str, min_pct=0.75)
    
    if Xs_f is None:
        print("❌ FATAL: Phase 2 (F&O) data pipeline failed. historical_fno.csv might be corrupted or lack rows.")
        return
        
    f_model, f_faiss, f_yp, f_yr = train_high_end_brain(Xs_f, Xm_f, Yp_f, Yr_f, epochs=30)
    
    print("🎯 Phase 3: Live Market Inference via Dynamic Attention...")
    final_data = []
    min_conv = float(os.environ.get("PARAM_MIN_CONVICTION", 80.00)) 
    fno_df = read_standard_csv("historical_fno.csv") if is_bt else None

    for asset in universe:
        seq, mac, entry = get_live_tensors(asset["symbol"], asset["key"], dt_str, is_bt, fno_df)
        if not is_bt: time.sleep(0.15) 
        if seq is None: continue
        
        f_model.eval()
        with torch.no_grad(): lat = f_model.encode(torch.tensor(seq).unsqueeze(0)).numpy()[0]
        
        fused = np.ascontiguousarray(np.hstack((lat, mac)).reshape(1, -1), dtype=np.float32)
        faiss.normalize_L2(fused)
        scores, idxs = f_faiss.search(fused, k=5)
        
        conv = (max(0.0, scores[0][0]) ** 0.5) * 100.0 
        if conv < min_conv: continue
            
        p_ret, p_rsk = f_yp[idxs[0]], f_yr[idxs[0]]
        pos, neg = sum(1 for r in p_ret if r > 0), sum(1 for r in p_ret if r < 0)
        
        if pos >= 3:
            dir_str, pct, rsk = "LONG 🟢", np.mean([r for r in p_ret if r > 0]), np.mean([r for r, pr in zip(p_rsk, p_ret) if pr > 0])
        elif neg >= 3:
            dir_str, pct, rsk = "SHORT 🔴", np.mean([r for r in p_ret if r < 0]), np.mean([r for r, pr in zip(p_rsk, p_ret) if pr < 0])
        else: continue
            
        if abs(pct) < 0.75: continue 
            
        tgt = entry * (1 + (pct / 100.0))
        sl = entry * (1 - (rsk / 100.0)) if pct > 0 else entry * (1 + (rsk / 100.0))
        
        out = "<b>Awaiting Market ⏳</b>"
        if is_bt and fno_df is not None:
            df_sym = fno_df[fno_df['Symbol'] == asset['symbol']].sort_values('Date').reset_index(drop=True)
            fut = df_sym[df_sym['Date'] >= dt_str]
            if len(fut) >= 2:
                fw = fut.iloc[:2] 
                if "LONG" in dir_str:
                    out = f"<span style='color: #dc3545;'>❌ STOP HIT (₹{sl:.2f})</span>" if fw['Low'].min() <= sl else f"<span style='color: #28a745;'>Closed ₹{fw['Close'].iloc[-1]:.2f} (+{((fw['Close'].iloc[-1]-entry)/entry)*100:.2f}%)</span>"
                else:
                    out = f"<span style='color: #dc3545;'>❌ STOP HIT (₹{sl:.2f})</span>" if fw['High'].max() >= sl else f"<span style='color: #28a745;'>Closed ₹{fw['Close'].iloc[-1]:.2f} ({((fw['Close'].iloc[-1]-entry)/entry)*100:.2f}%)</span>"
        
        final_data.append({'asset': asset["symbol"], 'direction': dir_str, 'conviction': conv, 'entry': entry, 'ai_stop': sl, 'risk_pct': rsk, 'target_display': f"₹{tgt:.2f} ({'+' if pct>0 else ''}{pct:.2f}%)", 'actual_outcome': out})

    send_mobile_alert(mac_rep, final_data, dt_str, is_bt)

if __name__ == "__main__":
    run_production_sweep()

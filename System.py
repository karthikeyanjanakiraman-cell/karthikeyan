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
from torch.utils.data import TensorDataset, DataLoader
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
# 1. UNIVERSAL CSV STANDARDIZER
# ==============================================================================
def read_and_standardize_csv(filename):
    if not os.path.exists(filename): return None
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print(f"⚠️ Error reading {filename}: {e}")
        return None
        
    rename_map = {}
    for c in df.columns:
        cl = str(c).lower().strip()
        if cl in ['date', 'time', 'timestamp', 'datetime']: rename_map[c] = 'Date'
        elif cl in ['symbol', 'ticker', 'asset', 'instrument']: rename_map[c] = 'Symbol'
        elif cl in ['open', 'o']: rename_map[c] = 'Open'
        elif cl in ['high', 'h']: rename_map[c] = 'High'
        elif cl in ['low', 'l']: rename_map[c] = 'Low'
        elif cl in ['close', 'c']: rename_map[c] = 'Close'
        elif cl in ['volume', 'vol', 'v']: rename_map[c] = 'Volume'
        
    df = df.rename(columns=rename_map)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True).dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=['Date'])
        
    return df

# ==============================================================================
# 2. INDICATOR ENGINE
# ==============================================================================
def add_holy_trinity(df):
    df = df.copy()
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/7, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/7, adjust=False).mean()
    rs = gain / (loss + 1e-8)
    df['RSI_7'] = 100 - (100 / (1 + rs))

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

    atr7 = tr.ewm(alpha=1/7, adjust=False).mean()
    hl2 = (df['High'] + df['Low']) / 2
    upperband = hl2 + (2 * atr7)
    lowerband = hl2 - (2 * atr7)
    
    st = [0.0] * len(df)
    st_dir = [1] * len(df)
    
    close_vals = df['Close'].values
    ub_vals = upperband.values
    lb_vals = lowerband.values
    
    for i in range(1, len(df)):
        st_dir[i] = st_dir[i-1]
        if close_vals[i] > ub_vals[i-1]: st_dir[i] = 1
        elif close_vals[i] < lb_vals[i-1]: st_dir[i] = -1
        
        if st_dir[i] == 1: st[i] = max(lb_vals[i], st[i-1] if st_dir[i-1]==1 else 0)
        else: st[i] = min(ub_vals[i], st[i-1] if st_dir[i-1]==-1 else float('inf'))
            
    df['ST_7_2'] = st
    df['ST_Dist'] = (df['Close'] - df['ST_7_2']) / (df['Close'] + 1e-8)
    
    df = df.ffill().bfill()
    return df

# ==============================================================================
# 3. PROPER MATRIX NORMALIZATION
# ==============================================================================
def normalize_window(window_matrix):
    """Jointly scales OHLC. Leaves fixed-bound oscillators mathematically absolute."""
    norm = np.zeros_like(window_matrix)
    
    p_min, p_max = window_matrix[:, 0:4].min(), window_matrix[:, 0:4].max()
    norm[:, 0:4] = (window_matrix[:, 0:4] - p_min) / (p_max - p_min + 1e-8)
    
    v_min, v_max = window_matrix[:, 4].min(), window_matrix[:, 4].max()
    norm[:, 4] = (window_matrix[:, 4] - v_min) / (v_max - v_min + 1e-8)
    
    norm[:, 5] = window_matrix[:, 5] / 100.0
    norm[:, 6] = window_matrix[:, 6] / 100.0
    norm[:, 7] = window_matrix[:, 7] * 10.0 
    return norm

# ==============================================================================
# 4. TEMPORAL AUTOENCODER
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
# 5. DATA LOADER (Fixed "Perfect Hindsight" Bug)
# ==============================================================================
def load_training_data(csv_filename, target_date_str=None, min_pct=2.0):
    df = read_and_standardize_csv(csv_filename)
    if df is None or 'Date' not in df.columns: return None, None, None
    if target_date_str: df = df[df['Date'] <= target_date_str]
    
    training_matrices, price_targets = [], []
    
    for symbol, group in df.groupby('Symbol') if 'Symbol' in df.columns else [('ASSET', df)]:
        group = group.sort_values('Date').reset_index(drop=True)
        group = add_holy_trinity(group)
        values = group[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_7', 'ADX_14', 'ST_Dist']].values.astype(np.float32)
        sl_values = group['ST_7_2'].values.astype(np.float32)
        
        if len(values) < 32: continue 
        for i in range(len(values) - 32 + 1):
            raw_window = values[i : i+30, :]
            norm_window = normalize_window(raw_window)
            
            close_t = values[i+29, 3]    # Yesterday's Close
            sl_t = sl_values[i+29]       # Yesterday's Stop Loss
            entry_price = values[i+30, 0] # Today's Open (Execution Price)
            
            is_long = close_t > sl_t
            
            if is_long and entry_price <= sl_t: continue
            if not is_long and entry_price >= sl_t: continue
            
            future_closes = values[i+30 : i+32, 3]
            future_highs = values[i+30 : i+32, 1]
            future_lows = values[i+30 : i+32, 2]
            
            # THE FIX: Train exclusively on the final closing price, NOT the peak high/low
            final_close = future_closes[-1]
            
            if is_long:
                if future_lows.min() <= sl_t:
                    actual_pct_move = ((sl_t - entry_price) / entry_price) * 100.0
                else:
                    actual_pct_move = ((final_close - entry_price) / entry_price) * 100.0
            else:
                if future_highs.max() >= sl_t:
                    actual_pct_move = ((sl_t - entry_price) / entry_price) * 100.0
                else:
                    actual_pct_move = ((final_close - entry_price) / entry_price) * 100.0
            
            if abs(actual_pct_move) < min_pct: continue
            
            training_matrices.append(norm_window.T)
            price_targets.append(actual_pct_move)
            
    return np.array(training_matrices, dtype=np.float32), np.array(price_targets, dtype=np.float32), None

# ==============================================================================
# 6. STATELESS AI BRAIN
# ==============================================================================
def get_ai_brain(X_raw, Y_price, prefix="fno"):
    print(f"⚙️ Initiating Deep Training (50 Epochs) for [{prefix}]...")
    if X_raw is None or len(X_raw) == 0: 
        print(f"❌ ERROR: Matrix for [{prefix}] is empty.")
        return None, None, None, None
        
    cnn_model = TemporalAutoencoder()
    X_tensor = torch.tensor(X_raw, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    cnn_model.train()
    for e in range(50):
        for batch_x, _ in dataloader:
            optimizer.zero_grad()
            reconstructed, _ = cnn_model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

    cnn_model.eval()
    with torch.no_grad(): latent_vectors = cnn_model.encode(X_tensor).numpy()
    latent_vectors = np.ascontiguousarray(latent_vectors, dtype=np.float32)
    xgb_price = xgb.XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42).fit(latent_vectors, Y_price)
    
    faiss.normalize_L2(latent_vectors)
    index = faiss.IndexFlatIP(12) 
    index.add(latent_vectors)
    return cnn_model, xgb_price, index, Y_price

# ==============================================================================
# 7. LIVE INGESTION
# ==============================================================================
def fetch_915_open_from_upstox(instrument_key, target_date_str):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token or not instrument_key: return None
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            candles = response.json().get('data', {}).get('candles', [])
            if candles:
                for c in candles:
                    if target_date_str in str(c[0]) and "09:15" in str(c[0]): return float(c[1]) 
                return float(candles[0][1]) 
    except: pass
    return None

def fetch_live_data_and_indicators(csv_filename, target_date_str, instrument_key=None, is_nifty=False, is_backtest=False):
    df = read_and_standardize_csv(csv_filename)
    if df is None: return None, None, None, None, None
    if is_nifty and 'Symbol' in df.columns: df = df[df['Symbol'].astype(str).str.contains("NIFTY")]
        
    df_window = df[df['Date'] < target_date_str].sort_values('Date').reset_index(drop=True)
    if len(df_window) < 30: return None, None, None, None, None
    
    df_ind = add_holy_trinity(df_window)
    values = df_ind[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_7', 'ADX_14', 'ST_Dist']].values.astype(np.float32)[-30:]
    
    close_t = float(df_ind['Close'].iloc[-1])
    current_sl = float(df_ind['ST_7_2'].iloc[-1]) 
    
    entry_price = None
    actual_date = target_date_str
    
    next_days = df[df['Date'] >= target_date_str].sort_values('Date')
    if not next_days.empty:
        entry_price = float(next_days.iloc[0]['Open'])
        actual_date = next_days.iloc[0]['Date'] 
    else:
        if not is_backtest and instrument_key:
            entry_price = fetch_915_open_from_upstox(instrument_key, target_date_str)
            
    if entry_price is None: return None, None, None, None, None 
        
    norm_window = normalize_window(values)
    return norm_window.T, entry_price, current_sl, close_t, actual_date

def get_dynamic_fno_universe():
    try:
        response = requests.get("https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz", timeout=10)
        if response.status_code != 200: return {}
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        fno_underlying = {i.get("underlying_symbol") for i in nse_data if i.get("segment") == "NSE_FO" and i.get("underlying_symbol")}
        return {i.get("trading_symbol"): i.get("instrument_key") for i in nse_data if i.get("segment") in ("NSE_EQ", "NSE_INDEX") and i.get("trading_symbol") in fno_underlying}
    except: return {}

# ==============================================================================
# 8. MASTER EXECUTION
# ==============================================================================
def run_production_sweep():
    set_deterministic_seeds(42)
    raw_date_str = os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    is_backtest = bool(raw_date_str)
    target_date_str = pd.to_datetime(raw_date_str, dayfirst=True).strftime("%Y-%m-%d") if is_backtest else datetime.now().strftime("%Y-%m-%d")
    
    env_conv = os.environ.get("PARAM_MIN_CONVICTION", "").strip()
    min_user_conviction = float(env_conv) if env_conv else 95.00
    print(f"⚙️ EXECUTING DATE: {target_date_str} | MODE: {'BACKTEST' if is_backtest else 'LIVE (9:15 AM Open LTP)'} | MIN CONVICTION: {min_user_conviction}%")
    
    nifty_file = "historical_indices.csv"
    if not os.path.exists(nifty_file):
        for r, d, f in os.walk("."):
            for file in f:
                if "nifty" in file.lower() or "indices" in file.lower(): nifty_file = os.path.join(r, file); break
    
    X_nifty, Y_np, _ = load_training_data(nifty_file, target_date_str, min_pct=0.75)
    nifty_cnn, nifty_xgb_p, nifty_faiss, nifty_hist_y = get_ai_brain(X_nifty, Y_np, prefix="macro")
    if not nifty_cnn: return 
    
    nifty_map = get_dynamic_fno_universe()
    nifty_key = nifty_map.get("NIFTY 50", nifty_map.get("NIFTY", None))
    n_mat, n_ltp, n_sl, n_close, act_date = fetch_live_data_and_indicators(nifty_file, target_date_str, nifty_key, True, is_backtest)
    
    macro_report = {'direction': "UNKNOWN", 'conviction': 0, 'target_display': "N/A"}
    if n_mat is not None and n_ltp is not None:
        live_t = torch.tensor(n_mat, dtype=torch.float32).unsqueeze(0)
        n_lat = np.ascontiguousarray(nifty_cnn.encode(live_t).detach().numpy(), dtype=np.float32)
        n_pct = float(nifty_xgb_p.predict(n_lat)[0])
        
        faiss.normalize_L2(n_lat)
        k_search = min(5, nifty_faiss.ntotal)
        n_conviction = max(0.0, nifty_faiss.search(n_lat, k_search)[0][0][0]) * 100 if k_search > 0 else 0
        
        macro_dir = "LONG 🟢" if n_close > n_sl else "SHORT 🔴"
        macro_report = {'direction': macro_dir, 'conviction': float(n_conviction), 'target_display': f"₹{n_ltp * (1 + (n_pct / 100)):.2f} ({'+' if n_pct>0 else ''}{n_pct:.2f}%)"}

    fno_file = "historical_fno.csv"
    if not os.path.exists(fno_file): return
        
    X_fno, Y_fp, _ = load_training_data(fno_file, target_date_str, min_pct=3.0)
    fno_cnn, fno_xgb_p, fno_faiss, fno_hist_y = get_ai_brain(X_fno, Y_fp, prefix="micro")
    if not fno_cnn: return 
    
    final_report = []
    fno_df = read_and_standardize_csv(fno_file)
    symbols = fno_df['Symbol'].unique() if fno_df is not None and 'Symbol' in fno_df.columns else []

    print(f"🎯 Phase 3: Sweeping Universe (Targeting Conviction >= {min_user_conviction}%)...")
    for sym in symbols:
        sym_file = f"temp_{sym}.csv"
        fno_df[fno_df['Symbol'] == sym].to_csv(sym_file, index=False)
        mat, entry_price, sl, close_t, actual_date = fetch_live_data_and_indicators(sym_file, target_date_str, nifty_map.get(sym), False, is_backtest)
        if os.path.exists(sym_file): os.remove(sym_file)
        if mat is None or entry_price is None or sl is None: continue
        
        live_t = torch.tensor(mat, dtype=torch.float32).unsqueeze(0)
        lat = np.ascontiguousarray(fno_cnn.encode(live_t).detach().numpy(), dtype=np.float32)
        pred_pct = float(fno_xgb_p.predict(lat)[0])
        
        # Physics Filters
        actual_trend = "LONG" if close_t > sl else "SHORT"
        predicted_trend = "LONG" if pred_pct > 0 else "SHORT"
        
        if actual_trend != predicted_trend:
            print(f"   [FILTERED] {sym}: AI predicted reversal. Rejected.")
            continue
        if actual_trend == "LONG" and entry_price <= sl:
            print(f"   [FILTERED] {sym}: Gapped down past Stop Loss. Rejected.")
            continue
        if actual_trend == "SHORT" and entry_price >= sl:
            print(f"   [FILTERED] {sym}: Gapped up past Stop Loss. Rejected.")
            continue
            
        direction = f"{actual_trend} " + ("🟢" if actual_trend == "LONG" else "🔴")
        target_price = entry_price * (1 + (pred_pct / 100))
        
        risk = abs(entry_price - sl)
        reward = abs(target_price - entry_price)
        if risk == 0 or reward < risk:
            print(f"   [FILTERED] {sym}: Risk (₹{risk:.2f}) > Reward (₹{reward:.2f}). Rejected.")
            continue

        faiss.normalize_L2(lat)
        k_search = min(5, fno_faiss.ntotal)
        if k_search == 0: continue
            
        scores, indices = fno_faiss.search(lat, k_search)
        raw_conviction = max(0.0, scores[0][0]) * 100
        historical_outcomes = fno_hist_y[indices[0]]
        consensus = sum(1 for y in historical_outcomes if (y>0 if pred_pct>0 else y<0)) / float(k_search)
        if consensus < 0.6: 
            print(f"   [FILTERED] {sym}: Historical consensus failed. Rejected.")
            continue 
        
        final_conviction = raw_conviction * consensus
        
        if final_conviction >= min_user_conviction:
            print(f"   🌟 [ACCEPTED] {sym} | Conviction: {final_conviction:.1f}%")
            outcome_text = "<b>Awaiting Market ⏳</b>"
            if is_backtest:
                df_sym = fno_df[fno_df['Date'] >= actual_date].sort_values('Date')
                if len(df_sym) >= 2:
                    fw = df_sym.iloc[:2]
                    mx, mn = fw['High'].max(), fw['Low'].min()
                    if "LONG" in direction: outcome_text = f"<span style='color: #dc3545;'>❌ STOP HIT (₹{sl:.2f})</span>" if mn <= sl else f"<span style='color: #28a745;'>✅ Closed ₹{mx:.2f}</span>"
                    else: outcome_text = f"<span style='color: #dc3545;'>❌ STOP HIT (₹{sl:.2f})</span>" if mx >= sl else f"<span style='color: #28a745;'>✅ Closed ₹{mn:.2f}</span>"
            
            final_report.append({
                'asset': sym, 'direction': direction, 'conviction': final_conviction,
                'entry': float(entry_price), 'sl': float(sl), 
                'target': f"₹{target_price:.2f} ({'+' if pred_pct>0 else ''}{pred_pct:.2f}%)", 'actual': outcome_text
            })
        else:
            print(f"   [FILTERED] {sym}: Conviction {final_conviction:.1f}% < {min_user_conviction}%. Rejected.")

    # Email Dispatch
    if not final_report: 
        print(f"\n⚠️ Result: NO TRADES MET THE {min_user_conviction}% THRESHOLD TODAY.")

    macro_color = "#28a745" if "LONG" in macro_report['direction'] else "#dc3545"
    sim_warning = f"<div style='background-color: #fff3cd; color: #856404; padding: 10px; text-align: center; font-weight: bold; margin-bottom: 15px;'>⚠️ VALIDATION MODE: ACTUAL OUTCOMES</div>" if is_backtest else ""

    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f7f6; padding: 10px;">
        {sim_warning}
        <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 6px solid {macro_color};">
            <h3 style="margin-top: 0; color: #333;">🌍 MACRO REGIME (NIFTY 50)</h3>
            <p style="font-size: 16px; color: #333; margin: 5px 0;">
                <b>Trend:</b> <span style="color: {macro_color}; font-weight: bold;">{macro_report['direction']}</span><br>
                <b>AI Target:</b> {macro_report['target_display']} | <b>Conviction:</b> {macro_report['conviction']:.2f}%
            </p>
        </div>
        <h3 style="color: #333;">⚡ MICRO F&O SWEEP (RR >= 1:1, Conviction >= {min_user_conviction}%)</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-size: 14px; background-color: white;">
          <tr bgcolor="#f8f9fa" style="color: #333; font-weight: bold;">
            <th>Asset</th><th>Signal</th><th>Conviction</th><th>Entry (9:15 Open)</th><th>Stop Loss</th><th>AI Target</th><th>Result</th>
          </tr>"""
          
    if not final_report:
        html += "<tr><td colspan='7' style='padding: 20px; color: #666;'>No assets passed the physics/conviction filters today. Cash is a position.</td></tr>"
    else:
        for r in sorted(final_report, key=lambda x: x['conviction'], reverse=True):
            dir_color = "#28a745" if "LONG" in r['direction'] else "#dc3545"
            html += f"""<tr>
                <td style="color: #0056b3;"><b>{r['asset']}</b></td>
                <td style="color: {dir_color}; font-weight: bold;">{r['direction']}</td>
                <td>{r['conviction']:.2f}%</td>
                <td>₹{r['entry']:.2f}</td>
                <td style="color: #dc3545;">₹{r['sl']:.2f}</td>
                <td style="color: {dir_color}; font-weight: bold;">{r['target']}</td>
                <td>{r['actual']}</td>
              </tr>"""
    html += "</table></body></html>"

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"{'⏪ BACKTEST' if is_backtest else '🚀 LIVE ALERT'} | {target_date_str}"
    sender, recipient, pswd = os.environ.get("SENDER_EMAIL"), os.environ.get("RECIPIENT_EMAIL"), os.environ.get("SENDER_PASSWORD")
    if sender and recipient and pswd:
        msg['From'], msg['To'] = sender, recipient
        msg.attach(MIMEText(html, 'html'))
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587); server.starttls(); server.login(sender, pswd)
            server.sendmail(sender, recipient, msg.as_string()); server.quit()
            print("\n✅ Alert Dispatched via Email!")
        except Exception as e: print(f"\n❌ Failed to send email: {e}")
    else: print("\n⚠️ Email skipped (Credentials missing).")

if __name__ == "__main__":
    run_production_sweep()

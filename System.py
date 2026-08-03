import os
import sys
import argparse
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
# 1. DUAL-INPUT TEMPORAL AUTOENCODER (The Siamese Multi-Timeframe Brain)
# ==============================================================================
class MultiTimeframeAutoencoder(nn.Module):
    def __init__(self, num_features=5, latent_dim_daily=12, latent_dim_weekly=12):
        super(MultiTimeframeAutoencoder, self).__init__()
        
        # Branch A: Processes the 30-day Daily Matrix (5 features x 30 steps)
        self.encoder_daily = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # 30 -> 15
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),  # 15 -> 5
            
            nn.Flatten(),
            nn.Linear(32 * 5, latent_dim_daily)
        )
        
        # Branch B: Processes the 15-week Weekly Matrix (5 features x 15 steps)
        self.encoder_weekly = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),  # 15 -> 5
            
            nn.Flatten(),
            nn.Linear(16 * 5, latent_dim_weekly)
        )
        
        # Decoder A: Reconstructs back to the original 30-day Daily Shape
        self.decoder_daily = nn.Sequential(
            nn.Linear(latent_dim_daily, 32 * 5),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (32, 5)),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=3, output_padding=0),  # 5 -> 15
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, num_features, kernel_size=2, stride=2, output_padding=0),  # 15 -> 30
            nn.Sigmoid()
        )
        
        # Decoder B: Reconstructs back to the original 15-week Weekly Shape
        self.decoder_weekly = nn.Sequential(
            nn.Linear(latent_dim_weekly, 16 * 5),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (16, 5)),
            nn.ConvTranspose1d(16, num_features, kernel_size=3, stride=3, output_padding=0),  # 5 -> 15
            nn.Sigmoid()
        )

    def encode(self, x_daily, x_weekly):
        ld = self.encoder_daily(x_daily)
        lw = self.encoder_weekly(x_weekly)
        return torch.cat((ld, lw), dim=1)  # Master 24-Dimensional Feature Map

    def forward(self, x_daily, x_weekly):
        ld = self.encoder_daily(x_daily)
        lw = self.encoder_weekly(x_weekly)
        recon_d = self.decoder_daily(ld)
        recon_w = self.decoder_weekly(lw)
        return recon_d, recon_w, torch.cat((ld, lw), dim=1)

# ==============================================================================
# 2. ALIGNED DUAL-TIMEFRAME TRAINING DATA LOADER
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
    if df is None or 'Date' not in df.columns:
        print(f"⚠️ Warning: Missing or invalid '{csv_filename}'")
        return None, None, None, None
        
    if target_date_str:
        df = df[df['Date'] <= target_date_str]
    
    training_daily = []
    training_weekly = []
    price_targets = []
    time_targets = []
    FUTURE_DAYS = 2 
    
    if "historical_indices" in csv_filename.lower() or "nifty" in csv_filename.lower():
        if 'Symbol' in df.columns:
            mask = df['Symbol'].astype(str).str.upper().str.replace("_", "").str.replace(" ", "").str.contains("NIFTY50|NIFTY")
            if mask.any(): df = df[mask]

    for symbol, group in df.groupby('Symbol') if 'Symbol' in df.columns else [('ASSET', df)]:
        group = group.sort_values('Date').reset_index(drop=True)
        group['Datetime'] = pd.to_datetime(group['Date'])
        
        # Generate the parallel macro Weekly resampled mapping
        group_w = group.set_index('Datetime').resample('W').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'Date': 'last'
        }).reset_index().sort_values('Date').reset_index(drop=True)
        
        values_d = group[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
        dates_d = group['Date'].values
        
        if len(values_d) < (30 + FUTURE_DAYS): continue
            
        for i in range(len(values_d) - (30 + FUTURE_DAYS) + 1):
            end_date_d = dates_d[i + 29]
            
            # Anti-leakage extraction of historical weeks ending on or before current daily date
            sub_w = group_w[group_w['Date'] <= end_date_d]
            if len(sub_w) < 15: continue
                
            raw_window_w = sub_w[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-15:]
            raw_window_d = values_d[i : i+30]
            
            # Normalize Daily Space (0 to 1)
            w_min_d = raw_window_d.min(axis=0)
            w_max_d = raw_window_d.max(axis=0)
            norm_window_d = (raw_window_d - w_min_d) / (w_max_d - w_min_d + 1e-8)
            
            # Normalize Weekly Space (0 to 1)
            w_min_w = raw_window_w.min(axis=0)
            w_max_w = raw_window_w.max(axis=0)
            norm_window_w = (raw_window_w - w_min_w) / (w_max_w - w_min_w + 1e-8)
            
            future_closes = values_d[i+30 : i+30+FUTURE_DAYS, 3]
            future_highs  = values_d[i+30 : i+30+FUTURE_DAYS, 1]
            future_lows   = values_d[i+30 : i+30+FUTURE_DAYS, 2]
            start_price   = values_d[i+29, 3] 
            
            max_close, min_close = future_closes.max(), future_closes.min()
            
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
                
            if abs(actual_pct_move) < min_pct or abs(actual_pct_move) > max_pct or start_price < 10.0: continue
            if is_long and (actual_drawdown < -max_dd or rejection_wick > (actual_pct_move * wick_ratio)): continue 
            if not is_long and (actual_drawdown > max_dd or rejection_wick > (abs(actual_pct_move) * wick_ratio)): continue 

            days_to_target = float(np.argmax(np.abs(future_closes - start_price)) + 1)
            
            training_daily.append(norm_window_d.T)
            training_weekly.append(norm_window_w.T)
            price_targets.append(actual_pct_move)
            time_targets.append(days_to_target)
            
    return np.array(training_daily, dtype=np.float32), np.array(training_weekly, dtype=np.float32), np.array(price_targets, dtype=np.float32), np.array(time_targets, dtype=np.float32)

# ==============================================================================
# 3. MULTI-TIMEFRAME BRAIN TRAINING ENGINE
# ==============================================================================
def train_ai_brain(X_daily, X_weekly, Y_price, Y_time, epochs=15):
    X_d_tensor = torch.tensor(X_daily)
    X_w_tensor = torch.tensor(X_weekly)
    
    model = MultiTimeframeAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    model.train()
    for _ in range(epochs): 
        optimizer.zero_grad()
        recon_d, recon_w, _ = model(X_d_tensor, X_w_tensor)
        loss = criterion(recon_d, X_d_tensor) + criterion(recon_w, X_w_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        latent_vectors = model.encode(X_d_tensor, X_w_tensor).numpy()
        
    xgb_price = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4).fit(latent_vectors, Y_price)
    xgb_time = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4).fit(latent_vectors, Y_time)

    faiss.normalize_L2(latent_vectors)
    index = faiss.IndexFlatIP(24)  # 12 (Daily Dimensionality) + 12 (Weekly Dimensionality) = 24
    index.add(latent_vectors)
    
    return model, xgb_price, xgb_time, index

# ==============================================================================
# 4. LIVE INGESTION PACKS (Local CSV & Upstox Multi-Timeframe Extractor)
# ==============================================================================
def get_live_tensor_from_csv(csv_filename, target_date_str):
    df = read_and_standardize_csv(csv_filename)
    if df is None or 'Date' not in df.columns: return None, None, None
    
    if 'Symbol' in df.columns:
        mask = df['Symbol'].astype(str).str.upper().str.replace("_", "").str.replace(" ", "").str.contains("NIFTY50|NIFTY")
        df = df[mask] if mask.any() else df[df['Symbol'] == df['Symbol'].unique()[0]]
            
    df['Datetime'] = pd.to_datetime(df['Date'])
    df_w = df.set_index('Datetime').resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'Date': 'last'
    }).reset_index().sort_values('Date').reset_index(drop=True)
    
    past_d = df[df['Date'] <= target_date_str].sort_values('Date').reset_index(drop=True)
    past_w = df_w[df_w['Date'] <= target_date_str].sort_values('Date').reset_index(drop=True)
    
    if len(past_d) < 30 or len(past_w) < 15: return None, None, None
    
    vals_d = past_d[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-30:]
    vals_w = past_w[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-15:]
    
    current_ltp = vals_d[-1, 3]
    norm_d = (vals_d - vals_d.min(axis=0)) / (vals_d.max(axis=0) - vals_d.min(axis=0) + 1e-8)
    norm_w = (vals_w - vals_w.min(axis=0)) / (vals_w.max(axis=0) - vals_w.min(axis=0) + 1e-8)
    
    return norm_d.T, norm_w.T, current_ltp

def get_dynamic_fno_universe():
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    response = requests.get(nse_url)
    if response.status_code != 200: return []
    try:
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        fno_underlying = {item.get("underlying_symbol") for item in nse_data if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")}
        return [{"symbol": item.get("trading_symbol"), "key": item.get("instrument_key")} for item in nse_data if item.get("segment") in ("NSE_EQ", "NSE_INDEX") and item.get("trading_symbol") in fno_underlying]
    except:
        return []

def fetch_upstox_data(instrument_key, target_date_str, days_back=180):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    to_date = target_date_str
    from_date = (target_dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
    
    url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(instrument_key)}/day/{to_date}/{from_date}"
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200: return None
        
    data = response.json().get('data', {}).get('candles', [])
    if not data or len(data) < 40: return None
        
    # Convert raw list into pandas framework to execute aligned timeframe conversion
    c_df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
    c_df = c_df.iloc[::-1].reset_index(drop=True)
    c_df['Date'] = c_df['Date'].astype(str).str[:10]
    c_df['Datetime'] = pd.to_datetime(c_df['Date'])
    
    past_d = c_df[c_df['Date'] <= target_date_str].sort_values('Date').reset_index(drop=True)
    if len(past_d) < 30: return None
        
    current_ltp = float(past_d.iloc[-1]['Close'])
    raw_d = past_d[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-30:]
    norm_d = (raw_d - raw_d.min(axis=0)) / (raw_d.max(axis=0) - raw_d.min(axis=0) + 1e-8)
    
    past_w = past_d.set_index('Datetime').resample('W').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'Date': 'last'
    }).reset_index().sort_values('Date').reset_index(drop=True)
    
    if len(past_w) < 15: return None
    raw_w = past_w[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-15:]
    norm_w = (raw_w - raw_w.min(axis=0)) / (raw_w.max(axis=0) - raw_w.min(axis=0) + 1e-8)
    
    return norm_d.T, norm_w.T, current_ltp

# ==============================================================================
# 5. EXECUTION CORE & EMAIL ALERTS SYSTEM
# ==============================================================================
def send_mobile_alert(macro_data, fno_data_list, target_date_str, is_backtest):
    sender_email, sender_pass, recipient_email = os.environ.get("SENDER_EMAIL"), os.environ.get("SENDER_PASSWORD"), os.environ.get("RECIPIENT_EMAIL")
    if not all([sender_email, sender_pass, recipient_email]): return

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"{'⏪ BACKTEST' if is_backtest else '🚀 MULTI-TIMEFRAME LIVE'} | {target_date_str}"
    msg['From'], msg['To'] = sender_email, recipient_email

    macro_color = "#28a745" if "LONG" in macro_data['direction'] else "#dc3545"
    sim_warning = "<div style='background-color: #fff3cd; color: #856404; padding: 10px; text-align: center; font-weight: bold; margin-bottom: 15px;'>⚠️ VALIDATION RUN ACTIVE</div>" if is_backtest else ""

    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #f4f7f6; padding: 10px;">
        {sim_warning}
        <div style="background-color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 6px solid {macro_color};">
            <h3 style="margin-top: 0; color: #333;">🌍 DUAL-TIMEFRAME MACRO REGIME (NIFTY 50)</h3>
            <p style="font-size: 16px; color: #333; margin: 5px 0;">
                <b>Direction:</b> <span style="color: {macro_color}; font-weight: bold;">{macro_data['direction']}</span><br>
                <b>AI Target:</b> {macro_data['target_display']} | <b>Conviction:</b> {macro_data['conviction']:.2f}%
            </p>
        </div>
        <h3 style="color: #333;">⚡ SNIPER F&O SCOPE SWEEP</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-size: 14px; background-color: white;">
          <tr bgcolor="#f8f9fa" style="color: #333; font-weight: bold;">
            <th>Asset</th><th>Signal</th><th>Trend Match?</th><th>Score</th><th>Current LTP</th><th>AI Target</th><th>Result (2-Day Close)</th>
          </tr>
    """
    fno_data_list.sort(key=lambda x: x['conviction'], reverse=True)
    for row in fno_data_list:
        dir_color = "#28a745" if "LONG" in row['direction'] else "#dc3545"
        html_content += f"""
          <tr>
            <td style="color: #0056b3;"><b>{row['asset']}</b></td>
            <td style="color: {dir_color}; font-weight: bold;">{row['direction']}</td>
            <td>{"✅" if row['direction'] == macro_data['direction'] else "⚠️"}</td>
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
        print(f"✅ Alert Dispatched with {len(fno_data_list)} dual-timeframe filtered targets.")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def run_production_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", type=str, default="")
    parser.add_argument("positional_date", nargs="?", default="")
    args, _ = parser.parse_known_args()

    raw_date_str = args.date or args.positional_date or os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    is_backtest = bool(raw_date_str)

    if not is_backtest:
        target_date_str = datetime.now().strftime("%Y-%m-%d")
    else:
        try:
            target_date_str = datetime.strptime(raw_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            print(f"❌ Critical Error: Date '{raw_date_str}' is invalid. Use YYYY-MM-DD.")
            return

    print(f"⚙️ METRICS DATE ACTIVE: {target_date_str} | MODE: {'BACKTEST' if is_backtest else 'LIVE'}")
    
    nifty_file = None
    for root, _, files in os.walk("."):
        for file in files:
            if ("nifty" in file.lower() or "historical_indices" in file.lower()) and file.lower().endswith(".csv"):
                nifty_file = os.path.join(root, file)
                break
        if nifty_file: break

    if not nifty_file:
        print("❌ Critical Error: No Nifty data sets located.")
        return

    # ==========================================
    # PHASE 1: MACRO NIFTY MATRIX (DUAL ENCODER)
    # ==========================================
    print(f"\n🧠 TRAINING PHASE 1: Processing Macro System on {nifty_file}...")
    X_d_nifty, X_w_nifty, Y_np, Y_nt = load_training_data(nifty_file, target_date_str, min_pct=0.75, max_pct=5.0, max_dd=0.5, wick_ratio=0.5)
    if X_d_nifty is None or len(X_d_nifty) == 0: return

    nifty_brain, nifty_xgb_p, nifty_xgb_t, nifty_faiss = train_ai_brain(X_d_nifty, X_w_nifty, Y_np, Y_nt)
    nifty_live_d, nifty_live_w, nifty_ltp = get_live_tensor_from_csv(nifty_file, target_date_str)
    
    if nifty_live_d is not None:
        with torch.no_grad():
            n_latent = nifty_brain.encode(torch.tensor(nifty_live_d).unsqueeze(0), torch.tensor(nifty_live_w).unsqueeze(0)).numpy()
        n_pct = nifty_xgb_p.predict(n_latent)[0]
        faiss.normalize_L2(n_latent)
        n_score, _ = nifty_faiss.search(n_latent, k=5)
        
        macro_report = {
            'direction': "LONG 🟢" if n_pct > 0 else "SHORT 🔴",
            'conviction': float(n_score[0][0] * 100),
            'target_display': f"₹{nifty_ltp * (1 + (n_pct / 100)):.2f} ({'+' if n_pct>0 else ''}{n_pct:.2f}%)"
        }
    else:
        macro_report = {'direction': "UNKNOWN", 'conviction': 0, 'target_display': "N/A"}

    # ==========================================
    # PHASE 2: MICRO F&O BRAIN (DUAL ENCODER)
    # ==========================================
    print("\n⚡ TRAINING PHASE 2: Processing Multi-Timeframe F&O Universe...")
    X_d_fno, X_w_fno, Y_fp, Y_ft = load_training_data("historical_fno.csv", target_date_str, min_pct=4.0, max_pct=50.0, max_dd=1.2, wick_ratio=0.4)
    if X_d_fno is None or len(X_d_fno) == 0: return

    fno_brain, fno_xgb_p, fno_xgb_t, fno_faiss = train_ai_brain(X_d_fno, X_w_fno, Y_fp, Y_ft)
    fno_universe = get_dynamic_fno_universe()
    
    final_report_data = []
    min_conviction = float(os.environ.get("PARAM_MIN_CONVICTION", 97.0))

    for asset in fno_universe:
        result = fetch_upstox_data(asset["key"], target_date_str, days_back=180)
        time.sleep(0.15)
        if result is None: continue
        live_d, live_w, current_ltp = result
        
        with torch.no_grad():
            l_vector = fno_brain.encode(torch.tensor(live_d).unsqueeze(0), torch.tensor(live_w).unsqueeze(0)).numpy()
        
        pred_pct = fno_xgb_p.predict(l_vector)[0]
        faiss.normalize_L2(l_vector)
        score, _ = fno_faiss.search(l_vector, k=5)
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
            
    # Automated Multi-Timeframe Backtest Engine Validation
    if is_backtest and os.path.exists("historical_fno.csv") and final_report_data:
        df_full = read_and_standardize_csv("historical_fno.csv")
        if df_full is not None and 'Symbol' in df_full.columns:
            for row in final_report_data:
                df_sym = df_full[df_full['Symbol'] == row['asset']].sort_values('Date').reset_index(drop=True)
                past = df_sym[df_sym['Date'] <= target_date_str]
                if len(past) > 0 and past.index[-1] + 1 < len(df_sym):
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
                            row['actual_outcome'] = f"<span style='color: {c};'>{mv:.2f}%</span><br><span style='color: #856404; font-size: 11px;'>Max DD: +{dd:.2f}%</span>"

    send_mobile_alert(macro_report, final_report_data, target_date_str, is_backtest)

if __name__ == "__main__":
    run_production_sweep()


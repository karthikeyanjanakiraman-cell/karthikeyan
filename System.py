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
# 0. EQUI-PERCENTILE EQUATING
# ==============================================================================
def convert_to_equi_percentile(raw_matrix):
    ranks = np.argsort(np.argsort(raw_matrix, axis=0), axis=0)
    percentile_matrix = ranks.astype(np.float32) / (raw_matrix.shape[0] - 1 + 1e-8)
    return percentile_matrix

# ==============================================================================
# 1. DUAL-INPUT TEMPORAL AUTOENCODER (The Siamese Multi-Timeframe Brain)
# ==============================================================================
class MultiTimeframeAutoencoder(nn.Module):
    def __init__(self, num_features=5, latent_dim_daily=12, latent_dim_weekly=12):
        super(MultiTimeframeAutoencoder, self).__init__()
        
        self.encoder_daily = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(32 * 5, latent_dim_daily)
        )
        
        self.encoder_weekly = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(16 * 5, latent_dim_weekly)
        )
        
        self.decoder_daily = nn.Sequential(
            nn.Linear(latent_dim_daily, 32 * 5),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (32, 5)),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=3, output_padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(16, num_features, kernel_size=2, stride=2, output_padding=0),
            nn.Sigmoid()
        )
        
        self.decoder_weekly = nn.Sequential(
            nn.Linear(latent_dim_weekly, 16 * 5),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (16, 5)),
            nn.ConvTranspose1d(16, num_features, kernel_size=3, stride=3, output_padding=0),
            nn.Sigmoid()
        )

    def encode(self, x_daily, x_weekly):
        ld = self.encoder_daily(x_daily)
        lw = self.encoder_weekly(x_weekly)
        return torch.cat((ld, lw), dim=1)

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
        
        group_w = group.set_index('Datetime').resample('W').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum', 'Date': 'last'
        }).reset_index().sort_values('Date').reset_index(drop=True)
        
        values_d = group[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)
        dates_d = group['Date'].values
        
        if len(values_d) < (30 + FUTURE_DAYS): continue
            
        for i in range(len(values_d) - (30 + FUTURE_DAYS) + 1):
            end_date_d = dates_d[i + 29]
            sub_w = group_w[group_w['Date'] <= end_date_d]
            if len(sub_w) < 15: continue
                
            raw_window_w = sub_w[['Open', 'High', 'Low', 'Close', 'Volume']].values.astype(np.float32)[-15:]
            raw_window_d = values_d[i : i+30]
            
            norm_window_d = convert_to_equi_percentile(raw_window_d)
            norm_window_w = convert_to_equi_percentile(raw_window_w)
            
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
    index = faiss.IndexFlatIP(24)
    index.add(latent_vectors)
    
    return model, xgb_price, xgb_time, index

# ==============================================================================
# 3. LIVE INGESTION & UNIVERSE PACKS
# ==============================================================================
def get_dynamic_fno_universe():
    nse_url = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
    try:
        response = requests.get(nse_url, timeout=5)
        if response.status_code != 200: return []
        nse_data = json.load(gzip.GzipFile(fileobj=io.BytesIO(response.content)))
        fno_underlying = {item.get("underlying_symbol") for item in nse_data if item.get("segment") == "NSE_FO" and item.get("underlying_symbol")}
        return [{"symbol": item.get("trading_symbol"), "key": item.get("instrument_key")} for item in nse_data if item.get("segment") in ("NSE_EQ", "NSE_INDEX") and item.get("trading_symbol") in fno_underlying]
    except:
        return []

def fetch_upstox_intraday_candles(instrument_key, target_date_str):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        return None
    
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    # FORCE IST TIME
    today_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    
    if target_date_str == today_str:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    else:
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(instrument_key)}/1minute/{target_date_str}/{target_date_str}"
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None
            
        data = response.json().get('data', {}).get('candles', [])
        if not data:
            return None
            
        c_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        c_df['Datetime'] = pd.to_datetime(c_df['Timestamp']).dt.tz_localize(None) 
        c_df = c_df.sort_values('Datetime').reset_index(drop=True)
        return c_df
    except Exception:
        return None

# ==============================================================================
# 4. FIXED CUMULATIVE SCANNER WITH LIVE TIME-GUARD
# ==============================================================================
def scan_cumulative_top_turnover(target_date_str):
    print(f"\n⏳ Initializing Cumulative Session-to-Date Scan (09:15 Open to Checkpoint) for {target_date_str}...")
    universe = get_dynamic_fno_universe()
    if not universe:
        print("⚠️ No F&O universe found. Please check Upstox API.")
        return
        
    target_dt = pd.to_datetime(target_date_str)
    
    # FORCE IST TIME HERE
    current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    
    # Identify if we are running in live conditions today
    is_live_today = (target_date_str == current_now.strftime("%Y-%m-%d"))

    # Cumulative Checkpoint Logic
    checkpoints = [
        ("09:15 - 10:15", target_dt + pd.Timedelta(hours=10, minutes=15)),
        ("09:15 - 11:15", target_dt + pd.Timedelta(hours=11, minutes=15)),
        ("09:15 - 12:15", target_dt + pd.Timedelta(hours=12, minutes=15)),
        ("09:15 - 13:15", target_dt + pd.Timedelta(hours=13, minutes=15)),
        ("09:15 - 14:15", target_dt + pd.Timedelta(hours=14, minutes=15)),
        ("09:15 - 15:15", target_dt + pd.Timedelta(hours=15, minutes=15)),
        ("09:15 - 15:30", target_dt + pd.Timedelta(hours=15, minutes=30))
    ]
    
    print(f"📡 Downloading intraday data and computing cumulative session metrics for {len(universe)} stocks...")
    
    master_intraday_list = []
    for item in universe:
        df = fetch_upstox_intraday_candles(item['key'], target_date_str)
        if df is not None and not df.empty:
            df['Symbol'] = item['symbol']
            df['Turnover'] = df['Volume'] * df['Close']
            master_intraday_list.append(df)
            
    if not master_intraday_list:
        print("⚠️ Warning: No valid intraday market volume found yet.")
        return
        
    master_df = pd.concat(master_intraday_list, ignore_index=True)
    
    print("\n" + "="*125)
    print(f"🔥 TOP 5 CUMULATIVE TRENDERS (From 09:15 Open | Turnover PR × Momentum PR) | DATE: {target_date_str}")
    print("="*125)
    
    for label, checkpoint_time in checkpoints:
        
        # ⚠️ CRITICAL TIME GUARD: Break out of the loop if the checkpoint is in the future
        if is_live_today and checkpoint_time > current_now:
            print(f"\n⏸️ Current time ({current_now.strftime('%H:%M')} IST) has not reached checkpoint {checkpoint_time.strftime('%H:%M')} IST. Halting future projections.")
            break
            
        # Filter all candles up to the checkpoint
        df_cum = master_df[master_df['Datetime'] <= checkpoint_time]
        if df_cum.empty: 
            continue
            
        # Aggregate Cumulative Values
        grouped = df_cum.groupby('Symbol').agg({
            'Turnover': 'sum',
            'Volume': 'sum',
            'Open': 'first',
            'Close': 'last'
        }).reset_index()
        
        grouped = grouped[grouped['Turnover'] > 0]
        if grouped.empty: continue
        
        # Calculate Percentiles
        grouped['Turnover_PR'] = grouped['Turnover'].rank(pct=True) * 100
        
        # Calculate Absolute % Move (Proxy for Momentum)
        grouped['Pct_Move'] = ((grouped['Close'] - grouped['Open']) / grouped['Open']) * 100
        grouped['Momentum_PR'] = grouped['Pct_Move'].abs().rank(pct=True) * 100
        
        # Combine into Power Score
        grouped['Power_Score'] = grouped['Turnover_PR'] * grouped['Momentum_PR']
        
        # Sort and select Top 5
        top5 = grouped.nlargest(5, 'Power_Score')
        
        print(f"\n⏰ CUMULATIVE WINDOW: {label} IST")
        print(f"{'Rank':<5} {'Symbol':<15} {'Power Score':<12} | {'Turnover PR':<12} | {'Momentum PR':<12} | {'% Move':<8} {'LTP (₹)':<10}")
        print("-" * 125)
        
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            move_sign = "+" if row['Pct_Move'] > 0 else ""
            print(f"{rank:<5} {row['Symbol']:<15} {row['Power_Score']:<12.1f} | {row['Turnover_PR']:>9.2f} PR | {row['Momentum_PR']:>9.2f} PR | {move_sign}{row['Pct_Move']:<6.2f}%   ₹{row['Close']:<10.2f}")

# ==============================================================================
# 5. MAIN CONTROLLER
# ==============================================================================
def run_production_sweep():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", type=str, default="")
    parser.add_argument("positional_date", nargs="?", default="")
    args, _ = parser.parse_known_args()

    raw_date_str = args.date or args.positional_date or os.environ.get("PARAM_BACKTEST_DATE", "").strip()
    is_backtest = bool(raw_date_str)

    # FORCE IST TIME
    if not is_backtest:
        target_date_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
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

    print(f"\n🧠 TRAINING PHASE 1: Processing Macro System on {nifty_file}...")
    X_d_nifty, X_w_nifty, Y_np, Y_nt = load_training_data(nifty_file, target_date_str, min_pct=0.75, max_pct=5.0, max_dd=0.5, wick_ratio=0.5)
    
    if X_d_nifty is not None and len(X_d_nifty) > 0:
        nifty_brain, nifty_xgb_p, nifty_xgb_t, nifty_faiss = train_ai_brain(X_d_nifty, X_w_nifty, Y_np, Y_nt)
    
    scan_cumulative_top_turnover(target_date_str)

if __name__ == "__main__":
    run_production_sweep()

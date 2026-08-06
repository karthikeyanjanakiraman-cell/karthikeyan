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
# 4. ACCELERATION DELTA SCANNER (Discrete Hourly vs. Cumulative Session)
# ==============================================================================
def scan_discrete_hourly_turnover(target_date_str):
    print(f"\n⏳ Initializing Acceleration Delta Scanner for {target_date_str}...")
    universe = get_dynamic_fno_universe()
    if not universe:
        print("⚠️ No F&O universe found. Please check Upstox API.")
        return
        
    target_dt = pd.to_datetime(target_date_str)
    start_of_day = target_dt + pd.Timedelta(hours=9, minutes=15)
    
    # STRICT IST CLOCK ENFORCEMENT
    current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = (target_date_str == current_now.strftime("%Y-%m-%d"))

    windows = [
        (target_dt + pd.Timedelta(hours=9, minutes=15), target_dt + pd.Timedelta(hours=10, minutes=15), "09:15 - 10:15"),
        (target_dt + pd.Timedelta(hours=10, minutes=15), target_dt + pd.Timedelta(hours=11, minutes=15), "10:15 - 11:15"),
        (target_dt + pd.Timedelta(hours=11, minutes=15), target_dt + pd.Timedelta(hours=12, minutes=15), "11:15 - 12:15"),
        (target_dt + pd.Timedelta(hours=12, minutes=15), target_dt + pd.Timedelta(hours=13, minutes=15), "12:15 - 13:15"),
        (target_dt + pd.Timedelta(hours=13, minutes=15), target_dt + pd.Timedelta(hours=14, minutes=15), "13:15 - 14:15"),
        (target_dt + pd.Timedelta(hours=14, minutes=15), target_dt + pd.Timedelta(hours=15, minutes=15), "14:15 - 15:15"),
        (target_dt + pd.Timedelta(hours=15, minutes=15), target_dt + pd.Timedelta(hours=15, minutes=30), "15:15 - 15:30")
    ]
    
    print(f"📡 Downloading intraday data for {len(universe)} stocks...")
    
    master_intraday_list = []
    for item in universe:
        df = fetch_upstox_intraday_candles(item['key'], target_date_str)
        if df is not None and not df.empty:
            df['Symbol'] = item['symbol']
            df['Turnover'] = df['Volume'] * df['Close']
            df['abs_move'] = (df['Close'] - df['Open']).abs()
            master_intraday_list.append(df)
            
    if not master_intraday_list:
        print("⚠️ Warning: No valid intraday market volume found yet.")
        return
        
    master_df = pd.concat(master_intraday_list, ignore_index=True)
    
    print("\n" + "="*145)
    print(f"🔥 TOP 5 ACCELERATION DELTA BREAKOUTS (Discrete Hourly Score minus Cumulative Session Score) | DATE: {target_date_str}")
    print("="*145)
    
    for start_time, end_time, base_label in windows:
        if is_live_today and current_now < start_time:
            break
            
        is_active_live = False
        label = base_label
        
        if is_live_today and start_time <= current_now < end_time:
            is_active_live = True
            end_time = current_now
            label = f"{start_time.strftime('%H:%M')} - {current_now.strftime('%H:%M')} (LIVE ONGOING)"
            
        # 1. ISOLATED DISCRETE WINDOW DATA
        df_discrete = master_df[(master_df['Datetime'] >= start_time) & (master_df['Datetime'] < end_time)]
        # 2. CUMULATIVE SESSION DATA (From 09:15 up to current window end)
        df_cumulative = master_df[(master_df['Datetime'] >= start_of_day) & (master_df['Datetime'] < end_time)]
        
        if df_discrete.empty or df_cumulative.empty: 
            if is_active_live: break
            continue
            
        # --- COMPUTE DISCRETE SCORES ---
        grouped_disc = df_discrete.groupby('Symbol').agg({
            'Turnover': 'sum', 'Volume': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'
        }).reset_index()
        grouped_disc = grouped_disc[grouped_disc['Turnover'] > 0]
        if grouped_disc.empty: 
            if is_active_live: break
            continue
            
        grouped_disc['Turnover_PR'] = grouped_disc['Turnover'].rank(pct=True) * 100
        grouped_disc['Pct_Move'] = ((grouped_disc['Close'] - grouped_disc['Open']) / grouped_disc['Open']) * 100
        grouped_disc['Momentum_PR'] = grouped_disc['Pct_Move'].abs().rank(pct=True) * 100
        grouped_disc['Net_Disp'] = (grouped_disc['Close'] - grouped_disc['Open']).abs()
        grouped_disc['Efficiency'] = grouped_disc['Net_Disp'] / (grouped_disc['abs_move'] + 1e-8)
        grouped_disc['Hurst_PR'] = grouped_disc['Efficiency'].rank(pct=True) * 100
        grouped_disc['Discrete_Power'] = (grouped_disc['Turnover_PR'] * grouped_disc['Momentum_PR'] * grouped_disc['Hurst_PR']) / 100.0

        # --- COMPUTE CUMULATIVE SCORES ---
        grouped_cum = df_cumulative.groupby('Symbol').agg({
            'Turnover': 'sum', 'Volume': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'
        }).reset_index()
        grouped_cum = grouped_cum[grouped_cum['Turnover'] > 0]
        
        grouped_cum['Cum_Turnover_PR'] = grouped_cum['Turnover'].rank(pct=True) * 100
        grouped_cum['Cum_Pct_Move'] = ((grouped_cum['Close'] - grouped_cum['Open']) / grouped_cum['Open']) * 100
        grouped_cum['Cum_Momentum_PR'] = grouped_cum['Cum_Pct_Move'].abs().rank(pct=True) * 100
        grouped_cum['Cum_Net_Disp'] = (grouped_cum['Close'] - grouped_cum['Open']).abs()
        grouped_cum['Cum_Efficiency'] = grouped_cum['Cum_Net_Disp'] / (grouped_cum['abs_move'] + 1e-8)
        grouped_cum['Cum_Hurst_PR'] = grouped_cum['Cum_Efficiency'].rank(pct=True) * 100
        grouped_cum['Cumulative_Power'] = (grouped_cum['Cum_Turnover_PR'] * grouped_cum['Cum_Momentum_PR'] * grouped_cum['Cum_Hurst_PR']) / 100.0

        # --- MERGE AND CALCULATE ACCELERATION DELTA ---
        # Now passing through the underlying PR values for display
        merged = pd.merge(grouped_disc[['Symbol', 'Discrete_Power', 'Turnover_PR', 'Momentum_PR', 'Hurst_PR', 'Pct_Move', 'Close']], 
                          grouped_cum[['Symbol', 'Cumulative_Power']], 
                          on='Symbol', how='inner')
        
        merged['Acceleration_Delta'] = merged['Discrete_Power'] - merged['Cumulative_Power']
        
        # Select Top 5 by highest acceleration delta
        top5 = merged.nlargest(5, 'Acceleration_Delta')
        
        print(f"\n⏰ DISCRETE WINDOW: {label} IST")
        print(f"{'Rank':<5} {'Symbol':<14} {'Accel Delta':<13} | {'Disc Pwr':<10} | {'Cum Pwr':<9} | {'Turnover':<11} | {'Momentum':<11} | {'Hurst':<9} | {'% Move':<8} {'LTP (₹)':<10}")
        print("-" * 145)
        
        for rank, (_, row) in enumerate(top5.iterrows(), 1):
            move_sign = "+" if row['Pct_Move'] > 0 else ""
            delta_sign = "+" if row['Acceleration_Delta'] > 0 else ""
            print(f"{rank:<5} {row['Symbol']:<14} {delta_sign}{row['Acceleration_Delta']:<12.1f} | {row['Discrete_Power']:>8.1f}   | {row['Cumulative_Power']:>7.1f}   | {row['Turnover_PR']:>7.2f} PR | {row['Momentum_PR']:>7.2f} PR | {row['Hurst_PR']:>5.2f} PR | {move_sign}{row['Pct_Move']:<6.2f}%   ₹{row['Close']:<10.2f}")

        if is_active_live:
            break

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
    
    scan_discrete_hourly_turnover(target_date_str)

if __name__ == "__main__":
    run_production_sweep()

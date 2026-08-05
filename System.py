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
# 0. EQUI-PERCENTILE EQUATING (Neutralizes Absolute Price Differences)
# ==============================================================================
def convert_to_equi_percentile(raw_matrix):
    """
    Transforms a raw price/volume matrix into a strict Equi-Percentile distribution (0.0 to 1.0).
    Converts every data point into its exact mathematical rank within the time window.
    """
    ranks = np.argsort(np.argsort(raw_matrix, axis=0), axis=0)
    percentile_matrix = ranks.astype(np.float32) / (raw_matrix.shape[0] - 1 + 1e-8)
    return percentile_matrix

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
            nn.MaxPool1d(2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            
            nn.Flatten(),
            nn.Linear(32 * 5, latent_dim_daily)
        )
        
        # Branch B: Processes the 15-week Weekly Matrix (5 features x 15 steps)
        self.encoder_weekly = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3),
            
            nn.Flatten(),
            nn.Linear(16 * 5, latent_dim_weekly)
        )
        
        # Decoder A: Reconstructs back to the original 30-day Daily Shape
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
        
        # Decoder B: Reconstructs back to the original 15-week Weekly Shape
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
# 2. ALIGNED DUAL-TIMEFRAME TRAINING DATA LOADER (Using Percentiles)
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
            
            # Application of Equi-Percentile method for AI Engine
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
    index = faiss.IndexFlatIP(24)
    index.add(latent_vectors)
    
    return model, xgb_price, xgb_time, index

# ==============================================================================
# 4. LIVE INGESTION & UNIVERSE PACKS
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
    
    # Application of Equi-Percentile method
    norm_d = convert_to_equi_percentile(vals_d)
    norm_w = convert_to_equi_percentile(vals_w)
    
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

# ==============================================================================
# 5. HOURLY TURNOVER SCANNER (Triple-Percentile Power Score with Hurst PR Filter)
# ==============================================================================
def calculate_hurst(price_series):
    """
    Calculates the Hurst Exponent to measure directional persistence (One-Way vs Whipsaw).
    """
    try:
        prices = np.array(price_series, dtype=np.float32)
        if len(prices) < 15:
            return 0.5
        lags = range(2, min(len(prices) // 2, 20))
        tau = [np.std(prices[lag:] - prices[:-lag]) for lag in lags]
        lags_arr = np.array(list(lags))
        tau_arr = np.array(tau)
        valid = tau_arr > 0
        if np.sum(valid) < 2:
            return 0.5
        poly = np.polyfit(np.log(lags_arr[valid]), np.log(tau_arr[valid]), 1)
        return float(poly[0] * 2.0)
    except:
        return 0.5

def fetch_upstox_intraday_candles(instrument_key, target_date_str):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token:
        return None
    
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    if target_date_str == today_str:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    else:
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(instrument_key)}/1minute/{target_date_str}/{target_date_str}"
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
        
    data = response.json().get('data', {}).get('candles', [])
    if not data:
        return None
        
    c_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
    c_df['Datetime'] = pd.to_datetime(c_df['Timestamp']).dt.tz_localize(None) 
    c_df = c_df.sort_values('Datetime').reset_index(drop=True)
    return c_df

def scan_hourly_top_turnover(target_date_str):
    print(f"\n⏳ Initializing Triple-Percentile (Turnover PR × Momentum PR × Hurst PR) Scan for {target_date_str}...")
    universe = get_dynamic_fno_universe()
    if not universe:
        print("⚠️ No F&O universe found. Please check Upstox API.")
        return
        
    all_hourly_records = []
    
    target_dt = pd.to_datetime(target_date_str)
    bins = [
        target_dt + pd.Timedelta(hours=9, minutes=15),
        target_dt + pd.Timedelta(hours=10, minutes=15),
        target_dt + pd.Timedelta(hours=11, minutes=15),
        target_dt + pd.Timedelta(hours=12, minutes=15),
        target_dt + pd.Timedelta(hours=13, minutes=15),
        target_dt + pd.Timedelta(hours=14, minutes=15),
        target_dt + pd.Timedelta(hours=15, minutes=15),
        target_dt + pd.Timedelta(hours=15, minutes=30)
    ]
    labels = [
        '09:15 - 10:15', '10:15 - 11:15', '11:15 - 12:15', 
        '12:15 - 13:15', '13:15 - 14:15', '14:15 - 15:15', '15:15 - 15:30'
    ]
    
    print(f"📡 Downloading intraday data and computing Hurst persistence for {len(universe)} stocks...")
    for item in universe:
        df = fetch_upstox_intraday_candles(item['key'], target_date_str)
        if df is None or df.empty: continue
            
        df['Turnover'] = df['Volume'] * df['Close']
        df['Time_Window'] = pd.cut(df['Datetime'], bins=bins, labels=labels, include_lowest=True, right=False)
        
        for tw, group_window in df.groupby('Time_Window', observed=False):
            if group_window.empty: continue
            turnover_sum = group_window['Turnover'].sum()
            if turnover_sum <= 0: continue
            
            open_val = group_window['Open'].iloc[0]
            close_val = group_window['Close'].iloc[-1]
            closes_1m = group_window['Close'].values
            
            hurst_val = calculate_hurst(closes_1m)
            
            all_hourly_records.append({
                'Time_Window': tw,
                'Symbol': item['symbol'],
                'Turnover': turnover_sum,
                'Open': open_val,
                'Close': close_val,
                'Hurst': hurst_val
            })
        
    if not all_hourly_records:
        print("❌ Could not retrieve valid intraday data.")
        return
        
    master_df = pd.DataFrame(all_hourly_records)
    
    # 1. Rank 1: Liquidity Percentile (Turnover PR)
    master_df['Turnover_PR'] = master_df.groupby('Time_Window', observed=False)['Turnover'].rank(pct=True) * 100
    
    # 2. Rank 2: Price Displacement Percentile (Momentum PR)
    master_df['Hourly_Pct_Move'] = ((master_df['Close'] - master_df['Open']) / master_df['Open']) * 100
    master_df['Abs_Move'] = master_df['Hourly_Pct_Move'].abs()
    master_df['Momentum_PR'] = master_df.groupby('Time_Window', observed=False)['Abs_Move'].rank(pct=True) * 100
    
    # 3. Rank 3: One-Way Volatility Percentile (Hurst PR - Filters out Whipsaw Traps)
    master_df['Hurst_PR'] = master_df.groupby('Time_Window', observed=False)['Hurst'].rank(pct=True) * 100
    
    # 4. Triple-Percentile Composite Power Score (Multiplicative Hurst Gatekeeper)
    master_df['Power_Score'] = master_df['Turnover_PR'] * master_df['Momentum_PR'] * (master_df['Hurst_PR'] / 100.0)
    
    # Sort strictly by the Composite Power Score
    master_df = master_df.sort_values(by=['Time_Window', 'Power_Score'], ascending=[True, False])
    top5_per_hour = master_df.groupby('Time_Window', observed=False).head(5)
    
    print("\n" + "="*125)
    print(f"🔥 TOP 5 UNIDIRECTIONAL TRENDERS (Turnover PR × Momentum PR × Hurst PR Filter) | DATE: {target_date_str}")
    print("="*125)
    
    for time_window, group in top5_per_hour.groupby('Time_Window', observed=False):
        if group.empty: continue
        print(f"\n⏰ TIME BLOCK: {time_window} IST")
        print(f"{'Rank':<5} {'Symbol':<15} {'Power Score':<12} | {'Turnover PR':<12} | {'Momentum PR':<12} | {'Hurst PR':<10} | {'% Move':<8} {'LTP':<10}")
        print("-" * 125)
        
        for rank, (_, row) in enumerate(group.iterrows(), 1):
            move_str = f"+{row['Hourly_Pct_Move']:.2f}%" if row['Hourly_Pct_Move'] > 0 else f"{row['Hourly_Pct_Move']:.2f}%"
            print(f"{rank:<5} {row['Symbol']:<15} {row['Power_Score']:<12.1f} | {row['Turnover_PR']:>8.2f} PR | {row['Momentum_PR']:>8.2f} PR | {row['Hurst_PR']:>6.2f} PR | {move_str:<8} ₹{row['Close']:<10.2f}")

# ==============================================================================
# 6. MAIN CONTROLLER
# ==============================================================================
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

    print(f"\n🧠 TRAINING PHASE 1: Processing Macro System on {nifty_file}...")
    X_d_nifty, X_w_nifty, Y_np, Y_nt = load_training_data(nifty_file, target_date_str, min_pct=0.75, max_pct=5.0, max_dd=0.5, wick_ratio=0.5)
    
    if X_d_nifty is not None and len(X_d_nifty) > 0:
        nifty_brain, nifty_xgb_p, nifty_xgb_t, nifty_faiss = train_ai_brain(X_d_nifty, X_w_nifty, Y_np, Y_nt)
    else:
        print("⚠️ Not enough historical Nifty data available for AI training.")
    
    # Execute the new Triple-Percentile Power Scan with Hurst PR
    scan_hourly_top_turnover(target_date_str)

if __name__ == "__main__":
    run_production_sweep()


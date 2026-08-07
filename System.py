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
# 0. TERMINAL COLOR CODES (For Intrusion Highlighting)
# ==============================================================================
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'
COLOR_DIM = '\033[2m'

# ==============================================================================
# 1. EQUI-PERCENTILE EQUATING
# ==============================================================================
def convert_to_equi_percentile(raw_matrix):
    ranks = np.argsort(np.argsort(raw_matrix, axis=0), axis=0)
    percentile_matrix = ranks.astype(np.float32) / (raw_matrix.shape[0] - 1 + 1e-8)
    return percentile_matrix

# ==============================================================================
# 2. DUAL-INPUT TEMPORAL AUTOENCODER (The Siamese Multi-Timeframe Brain)
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
# 3. ALIGNED DUAL-TIMEFRAME TRAINING DATA LOADER
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
# 4. LIVE INGESTION & UNIVERSE PACKS
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
# 5. CONTINUOUS ROLLING VELOCITY LEADERBOARD (Delta Jump / Intrusion Scanner)
# ==============================================================================
def calculate_velocity_leaderboard(master_df, current_eval_time, window_mins=15):
    start_of_day = pd.to_datetime(current_eval_time.date()) + pd.Timedelta(hours=9, minutes=15)
    recent_start = current_eval_time - pd.Timedelta(minutes=window_mins)
    
    # If we are too close to market open, fallback gracefully
    if recent_start <= start_of_day:
        return pd.DataFrame()
        
    cum_df = master_df[(master_df['Datetime'] >= start_of_day) & (master_df['Datetime'] < recent_start)]
    rec_df = master_df[(master_df['Datetime'] >= recent_start) & (master_df['Datetime'] <= current_eval_time)]
    
    if cum_df.empty or rec_df.empty:
        return pd.DataFrame()
        
    # --- BASELINE SCORE (Morning up to 15 mins ago) ---
    g_cum = cum_df.groupby('Symbol').agg({'Turnover': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'}).reset_index()
    g_cum = g_cum[g_cum['Turnover'] > 0]
    if g_cum.empty: return pd.DataFrame()
    
    g_cum['Cum_Turnover_PR'] = g_cum['Turnover'].rank(pct=True) * 100
    g_cum['Cum_Pct_Move'] = ((g_cum['Close'] - g_cum['Open']) / (g_cum['Open'] + 1e-8)) * 100
    g_cum['Cum_Momentum_PR'] = g_cum['Cum_Pct_Move'].abs().rank(pct=True) * 100
    g_cum['Cum_Efficiency'] = (g_cum['Close'] - g_cum['Open']).abs() / (g_cum['abs_move'] + 1e-8)
    g_cum['Cum_Hurst_PR'] = g_cum['Cum_Efficiency'].rank(pct=True) * 100
    g_cum['Cum_Score'] = (g_cum['Cum_Turnover_PR'] * g_cum['Cum_Momentum_PR'] * g_cum['Cum_Hurst_PR']) / 100.0

    # --- RECENT SCORE (Last 15 minutes) ---
    g_rec = rec_df.groupby('Symbol').agg({'Turnover': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'}).reset_index()
    g_rec = g_rec[g_rec['Turnover'] > 0]
    if g_rec.empty: return pd.DataFrame()
    
    g_rec['Rec_Turnover_PR'] = g_rec['Turnover'].rank(pct=True) * 100
    g_rec['Rec_Pct_Move'] = ((g_rec['Close'] - g_rec['Open']) / (g_rec['Open'] + 1e-8)) * 100
    g_rec['Rec_Momentum_PR'] = g_rec['Rec_Pct_Move'].abs().rank(pct=True) * 100
    g_rec['Rec_Efficiency'] = (g_rec['Close'] - g_rec['Open']).abs() / (g_rec['abs_move'] + 1e-8)
    g_rec['Rec_Hurst_PR'] = g_rec['Rec_Efficiency'].rank(pct=True) * 100
    g_rec['Rec_Score'] = (g_rec['Rec_Turnover_PR'] * g_rec['Rec_Momentum_PR'] * g_rec['Rec_Hurst_PR']) / 100.0

    # --- THE DELTA JUMP ("Points gone up") ---
    merged = pd.merge(g_rec[['Symbol', 'Rec_Score', 'Rec_Pct_Move', 'Close']], g_cum[['Symbol', 'Cum_Score']], on='Symbol', how='inner')
    merged['Points_Jump'] = merged['Rec_Score'] - merged['Cum_Score']
    
    # Sort strictly by Delta Jump descending and grab Top 10
    top10 = merged.nlargest(10, 'Points_Jump')
    return top10

def scan_live_rolling_leaderboard(target_date_str):
    print(f"\n⏳ Initializing Live Rolling Velocity Radar for {target_date_str}...")
    universe = get_dynamic_fno_universe()
    if not universe:
        print("⚠️ No F&O universe found. Please check Upstox API.")
        return
        
    print(f"📡 Downloading active data for {len(universe)} stocks... (Respecting API limits)")
    master_intraday_list = []
    for item in universe:
        df = fetch_upstox_intraday_candles(item['key'], target_date_str)
        if df is not None and not df.empty:
            df['Symbol'] = item['symbol']
            df['Turnover'] = df['Volume'] * df['Close']
            df['abs_move'] = (df['Close'] - df['Open']).abs()
            master_intraday_list.append(df)
        time.sleep(0.05) 
            
    if not master_intraday_list:
        print("⚠️ Warning: No valid intraday market volume found yet.")
        return
        
    master_df = pd.concat(master_intraday_list, ignore_index=True)
    
    # Define "Now" (If backtesting, simulate as End of Day. If Live, use actual IST time)
    current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = (target_date_str == current_now.strftime("%Y-%m-%d"))
    
    if not is_live_today or current_now.hour >= 16:
        target_dt = pd.to_datetime(target_date_str)
        eval_time_current = target_dt + pd.Timedelta(hours=15, minutes=15) # Market Close Backtest
    else:
        eval_time_current = current_now

    # We simulate a "memory" by taking a snapshot 5 minutes prior to track "Intruders"
    eval_time_previous = eval_time_current - pd.Timedelta(minutes=5)
    
    prev_top10 = calculate_velocity_leaderboard(master_df, eval_time_previous, window_mins=15)
    curr_top10 = calculate_velocity_leaderboard(master_df, eval_time_current, window_mins=15)
    
    if curr_top10.empty:
        print("⚠️ Not enough data collected today to form a leaderboard (Need > 15 mins of live action).")
        return

    prev_symbols = prev_top10['Symbol'].tolist() if not prev_top10.empty else []

    print("\n" + "="*85)
    print(f"⚡ LIVE INTRUSION LEADERBOARD | WINDOW: LAST 15 MINS | TIME: {eval_time_current.strftime('%H:%M')} IST")
    print("   (Ranked strictly by violent behavior change - Intruders highlighted)")
    print("="*85)
    print(f" {'Rank':<4} {'Symbol':<15} {'Points Jump':<15} {'Direction':<12} {'LTP (₹)':<10}")
    print("-" * 85)

    for rank, (_, row) in enumerate(curr_top10.iterrows(), 1):
        sym = row['Symbol']
        jump = row['Points_Jump']
        pct_m = row['Rec_Pct_Move']
        ltp = row['Close']
        
        move_dir = "BULLISH" if pct_m > 0 else "BEARISH"
        
        # Color coding logic: If the stock was NOT in the top 10 five minutes ago, highlight it!
        if sym not in prev_symbols:
            color = COLOR_GREEN if pct_m > 0 else COLOR_RED
            prefix = "🚨 "
            reset = COLOR_RESET
            bold = COLOR_BOLD
        else:
            color = ""
            prefix = "   "
            reset = ""
            bold = COLOR_DIM # Dim the ones already on the board to clear visual noise

        print(f"{bold}{color}{prefix}{rank:<4} {sym:<15} +{jump:<14.1f} {move_dir:<12} ₹{ltp:<10.2f}{reset}")

    print("-" * 85)
    print("Wait for colored intrusions. Trade the momentum change, not the rank.\n")

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
        print("⚠️ Warning: No Nifty data sets located for AI Training phase. Bypassing ML Engine.")
    else:
        print(f"\n🧠 TRAINING PHASE 1: Processing Macro System on {nifty_file}...")
        X_d_nifty, X_w_nifty, Y_np, Y_nt = load_training_data(nifty_file, target_date_str, min_pct=0.75, max_pct=5.0, max_dd=0.5, wick_ratio=0.5)
        
        if X_d_nifty is not None and len(X_d_nifty) > 0:
            nifty_brain, nifty_xgb_p, nifty_xgb_t, nifty_faiss = train_ai_brain(X_d_nifty, X_w_nifty, Y_np, Y_nt)
    
    scan_live_rolling_leaderboard(target_date_str)

if __name__ == "__main__":
    run_production_sweep()


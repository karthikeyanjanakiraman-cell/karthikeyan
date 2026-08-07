import os
import sys
import argparse
import urllib.parse
import json
import gzip
import io
import time
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
# 0. TERMINAL COLOR CODES (For pure signal isolation)
# ==============================================================================
COLOR_GREEN = '\033[92m'
COLOR_RED = '\033[91m'
COLOR_CYAN = '\033[96m'
COLOR_YELLOW = '\033[93m'
COLOR_RESET = '\033[0m'
COLOR_BOLD = '\033[1m'

# ==============================================================================
# 1. EQUI-PERCENTILE EQUATING
# ==============================================================================
def convert_to_equi_percentile(raw_matrix):
    ranks = np.argsort(np.argsort(raw_matrix, axis=0), axis=0)
    percentile_matrix = ranks.astype(np.float32) / (raw_matrix.shape[0] - 1 + 1e-8)
    return percentile_matrix

# ==============================================================================
# 2. DUAL-INPUT TEMPORAL AUTOENCODER (The AI Engine)
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
    if not access_token: return None
    
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    today_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    
    if target_date_str == today_str:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    else:
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(instrument_key)}/1minute/{target_date_str}/{target_date_str}"
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200: return None
        data = response.json().get('data', {}).get('candles', [])
        if not data: return None
        c_df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'OI'])
        c_df['Datetime'] = pd.to_datetime(c_df['Timestamp']).dt.tz_localize(None) 
        c_df = c_df.sort_values('Datetime').reset_index(drop=True)
        return c_df
    except:
        return None

# ==============================================================================
# 4. VELOCITY ENGINE: DELTA CALCULATOR
# ==============================================================================
def calculate_velocity_leaderboard(master_df, current_eval_time, window_mins=15):
    start_of_day = pd.to_datetime(current_eval_time.date()) + pd.Timedelta(hours=9, minutes=15)
    recent_start = current_eval_time - pd.Timedelta(minutes=window_mins)
    
    if recent_start <= start_of_day:
        return pd.DataFrame()
        
    cum_df = master_df[(master_df['Datetime'] >= start_of_day) & (master_df['Datetime'] < recent_start)]
    rec_df = master_df[(master_df['Datetime'] >= recent_start) & (master_df['Datetime'] <= current_eval_time)]
    
    if cum_df.empty or rec_df.empty:
        return pd.DataFrame()
        
    # Baseline (Morning to 15m ago)
    g_cum = cum_df.groupby('Symbol').agg({'Turnover': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'}).reset_index()
    g_cum = g_cum[g_cum['Turnover'] > 0]
    if g_cum.empty: return pd.DataFrame()
    g_cum['Cum_Score'] = (g_cum['Turnover'].rank(pct=True) * 100 * 
                          (((g_cum['Close'] - g_cum['Open']) / (g_cum['Open'] + 1e-8)).abs().rank(pct=True) * 100) * 
                          ((g_cum['Close'] - g_cum['Open']).abs() / (g_cum['abs_move'] + 1e-8)).rank(pct=True) * 100) / 100.0

    # Recent (Last 15m)
    g_rec = rec_df.groupby('Symbol').agg({'Turnover': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'}).reset_index()
    g_rec = g_rec[g_rec['Turnover'] > 0]
    if g_rec.empty: return pd.DataFrame()
    
    g_rec['Rec_Pct_Move'] = ((g_rec['Close'] - g_rec['Open']) / (g_rec['Open'] + 1e-8)) * 100
    g_rec['Rec_Score'] = (g_rec['Turnover'].rank(pct=True) * 100 * 
                          g_rec['Rec_Pct_Move'].abs().rank(pct=True) * 100 * 
                          ((g_rec['Close'] - g_rec['Open']).abs() / (g_rec['abs_move'] + 1e-8)).rank(pct=True) * 100) / 100.0

    # Delta
    merged = pd.merge(g_rec[['Symbol', 'Rec_Score', 'Rec_Pct_Move', 'Close']], g_cum[['Symbol', 'Cum_Score']], on='Symbol', how='inner')
    merged['Points_Jump'] = merged['Rec_Score'] - merged['Cum_Score']
    
    # Sort purely by Delta Jump descending and grab Top 10
    top10 = merged.nlargest(10, 'Points_Jump')
    return top10

# ==============================================================================
# 5. LIVE BURNED LIST & TAPE PRINTER (Zero Noise, Dual-Table Radar)
# ==============================================================================
def scan_institutional_tape(target_date_str):
    print(f"\n📡 Initiating Institutional Stealth Tape for {target_date_str}...")
    universe = get_dynamic_fno_universe()
    if not universe:
        print("⚠️ No F&O universe found. Please check Upstox API.")
        return
        
    master_intraday_list = []
    for item in universe:
        df = fetch_upstox_intraday_candles(item['key'], target_date_str)
        if df is not None and not df.empty:
            df['Symbol'] = item['symbol']
            df['Turnover'] = df['Volume'] * df['Close']
            df['abs_move'] = (df['Close'] - df['Open']).abs()
            master_intraday_list.append(df)
        time.sleep(0.02) # Respecting rate limits
            
    if not master_intraday_list:
        print("⚠️ Warning: No valid intraday market volume found.")
        return
        
    master_df = pd.concat(master_intraday_list, ignore_index=True)
    
    # Determine Current Evaluation Time
    current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = (target_date_str == current_now.strftime("%Y-%m-%d"))
    
    if not is_live_today or current_now.hour >= 16:
        target_dt = pd.to_datetime(target_date_str)
        eval_time_current = target_dt + pd.Timedelta(hours=15, minutes=15) 
    else:
        eval_time_current = current_now

    start_of_day = pd.to_datetime(target_date_str) + pd.Timedelta(hours=9, minutes=15)
    
    # ----------------------------------------------------------------------
    # SILENTLY BUILD THE MASTER LOG ("THE BURNED LIST") FROM 9:15 AM
    # ----------------------------------------------------------------------
    historical_burned_list = {}
    last_known_top10 = []
    
    # Simulate the day in 5-minute chunks to build perfect historical memory
    time_steps = pd.date_range(start=start_of_day + pd.Timedelta(minutes=15), 
                               end=eval_time_current - pd.Timedelta(minutes=5), 
                               freq='5min')
                               
    for t in time_steps:
        historical_top10 = calculate_velocity_leaderboard(master_df, t, window_mins=15)
        if not historical_top10.empty:
            symbols = historical_top10['Symbol'].tolist()
            for sym in symbols:
                if sym not in historical_burned_list:
                    historical_burned_list[sym] = t.strftime('%H:%M')
            last_known_top10 = symbols
        else:
            last_known_top10 = []

    # ----------------------------------------------------------------------
    # ANALYZE THE CURRENT MINUTE
    # ----------------------------------------------------------------------
    curr_top10 = calculate_velocity_leaderboard(master_df, eval_time_current, window_mins=15)
    
    if curr_top10.empty:
        print(f"[{eval_time_current.strftime('%H:%M')} IST] Market compiling... insufficient data window.")
        return

    fresh_intrusions = []
    algorithmic_reloads = []

    for _, row in curr_top10.iterrows():
        sym = row['Symbol']
        
        # 1. Is it just stale momentum from 5 minutes ago? (Ignore completely)
        if sym in last_known_top10:
            continue
            
        # 2. Is it a Reload? (It's on the Burned List from earlier today)
        elif sym in historical_burned_list:
            row['First_Seen'] = historical_burned_list[sym]
            algorithmic_reloads.append(row)
            
        # 3. Is it a Virgin Alert? (Never seen today)
        else:
            fresh_intrusions.append(row)

    # ----------------------------------------------------------------------
    # TERMINAL OUTPUT: TWO CLEAN TABLES
    # ----------------------------------------------------------------------
    print(f"\n{COLOR_CYAN}========================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}LIVE INSTITUTIONAL TAPE | TIME: {eval_time_current.strftime('%H:%M')} IST{COLOR_RESET}")
    print(f"{COLOR_CYAN}========================================================================{COLOR_RESET}\n")

    if not fresh_intrusions and not algorithmic_reloads:
        print(f"{COLOR_DIM}[Terminal Silent] No new block sweeps or reloads detected.{COLOR_RESET}\n")
        return

    # TABLE 1: THE VIRGIN TAPE
    if fresh_intrusions:
        print(f"{COLOR_BOLD}⚡ FRESH INTRUSIONS (First Time Today since 09:15){COLOR_RESET}")
        for row in fresh_intrusions:
            sym = row['Symbol']
            jump = row['Points_Jump']
            dir_str = "BULLISH" if row['Rec_Pct_Move'] > 0 else "BEARISH"
            color = COLOR_GREEN if row['Rec_Pct_Move'] > 0 else COLOR_RED
            print(f"  {color}🚨 [{eval_time_current.strftime('%H:%M')}] {sym:<12} +{jump:<8.1f} points ({dir_str}) LTP: ₹{row['Close']:.2f}{COLOR_RESET}")
        print("")

    # TABLE 2: THE RELOAD TAPE
    if algorithmic_reloads:
        print(f"{COLOR_BOLD}🔄 ALGORITHMIC RELOADS (Second Waves){COLOR_RESET}")
        for row in algorithmic_reloads:
            sym = row['Symbol']
            jump = row['Points_Jump']
            dir_str = "BULLISH" if row['Rec_Pct_Move'] > 0 else "BEARISH"
            color = COLOR_GREEN if row['Rec_Pct_Move'] > 0 else COLOR_RED
            first_seen = row['First_Seen']
            print(f"  {color}🔄 [{eval_time_current.strftime('%H:%M')}] {sym:<12} +{jump:<8.1f} points ({dir_str}) --> (First footprint at {first_seen}){COLOR_RESET}")
        print("")

# ==============================================================================
# 6. RUN EXECUTOR
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
        target_date_str = datetime.strptime(raw_date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    # Ensure Upstox Token exists before running
    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print("❌ Error: UPSTOX_ACCESS_TOKEN not found in environment variables.")
        return

    scan_institutional_tape(target_date_str)

if __name__ == "__main__":
    # Disable warnings for cleaner output
    import warnings
    warnings.filterwarnings("ignore")
    run_production_sweep()

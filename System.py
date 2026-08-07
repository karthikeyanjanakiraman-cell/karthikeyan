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
COLOR_DIM = '\033[2m'
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

# ==============================================================================
# 3. LIVE INGESTION & ROLLING 1-WEEK UNIVERSE PACKS
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

def fetch_upstox_candles_for_date(instrument_key, date_str):
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if not access_token: return None
    
    headers = {'Accept': 'application/json', 'Authorization': f'Bearer {access_token}'}
    today_str = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
    
    if date_str == today_str:
        url = f"https://api.upstox.com/v2/historical-candle/intraday/{urllib.parse.quote(instrument_key)}/1minute"
    else:
        url = f"https://api.upstox.com/v2/historical-candle/{urllib.parse.quote(instrument_key)}/1minute/{date_str}/{date_str}"
    
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

def get_past_trading_days(target_date_str, num_days=5):
    target_dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    trading_days = []
    current_dt = target_dt
    while len(trading_days) < num_days:
        if current_dt.weekday() < 5:  # Skip weekends
            trading_days.append(current_dt.strftime("%Y-%m-%d"))
        current_dt -= timedelta(days=1)
    trading_days.reverse()
    return trading_days

# ==============================================================================
# 4. TRI-DELTA VELOCITY ENGINE (Independent Percentile Changes)
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
        
    # --- BASELINE (Morning up to window start) ---
    g_cum = cum_df.groupby('Symbol').agg({'Turnover': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'}).reset_index()
    g_cum = g_cum[g_cum['Turnover'] > 0]
    if g_cum.empty: return pd.DataFrame()
    
    g_cum['Cum_Pct_Move'] = ((g_cum['Close'] - g_cum['Open']) / (g_cum['Open'] + 1e-8)) * 100
    g_cum['Cum_Efficiency'] = (g_cum['Close'] - g_cum['Open']).abs() / (g_cum['abs_move'] + 1e-8)
    
    g_cum['Cum_Vol_Rank'] = g_cum['Turnover'].rank(pct=True) * 100
    g_cum['Cum_Mom_Rank'] = g_cum['Cum_Pct_Move'].abs().rank(pct=True) * 100
    g_cum['Cum_Eff_Rank'] = g_cum['Cum_Efficiency'].rank(pct=True) * 100

    # --- RECENT (Last window minutes) ---
    g_rec = rec_df.groupby('Symbol').agg({'Turnover': 'sum', 'Open': 'first', 'Close': 'last', 'abs_move': 'sum'}).reset_index()
    g_rec = g_rec[g_rec['Turnover'] > 0]
    if g_rec.empty: return pd.DataFrame()
    
    g_rec['Rec_Pct_Move'] = ((g_rec['Close'] - g_rec['Open']) / (g_rec['Open'] + 1e-8)) * 100
    g_rec['Rec_Efficiency'] = (g_rec['Close'] - g_rec['Open']).abs() / (g_rec['abs_move'] + 1e-8)
    
    g_rec['Rec_Vol_Rank'] = g_rec['Turnover'].rank(pct=True) * 100
    g_rec['Rec_Mom_Rank'] = g_rec['Rec_Pct_Move'].abs().rank(pct=True) * 100
    g_rec['Rec_Eff_Rank'] = g_rec['Rec_Efficiency'].rank(pct=True) * 100

    # --- THE TRI-DELTA CALCULATION ---
    merged = pd.merge(g_rec[['Symbol', 'Rec_Pct_Move', 'Close', 'Datetime', 'Rec_Vol_Rank', 'Rec_Mom_Rank', 'Rec_Eff_Rank']], 
                      g_cum[['Symbol', 'Cum_Vol_Rank', 'Cum_Mom_Rank', 'Cum_Eff_Rank']], on='Symbol', how='inner')
    
    merged['Vol_Delta'] = merged['Rec_Vol_Rank'] - merged['Cum_Vol_Rank']
    merged['Mom_Delta'] = merged['Rec_Mom_Rank'] - merged['Cum_Mom_Rank']
    merged['Eff_Delta'] = merged['Rec_Eff_Rank'] - merged['Cum_Eff_Rank']
    
    merged['Velocity_Jump'] = merged['Vol_Delta'] + merged['Mom_Delta'] + merged['Eff_Delta']
    
    top10 = merged.nlargest(10, 'Velocity_Jump')
    return top10

# ==============================================================================
# 5. STATELESS STATE MACHINE & TAPE PRINTER
# ==============================================================================
def scan_institutional_tape(target_date_str):
    print(f"\n📡 Initiating Stateless Rolling-Memory Tape for {target_date_str}...")
    universe = get_dynamic_fno_universe()
    if not universe:
        print("⚠️ No F&O universe found. Please check Upstox API.")
        return
        
    trading_days = get_past_trading_days(target_date_str, num_days=5)
    print(f"🔄 Backtracing rolling 1-week window across: {trading_days}")

    # Fetch multi-day historical candles into RAM
    historical_dfs = []
    for day in trading_days:
        day_list = []
        for item in universe:
            df = fetch_upstox_candles_for_date(item['key'], day)
            if df is not None and not df.empty:
                df['Symbol'] = item['symbol']
                df['Turnover'] = df['Volume'] * df['Close']
                df['abs_move'] = (df['Close'] - df['Open']).abs()
                day_list.append(df)
            time.sleep(0.01)
        if day_list:
            historical_dfs.append(pd.concat(day_list, ignore_index=True))

    if not historical_dfs:
        print("⚠️ Warning: No valid market volume found across the rolling window.")
        return

    rolling_master_df = pd.concat(historical_dfs, ignore_index=True)
    
    # Strict Current Minute - 1 Boundary Rule for Live Execution
    current_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    is_live_today = (target_date_str == current_now.strftime("%Y-%m-%d"))
    
    if not is_live_today or current_now.hour >= 16:
        target_dt = pd.to_datetime(target_date_str)
        eval_time_current = target_dt + pd.Timedelta(hours=15, minutes=15) 
    else:
        eval_time_current = current_now.replace(second=0, microsecond=0) - timedelta(minutes=1)

    # ----------------------------------------------------------------------
    # BUILD ROLLING RAM MEMORY BANK (1-WEEK FOOTPRINTS)
    # ----------------------------------------------------------------------
    historical_burned_list = {}
    
    for day in trading_days:
        day_start = pd.to_datetime(day) + pd.Timedelta(hours=9, minutes=15)
        day_end = pd.to_datetime(day) + pd.Timedelta(hours=15, minutes=15)
        if day == target_date_str:
            day_end = eval_time_current

        time_steps = pd.date_range(start=day_start + pd.Timedelta(minutes=15), 
                                   end=day_end, 
                                   freq='5min')
                                   
        for t in time_steps:
            top_historical = calculate_velocity_leaderboard(rolling_master_df, t, window_mins=15)
            if not top_historical.empty:
                for _, row in top_historical.iterrows():
                    sym = row['Symbol']
                    if sym not in historical_burned_list:
                        historical_burned_list[sym] = {
                            'date': day,
                            'time': t.strftime('%H:%M'),
                            'price': row['Close']
                        }

    # ----------------------------------------------------------------------
    # EVALUATE CURRENT MINUTE (-1)
    # ----------------------------------------------------------------------
    today_master_df = rolling_master_df[rolling_master_df['Datetime'].dt.strftime('%Y-%m-%d') == target_date_str]
    curr_top10 = calculate_velocity_leaderboard(today_master_df, eval_time_current, window_mins=15)
    
    if curr_top10.empty:
        print(f"[{eval_time_current.strftime('%H:%M')} IST] Market compiling... insufficient data window.")
        return

    fresh_intrusions = []
    algorithmic_reloads = []

    for _, row in curr_top10.iterrows():
        sym = row['Symbol']
        
        # Check against rolling RAM memory bank
        if sym in historical_burned_list:
            footprint = historical_burned_list[sym]
            first_date = footprint['date']
            first_time = footprint['time']
            first_price = footprint['price']
            current_price = row['Close']
            
            pct_change = ((current_price - first_price) / first_price) * 100
            
            row['First_Date'] = first_date
            row['First_Seen'] = first_time
            row['First_Price'] = first_price
            row['Pct_Change_Since_First'] = pct_change
            
            algorithmic_reloads.append(row)
        else:
            fresh_intrusions.append(row)

    # ----------------------------------------------------------------------
    # TERMINAL OUTPUT: DUAL TABLES
    # ----------------------------------------------------------------------
    print(f"\n{COLOR_CYAN}================================================================================================{COLOR_RESET}")
    print(f"{COLOR_BOLD}STATELESS TRI-DELTA TAPE | TIME: {eval_time_current.strftime('%H:%M')} IST{COLOR_RESET}")
    print(f"{COLOR_CYAN}================================================================================================{COLOR_RESET}\n")

    if not fresh_intrusions and not algorithmic_reloads:
        print(f"{COLOR_DIM}[Terminal Silent] No new block sweeps or reloads detected.{COLOR_RESET}\n")
        return

    # TABLE 1: FRESH INTRUSIONS
    if fresh_intrusions:
        print(f"{COLOR_BOLD}⚡ FRESH INTRUSIONS (First Time Seen in Rolling Window){COLOR_RESET}")
        for row in fresh_intrusions:
            sym = row['Symbol']
            jump = row['Velocity_Jump']
            dir_str = "BULLISH" if row['Rec_Pct_Move'] > 0 else "BEARISH"
            color = COLOR_GREEN if row['Rec_Pct_Move'] > 0 else COLOR_RED
            ltp = row['Close']
            
            v_del, m_del, e_del = row['Vol_Delta'], row['Mom_Delta'], row['Eff_Delta']
            delta_str = f"[V:{v_del:+.0f} M:{m_del:+.0f} E:{e_del:+.0f}]"
            
            print(f"  {color}🚨 [{eval_time_current.strftime('%H:%M')}] {sym:<12} +{jump:<5.0f} pts {delta_str:<20} ({dir_str:<7}) | LTP: ₹{ltp:<8.2f}{COLOR_RESET}")
        print("")

    # TABLE 2: ALGORITHMIC RELOADS (SECOND WAVES)
    if algorithmic_reloads:
        print(f"{COLOR_BOLD}🔄 ALGORITHMIC RELOADS (Second Waves / Multi-Day Footprints){COLOR_RESET}")
        for row in algorithmic_reloads:
            sym = row['Symbol']
            jump = row['Velocity_Jump']
            dir_str = "BULLISH" if row['Rec_Pct_Move'] > 0 else "BEARISH"
            color = COLOR_GREEN if row['Rec_Pct_Move'] > 0 else COLOR_RED
            
            v_del, m_del, e_del = row['Vol_Delta'], row['Mom_Delta'], row['Eff_Delta']
            delta_str = f"[V:{v_del:+.0f} M:{m_del:+.0f} E:{e_del:+.0f}]"
            
            curr_ltp = row['Close']
            f_date = row['First_Date']
            first_seen = row['First_Seen']
            first_price = row['First_Price']
            pct_chg = row['Pct_Change_Since_First']
            
            print(f"  {color}🔄 [{eval_time_current.strftime('%H:%M')}] {sym:<12} +{jump:<5.0f} pts {delta_str:<20} ({dir_str:<7}) | LTP: ₹{curr_ltp:<8.2f} --> [1st Footprint on {f_date} @ {first_seen} @ ₹{first_price:.2f} | {pct_chg:+.2f}%]{COLOR_RESET}")
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
    
    if not os.environ.get("UPSTOX_ACCESS_TOKEN"):
        print("❌ Error: UPSTOX_ACCESS_TOKEN not found in environment variables.")
        return

    scan_institutional_tape(target_date_str)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run_production_sweep()

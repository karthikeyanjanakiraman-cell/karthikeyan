import os
import random
import warnings
from datetime import datetime

import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. ALL NSE F&O SYMBOLS (LIVE TICKERS)
# ==============================================================================
FO_SYMBOLS = [
    f"{sym}.NS" for sym in [
        "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT", "ADANIPORTS", 
        "ALKEM", "AMBUJACEM", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT", "ASTRAL", 
        "ATUL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", 
        "BALKRISIND", "BALRAMCHIN", "BANDHANBNK", "BANKBARODA", "BATAINDIA", "BEL", "BERGEPAINT", 
        "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA", "CANBK", 
        "CANFINHOME", "CHAMBLFERT", "CHOLAFIN", "CIPLA", "COALINDIA", "COFORGE", "COLPAL", "CONCOR", 
        "COROMANDEL", "CROMPTON", "CUB", "CUMMINSIND", "DABUR", "DALBHARAT", "DEEPAKNTR", "DIVISLAB", 
        "DIXON", "DLF", "DRREDDY", "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL", 
        "GLENMARK", "GMRINFRA", "GNFC", "GODREJCP", "GODREJPROP", "GRANULES", "GRASIM", "GUJGASLTD", 
        "HAL", "HAVELLS", "HCLTECH", "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", 
        "HINDCOPPER", "HINDPETRO", "HINDUNILVR", "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", 
        "IDFCFIRSTB", "IEX", "IGL", "INDHOTEL", "INDIACEM", "INDIAMART", "INDIGO", "INDUSINDBK", 
        "INDUSTOWER", "INFY", "INTELLECT", "IOC", "IPCALAB", "IRCTC", "ITC", "JINDALSTEL", 
        "JKCEMENT", "JSWSTEEL", "JUBLFOOD", "KOTAKBANK", "LALPATHLAB", "LAURUSLABS", "LICHSGFIN", 
        "LT", "LTIM", "LTTS", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MARICO", "MARUTI", "MCDOWELL-N", 
        "MCX", "METROPOLIS", "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN", "NATIONALUM", 
        "NAUKRI", "NAVINFLUOR", "NESTLEIND", "NMDC", "NTPC", "OBEROIRLTY", "OFSS", "ONGC", "PAGEIND", 
        "PEL", "PERSISTENT", "PETRONET", "PFC", "PIDILITIND", "PIIND", "PNB", "POLYCAB", "POWERGRID", 
        "PVRINOX", "RAMCOCEM", "RBLBANK", "RECLTD", "RELIANCE", "SAIL", "SBICARD", "SBILIFE", "SBIN", 
        "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SRF", "SUNTV", "SUNPHARMA", "SYNGENE", "TATACHEM", 
        "TATACOMM", "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL", "TCS", "TECHM", "TITAN", 
        "TORNTPHARM", "TRENT", "TVSMOTOR", "UBL", "ULTRACEMCO", "UPL", "VEDL", "VOLTAS", "WIPRO", 
        "ZEEL", "ZYDUSLIFE"
    ]
]

# ==============================================================================
# 1. DETERMINISTIC QUANTUM ENVIRONMENT
# ==============================================================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ==============================================================================
# 2. ROUGH PATH SIGNATURES
# ==============================================================================
def compute_path_signatures(path):
    dX = path[:, -1, :] - path[:, 0, :]
    X_shifted = path[:, :-1, :] - path[:, 0:1, :]
    dX_t = path[:, 1:, :] - path[:, :-1, :]
    sig2 = torch.matmul(X_shifted.unsqueeze(-1), dX_t.unsqueeze(-2)).sum(dim=1) 
    sig_flat = torch.cat([dX, sig2.reshape(sig2.shape[0], -1)], dim=1)
    return torch.clamp(sig_flat, -50.0, 50.0)

# ==============================================================================
# 3. MATRIX PRODUCT STATE (Tensor Network) & QUANTUM BRAIN
# ==============================================================================
class MatrixProductState(nn.Module):
    def __init__(self, num_nodes, phys_dim, bond_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.left_core = nn.Parameter(torch.randn(phys_dim, bond_dim) * 0.01)
        if num_nodes > 1:
            self.middle_cores = nn.ParameterList([
                nn.Parameter(torch.randn(bond_dim, phys_dim, bond_dim) * 0.01)
                for _ in range(num_nodes - 1)
            ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(bond_dim) for _ in range(num_nodes)])
            
    def forward(self, x):
        state = torch.matmul(x[:, 0, :], self.left_core) 
        state = self.norm_layers[0](state)
        if self.num_nodes > 1:
            for i, core in enumerate(self.middle_cores):
                state = torch.einsum('bd,dpD,bp->bD', state, core, x[:, i+1, :])
                state = self.norm_layers[i+1](state)
        return state 

class EntangledQuantumBrain(nn.Module):
    def __init__(self, num_assets, num_features, bond_dim=16):
        super().__init__()
        self.num_assets = num_assets
        self.phys_dim = num_features + (num_features ** 2)
        
        self.mps = MatrixProductState(num_assets, self.phys_dim, bond_dim)
        self.measurement_operator = nn.Sequential(
            nn.Linear(bond_dim, bond_dim),
            nn.Tanh(),
            nn.Linear(bond_dim, num_assets)
        )
        
    def forward(self, x):
        batch_size, num_assets, seq_len, features = x.shape
        x_flat = x.reshape(batch_size * num_assets, seq_len, features)
        signatures = compute_path_signatures(x_flat)
        quantum_state_input = signatures.reshape(batch_size, num_assets, self.phys_dim)
        
        entangled_state = self.mps(quantum_state_input)
        amplitudes = self.measurement_operator(entangled_state)
        
        raw_signals = torch.tanh(amplitudes)
        probabilities = raw_signals / (torch.sum(torch.abs(raw_signals), dim=1, keepdim=True) + 1e-8)
        return probabilities

class HamiltonianEnergyLoss(nn.Module):
    def __init__(self, risk_penalty=0.5):
        super().__init__()
        self.risk_penalty = risk_penalty
        
    def forward(self, allocations, future_returns):
        port_return = torch.sum(allocations * future_returns, dim=1)
        port_variance = torch.sum((allocations ** 2) * (future_returns ** 2), dim=1)
        hamiltonian = (self.risk_penalty * port_variance) - port_return
        return torch.mean(hamiltonian)

# ==============================================================================
# 4. INSTANT LIVE DATA COMPILER (WITH QUARANTINE PROTOCOL)
# ==============================================================================
def fetch_live_fo_data(seq_len=10):
    print(f"\n📥 Fetching LIVE data for {len(FO_SYMBOLS)} F&O Assets from NSE feeds...")
    print("⏳ This will take ~10 seconds. No files or tokens required.")
    
    # Download 1-year bulk data instantly
    df = yf.download(FO_SYMBOLS, period="1y", interval="1d", progress=False)
    
    # Clean the data (forward fill missing days, backward fill early gaps)
    df = df.ffill().bfill()
    
    # QUARANTINE PROTOCOL: Strictly filter out any ticker containing NaN values
    features = ['Open', 'High', 'Low', 'Close']
    valid_tickers = []
    
    # Check all columns to ensure no NaNs survived the fill
    for ticker in df['Close'].columns:
        is_corrupted = False
        for feat in features:
            if df[feat][ticker].isna().any():
                is_corrupted = True
                break
        
        if not is_corrupted:
            valid_tickers.append(ticker)

    failed_count = len(FO_SYMBOLS) - len(valid_tickers)
    print(f"✅ Download complete. Automatically dropped {failed_count} broken/delisted symbols to protect the Tensor Math.")
    print(f"✅ Proceeding with {len(valid_tickers)} mathematically clean assets.")
    
    data_3d = np.zeros((len(df), len(valid_tickers), len(features)))
    latest_prices = {}
    
    # Construct Tensor Matrix using ONLY clean valid tickers
    for j, ticker in enumerate(valid_tickers):
        for k, feat in enumerate(features):
            data_3d[:, j, k] = df[feat][ticker].values
        # Save the current real-time closing price
        latest_prices[ticker.replace('.NS', '')] = float(df['Close'][ticker].iloc[-1])

    # Normalize data
    mean = np.mean(data_3d, axis=0, keepdims=True)
    std = np.std(data_3d, axis=0, keepdims=True) + 1e-8
    data_3d_norm = (data_3d - mean) / std
    
    X_list, Y_list = [], []
    
    # 48-HOUR SHIFT WINDOW
    for i in range(len(data_3d_norm) - seq_len):
        x_window = data_3d_norm[i : i + seq_len]
        current_close = data_3d_norm[i + seq_len - 1, :, 3]
        target_idx = i + seq_len + 1 
        if target_idx >= len(data_3d_norm):
            break 
            
        future_close_t2 = data_3d_norm[target_idx, :, 3]
        y_target = (future_close_t2 - current_close) / (np.abs(current_close) + 1e-8)
        
        X_list.append(x_window)
        Y_list.append(y_target)
        
    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
    X = X.permute(0, 2, 1, 3).contiguous()
    
    return X, Y, [t.replace('.NS', '') for t in valid_tickers], latest_prices

# ==============================================================================
# 5. MASTER EXECUTION
# ==============================================================================
def run_quantum_desk():
    set_seeds(42)
    SEQ_LEN = 10        
    FEATURES = 4        
    BOND_DIM = 16       
    EPOCHS = 10
    
    # FETCH LIVE DATA
    X_data, Y_data, asset_names, latest_prices = fetch_live_fo_data(seq_len=SEQ_LEN)
    actual_assets = X_data.shape[1] 
    
    brain = EntangledQuantumBrain(num_assets=actual_assets, num_features=FEATURES, bond_dim=BOND_DIM)
    optimizer = optim.AdamW(brain.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = HamiltonianEnergyLoss(risk_penalty=1.0)
    
    print("\n🌌 WAKING THE ENTANGLED QUANTUM BRAIN (LONG/SHORT HORIZON)")
    print(f"-> Crunching Multi-Dimensional Tensors for {actual_assets} Assets")
    print("-" * 65)
    
    brain.train()
    batch_size = 64
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for i in range(0, len(X_data), batch_size):
            X_batch = X_data[i:i+batch_size]
            Y_batch = Y_data[i:i+batch_size]
            
            optimizer.zero_grad()
            allocations = brain(X_batch)
            loss = loss_function(allocations, Y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / (len(X_data) / batch_size + 1e-8)
        print(f"Epoch {epoch:02d} | 2-Day Hamiltonian Energy State: {avg_loss:>8.5f} | Convergence Optimal")

    print("-" * 65)
    print("🚀 LIVE INFERENCE: EXECUTING LONG/SHORT WAVEFUNCTION COLLAPSE FOR T+2")
    brain.eval()
    with torch.no_grad():
        live_allocations = brain(X_data[-1].unsqueeze(0))[0] 
        
    sorted_indices = torch.argsort(live_allocations, descending=True)
    
    # Absolute best 5 Longs and best 5 Shorts
    top_longs = sorted_indices[:5]   
    top_shorts = sorted_indices[-5:] 
    
    target_indices = torch.cat((top_longs, top_shorts))
    
    print("\n==================================================================")
    for idx in target_indices:
        asset_idx = idx.item()
        asset_symbol = asset_names[asset_idx]
        raw_alloc = live_allocations[asset_idx].item()
        
        abs_weight = abs(raw_alloc) * 100
        is_long = raw_alloc > 0
        direction_text = "LONG 🟢 " if is_long else "SHORT 🔴"
        
        entry_price = latest_prices.get(asset_symbol, 100.0)
        target_pct = (abs_weight / 100.0) * 15.0 
        sl_pct = target_pct / 1.5               
        
        if is_long:
            target = entry_price * (1 + (target_pct / 100.0))
            sl = entry_price * (1 - (sl_pct / 100.0))
        else:
            target = entry_price * (1 - (target_pct / 100.0))
            sl = entry_price * (1 + (sl_pct / 100.0))
        
        print(f"-> {direction_text} | ALLOCATE {abs_weight:05.2f}% TO [ {asset_symbol:<10} ] | CMP: ₹{entry_price:8.2f} | TGT: ₹{target:8.2f} | SL: ₹{sl:8.2f}")
    print("==================================================================\n")

if __name__ == "__main__":
    run_quantum_desk()


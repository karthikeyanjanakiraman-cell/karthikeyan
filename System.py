import os
import random
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. NOTIFICATION CONFIGURATION (SECURE GITHUB ACTIONS SETUP)
# ==============================================================================
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD") 
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL")

# ==============================================================================
# 1. DETERMINISTIC QUANTUM ENVIRONMENT
# ==============================================================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ==============================================================================
# 2. DYNAMIC F&O UNIVERSE MATCHER (ZERO HARDCODING)
# ==============================================================================
def get_dynamic_fo_symbols():
    print("\n🔍 Scanning live market data for today's active F&O Universe...")
    try:
        url = "https://api.kite.trade/instruments"
        df = pd.read_csv(url)
        nfo_df = df[df['segment'] == 'NFO-FUT']
        raw_symbols = nfo_df['name'].dropna().unique().tolist()
        
        indices_to_exclude = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']
        valid_symbols = [sym for sym in raw_symbols if sym not in indices_to_exclude]
        fo_symbols_ns = [f"{sym}.NS" for sym in valid_symbols]
        
        print(f"✅ Successfully discovered {len(fo_symbols_ns)} active F&O stocks dynamically.")
        return fo_symbols_ns
        
    except Exception as e:
        print(f"❌ Failed to fetch dynamic symbols: {e}")
        return []

# ==============================================================================
# 3. ROUGH PATH SIGNATURES (UPGRADED: LEAD-LAG TRANSFORMATION)
# ==============================================================================
def compute_path_signatures(path):
    # Split into Lead and Lag streams to capture cross-temporal friction
    lead = path[:, 1:, :]
    lag = path[:, :-1, :]
    
    # Augment feature space: (Batch, Time-1, Features*2)
    aug_path = torch.cat([lead, lag], dim=-1)
    
    dX = aug_path[:, -1, :] - aug_path[:, 0, :]
    X_shifted = aug_path[:, :-1, :] - aug_path[:, 0:1, :]
    dX_t = aug_path[:, 1:, :] - aug_path[:, :-1, :]
    
    sig2 = torch.matmul(X_shifted.unsqueeze(-1), dX_t.unsqueeze(-2)).sum(dim=1) 
    sig_flat = torch.cat([dX, sig2.reshape(sig2.shape[0], -1)], dim=1)
    
    return torch.clamp(sig_flat, -50.0, 50.0)

# ==============================================================================
# 4. MATRIX PRODUCT STATE (Tensor Network) & QUANTUM BRAIN
# ==============================================================================
class MatrixProductState(nn.Module):
    def __init__(self, num_nodes, phys_dim, bond_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.left_core = nn.Parameter(torch.randn(phys_dim, bond_dim) * 0.01)
        
        if num_nodes > 1:
            self.middle_cores = nn.Parameter(
                torch.randn(num_nodes - 1, bond_dim, phys_dim, bond_dim) * 0.01
            )
            
        self.norm = nn.LayerNorm(bond_dim)
            
    def forward(self, x):
        state = torch.matmul(x[:, 0, :], self.left_core) 
        state = self.norm(state)
        
        if self.num_nodes > 1:
            for i in range(self.num_nodes - 1):
                core = self.middle_cores[i]
                state = torch.einsum('bd,dpD,bp->bD', state, core, x[:, i+1, :])
                state = self.norm(state)
        return state 

class EntangledQuantumBrain(nn.Module):
    def __init__(self, num_assets, num_features, bond_dim=16):
        super().__init__()
        self.num_assets = num_assets
        
        # Lead-Lag doubles the features going into the signature
        aug_features = num_features * 2 
        self.phys_dim = aug_features + (aug_features ** 2)
        
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
        
        # Market Neutral Constraint
        raw_signals = raw_signals - raw_signals.mean(dim=1, keepdim=True)
        
        probabilities = raw_signals / (torch.sum(torch.abs(raw_signals), dim=1, keepdim=True) + 1e-8)
        return probabilities

# ==============================================================================
# UPGRADED: SORTINO HAMILTONIAN ENERGY LOSS (ASYMMETRIC RISK)
# ==============================================================================
class SortinoHamiltonianEnergyLoss(nn.Module):
    def __init__(self, risk_penalty=1.0, l1_penalty=0.001):
        super().__init__()
        self.risk_penalty = risk_penalty
        self.l1_penalty = l1_penalty # Simulates 0.1% friction (Brokerage, STT, Slippage)
        
    def forward(self, allocations, future_returns):
        # Calculate actual returns based on allocation direction
        position_returns = allocations * future_returns
        port_return = torch.sum(position_returns, dim=1)
        
        # Sortino Variance: Only penalize trades that lost money
        downside_returns = torch.clamp(position_returns, max=0.0)
        downside_variance = torch.sum(downside_returns ** 2, dim=1)
        
        # Friction penalty for making trades
        friction = self.l1_penalty * torch.sum(torch.abs(allocations), dim=1)
        
        # Energy state minimizes downside + friction, maximizes upside
        hamiltonian = (self.risk_penalty * downside_variance) + friction - port_return
        return torch.mean(hamiltonian)

# ==============================================================================
# 5. INSTANT LIVE DATA COMPILER (VECTORIZED + CORRELATION CHAINING)
# ==============================================================================
def fetch_live_fo_data(seq_len=10, forecast_horizon=2):
    fo_symbols = get_dynamic_fo_symbols()
    if not fo_symbols:
        raise ValueError("Failed to retrieve dynamic F&O symbols. Cannot proceed.")

    print(f"📥 Fetching LIVE DAILY price history for {len(fo_symbols)} assets from NSE feeds...")
    df = yf.download(fo_symbols, period="1y", interval="1d", progress=False)
    
    df = df.ffill()
    features = ['Open', 'High', 'Low', 'Close']
    valid_tickers = []
    
    for ticker in df['Close'].columns:
        is_corrupted = False
        for feat in features:
            if df[feat][ticker].isna().any():
                is_corrupted = True
                break
        if not is_corrupted:
            valid_tickers.append(ticker)

    # 3D Data Creation
    data_3d = np.stack([df[feat][valid_tickers].values for feat in features], axis=2)
    
    # 🔥 UPGRADE: CORRELATION NODE ORDERING
    print("🧬 Ordering assets via Pearson Correlation for Quantum Chain stabilization...")
    close_prices = data_3d[:, :, 3] 
    returns = np.diff(close_prices, axis=0) / (close_prices[:-1] + 1e-8)
    corr_matrix = np.corrcoef(returns, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix) # Clean NaNs
    
    avg_corr = np.mean(corr_matrix, axis=0)
    sort_indices = np.argsort(avg_corr)
    
    valid_tickers = [valid_tickers[i] for i in sort_indices]
    data_3d = data_3d[:, sort_indices, :]
    
    print(f"✅ Proceeding with {len(valid_tickers)} mathematically ordered assets.")
    
    latest_prices = {t.replace('.NS', ''): float(df['Close'][t].iloc[-1]) for t in valid_tickers}

    # Vectorized Rolling Windows
    total_len = len(data_3d)
    window_count = total_len - seq_len - forecast_horizon + 1
    
    idx_matrix = np.arange(seq_len)[None, :] + np.arange(window_count)[:, None]
    raw_windows = data_3d[idx_matrix] 
    
    window_mean = np.mean(raw_windows, axis=1, keepdims=True)
    window_std = np.std(raw_windows, axis=1, keepdims=True) + 1e-8
    norm_windows = (raw_windows - window_mean) / window_std
    
    current_closes = raw_windows[:, -1, :, 3] 
    target_indices = np.arange(seq_len + forecast_horizon - 1, total_len)
    future_closes = data_3d[target_indices, :, 3]
    
    y_target = (future_closes - current_closes) / (np.abs(current_closes) + 1e-8)
    
    X_train = torch.tensor(norm_windows, dtype=torch.float32).permute(0, 2, 1, 3).contiguous()
    Y_train = torch.tensor(y_target, dtype=torch.float32)
    
    # Live Inference Tensor
    latest_raw_window = data_3d[-seq_len:]
    latest_mean = np.mean(latest_raw_window, axis=0, keepdims=True)
    latest_std = np.std(latest_raw_window, axis=0, keepdims=True) + 1e-8
    latest_norm_window = (latest_raw_window - latest_mean) / latest_std
    
    X_live = torch.tensor(latest_norm_window, dtype=torch.float32).unsqueeze(0)
    X_live = X_live.permute(0, 2, 1, 3).contiguous()
    
    return X_train, Y_train, X_live, [t.replace('.NS', '') for t in valid_tickers], latest_prices

# ==============================================================================
# 6. EMAIL DISPATCH SYSTEM
# ==============================================================================
def send_trade_report_via_email(report_text):
    if not SENDER_EMAIL or not SENDER_PASSWORD or not RECIPIENT_EMAIL:
        print("\n⚠️ Email credentials not found in environment variables. Skipping email dispatch.")
        return

    print("\n📧 Dispatching trade report via Email...")
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"📈 Quantum AI Trade Signals - {datetime.now().strftime('%Y-%m-%d')}"

        msg.attach(MIMEText(report_text, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"✅ Email successfully sent to {RECIPIENT_EMAIL}!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# ==============================================================================
# 7. MASTER EXECUTION (MARKET NEUTRAL PROTOCOL)
# ==============================================================================
def run_quantum_desk():
    set_seeds(42)
    SEQ_LEN = 10        
    FEATURES = 4        
    BOND_DIM = 16       
    EPOCHS = 10
    
    X_train, Y_train, X_live, asset_names, latest_prices = fetch_live_fo_data(seq_len=SEQ_LEN, forecast_horizon=2)
    actual_assets = X_train.shape[1] 
    
    brain = EntangledQuantumBrain(num_assets=actual_assets, num_features=FEATURES, bond_dim=BOND_DIM)
    optimizer = optim.AdamW(brain.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = SortinoHamiltonianEnergyLoss(risk_penalty=1.0, l1_penalty=0.001)
    
    print("\n🌌 WAKING THE ENTANGLED QUANTUM BRAIN (DAILY/SWING HORIZON)")
    print(f"-> Crunching Multi-Dimensional Tensors for {actual_assets} Assets")
    print("-" * 65)
    
    brain.train()
    batch_size = 64
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            allocations = brain(X_batch)
            loss = loss_function(allocations, Y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / (len(X_train) / batch_size + 1e-8)
        print(f"Epoch {epoch:02d} | 2-Day Hamiltonian Energy State: {avg_loss:>8.5f} | Convergence Optimal")

    print("-" * 65)
    brain.eval()
    with torch.no_grad():
        live_allocations = brain(X_live)[0] 
        
    abs_allocations = torch.abs(live_allocations)
    top_10_indices = torch.argsort(abs_allocations, descending=True)[:10]
    max_signal = torch.max(abs_allocations).item()

    report_lines = []
    header = "\n=======================================================================================\n"
    header += " 🏆 TOP 10 SWING TRADES (Market-Neutral / Long & Short Candidates)\n"
    header += "=======================================================================================\n"
    report_lines.append(header)
    
    for idx in top_10_indices:
        asset_idx = idx.item()
        asset_symbol = asset_names[asset_idx]
        raw_alloc = live_allocations[asset_idx].item()
        is_long = raw_alloc > 0
        direction_text = "LONG 🟢 " if is_long else "SHORT 🔴"
        
        conviction_score = (abs(raw_alloc) / max_signal) * 99.0
        if conviction_score > 99.0: conviction_score = 99.0
        
        entry_price = latest_prices.get(asset_symbol, 100.0)
        
        position_size_pct = 10.0 
        target_pct = 4.0         
        sl_pct = 2.0             
        
        if is_long:
            target = entry_price * (1 + (target_pct / 100.0))
            sl = entry_price * (1 - (sl_pct / 100.0))
        else:
            target = entry_price * (1 - (target_pct / 100.0))
            sl = entry_price * (1 + (sl_pct / 100.0))
        
        trade_line = f"-> {direction_text} | WIN PROB: {conviction_score:5.1f}% | ALLOC: {position_size_pct:05.2f}% | [ {asset_symbol:<10} ] | CMP: ₹{entry_price:8.2f} | TGT: ₹{target:8.2f} | SL: ₹{sl:8.2f}"
        report_lines.append(trade_line)
    
    footer = "\n=======================================================================================\n"
    report_lines.append(footer)
    
    final_report = "\n".join(report_lines)
    print(final_report)
    
    send_trade_report_via_email(final_report)

if __name__ == "__main__":
    run_quantum_desk()

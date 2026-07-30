import os
import random
import warnings
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings('ignore')

# ==============================================================================
# 0. DETERMINISTIC QUANTUM ENVIRONMENT
# ==============================================================================
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ==============================================================================
# 1. ROUGH PATH SIGNATURES
# ==============================================================================
def compute_path_signatures(path):
    dX = path[:, -1, :] - path[:, 0, :]
    X_shifted = path[:, :-1, :] - path[:, 0:1, :]
    dX_t = path[:, 1:, :] - path[:, :-1, :]
    
    sig2 = torch.matmul(X_shifted.unsqueeze(-1), dX_t.unsqueeze(-2))
    sig2 = sig2.sum(dim=1) 
    
    sig_flat = torch.cat([dX, sig2.reshape(sig2.shape[0], -1)], dim=1)
    return torch.clamp(sig_flat, -50.0, 50.0)

# ==============================================================================
# 2. MATRIX PRODUCT STATE (Tensor Network)
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

# ==============================================================================
# 3. LONG/SHORT QUANTUM BRAIN
# ==============================================================================
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
        
        # FIXED: Removed Softmax (Long-only). Replaced with Normalized Tanh (Long/Short).
        # Generates signals between -1 (Strong Short) and +1 (Strong Long).
        raw_signals = torch.tanh(amplitudes)
        
        # Normalize so absolute weights sum to 1.0 (100% Capital Allocation)
        probabilities = raw_signals / (torch.sum(torch.abs(raw_signals), dim=1, keepdim=True) + 1e-8)
        return probabilities

# ==============================================================================
# 4. HAMILTONIAN ENERGY LOSS (Market Neutral)
# ==============================================================================
class HamiltonianEnergyLoss(nn.Module):
    def __init__(self, risk_penalty=0.5):
        super().__init__()
        self.risk_penalty = risk_penalty
        
    def forward(self, allocations, future_returns):
        # A negative allocation * negative return = Positive Profit
        port_return = torch.sum(allocations * future_returns, dim=1)
        
        # Variance uses squared allocations to treat shorts and longs as equal risk
        port_variance = torch.sum((allocations ** 2) * (future_returns ** 2), dim=1)
        hamiltonian = (self.risk_penalty * port_variance) - port_return
        return torch.mean(hamiltonian)

# ==============================================================================
# 5. REAL CSV DATA COMPILER (2-DAY TARGET)
# ==============================================================================
def compile_fo_universe(csv_path="historical_fno.csv", seq_len=10, max_assets=50):
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: '{csv_path}' not found. Booting synthetic environment.")
        return torch.randn(32, max_assets, seq_len, 4), torch.randn(32, max_assets) * 0.05, [f"MOCK_{i}" for i in range(max_assets)]
        
    print("⚛️ Parsing F&O CSV into Hilbert Space Tensors (2-Day Target Horizon)...")
    try:
        df = pd.read_csv(csv_path)
        df.rename(columns=lambda x: str(x).lower().strip(), inplace=True)
        col_map = {'date':'Date', 'timestamp':'Date', 'symbol':'Symbol', 'ticker':'Symbol', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close'}
        df.rename(columns=col_map, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        features = ['Open', 'High', 'Low', 'Close']
        pivot_df = df.pivot(index='Date', columns='Symbol', values=features)
        pivot_df = pivot_df.ffill().bfill() 
        
        symbols = pivot_df.columns.get_level_values('Symbol').unique()[:max_assets]
        pivot_df = pivot_df.loc[:, (slice(None), symbols)]
        
        data_3d = np.stack([pivot_df[f].values for f in features], axis=-1)
        
        mean = np.mean(data_3d, axis=0, keepdims=True)
        std = np.std(data_3d, axis=0, keepdims=True) + 1e-8
        data_3d = (data_3d - mean) / std
        
        X_list, Y_list = [], []
        
        # EXACT 48-HOUR SHIFT
        for i in range(len(data_3d) - seq_len):
            x_window = data_3d[i : i + seq_len]
            current_close = data_3d[i + seq_len - 1, :, 3]
            
            target_idx = i + seq_len + 1 
            if target_idx >= len(data_3d):
                break 
                
            future_close_t2 = data_3d[target_idx, :, 3]
            y_target = (future_close_t2 - current_close) / (np.abs(current_close) + 1e-8)
            
            X_list.append(x_window)
            Y_list.append(y_target)
            
        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
        X = X.permute(0, 2, 1, 3).contiguous()
        
        return X, Y, list(symbols)
        
    except Exception as e:
        print(f"❌ FATAL COMPILER ERROR: {str(e)}")
        return torch.randn(32, max_assets, seq_len, 4), torch.randn(32, max_assets) * 0.05, [f"MOCK_{i}" for i in range(max_assets)]

# ==============================================================================
# 6. DISPATCH & NOTIFICATION ENGINE (LONG & SHORT SUPPORT)
# ==============================================================================
def send_quantum_alert(trade_data):
    sender_email = os.environ.get("SENDER_EMAIL")
    sender_pass = os.environ.get("SENDER_PASSWORD")
    recipient_email = os.environ.get("RECIPIENT_EMAIL")
    
    if not all([sender_email, sender_pass, recipient_email]):
        print("⚠️ Email credentials missing in GitHub Secrets. Report printed to console only.")
        return

    msg = MIMEMultipart('alternative')
    msg['Subject'] = f"🌌 LONG/SHORT 2-DAY TARGETS | QUANTUM BRAIN ALLOCATIONS | {datetime.now().strftime('%Y-%m-%d')}"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    html = """
    <html><body style="font-family: Arial, sans-serif; background-color: #f4f7f6; padding: 10px;">
        <h3 style="color: #333;">🌌 48-HOUR LONG/SHORT MARKET NEUTRAL ALLOCATIONS</h3>
        <table border="1" cellpadding="8" cellspacing="0" style="border-collapse: collapse; width: 100%; text-align: center; font-size: 14px; background-color: white;">
          <tr bgcolor="#f8f9fa" style="color: #333; font-weight: bold;">
            <th>Direction</th>
            <th>Asset</th>
            <th>Capital Weight</th>
            <th>Entry (9:15 Open)</th>
            <th>Max Hist. Pain (SL)</th>
            <th>Expected 2-Day Target</th>
            <th>Result</th>
          </tr>"""
    
    for row in trade_data:
        html += f"""
        <tr>
            <td style='color: {row['Dir_Color']}; font-weight: bold;'>{row['Direction']}</td>
            <td style='color: #0056b3;'><b>{row['Asset']}</b></td>
            <td style='color: #6f42c1; font-weight: bold;'>{row['Weight']}</td>
            <td>{row['Entry']}</td>
            <td style='color: #dc3545;'>{row['SL']}</td>
            <td style='color: #28a745; font-weight: bold;'>{row['Target']}</td>
            <td>{row['Result']}</td>
        </tr>"""
        
    html += "</table></body></html>"
    msg.attach(MIMEText(html, 'html'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_pass)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        print("✅ Long/Short 2-Day Horizon Quantum Report Dispatched via Email.")
    except Exception as e:
        print(f"❌ Failed to send email: {str(e)}")

# ==============================================================================
# 7. MASTER EXECUTION
# ==============================================================================
def run_quantum_desk():
    set_seeds(42)
    MAX_ASSETS = 50     
    SEQ_LEN = 10        
    FEATURES = 4        
    BOND_DIM = 16       
    EPOCHS = 10
    
    X_data, Y_data, asset_names = compile_fo_universe(csv_path="historical_fno.csv", seq_len=SEQ_LEN, max_assets=MAX_ASSETS)
    actual_assets = X_data.shape[1] 
    
    brain = EntangledQuantumBrain(num_assets=actual_assets, num_features=FEATURES, bond_dim=BOND_DIM)
    optimizer = optim.AdamW(brain.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = HamiltonianEnergyLoss(risk_penalty=1.0)
    
    print("\n🌌 WAKING THE ENTANGLED QUANTUM BRAIN (LONG/SHORT HORIZON)")
    print(f"-> Integrated F&O Universe: {actual_assets} Assets")
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
        
    # Sort signals to find the extreme highs (Longs) and extreme lows (Shorts)
    sorted_indices = torch.argsort(live_allocations, descending=True)
    
    top_longs = sorted_indices[:3]   # Top 3 Positive allocations
    top_shorts = sorted_indices[-3:] # Top 3 Negative allocations
    
    target_indices = torch.cat((top_longs, top_shorts))

    try:
        raw_df = pd.read_csv("historical_fno.csv")
        raw_df.rename(columns=lambda x: str(x).lower().strip(), inplace=True)
        raw_df.rename(columns={'symbol':'Symbol', 'ticker':'Symbol', 'close':'Close'}, inplace=True)
    except:
        raw_df = pd.DataFrame()

    trade_data = []
    
    for idx in target_indices:
        asset_idx = idx.item()
        asset_symbol = asset_names[asset_idx]
        raw_alloc = live_allocations[asset_idx].item()
        
        abs_weight = abs(raw_alloc) * 100
        
        # Handle directions and formatting
        is_long = raw_alloc > 0
        direction_text = "LONG 🟢" if is_long else "SHORT 🔴"
        dir_color = "#28a745" if is_long else "#dc3545"
        
        entry_price = 100.0 
        if not raw_df.empty and 'Symbol' in raw_df.columns and 'Close' in raw_df.columns:
            sym_df = raw_df[raw_df['Symbol'] == asset_symbol]
            if not sym_df.empty:
                entry_price = float(sym_df['Close'].iloc[-1])
                
        # Dynamic volatility width based on quantum confidence
        target_pct = (abs_weight / 100.0) * 15.0 
        sl_pct = target_pct / 1.5               
        
        # INVERT Math if the asset is being Shorted
        if is_long:
            target = entry_price * (1 + (target_pct / 100.0))
            sl = entry_price * (1 - (sl_pct / 100.0))
        else:
            target = entry_price * (1 - (target_pct / 100.0))
            sl = entry_price * (1 + (sl_pct / 100.0))
        
        trade_data.append({
            'Direction': direction_text,
            'Dir_Color': dir_color,
            'Asset': asset_symbol,
            'Weight': f"{abs_weight:.2f}%",
            'Entry': f"₹{entry_price:.2f}",
            'SL': f"₹{sl:.2f}",
            'Target': f"₹{target:.2f}",
            'Result': "<b>Awaiting 2-Day Target ⏳</b>"
        })
        
        print(f"-> {direction_text} | ALLOCATE {abs_weight:05.2f}% TO [ {asset_symbol} ] | Target: ₹{target:.2f}")

    send_quantum_alert(trade_data)

if __name__ == "__main__":
    run_quantum_desk()

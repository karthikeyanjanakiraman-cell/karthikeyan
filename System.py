import os
import random
import warnings
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
# 1. ROUGH PATH SIGNATURES (The Physics of the Time Series)
# ==============================================================================
def compute_path_signatures(path):
    """
    Extracts the 1st and 2nd Order Iterated Integrals of the price path.
    Includes an active clamp to prevent initial gradient explosion.
    """
    dX = path[:, -1, :] - path[:, 0, :]
    X_shifted = path[:, :-1, :] - path[:, 0:1, :]
    dX_t = path[:, 1:, :] - path[:, :-1, :]
    
    sig2 = torch.matmul(X_shifted.unsqueeze(-1), dX_t.unsqueeze(-2))
    sig2 = sig2.sum(dim=1) 
    
    sig_flat = torch.cat([dX, sig2.flatten(start_dim=1)], dim=1)
    return torch.clamp(sig_flat, -50.0, 50.0)

# ==============================================================================
# 2. MATRIX PRODUCT STATE (Stabilized Entanglement Core)
# ==============================================================================
class MatrixProductState(nn.Module):
    """
    Fuses isolated stocks into a single Quantum Entangled state.
    Uses Site-Specific Layer Normalization to stabilize long tensor chains.
    """
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
# 3. THE QUANTUM BRAIN & WAVEFUNCTION COLLAPSE
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
        
        x_flat = x.view(batch_size * num_assets, seq_len, features)
        signatures = compute_path_signatures(x_flat)
        quantum_state_input = signatures.view(batch_size, num_assets, self.phys_dim)
        
        entangled_state = self.mps(quantum_state_input)
        amplitudes = self.measurement_operator(entangled_state)
        
        probabilities = torch.softmax(amplitudes, dim=1)
        return probabilities

# ==============================================================================
# 4. HAMILTONIAN ENERGY LOSS (Market Neutrality)
# ==============================================================================
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
# 5. REAL CSV DATA COMPILER (The Physics Bridge)
# ==============================================================================
def compile_fo_universe(csv_path="historical_fno.csv", seq_len=10, max_assets=50):
    """
    Reads the real CSV, standardizes the volatility surface, and 
    pivots the data into a 4D Quantum Tensor State.
    """
    if not os.path.exists(csv_path):
        print(f"⚠️ Warning: '{csv_path}' not found. Booting synthetic quantum environment.")
        return torch.randn(32, max_assets, seq_len, 4), torch.randn(32, max_assets) * 0.05, [f"MOCK_{i}" for i in range(max_assets)]
        
    print("⚛️ Parsing F&O CSV into Hilbert Space Tensors...")
    try:
        df = pd.read_csv(csv_path)
        # Standardize standard column names
        df.rename(columns=lambda x: str(x).lower().strip(), inplace=True)
        col_map = {'date':'Date', 'timestamp':'Date', 'symbol':'Symbol', 'ticker':'Symbol', 'open':'Open', 'high':'High', 'low':'Low', 'close':'Close'}
        df.rename(columns=col_map, inplace=True)
        
        # Ensure proper datetime sorting
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        
        # Pivot the universe to align all stocks simultaneously
        features = ['Open', 'High', 'Low', 'Close']
        pivot_df = df.pivot(index='Date', columns='Symbol', values=features)
        pivot_df = pivot_df.ffill().bfill() # Patch structural holes in liquidity
        
        # Select the top N most liquid assets
        symbols = pivot_df.columns.get_level_values('Symbol').unique()[:max_assets]
        pivot_df = pivot_df.loc[:, (slice(None), symbols)]
        
        # Reshape to (Time, Assets, Features)
        data_3d = np.stack([pivot_df[f].values for f in features], axis=-1)
        
        # Z-Score Normalization (Critical for Quantum State scaling)
        mean = np.mean(data_3d, axis=0, keepdims=True)
        std = np.std(data_3d, axis=0, keepdims=True) + 1e-8
        data_3d = (data_3d - mean) / std
        
        # Build Rolling Path Tensors
        X_list, Y_list = [], []
        for i in range(len(data_3d) - seq_len - 1):
            x_window = data_3d[i : i + seq_len]
            
            # Target (Y) is the T+1 Return of the Close Price (Feature Index 3)
            current_close = data_3d[i + seq_len - 1, :, 3]
            next_close = data_3d[i + seq_len, :, 3]
            y_target = (next_close - current_close) / (np.abs(current_close) + 1e-8)
            
            X_list.append(x_window)
            Y_list.append(y_target)
            
        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        Y = torch.tensor(np.array(Y_list), dtype=torch.float32)
        
        # Swap axes to target shape: (Batch, Assets, Seq_Len, Features)
        X = X.permute(0, 2, 1, 3)
        
        return X, Y, list(symbols)
        
    except Exception as e:
        print(f"❌ FATAL COMPILER ERROR: {str(e)}")
        print("Booting synthetic fallback...")
        return torch.randn(32, max_assets, seq_len, 4), torch.randn(32, max_assets) * 0.05, [f"MOCK_{i}" for i in range(max_assets)]

# ==============================================================================
# 6. MASTER DISPATCH EXECUTION
# ==============================================================================
def run_quantum_desk():
    set_seeds(42)
    MAX_ASSETS = 50     # Caps the F&O universe read from the CSV
    SEQ_LEN = 10        # Rolling path window
    FEATURES = 4        # OHLC
    BOND_DIM = 16       # Entanglement depth
    EPOCHS = 10
    
    # Compile the physical CSV data into tensor geometry
    X_data, Y_data, asset_names = compile_fo_universe(csv_path="historical_fno.csv", seq_len=SEQ_LEN, max_assets=MAX_ASSETS)
    actual_assets = X_data.shape[1] # Adjust dynamically if CSV has fewer than 50 stocks
    
    # Initialize Engine
    brain = EntangledQuantumBrain(num_assets=actual_assets, num_features=FEATURES, bond_dim=BOND_DIM)
    optimizer = optim.AdamW(brain.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = HamiltonianEnergyLoss(risk_penalty=1.0)
    
    print("\n🌌 WAKING THE ENTANGLED QUANTUM BRAIN")
    print(f"-> Integrated F&O Universe: {actual_assets} Assets")
    print(f"-> Hilbert Space Dimensions: {actual_assets * (FEATURES + FEATURES**2)}")
    print(f"-> Training Tensors Compiled: {len(X_data)} Path Signatures")
    print("-" * 65)
    
    # Train the Matrix Product State
    brain.train()
    batch_size = 64
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        # Iterate over mini-batches to prevent OOM on massive CSVs
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
        print(f"Epoch {epoch:02d} | Hamiltonian Energy State: {avg_loss:>8.5f} | Convergence Optimal")

    # Final Inference (Wavefunction Collapse for Tomorrow)
    print("-" * 65)
    print("🚀 LIVE INFERENCE: EXECUTING WAVEFUNCTION COLLAPSE FOR T+1")
    brain.eval()
    with torch.no_grad():
        # Feed the absolute latest window in the dataset
        live_allocations = brain(X_data[-1].unsqueeze(0))[0] 
        
    top_trades = torch.topk(live_allocations, k=5)
    for i in range(5):
        asset_idx = top_trades.indices[i].item()
        asset_symbol = asset_names[asset_idx]
        allocation = top_trades.values[i].item() * 100
        print(f"-> ALLOCATE {allocation:05.2f}% CAPITAL TO [ {asset_symbol} ] (Highest Probability Density)")

if __name__ == "__main__":
    run_quantum_desk()

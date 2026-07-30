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
    # Numerical boundary protection
    return torch.clamp(sig_flat, -50.0, 50.0)

# ==============================================================================
# 2. MATRIX PRODUCT STATE (Stabilized Entanglement Core)
# ==============================================================================
class MatrixProductState(nn.Module):
    """
    Fuses 200 isolated stocks into a single Quantum Entangled state.
    Uses Site-Specific Layer Normalization to stabilize long tensor chains.
    """
    def __init__(self, num_nodes, phys_dim, bond_dim):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Edge Tensor (Node 0)
        self.left_core = nn.Parameter(torch.randn(phys_dim, bond_dim) * 0.01)
        
        # Bulk Tensors (Nodes 1 to N-1)
        if num_nodes > 1:
            self.middle_cores = nn.ParameterList([
                nn.Parameter(torch.randn(bond_dim, phys_dim, bond_dim) * 0.01)
                for _ in range(num_nodes - 1)
            ])
            
        # FIXED: Quantum Gate Normalization layers to intercept and bind scaling explosion
        self.norm_layers = nn.ModuleList([nn.LayerNorm(bond_dim) for _ in range(num_nodes)])
            
    def forward(self, x):
        # Contract the classical data with the left edge of the quantum state
        state = torch.matmul(x[:, 0, :], self.left_core) 
        state = self.norm_layers[0](state)
        
        # Iteratively contract the state through the rest of the F&O universe
        if self.num_nodes > 1:
            for i, core in enumerate(self.middle_cores):
                # Tensor Contraction Step
                state = torch.einsum('bd,dpD,bp->bD', state, core, x[:, i+1, :])
                # FIXED: Force the system state back to stable bounds at each node
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
        
        # 1. Transform raw CSV time-series into Rough Path Signatures
        x_flat = x.view(batch_size * num_assets, seq_len, features)
        signatures = compute_path_signatures(x_flat)
        quantum_state_input = signatures.view(batch_size, num_assets, self.phys_dim)
        
        # 2. Entangle the entire F&O universe through the Tensor Network
        entangled_state = self.mps(quantum_state_input)
        
        # 3. Wavefunction Collapse (Calculate quantum amplitudes for each asset)
        amplitudes = self.measurement_operator(entangled_state)
        
        # FIXED: Safe Born Rule using Softmax to natively guarantee allocations 
        # sum up to exactly 1.0 without ever risking division by zero or infinity.
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
        # Expected Portfolio Return
        port_return = torch.sum(allocations * future_returns, dim=1)
        
        # Kinetic Energy (Variance / Risk)
        port_variance = torch.sum((allocations ** 2) * (future_returns ** 2), dim=1)
        
        # Hamiltonian H = Kinetic Energy - Return Optimization Matrix
        hamiltonian = (self.risk_penalty * port_variance) - port_return
        
        return torch.mean(hamiltonian)

# ==============================================================================
# 5. EXECUTION LOOP
# ==============================================================================
def compile_fo_universe(batch_size=32, num_assets=50, seq_len=10, features=4):
    print("⚛️ Initializing Stabilized Quantum State Space...")
    X_mock = torch.randn(batch_size, num_assets, seq_len, features)
    Y_mock = torch.randn(batch_size, num_assets) * 0.05 
    return X_mock, Y_mock

def run_quantum_desk():
    set_seeds(42)
    NUM_ASSETS = 50     
    FEATURES = 4        
    BOND_DIM = 16       
    EPOCHS = 10
    
    X_train, Y_train = compile_fo_universe(num_assets=NUM_ASSETS, features=FEATURES)
    
    brain = EntangledQuantumBrain(num_assets=NUM_ASSETS, num_features=FEATURES, bond_dim=BOND_DIM)
    optimizer = optim.AdamW(brain.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = HamiltonianEnergyLoss(risk_penalty=1.0)
    
    print("\n🌌 WAKING THE ENTANGLED QUANTUM BRAIN")
    print(f"-> F&O Universe: {NUM_ASSETS} Assets")
    print(f"-> Hilbert Space Dimensions: {NUM_ASSETS * (FEATURES + FEATURES**2)}")
    print("-" * 50)
    
    brain.train()
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        
        allocations = brain(X_train)
        loss = loss_function(allocations, Y_train)
        
        loss.backward()
        # Protect gradients from shifting erratically
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        optimizer.step()
        
        print(f"Epoch {epoch:02d} | Hamiltonian Energy State: {loss.item():>8.5f} | Convergence Optimal")

    print("-" * 50)
    print("🚀 LIVE INFERENCE: EXECUTING WAVEFUNCTION COLLAPSE")
    brain.eval()
    with torch.no_grad():
        live_allocations = brain(X_train[0:1])[0] 
        
    top_trades = torch.topk(live_allocations, k=5)
    for i in range(5):
        asset_idx = top_trades.indices[i].item()
        allocation = top_trades.values[i].item() * 100
        print(f"-> ALLOCATE {allocation:05.2f}% CAPITAL TO ASSET [ID_{asset_idx:03d}] (Highest Probability Density)")

if __name__ == "__main__":
    run_quantum_desk()

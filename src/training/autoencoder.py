import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import fbeta_score, precision_score, recall_score
import optuna

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ───────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_rate):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder (mirror)
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        for h_dim in reversed_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = h_dim
        # Replace last ReLU+Dropout with just Linear
        decoder_layers = decoder_layers[:-2]  # remove last ReLU+Dropout
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ── Training ────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch, in loader:
        batch = batch.to(DEVICE)
        output = model(batch)
        loss = criterion(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
    return total_loss / len(loader.dataset)

def compute_reconstruction_error(model, data):
    model.eval()
    with torch.no_grad():
        tensor = torch.FloatTensor(data).to(DEVICE)
        output = model(tensor)
        errors = torch.mean((tensor - output) ** 2, dim=1)
    return errors.cpu().numpy()

def find_best_threshold(errors_normal, errors_all, labels, beta=2):
    best_score, best_threshold = 0, 0
    
    for p in np.arange(90, 99.9, 0.5):
        threshold = np.percentile(errors_normal, p)
        preds = (errors_all > threshold).astype(int)
        score = fbeta_score(labels, preds, beta=beta, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score

# ── Optuna objective ────────────────────────────────────────
def objective(trial, train_data, val_X, val_y):
    # Architecture search space
    n_layers = trial.suggest_int("n_layers", 2, 3)
    bottleneck = trial.suggest_int("bottleneck", 8, 24, step=4)
    dropout = trial.suggest_float("dropout", 0.1, 0.4, step=0.05)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # Build hidden dims: linearly interpolate from input to bottleneck
    input_dim = train_data.shape[1]
    hidden_dims = []
    for i in range(1, n_layers + 1):
        dim = int(input_dim - (input_dim - bottleneck) * i / n_layers)
        hidden_dims.append(max(dim, bottleneck))
    
    # Model
    model = Autoencoder(input_dim, hidden_dims, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # DataLoader
    train_tensor = torch.FloatTensor(train_data)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
    
    # Train with early stopping
    best_val_loss = float("inf")
    patience, patience_counter = 10, 0
    
    for epoch in range(100):
        train_loss = train_epoch(model, loader, optimizer, criterion)
        
        # Val loss on normal events only (for early stopping)
        val_normal = val_X[val_y == 0]
        val_errors = compute_reconstruction_error(model, val_normal)
        val_loss = val_errors.mean()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        # Optuna pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # Evaluate: find threshold and compute F1
    errors_normal = compute_reconstruction_error(model, val_X[val_y == 0])
    errors_all = compute_reconstruction_error(model, val_X)
    _, best_fbeta = find_best_threshold(errors_normal, errors_all, val_y, beta=2)
    return best_fbeta

# ── Main ────────────────────────────────────────────────────
def run_optuna(n_trials=50):
    data = np.load(PROJECT_ROOT / "data" / "training" / "autoencoder.npz")
    train_data = data["train"]
    val_X, val_y = data["val_X"], data["val_y"]
    
    study = optuna.create_study(direction="maximize")  # maximize F1
    study.optimize(lambda trial: objective(trial, train_data, val_X, val_y),
                   n_trials=n_trials)
    
    print(f"\nBest F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study

if __name__ == "__main__":
    study = run_optuna(n_trials=50)
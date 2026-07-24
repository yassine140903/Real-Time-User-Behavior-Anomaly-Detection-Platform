"""
Runner: data prep → LSTM training → evaluation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.training.data_prep import prepare
from src.training.lstm import train, evaluate

# ── Data Prep ─────────────────────────────────────────────────────────
print("=" * 60)
print("DATA PREP")
print("=" * 60)

data = prepare("data/enriched_events.csv")

# ── Train ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LSTM TRAINING")
print("=" * 60)

model, test_data = train(
    data,
    seq_len=10,
    hidden_dim=64,
    num_layers=2,
    epochs=50,
    batch_size=256,
    lr=1e-3,
)

# ── Evaluate ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LSTM TEST EVALUATION")
print("=" * 60)

test_auc, scores = evaluate(model, test_data)

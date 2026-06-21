"""
Runner: data prep → Autoencoder training → evaluation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.training.data_prep import prepare
from src.training.autoencoder import train, evaluate

# ── Data Prep ─────────────────────────────────────────────────────────
print("=" * 60)
print("DATA PREP")
print("=" * 60)

data = prepare("data/enriched_events.csv")

# ── Train ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("AUTOENCODER TRAINING")
print("=" * 60)

model = train(data, hidden_dims=[20, 10], epochs=50, batch_size=256, lr=1e-3)

# ── Evaluate ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)

test_auc, scores = evaluate(model, data)

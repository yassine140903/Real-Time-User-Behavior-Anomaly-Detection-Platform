# src/scoring/fusion.py
import numpy as np
import torch
import json
import joblib
from pathlib import Path
from src.training.autoencoder import Autoencoder
from src.training.lstm import LSTMPredictor


class ScoreFusion:
    def __init__(self, models_dir="models"):
        models_dir = Path(models_dir)

        # ── Load shared artifacts ───────────────────────────
        with open(models_dir / "feature_cols.json") as f:
            self.feature_cols = json.load(f)
        self.scaler = joblib.load(models_dir / "scaler.pkl")
        self.input_dim = len(self.feature_cols)

        # ── Load AE ─────────────────────────────────────────
        self.ae = Autoencoder(self.input_dim, hidden_dims=[20, 10])
        self.ae.load_state_dict(
            torch.load(models_dir / "autoencoder.pt", weights_only=True))
        self.ae.eval()

        # ── Load LSTM ───────────────────────────────────────
        with open(models_dir / "lstm_config.json") as f:
            cfg = json.load(f)
        self.seq_len = cfg["seq_len"]
        self.lstm = LSTMPredictor(
            cfg["input_dim"], cfg["hidden_dim"], cfg["num_layers"])
        self.lstm.load_state_dict(
            torch.load(models_dir / "lstm.pt", weights_only=True))
        self.lstm.eval()

        # ── Load reference distributions ────────────────────
        self.ae_ref = np.load(models_dir / "ae_ref_scores.npy")
        self.lstm_ref = np.load(models_dir / "lstm_ref_scores.npy")

    # ── Percentile normalization ────────────────────────────
    def _percentile(self, score, ref_sorted):
        """Percentile rank of score against sorted reference."""
        idx = np.searchsorted(ref_sorted, score, side="right")
        return idx / len(ref_sorted)

    # ── Maturity weight ─────────────────────────────────────
    def _lstm_weight(self, days_since_opening, midpoint=90, steepness=15):
        """Sigmoid ramp: 0 for new clients → 1 for mature."""
        z = (days_since_opening - midpoint) / steepness
        return 1.0 / (1.0 + np.exp(-z))

    # ── Score a single event ────────────────────────────────
    def score(self, features_scaled, sequence_scaled=None,
              days_since_opening=None):
        """
        Compute fused anomaly score.

        Args:
            features_scaled:   np.array (n_features,) — already scaled
            sequence_scaled:   np.array (seq_len, n_features) or None
            days_since_opening: int or None

        Returns:
            dict with ae_raw, lstm_raw, ae_pct, lstm_pct, w_lstm, fused
        """
        # AE score
        with torch.no_grad():
            x = torch.FloatTensor(features_scaled).unsqueeze(0)
            recon = self.ae(x)
            ae_raw = (recon - x).pow(2).mean().item()
        ae_pct = self._percentile(ae_raw, self.ae_ref)

        # LSTM score (only if we have enough history)
        if sequence_scaled is not None and len(sequence_scaled) >= self.seq_len:
            with torch.no_grad():
                seq = torch.FloatTensor(sequence_scaled[-self.seq_len:])\
                           .unsqueeze(0)
                pred = self.lstm(seq)
                target = torch.FloatTensor(features_scaled).unsqueeze(0)
                lstm_raw = (pred - target).pow(2).mean().item()
            lstm_pct = self._percentile(lstm_raw, self.lstm_ref)
        else:
            lstm_raw = None
            lstm_pct = None

        # Fusion
        if lstm_pct is not None and days_since_opening is not None:
            w = self._lstm_weight(days_since_opening)
            fused = (1 - w) * ae_pct + w * lstm_pct
        else:
            # No LSTM available — AE only
            w = 0.0
            fused = ae_pct

        return {
            "ae_raw": ae_raw,
            "lstm_raw": lstm_raw,
            "ae_pct": ae_pct,
            "lstm_pct": lstm_pct,
            "w_lstm": round(w, 4),
            "fused_score": round(fused, 6),
        }
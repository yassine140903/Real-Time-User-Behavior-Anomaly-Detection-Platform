"""
LSTM Scoring wrapper with expected-vs-actual explanations.
"""

import numpy as np
import torch


LSTM_EXPLAINABLE_FEATURES = [
    "amount", "hour", "amount_to_mean_ratio", "op_type_probability",
    "day_distance_from_preferred", "tx_count_24h", "tx_count_7d",
    "cumulative_amount_24h", "near_threshold_count_7d",
    "employee_tx_count_24h", "same_employee_client_count_24h",
]


class LSTMScorer:
    def __init__(self, model, scaler, feature_cols, seq_len=10):
        """
        Args:
            model: trained LSTMPredictor instance (already loaded)
            scaler: fitted StandardScaler (already loaded)
            feature_cols: list of 30 feature names
            seq_len: sequence length the LSTM expects
        """
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.seq_len = seq_len

        self.explainable_idx = [
            self.feature_cols.index(f) for f in LSTM_EXPLAINABLE_FEATURES
            if f in self.feature_cols
        ]

    def score(self, sequence_scaled, target_scaled):
        with torch.no_grad():
            seq_t = torch.FloatTensor(sequence_scaled).unsqueeze(0)
            pred = self.model(seq_t).squeeze(0).numpy()
        return float(((pred - target_scaled) ** 2).mean())

    def explain(self, sequence_scaled, target_scaled, top_k=3):
        with torch.no_grad():
            seq_t = torch.FloatTensor(sequence_scaled).unsqueeze(0)
            pred_scaled = self.model(seq_t).squeeze(0).numpy()

        anomaly_score = float(((pred_scaled - target_scaled) ** 2).mean())

        per_feature_error = (pred_scaled - target_scaled) ** 2

        explainable_errors = [(i, per_feature_error[i]) for i in self.explainable_idx]
        explainable_errors.sort(key=lambda x: x[1], reverse=True)
        top = explainable_errors[:top_k]

        pred_raw = self.scaler.inverse_transform(pred_scaled.reshape(1, -1))[0]
        actual_raw = self.scaler.inverse_transform(target_scaled.reshape(1, -1))[0]

        top_deviations = {}
        for idx, _ in top:
            feature_name = self.feature_cols[idx]
            top_deviations[feature_name] = (
                round(float(pred_raw[idx]), 4),
                round(float(actual_raw[idx]), 4)
            )

        return {
            "score": round(anomaly_score, 6),
            "top_deviations": top_deviations
        }
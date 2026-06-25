"""
AE Scoring wrapper with SHAP explanations.
"""

import numpy as np
import torch
import shap


class AEScorer:
    def __init__(self, model, scaler, feature_cols, 
                 background_data=None, n_background=100):
        """
        Args:
            model: trained Autoencoder instance (already loaded)
            scaler: fitted StandardScaler (already loaded)
            feature_cols: list of 30 feature names
            background_data: numpy array of SCALED normal events
            n_background: samples to use from background_data
        """
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols

        self.explainer = None
        if background_data is not None:
            self.set_background(background_data, n_background)

    def set_background(self, data, n_samples=100):
        if len(data) > n_samples:
            idx = np.random.choice(len(data), n_samples, replace=False)
            data = data[idx]
        self.explainer = shap.KernelExplainer(self._score_fn, data)

    def _score_fn(self, x):
        with torch.no_grad():
            x_t = torch.FloatTensor(x)
            recon = self.model(x_t)
            return (recon - x_t).pow(2).mean(dim=1).numpy()

    def score(self, x_scaled):
        return float(self._score_fn(x_scaled.reshape(1, -1))[0])

    def explain(self, x_scaled, top_k=3):
        if self.explainer is None:
            raise RuntimeError("No background dataset. Call set_background() first.")

        anomaly_score = self.score(x_scaled)
        raw_values = self.scaler.inverse_transform(x_scaled.reshape(1, -1))[0]
        shap_values = self.explainer.shap_values(x_scaled.reshape(1, -1))[0]

        top_idx = np.argsort(np.abs(shap_values))[::-1][:top_k]

        top_features = {}
        for i in top_idx:
            feature_name = self.feature_cols[i]
            top_features[feature_name] = (
                round(float(shap_values[i]), 6),
                round(float(raw_values[i]), 4)
            )

        return {
            "score": round(anomaly_score, 6),
            "top_features": top_features
        }
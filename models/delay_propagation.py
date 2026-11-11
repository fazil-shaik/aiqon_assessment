"""
Multi-Stage Delay Propagation and ETA Prediction Engine.
Models cascading delays, berth window penalties, buffer absorption, and in-flight milestone inference.
Delegates feature extraction to LogisticsFeaturePipeline.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, f1_score, accuracy_score
import os

from models.feature_pipeline import (
    LogisticsFeaturePipeline,
    FEATURE_COLS_STATIC,
    CATEGORICAL_COLS
)

class DelayPropagationEngine:
    """
    Core predictive engine for shipment ETA and delay propagation.
    Supports in-flight inference across multiple operational milestones:
      - Milestone 0: Booking / Pre-departure (No realized stage delays)
      - Milestone 1: Origin Departed (Stage 1 realized)
      - Milestone 2: Ocean Arrived / Port Anchorage (Stage 1 & 2 realized)
      - Milestone 3: Customs Cleared (Stage 1, 2, 3 realized)
    """

    def __init__(self):
        self.pipeline = LogisticsFeaturePipeline()
        self.regressor = GradientBoostingRegressor(
            n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42
        )
        self.classifier = RandomForestClassifier(
            n_estimators=120, max_depth=5, random_state=42
        )
        # Stage duration estimators for granular cascade modeling
        self.stage_models = {
            "origin": RandomForestRegressor(n_estimators=60, max_depth=4, random_state=42),
            "transit": RandomForestRegressor(n_estimators=60, max_depth=4, random_state=42),
            "port": RandomForestRegressor(n_estimators=60, max_depth=4, random_state=42),
            "inland": RandomForestRegressor(n_estimators=60, max_depth=4, random_state=42),
        }
        self.is_fitted = False
        self.train_metrics: Dict[str, float] = {}

    @property
    def feature_names_(self) -> List[str]:
        return self.pipeline.feature_names_

    def _prepare_matrix(self, df: pd.DataFrame, fit_encoder: bool = False, milestone: int = 3) -> np.ndarray:
        """Internal delegation to feature pipeline for matrix preparation."""
        return self.pipeline.prepare_matrix(df, fit_encoder=fit_encoder, milestone=milestone)

    def _engineer_features(self, df: pd.DataFrame, milestone: int = 3) -> pd.DataFrame:
        """Internal delegation to feature pipeline for feature engineering."""
        return self.pipeline.engineer_features(df, milestone=milestone)

    def fit(self, df: pd.DataFrame) -> "DelayPropagationEngine":
        """
        Trains models across synthetic or real shipment history.
        Trains both main regressor/classifier and stage-specific models.
        """
        # Split train/test
        train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)

        # Augment training across all milestones for in-flight generalization
        X_train_list, y_reg_train_list, y_cls_train_list = [], [], []
        for m in [0, 1, 2, 3]:
            X_m = self._prepare_matrix(train_df, fit_encoder=(m == 0), milestone=m)
            X_train_list.append(X_m)
            y_reg_train_list.append(train_df["total_delay_hours"].values)
            y_cls_train_list.append(train_df["is_delayed"].values)

        X_train = np.vstack(X_train_list)
        y_reg_train = np.concatenate(y_reg_train_list)
        y_cls_train = np.concatenate(y_cls_train_list)

        # Fit main models
        self.regressor.fit(X_train, y_reg_train)
        self.classifier.fit(X_train, y_cls_train)

        # Fit stage-specific models
        X_stage_base = self._prepare_matrix(train_df, fit_encoder=False, milestone=0)
        self.stage_models["origin"].fit(X_stage_base, train_df["stage_origin_delay_hours"])
        self.stage_models["transit"].fit(X_stage_base, train_df["stage_transit_delay_hours"])
        self.stage_models["port"].fit(
            X_stage_base, train_df["stage_port_waiting_hours"] + train_df["stage_customs_dwell_hours"]
        )
        self.stage_models["inland"].fit(X_stage_base, train_df["stage_inland_delay_hours"])

        self.is_fitted = True

        # Compute validation metrics
        X_val = self._prepare_matrix(val_df, fit_encoder=False, milestone=3)
        y_val_reg = val_df["total_delay_hours"].values
        y_val_cls = val_df["is_delayed"].values

        pred_reg = self.regressor.predict(X_val)
        pred_cls_prob = self.classifier.predict_proba(X_val)[:, 1]
        pred_cls = (pred_cls_prob >= 0.5).astype(int)

        self.train_metrics = {
            "reg_mae": float(mean_absolute_error(y_val_reg, pred_reg)),
            "reg_rmse": float(np.sqrt(mean_squared_error(y_val_reg, pred_reg))),
            "reg_r2": float(r2_score(y_val_reg, pred_reg)),
            "cls_accuracy": float(accuracy_score(y_val_cls, pred_cls)),
            "cls_f1": float(f1_score(y_val_cls, pred_cls)),
            "cls_auc": float(roc_auc_score(y_val_cls, pred_cls_prob)),
            "n_train_samples": len(train_df),
            "n_val_samples": len(val_df),
        }
        return self

    def predict_shipment(self, shipment_dict: Dict[str, Any], milestone: int = 3) -> Dict[str, Any]:
        """
        Runs full predictive inference for a single shipment journey at a given milestone.
        Calculates:
          - Expected Final ETA (Datetime string)
          - Delay Magnitude (hours)
          - Delay Likelihood / Risk Probability (0 to 1)
          - Delay Severity Tier (On-Time, Minor Delay, Moderate Delay, Critical Delay)
          - Stage-by-stage delay cascade breakdown (realized vs remaining forecasted)
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        df_single = pd.DataFrame([shipment_dict])
        X = self._prepare_matrix(df_single, fit_encoder=False, milestone=milestone)

        pred_delay_hours = float(self.regressor.predict(X)[0])
        pred_delay_prob = float(self.classifier.predict_proba(X)[0, 1])

        # Calculate ETA
        sched_arrival = pd.to_datetime(shipment_dict["sched_arrival"])
        pred_eta = sched_arrival + timedelta(hours=pred_delay_hours)

        # Risk classification
        if pred_delay_hours <= 1.0 and pred_delay_prob < 0.35:
            severity = "On-Time / Early"
            badge_color = "#10B981"
        elif pred_delay_hours <= 8.0:
            severity = "Minor Delay (Low Risk)"
            badge_color = "#3B82F6"
        elif pred_delay_hours <= 24.0:
            severity = "Moderate Delay (Medium Risk)"
            badge_color = "#F59E0B"
        else:
            severity = "Critical Delay (High Risk)"
            badge_color = "#EF4444"

        # Stage Cascade Breakdown
        X_base = self._prepare_matrix(df_single, fit_encoder=False, milestone=0)

        s1_origin = (
            float(shipment_dict.get("stage_origin_delay_hours", 0.0))
            if milestone >= 1
            else float(self.stage_models["origin"].predict(X_base)[0])
        )
        s2_transit = (
            float(shipment_dict.get("stage_transit_delay_hours", 0.0))
            if milestone >= 2
            else float(self.stage_models["transit"].predict(X_base)[0])
        )
        s3_port = (
            (float(shipment_dict.get("stage_port_waiting_hours", 0.0)) + float(shipment_dict.get("stage_customs_dwell_hours", 0.0)))
            if milestone >= 3
            else float(self.stage_models["port"].predict(X_base)[0])
        )
        s4_inland = (
            float(shipment_dict.get("stage_inland_delay_hours", 0.0))
            if milestone >= 4
            else float(self.stage_models["inland"].predict(X_base)[0])
        )

        # Buffer absorption modeling
        absorbed = float(shipment_dict.get("buffer_absorbed_hours", 0.0))

        cascade = [
            {"stage": "Origin Departure", "delay_hours": round(s1_origin, 1), "status": "Realized" if milestone >= 1 else "Forecasted"},
            {"stage": "Ocean Transit (Net)", "delay_hours": round(s2_transit, 1), "status": "Realized" if milestone >= 2 else "Forecasted"},
            {"stage": "Port Berth & Customs", "delay_hours": round(s3_port, 1), "status": "Realized" if milestone >= 3 else "Forecasted"},
            {"stage": "Final Inland Leg", "delay_hours": round(s4_inland, 1), "status": "Realized" if milestone >= 4 else "Forecasted"},
        ]

        return {
            "shipment_id": shipment_dict.get("shipment_id", "N/A"),
            "current_milestone": milestone,
            "scheduled_arrival": sched_arrival.strftime("%Y-%m-%d %H:%M"),
            "predicted_eta": pred_eta.strftime("%Y-%m-%d %H:%M"),
            "predicted_delay_hours": round(pred_delay_hours, 1),
            "delay_probability": round(pred_delay_prob * 100, 1),
            "severity_tier": severity,
            "badge_color": badge_color,
            "stage_cascade": cascade,
            "berth_window_miss_risk": bool(
                pred_delay_hours > 12.0 and float(shipment_dict.get("dest_port_congestion_idx", 0.5)) > 0.65
            ),
            "buffer_absorbed_hours": round(absorbed, 1),
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns normalized feature importances for regression model."""
        if not self.is_fitted:
            return pd.DataFrame()
        importances = self.regressor.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": self.feature_names_,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        fi_df["Importance_Pct"] = (fi_df["Importance"] / fi_df["Importance"].sum()) * 100
        return fi_df

def train_and_cache_model(csv_path: str = "data/shipments_dataset.csv") -> DelayPropagationEngine:
    """Helper to train or load pre-trained engine."""
    engine = DelayPropagationEngine()
    df = pd.read_csv(csv_path)
    engine.fit(df)
    return engine

"""
Feature Engineering and Matrix Assembly Pipeline for Delay Propagation.
Handles multi-milestone data transformations, non-linear interaction terms, and categorical encoding.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLS_STATIC = [
    "distance_nm",
    "total_sched_hours",
    "sched_transit_hours",
    "sched_port_dwell_hours",
    "sched_inland_hours",
    "origin_congestion_idx",
    "transit_weather_severity",
    "dest_port_congestion_idx",
    "inland_traffic_idx",
]

CATEGORICAL_COLS = ["route_id", "carrier", "cargo_priority", "customs_risk_tier"]

class LogisticsFeaturePipeline:
    """
    Transforms raw shipment journey records into modeled interaction features
    conditioned on the active operational milestone (0 to 3).
    """

    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.feature_names_: List[str] = []

    def engineer_features(self, df: pd.DataFrame, milestone: int = 3) -> pd.DataFrame:
        """
        Engineers interaction and propagation features according to current shipment milestone.
        If milestone < k, subsequent realized stage delays are masked out (set to 0 / imputed).
        """
        feat_df = df.copy()

        # Non-linear congestion and risk interaction terms
        feat_df["weather_x_distance"] = feat_df["transit_weather_severity"] * (feat_df["distance_nm"] / 1000.0)
        feat_df["port_congestion_x_dwell"] = feat_df["dest_port_congestion_idx"] * feat_df["sched_port_dwell_hours"]
        feat_df["inland_traffic_x_distance"] = feat_df["inland_traffic_idx"] * feat_df["sched_inland_hours"]

        # Milestone-conditioned realized stage features
        if milestone >= 1:
            feat_df["known_origin_delay"] = feat_df["stage_origin_delay_hours"]
        else:
            feat_df["known_origin_delay"] = 0.0

        if milestone >= 2:
            feat_df["known_transit_delay"] = feat_df["stage_transit_delay_hours"]
            feat_df["cum_delay_at_port"] = feat_df["known_origin_delay"] + feat_df["known_transit_delay"]
            # Berth window risk multiplier
            feat_df["berth_risk_factor"] = (feat_df["cum_delay_at_port"] > 10.0).astype(float) * feat_df["dest_port_congestion_idx"]
        else:
            feat_df["known_transit_delay"] = 0.0
            feat_df["cum_delay_at_port"] = feat_df["known_origin_delay"]
            feat_df["berth_risk_factor"] = 0.0

        if milestone >= 3:
            feat_df["known_port_waiting"] = feat_df["stage_port_waiting_hours"]
            feat_df["known_customs_dwell"] = feat_df["stage_customs_dwell_hours"]
            feat_df["cum_delay_post_customs"] = (
                feat_df["cum_delay_at_port"] + feat_df["known_port_waiting"] + feat_df["known_customs_dwell"]
            )
        else:
            feat_df["known_port_waiting"] = 0.0
            feat_df["known_customs_dwell"] = 0.0
            feat_df["cum_delay_post_customs"] = feat_df["cum_delay_at_port"]

        feat_df["milestone_stage"] = milestone
        return feat_df

    def prepare_matrix(self, df: pd.DataFrame, fit_encoder: bool = False, milestone: int = 3) -> np.ndarray:
        """
        Transforms input dataframe into numerical + one-hot encoded numpy feature matrix.
        """
        engineered = self.engineer_features(df, milestone=milestone)

        num_cols = FEATURE_COLS_STATIC + [
            "weather_x_distance",
            "port_congestion_x_dwell",
            "inland_traffic_x_distance",
            "known_origin_delay",
            "known_transit_delay",
            "cum_delay_at_port",
            "berth_risk_factor",
            "known_port_waiting",
            "known_customs_dwell",
            "cum_delay_post_customs",
            "milestone_stage",
        ]

        X_num = engineered[num_cols].values

        if fit_encoder:
            X_cat = self.encoder.fit_transform(engineered[CATEGORICAL_COLS])
            cat_feat_names = list(self.encoder.get_feature_names_out(CATEGORICAL_COLS))
            self.feature_names_ = num_cols + cat_feat_names
        else:
            X_cat = self.encoder.transform(engineered[CATEGORICAL_COLS])

        return np.hstack([X_num, X_cat])

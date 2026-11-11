"""
Evaluation Suite and Benchmark Comparator for ETA Prediction and Delay Propagation.
Computes multi-metric performance, baselines, and ablation across shipment milestones.
"""

import numpy as np
import pandas as pd
import sys
import os

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, Any, List
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

def evaluate_models_and_baselines(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates:
      1. Naive Baseline (Static Scheduled ETA - assumes 0 delay)
      2. Historical Route Average Baseline
      3. Standard Linear Regression / Logistic Baseline
      4. Proposed Multi-Stage Delay Propagation AI Model
    """
    y_reg = df["total_delay_hours"].values
    y_cls = df["is_delayed"].values
    
    # --- 1. Naive Baseline: Assume 0 Delay ---
    pred_naive_reg = np.zeros_like(y_reg)
    pred_naive_cls = np.zeros_like(y_cls)  # Always predicts on-time
    
    naive_metrics = {
        "Model": "Static Schedule (Zero Delay)",
        "MAE (Hours)": round(float(mean_absolute_error(y_reg, pred_naive_reg)), 2),
        "RMSE (Hours)": round(float(np.sqrt(mean_squared_error(y_reg, pred_naive_reg))), 2),
        "R2 Score": round(float(r2_score(y_reg, pred_naive_reg)), 3),
        "Accuracy": round(float(accuracy_score(y_cls, pred_naive_cls)), 3),
        "Precision": 0.0,
        "Recall": 0.0,
        "F1 Score": 0.0,
        "ROC-AUC": 0.500,
    }
    
    # --- 2. Historical Route Mean Baseline ---
    route_means = df.groupby("route_id")["total_delay_hours"].mean().to_dict()
    route_delay_rates = df.groupby("route_id")["is_delayed"].mean().to_dict()
    
    pred_route_reg = df["route_id"].map(route_means).values
    pred_route_prob = df["route_id"].map(route_delay_rates).values
    pred_route_cls = (pred_route_prob >= 0.5).astype(int)
    
    route_metrics = {
        "Model": "Historical Route Average",
        "MAE (Hours)": round(float(mean_absolute_error(y_reg, pred_route_reg)), 2),
        "RMSE (Hours)": round(float(np.sqrt(mean_squared_error(y_reg, pred_route_reg))), 2),
        "R2 Score": round(float(r2_score(y_reg, pred_route_reg)), 3),
        "Accuracy": round(float(accuracy_score(y_cls, pred_route_cls)), 3),
        "Precision": round(float(precision_score(y_cls, pred_route_cls, zero_division=0)), 3),
        "Recall": round(float(recall_score(y_cls, pred_route_cls, zero_division=0)), 3),
        "F1 Score": round(float(f1_score(y_cls, pred_route_cls, zero_division=0)), 3),
        "ROC-AUC": round(float(roc_auc_score(y_cls, pred_route_prob)), 3),
    }
    
    # --- 3 & 4. Cross-Validated ML Models ---
    from models.delay_propagation import DelayPropagationEngine
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_reg_preds = np.zeros(len(df))
    cv_cls_probs = np.zeros(len(df))
    
    # Linear baseline CV preds
    cv_linear_preds = np.zeros(len(df))
    cv_linear_probs = np.zeros(len(df))
    
    for train_idx, val_idx in kf.split(df):
        train_sub = df.iloc[train_idx]
        val_sub = df.iloc[val_idx]
        
        # Train engine
        engine = DelayPropagationEngine()
        engine.fit(train_sub)
        
        # In-flight validation matrix (milestone 3)
        X_val = engine._prepare_matrix(val_sub, fit_encoder=False, milestone=3)
        cv_reg_preds[val_idx] = engine.regressor.predict(X_val)
        cv_cls_probs[val_idx] = engine.classifier.predict_proba(X_val)[:, 1]
        
        # Linear Baseline
        X_train_base = engine._prepare_matrix(train_sub, fit_encoder=False, milestone=0)
        X_val_base = engine._prepare_matrix(val_sub, fit_encoder=False, milestone=0)
        
        lin = LinearRegression().fit(X_train_base, train_sub["total_delay_hours"].values)
        cv_linear_preds[val_idx] = lin.predict(X_val_base)
        
        logreg = LogisticRegression(solver="liblinear", max_iter=1000).fit(X_train_base, train_sub["is_delayed"].values)
        cv_linear_probs[val_idx] = logreg.predict_proba(X_val_base)[:, 1]
        
    cv_linear_cls = (cv_linear_probs >= 0.5).astype(int)
    linear_metrics = {
        "Model": "Standard Linear Regression",
        "MAE (Hours)": round(float(mean_absolute_error(y_reg, cv_linear_preds)), 2),
        "RMSE (Hours)": round(float(np.sqrt(mean_squared_error(y_reg, cv_linear_preds))), 2),
        "R2 Score": round(float(r2_score(y_reg, cv_linear_preds)), 3),
        "Accuracy": round(float(accuracy_score(y_cls, cv_linear_cls)), 3),
        "Precision": round(float(precision_score(y_cls, cv_linear_cls, zero_division=0)), 3),
        "Recall": round(float(recall_score(y_cls, cv_linear_cls, zero_division=0)), 3),
        "F1 Score": round(float(f1_score(y_cls, cv_linear_cls, zero_division=0)), 3),
        "ROC-AUC": round(float(roc_auc_score(y_cls, cv_linear_probs)), 3),
    }
    
    cv_cls = (cv_cls_probs >= 0.5).astype(int)
    ai_metrics = {
        "Model": "Proposed Delay Propagation AI (Ours)",
        "MAE (Hours)": round(float(mean_absolute_error(y_reg, cv_reg_preds)), 2),
        "RMSE (Hours)": round(float(np.sqrt(mean_squared_error(y_reg, cv_reg_preds))), 2),
        "R2 Score": round(float(r2_score(y_reg, cv_reg_preds)), 3),
        "Accuracy": round(float(accuracy_score(y_cls, cv_cls)), 3),
        "Precision": round(float(precision_score(y_cls, cv_cls, zero_division=0)), 3),
        "Recall": round(float(recall_score(y_cls, cv_cls, zero_division=0)), 3),
        "F1 Score": round(float(f1_score(y_cls, cv_cls, zero_division=0)), 3),
        "ROC-AUC": round(float(roc_auc_score(y_cls, cv_cls_probs)), 3),
    }
    
    benchmark_df = pd.DataFrame([naive_metrics, route_metrics, linear_metrics, ai_metrics])
    
    # Milestone-progression evaluation (MAE by journey milestone)
    milestone_maes = {}
    engine_full = DelayPropagationEngine().fit(df)
    for m in [0, 1, 2, 3]:
        X_m = engine_full._prepare_matrix(df, fit_encoder=False, milestone=m)
        pred_m = engine_full.regressor.predict(X_m)
        milestone_maes[f"Milestone {m}"] = round(float(mean_absolute_error(y_reg, pred_m)), 2)
        
    return {
        "benchmark_df": benchmark_df,
        "milestone_maes": milestone_maes,
        "cv_reg_preds": cv_reg_preds,
        "cv_cls_probs": cv_cls_probs,
        "actual_y_reg": y_reg,
        "actual_y_cls": y_cls,
    }

if __name__ == "__main__":
    df = pd.read_csv("data/shipments_dataset.csv")
    res = evaluate_models_and_baselines(df)
    print("=== Model Benchmark Results ===")
    print(res["benchmark_df"].to_string(index=False))
    print("\n=== Milestone Error Progression (In-flight Convergence) ===")
    print(res["milestone_maes"])

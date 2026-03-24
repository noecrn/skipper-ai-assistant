# src/skipper_ai/analyze.py

import pandas as pd
import joblib
import shap
import numpy as np
import os

def run_analysis(data_path, model_path='models/performance_model.joblib'):
    """
    Performs SHAP analysis on a processed run data.
    """
    # Load model
    model = joblib.load(model_path)
    
    # Load data
    df = pd.read_csv(data_path)
    
    from skipper_ai.ingest import SAIL_MAPPING
    if 'sail_id_numeric' not in df.columns:
        if 'sail_id' in df.columns:
            df['sail_id_numeric'] = df['sail_id'].map(SAIL_MAPPING).fillna(-1).astype(int)
        else:
            df['sail_id_numeric'] = -1
    
    # Define features (must match train.py)
    features = ['tws', 'twa', 'heel', 'sail_id_numeric']
    
    # Prepare input for model
    x = df[features]
    
    # Run predictions
    df['predicted_performance'] = model.predict(x)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)
    
    # If shap_values is a list (multi-output), take the first one
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Calculate global importance
    global_importance = np.abs(shap_values).mean(axis=0)
    importance_dict = dict(zip(features, global_importance.tolist()))
    
    # Calculate local importance summary (average impact per feature)
    # This helps identify if a feature was consistently pushing performance down
    impact_dict = dict(zip(features, shap_values.mean(axis=0).tolist()))
    
    # Create a summary of the run
    summary = {
        "run_id": os.path.basename(os.path.dirname(data_path)),
        "avg_performance": float(df['performance_ratio'].mean()),
        "avg_expected_speed": float(df['expected_speed'].mean()),
        "avg_boat_speed": float(df['boat_speed'].mean()),
        "feature_importance": importance_dict,
        "feature_impact": impact_dict,
        "total_rows": len(df)
    }
    
    # We can also add top 3 loss segments
    df['loss'] = df['performance_ratio'] - 1.0
    under_perf = df[df['loss'] < -0.05].copy() # More than 5% loss
    
    if not under_perf.empty:
        # Most impactful features for under-performing rows
        under_perf_x = under_perf[features]
        shap_values_under = explainer.shap_values(under_perf_x)
        if isinstance(shap_values_under, list):
            shap_values_under = shap_values_under[0]
        
        under_perf_impact = dict(zip(features, shap_values_under.mean(axis=0).tolist()))
        summary["under_performance_impact"] = under_perf_impact
    else:
        summary["under_performance_impact"] = {}

    return summary

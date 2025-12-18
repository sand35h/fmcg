# -*- coding: utf-8 -*-
"""
ADD UNCERTAINTY QUANTIFICATION TO YOUR EXISTING PIPELINE
=========================================================
Add this code to your existing training script to generate prediction intervals.

This uses quantile regression to provide 90% confidence bands.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

def train_quantile_models(X_train, y_train, X_val, y_val, quantiles=[0.05, 0.5, 0.95]):
    """
    Train quantile regression models for uncertainty estimation.
    
    Returns:
        dict: Dictionary of trained quantile models
    """
    quantile_models = {}
    
    for q in quantiles:
        print(f"Training quantile model for q={q}...")
        
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=q,  # Quantile level
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=64,
            max_depth=8,
            min_child_samples=100,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        quantile_models[q] = model
        print(f"  Best iteration: {model.best_iteration_}")
    
    return quantile_models


def predict_with_intervals(models, X, quantiles=[0.05, 0.95]):
    """
    Generate predictions with confidence intervals.
    
    Args:
        models: Dict of quantile models
        X: Input features
        quantiles: Lower and upper quantile bounds
        
    Returns:
        DataFrame with mean, lower, upper predictions
    """
    predictions = {}
    
    # Get predictions from all quantile models
    for q, model in models.items():
        pred = np.clip(model.predict(X), 0, None)
        predictions[f'q{int(q*100)}'] = pred
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'mean': predictions.get('q50', predictions[list(predictions.keys())[1]]),
        'lower': predictions[f'q{int(quantiles[0]*100)}'],
        'upper': predictions[f'q{int(quantiles[1]*100)}']
    })
    
    # Calculate interval width
    result_df['interval_width'] = result_df['upper'] - result_df['lower']
    result_df['relative_width'] = result_df['interval_width'] / (result_df['mean'] + 1)
    
    return result_df


def evaluate_interval_coverage(y_true, predictions_df):
    """
    Evaluate how well prediction intervals cover actual values.
    
    Target: 90% coverage for 90% confidence intervals.
    """
    coverage = np.mean(
        (y_true >= predictions_df['lower']) & 
        (y_true <= predictions_df['upper'])
    )
    
    print(f"Prediction Interval Coverage: {coverage*100:.2f}%")
    print(f"Target Coverage: 90%")
    
    if coverage < 0.85:
        print("⚠️ WARNING: Coverage too low - intervals are too narrow")
    elif coverage > 0.95:
        print("⚠️ WARNING: Coverage too high - intervals are too wide")
    else:
        print("✓ Coverage is acceptable")
    
    return coverage


# =============================================================================
# INTEGRATION WITH YOUR EXISTING CODE
# =============================================================================

def add_uncertainty_to_existing_pipeline(train_df, val_df, test_df, feature_cols):
    """
    This function shows how to integrate uncertainty quantification
    into your existing training pipeline.
    
    Add this AFTER training your main LightGBM model.
    """
    
    print("\n" + "="*70)
    print("UNCERTAINTY QUANTIFICATION")
    print("="*70)
    
    # Prepare data
    X_train, y_train = train_df[feature_cols], train_df['true_demand']
    X_val, y_val = val_df[feature_cols], val_df['true_demand']
    X_test, y_test = test_df[feature_cols], test_df['true_demand']
    
    # Train quantile models
    quantile_models = train_quantile_models(X_train, y_train, X_val, y_val)
    
    # Generate predictions with intervals
    test_predictions = predict_with_intervals(quantile_models, X_test)
    
    # Evaluate coverage
    coverage = evaluate_interval_coverage(y_test, test_predictions)
    
    # Save predictions with intervals
    output_df = test_df[['date', 'sku_id', 'location_id']].copy()
    output_df['actual'] = y_test.values
    output_df['predicted_mean'] = test_predictions['mean'].values
    output_df['predicted_lower'] = test_predictions['lower'].values
    output_df['predicted_upper'] = test_predictions['upper'].values
    output_df['interval_width'] = test_predictions['interval_width'].values
    
    output_df.to_csv('/kaggle/working/predictions_with_uncertainty.csv', index=False)
    print(f"\n✓ Saved predictions with uncertainty intervals")
    
    # Calculate metrics
    test_mae = mean_absolute_error(y_test, test_predictions['mean'])
    print(f"\nTest MAE (median forecast): {test_mae:.2f}")
    print(f"Average interval width: {test_predictions['interval_width'].mean():.2f}")
    print(f"Coverage: {coverage*100:.1f}%")
    
    return quantile_models, output_df


# =============================================================================
# EXAMPLE: HOW TO ADD TO YOUR EXISTING SCRIPT
# =============================================================================

"""
In your existing main() function, add this after model training:

def main(run_forecast=True):
    # ... your existing code ...
    
    # STEP 5: TRAIN MODEL
    train_results = train_baseline(df, feature_cols)
    
    # 🆕 ADD THIS: Train uncertainty models
    df_sorted = df.sort_values('date').reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]
    
    quantile_models, uncertainty_results = add_uncertainty_to_existing_pipeline(
        train_df, val_df, test_df, feature_cols
    )
    
    # Save quantile models
    for q, model in quantile_models.items():
        model.booster_.save_model(f'/kaggle/working/quantile_model_q{int(q*100)}.txt')
    
    # ... rest of your code ...
"""

if __name__ == "__main__":
    print("This module adds uncertainty quantification to your existing pipeline.")
    print("Copy the add_uncertainty_to_existing_pipeline() function to your main script.")

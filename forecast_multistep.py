# -*- coding: utf-8 -*-
"""
Multi-Step Forecasting with Recursive Strategy
===============================================
Uses trained baseline model to generate 7-day and 14-day forecasts.
"""

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Auto-detect environment
if Path("/kaggle/input").exists():
    DATA_ROOT = Path("/kaggle/input/fmcgparquet")
    MODEL_PATH = Path("/kaggle/working/baseline_model.txt")
    OUTPUT_ROOT = Path("/kaggle/working")
else:
    DATA_ROOT = Path("./data")
    MODEL_PATH = Path("./output/baseline_model.txt")
    OUTPUT_ROOT = Path("./output")

def calculate_wmape(y_true, y_pred):
    """Calculate WMAPE metric."""
    y_pred = np.clip(y_pred, 0, None)
    return np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(np.abs(y_true)), 1) * 100

def recursive_forecast(model, initial_data, feature_cols, horizon=7):
    """
    Generate multi-step forecast using recursive strategy.
    
    Args:
        model: Trained LightGBM model
        initial_data: DataFrame with last known data for each SKU-location
        feature_cols: List of feature column names
        horizon: Number of days to forecast
    
    Returns:
        DataFrame with forecasts for each day
    """
    logger.info(f"Generating {horizon}-day recursive forecast...")
    
    forecasts = []
    current_data = initial_data.copy()
    
    for day in range(1, horizon + 1):
        # Predict next day
        X = current_data[feature_cols]
        pred = np.clip(model.predict(X), 0, None)
        
        # Store forecast
        forecast_date = current_data['date'] + pd.Timedelta(days=day)
        forecast_df = current_data[['sku_id', 'location_id']].copy()
        forecast_df['date'] = forecast_date
        forecast_df['horizon'] = day
        forecast_df['forecast'] = pred
        forecasts.append(forecast_df)
        
        # Update lag features for next iteration
        if 'demand_lag_1' in feature_cols:
            current_data['demand_lag_7'] = current_data.get('demand_lag_1', pred)
            current_data['demand_lag_1'] = pred
        
        if 'demand_rolling_7_mean' in feature_cols:
            # Update rolling mean (simplified - uses last prediction)
            current_data['demand_rolling_7_mean'] = (
                current_data['demand_rolling_7_mean'] * 0.85 + pred * 0.15
            )
        
        if 'demand_rolling_7_std' in feature_cols:
            # Keep std relatively stable
            current_data['demand_rolling_7_std'] = current_data['demand_rolling_7_std']
        
        # Update calendar features
        if 'day_of_week' in feature_cols:
            current_data['day_of_week'] = (current_data['day_of_week'] + 1) % 7
            current_data['day_of_week_sin'] = np.sin(2 * np.pi * current_data['day_of_week'] / 7)
            current_data['day_of_week_cos'] = np.cos(2 * np.pi * current_data['day_of_week'] / 7)
            current_data['is_weekend'] = (current_data['day_of_week'] >= 5).astype(np.int8)
        
        if 'day_of_month' in feature_cols:
            current_data['day_of_month'] = (current_data['day_of_month'] % 31) + 1
        
        logger.info(f"  Day {day}/{horizon}: Avg forecast = {pred.mean():.1f}")
    
    return pd.concat(forecasts, ignore_index=True)

def evaluate_multistep(test_data, forecasts, horizons=[1, 7, 14]):
    """Evaluate forecast accuracy at different horizons."""
    logger.info("\n" + "="*70)
    logger.info("MULTI-STEP FORECAST EVALUATION")
    logger.info("="*70)
    
    results = []
    
    for h in horizons:
        if h > forecasts['horizon'].max():
            continue
        
        # Get forecasts for this horizon
        h_forecasts = forecasts[forecasts['horizon'] == h].copy()
        
        # Merge with actuals
        h_forecasts = h_forecasts.merge(
            test_data[['date', 'sku_id', 'location_id', 'true_demand']],
            on=['date', 'sku_id', 'location_id'],
            how='inner'
        )
        
        if len(h_forecasts) == 0:
            logger.warning(f"No matching actuals for horizon {h}")
            continue
        
        # Calculate WMAPE
        wmape = calculate_wmape(h_forecasts['true_demand'], h_forecasts['forecast'])
        mae = np.mean(np.abs(h_forecasts['true_demand'] - h_forecasts['forecast']))
        
        logger.info(f"Horizon {h:2d}-day: WMAPE = {wmape:5.2f}%, MAE = {mae:6.1f}")
        
        results.append({
            'horizon': h,
            'wmape': wmape,
            'mae': mae,
            'n_forecasts': len(h_forecasts)
        })
    
    logger.info("="*70)
    
    return pd.DataFrame(results)

def main():
    """Main forecasting pipeline."""
    logger.info("="*70)
    logger.info("MULTI-STEP RECURSIVE FORECASTING")
    logger.info("="*70)
    
    # Load model
    logger.info(f"\nLoading model from {MODEL_PATH}")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    
    # Load feature importance to get exact training features
    fi_path = OUTPUT_ROOT / 'feature_importance.csv'
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        feature_cols = fi['feature'].tolist()
        logger.info(f"Loaded {len(feature_cols)} features from feature_importance.csv")
    else:
        logger.error(f"Feature importance file not found: {fi_path}")
        return
    
    # Load test predictions (already has all features)
    pred_path = OUTPUT_ROOT / 'test_predictions.csv'
    if not pred_path.exists():
        logger.error(f"Test predictions file not found: {pred_path}")
        logger.info("Run train_baseline_only.py first to generate predictions.")
        return
    
    # Load full processed data from training
    logger.info("Loading and processing data...")
    df = pd.read_parquet(DATA_ROOT / "daily_timeseries.parquet")
    df['date'] = pd.to_datetime(df['date'])
    
    # Apply same preprocessing as training
    from train_baseline_only import correct_stockouts, engineer_features, reduce_mem_usage
    df = correct_stockouts(df)
    df = engineer_features(df)
    df = reduce_mem_usage(df)
    
    # Get last 30 days for evaluation
    test_start = df['date'].max() - pd.Timedelta(days=30)
    test_data = df[df['date'] > test_start].copy()
    
    logger.info(f"Test period: {test_data['date'].min().date()} to {test_data['date'].max().date()}")
    logger.info(f"Test data: {len(test_data):,} rows")
    
    # Get initial state (last known day before forecast period)
    forecast_start = test_data['date'].min()
    initial_data = df[df['date'] == forecast_start - pd.Timedelta(days=1)].copy()
    
    if len(initial_data) == 0:
        logger.error("No initial data found. Cannot generate forecast.")
        return
    
    logger.info(f"Initial state: {len(initial_data):,} SKU-location combinations")
    logger.info(f"Using {len(feature_cols)} features (exact match with training)")
    
    # Generate 14-day forecast
    forecasts = recursive_forecast(model, initial_data, feature_cols, horizon=14)
    
    # Evaluate at different horizons
    eval_results = evaluate_multistep(test_data, forecasts, horizons=[1, 3, 7, 14])
    
    # Save results
    forecasts.to_csv(OUTPUT_ROOT / 'multistep_forecasts.csv', index=False)
    eval_results.to_csv(OUTPUT_ROOT / 'multistep_evaluation.csv', index=False)
    
    logger.info(f"\nSaved forecasts to {OUTPUT_ROOT / 'multistep_forecasts.csv'}")
    logger.info(f"Saved evaluation to {OUTPUT_ROOT / 'multistep_evaluation.csv'}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"1-day WMAPE:  {eval_results[eval_results['horizon']==1]['wmape'].values[0]:.2f}%")
    logger.info(f"7-day WMAPE:  {eval_results[eval_results['horizon']==7]['wmape'].values[0]:.2f}%")
    logger.info(f"14-day WMAPE: {eval_results[eval_results['horizon']==14]['wmape'].values[0]:.2f}%")
    logger.info("="*70)
    
    return forecasts, eval_results

if __name__ == "__main__":
    forecasts, eval_results = main()

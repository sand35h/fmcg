# -*- coding: utf-8 -*-
"""
FMCG Baseline Model - Simplified Training
==========================================
Single LightGBM model with strict time-based validation.
No advanced models, no ensemble, no complexity.
"""

import logging
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Auto-detect Kaggle environment
if Path("/kaggle/input").exists():
    DATA_ROOT = Path("/kaggle/input/fmcgparquet")
    OUTPUT_ROOT = Path("/kaggle/working")
else:
    DATA_ROOT = Path("./data")
    OUTPUT_ROOT = Path("./output")
    OUTPUT_ROOT.mkdir(exist_ok=True)

# Files
DAILY_TS_FILE = "daily_timeseries.parquet"
SKU_FILE = "sku_master.parquet"
LOCATION_FILE = "location_master.parquet"
FESTIVAL_FILE = "festival_calendar.parquet"
WEATHER_FILE = "weather_data.parquet"
SHOCK_FILE = "external_shocks.parquet"
COMPETITOR_FILE = "competitor_activity.parquet"

# Target
TARGET_COL = "true_demand"  # Stockout-corrected target

# Leaky columns to exclude
LEAK_COLS = ["expected_demand", "unfulfilled_demand", "incoming_stock", 
             "closing_stock", "reorder_point", "safety_stock", "actual_demand"]

# =============================================================================
# UTILITIES
# =============================================================================

def reduce_mem_usage(df):
    """Reduce memory usage by downcasting dtypes."""
    for col in df.columns:
        col_type = df[col].dtype
        if 'datetime' in str(col_type) or 'category' in str(col_type):
            continue
        if str(col_type)[:3] == 'int':
            df[col] = df[col].astype(np.int32)
        elif str(col_type)[:5] == 'float':
            df[col] = df[col].astype(np.float32)
    return df

def calculate_wmape(y_true, y_pred):
    """Calculate WMAPE metric."""
    y_pred = np.clip(y_pred, 0, None)
    return np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(np.abs(y_true)), 1) * 100

# =============================================================================
# STOCKOUT CORRECTION
# =============================================================================

def correct_stockouts(df):
    """Correct censored demand from stockouts."""
    logger.info("Applying stockout correction...")
    
    if 'stockout_flag' not in df.columns:
        logger.warning("No stockout_flag found. Skipping correction.")
        df['true_demand'] = df['actual_demand']
        return df
    
    df = df.sort_values(['sku_id', 'location_id', 'date']).reset_index(drop=True)
    g = df.groupby(['sku_id', 'location_id'], observed=True)
    
    # Calculate 7-day velocity
    velocity = g['actual_demand'].shift(1).rolling(7, min_periods=3).mean()
    
    # Cap at p95 to prevent promo inflation (per SKU-location)
    df['_velocity'] = velocity.values
    velocity_p95 = df.groupby(['sku_id', 'location_id'], observed=True)['_velocity'].transform(lambda x: x.quantile(0.95))
    velocity_capped = np.minimum(df['_velocity'].values, velocity_p95.values)
    df = df.drop(columns=['_velocity'])
    
    # Identify censored rows
    fully_censored = (df.get('opening_stock', 0) == 0)
    partially_censored = (
        (df['stockout_flag'] == 1) & 
        (df.get('closing_stock', 1) == 0) &
        (df['actual_demand'] > 0)
    )
    
    # Create corrected target
    df['true_demand'] = df['actual_demand'].astype(np.float32)
    df.loc[partially_censored, 'true_demand'] = np.maximum(
        df.loc[partially_censored, 'actual_demand'].values.astype(np.float32),
        velocity_capped[partially_censored].astype(np.float32)
    )
    
    # Exclude fully censored rows
    df = df[~fully_censored].reset_index(drop=True)
    
    logger.info(f"Excluded {fully_censored.sum():,} fully censored rows")
    logger.info(f"Corrected {partially_censored.sum():,} partially censored rows")
    
    return df

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_calendar_features(df):
    """Add temporal features."""
    df['day_of_week'] = df['date'].dt.weekday
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(np.int8)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def add_lag_features(df):
    """Add lag features with proper shifting (no leakage)."""
    logger.info("Creating lag features...")
    df = df.sort_values(['sku_id', 'location_id', 'date']).reset_index(drop=True)
    g = df.groupby(['sku_id', 'location_id'], observed=True)
    
    # Lags
    df['demand_lag_1'] = g[TARGET_COL].shift(1).astype(np.float32)
    df['demand_lag_7'] = g[TARGET_COL].shift(7).astype(np.float32)
    
    # Rolling stats
    df['demand_rolling_7_mean'] = g[TARGET_COL].shift(1).rolling(7, min_periods=1).mean().astype(np.float32)
    df['demand_rolling_7_std'] = g[TARGET_COL].shift(1).rolling(7, min_periods=1).std().astype(np.float32)
    
    return df

def add_price_promo_features(df):
    """Add price and promo features."""
    if 'base_price' in df.columns and 'price' in df.columns:
        df['discount_depth'] = ((df['base_price'] - df['price']) / df['base_price'].replace(0, np.nan)).fillna(0).clip(0, 1).astype(np.float32)
        df['price_to_base'] = (df['price'] / df['base_price'].replace(0, np.nan)).fillna(1.0).astype(np.float32)
    
    if 'promo_flag' in df.columns:
        df = df.sort_values(['sku_id', 'location_id', 'date']).reset_index(drop=True)
        g = df.groupby(['sku_id', 'location_id'], observed=True)
        promo_rolling = g['promo_flag'].rolling(7, min_periods=1).mean()
        df['promo_intensity_7d'] = promo_rolling.reset_index(level=[0,1], drop=True).astype(np.float32)
    
    return df

def engineer_features(df):
    """Full feature engineering pipeline."""
    logger.info("Starting feature engineering...")
    
    # Load and merge reference data
    for file, name in [(SKU_FILE, 'sku'), (LOCATION_FILE, 'location')]:
        path = DATA_ROOT / file
        if path.exists():
            ref_df = pd.read_parquet(path)
            merge_col = f"{name}_id"
            if merge_col in df.columns and merge_col in ref_df.columns:
                df = df.merge(ref_df, on=merge_col, how='left', suffixes=('', f'_{name}'))
                logger.info(f"Merged {name} master")
    
    # Festival flags
    festival_path = DATA_ROOT / FESTIVAL_FILE
    if festival_path.exists():
        fest_df = pd.read_parquet(festival_path)
        fest_df['is_festival'] = 1
        df = df.merge(fest_df[['date', 'is_festival']], on='date', how='left')
        df['is_festival'] = df['is_festival'].fillna(0).astype(np.int8)
        logger.info("Added festival features")
    
    # Weather (gated by category)
    weather_path = DATA_ROOT / WEATHER_FILE
    if weather_path.exists() and 'category' in df.columns:
        weather_df = pd.read_parquet(weather_path)
        df = df.merge(weather_df, on='date', how='left')
        weather_sensitive = ['BEVERAGES', 'DAIRY', 'SNACKS']
        for col in ['avg_temp_c', 'precipitation_mm']:
            if col in df.columns:
                df[f'{col}_masked'] = np.where(
                    df['category'].isin(weather_sensitive),
                    df[col], 0.0
                ).astype(np.float32)
                df = df.drop(columns=[col])
        logger.info("Added weather features (gated)")
    
    # Calendar features
    df = add_calendar_features(df)
    
    # Price/promo features
    df = add_price_promo_features(df)
    
    # Lag features (LAST - to avoid leakage)
    df = add_lag_features(df)
    
    # Drop rows with missing lags
    df = df.dropna(subset=['demand_lag_1', 'demand_lag_7']).reset_index(drop=True)
    
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df

# =============================================================================
# TRAINING
# =============================================================================

def train_baseline(df, feature_cols):
    """Train single LightGBM baseline model."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING BASELINE MODEL")
    logger.info("="*70)
    
    # Time-based split: 70% train, 15% val, 15% test
    df = df.sort_values('date').reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Train: {len(train_df):,} rows ({train_df['date'].min()} to {train_df['date'].max()})")
    logger.info(f"Val:   {len(val_df):,} rows ({val_df['date'].min()} to {val_df['date'].max()})")
    logger.info(f"Test:  {len(test_df):,} rows ({test_df['date'].min()} to {test_df['date'].max()})")
    
    # Prepare data
    X_train, y_train = train_df[feature_cols], train_df[TARGET_COL].astype(np.float32)
    X_val, y_val = val_df[feature_cols], val_df[TARGET_COL].astype(np.float32)
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COL].astype(np.float32)
    
    # Model parameters
    params = {
        'objective': 'tweedie',
        'tweedie_variance_power': 1.3,
        'metric': 'None',
        'n_estimators': 3000,
        'learning_rate': 0.03,
        'num_leaves': 128,
        'max_depth': 8,
        'min_child_samples': 100,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'reg_lambda': 1.0,
        'reg_alpha': 0.5,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # Train
    logger.info("\nTraining LightGBM with Tweedie loss...")
    model = lgb.LGBMRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    logger.info(f"Best iteration: {model.best_iteration_}")
    
    # Evaluate
    y_train_pred = np.clip(model.predict(X_train), 0, None)
    y_val_pred = np.clip(model.predict(X_val), 0, None)
    y_test_pred = np.clip(model.predict(X_test), 0, None)
    
    train_wmape = calculate_wmape(y_train, y_train_pred)
    val_wmape = calculate_wmape(y_val, y_val_pred)
    test_wmape = calculate_wmape(y_test, y_test_pred)
    
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    logger.info(f"Train WMAPE: {train_wmape:.2f}%")
    logger.info(f"Val WMAPE:   {val_wmape:.2f}%")
    logger.info(f"Test WMAPE:  {test_wmape:.2f}%  ⭐ PRIMARY METRIC")
    logger.info("="*70)
    
    # Feature importance
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 15 Features:")
    logger.info(fi.head(15).to_string(index=False))
    
    # Save
    model.booster_.save_model(str(OUTPUT_ROOT / 'baseline_model.txt'))
    fi.to_csv(OUTPUT_ROOT / 'feature_importance.csv', index=False)
    
    # Save predictions
    pred_df = test_df[['date', 'sku_id', 'location_id']].copy()
    pred_df['actual'] = y_test.values
    pred_df['predicted'] = y_test_pred
    pred_df.to_csv(OUTPUT_ROOT / 'test_predictions.csv', index=False)
    
    logger.info(f"\nSaved artifacts to {OUTPUT_ROOT}")
    
    return {
        'train_wmape': train_wmape,
        'val_wmape': val_wmape,
        'test_wmape': test_wmape,
        'model': model,
        'feature_importance': fi
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training pipeline."""
    logger.info("="*70)
    logger.info("FMCG BASELINE MODEL TRAINING")
    logger.info("="*70)
    
    # Load data
    data_path = DATA_ROOT / DAILY_TS_FILE
    logger.info(f"\nLoading data from {data_path}")
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Loaded {len(df):,} rows")
    
    # Stockout correction
    df = correct_stockouts(df)
    
    # Feature engineering
    df = engineer_features(df)
    df = reduce_mem_usage(df)
    
    # Select features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_cols = ['date', 'sku_id', 'location_id']
    exclude = [TARGET_COL] + LEAK_COLS + id_cols
    feature_cols = [c for c in numeric_cols if c not in exclude and c in df.columns]
    
    logger.info(f"\nUsing {len(feature_cols)} features")
    logger.info(f"Target: {TARGET_COL}")
    
    # Train
    results = train_baseline(df, feature_cols)
    
    # Final summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Test WMAPE: {results['test_wmape']:.2f}%")
    logger.info(f"Target: 35-40% (Acceptable: <45%)")
    
    if results['test_wmape'] < 45:
        logger.info("ACCEPTABLE - Ready for hyperparameter tuning")
    else:
        logger.info("NEEDS INVESTIGATION - Check data/features")
    
    return results

if __name__ == "__main__":
    results = main()

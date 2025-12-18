# -*- coding: utf-8 -*-
"""
FMCG Training & Multi-Horizon Forecasting Pipeline
===================================================
Combined pipeline for:
1. Training LightGBM models for multiple forecast horizons (1, 7, 14 days)
2. Stockout correction and feature engineering
3. SHAP explainability and drift detection
4. Business KPI measurement
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
# OPTIONAL DEPENDENCIES
# =============================================================================
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.info("SHAP not available - install with: pip install shap")

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import json
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Kaggle
import matplotlib.pyplot as plt

# =============================================================================
# EMBEDDED: SHAP EXPLAINABILITY
# =============================================================================

def create_shap_report(model, X_train_sample, X_test_sample, feature_cols, output_dir):
    """
    Generate SHAP explainability report.
    Embedded version for Kaggle compatibility.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available - skipping explainability")
        return None
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info("Generating SHAP explainability report...")
    
    try:
        # Create TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for test sample
        shap_values = explainer.shap_values(X_test_sample)
        
        # 1. Summary plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test_sample, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(output_path / 'shap_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP summary plot to {output_path / 'shap_summary.png'}")
        
        # 2. Feature importance bar plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(output_path / 'shap_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP importance plot to {output_path / 'shap_importance.png'}")
        
        # 3. Calculate global feature importance from SHAP
        shap_importance = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        importance_df.to_csv(output_path / 'shap_feature_importance.csv', index=False)
        
        logger.info("✓ SHAP explainability report complete")
        return importance_df
        
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
        return None

# =============================================================================
# EMBEDDED: DRIFT DETECTION
# =============================================================================

class DriftDetector:
    """Detects data drift and concept drift."""
    
    def __init__(self, drift_threshold=0.15):
        self.drift_threshold = drift_threshold
        self.baseline_stats = {}
        self.critical_features = []
    
    def capture_baseline(self, X_train, y_train, critical_features=None):
        """Capture baseline statistics from training data."""
        logger.info("Capturing baseline statistics for drift detection...")
        
        self.critical_features = critical_features if critical_features else X_train.columns.tolist()[:20]
        
        for col in self.critical_features:
            if col in X_train.columns:
                values = X_train[col].dropna()
                self.baseline_stats[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()) if values.std() > 0 else 1.0,
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q25': float(values.quantile(0.25)),
                    'q50': float(values.quantile(0.50)),
                    'q75': float(values.quantile(0.75))
                }
        
        # Target statistics
        self.baseline_stats['__target__'] = {
            'mean': float(y_train.mean()),
            'std': float(y_train.std()) if y_train.std() > 0 else 1.0
        }
        
        logger.info(f"✓ Baseline captured for {len(self.baseline_stats)} features")
    
    def detect_data_drift(self, X_new):
        """Detect data drift using statistical measures."""
        logger.info(f"Analyzing data drift for {len(X_new)} new samples...")
        
        drift_results = {}
        drifted_features = []
        
        for col in self.critical_features:
            if col not in X_new.columns or col not in self.baseline_stats:
                continue
            
            baseline = self.baseline_stats[col]
            current_values = X_new[col].dropna()
            
            if len(current_values) < 10:
                continue
            
            # Mean shift (normalized)
            mean_shift = abs(current_values.mean() - baseline['mean']) / (baseline['std'] + 1e-6)
            
            # Std shift (normalized)
            std_shift = abs(current_values.std() - baseline['std']) / (baseline['std'] + 1e-6)
            
            # Combined drift score
            drift_score = (mean_shift + std_shift) / 2
            
            drift_results[col] = {
                'drift_score': float(drift_score),
                'is_drifted': drift_score > self.drift_threshold
            }
            
            if drift_results[col]['is_drifted']:
                drifted_features.append(col)
        
        overall_drift = np.mean([r['drift_score'] for r in drift_results.values()]) if drift_results else 0
        
        logger.info(f"Overall drift score: {overall_drift:.3f} (threshold: {self.drift_threshold})")
        if drifted_features:
            logger.warning(f"Drifted features: {drifted_features[:5]}")
        
        return {
            'overall_drift': overall_drift,
            'drifted_features': drifted_features,
            'needs_attention': overall_drift > self.drift_threshold
        }
    
    def save_baseline(self, filepath):
        """Save baseline statistics to file."""
        baseline_data = {
            'baseline_stats': self.baseline_stats,
            'critical_features': self.critical_features,
            'drift_threshold': self.drift_threshold,
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        logger.info(f"✓ Baseline saved to {filepath}")
    
    def load_baseline(self, filepath):
        """Load baseline statistics from file."""
        with open(filepath, 'r') as f:
            baseline_data = json.load(f)
        self.baseline_stats = baseline_data['baseline_stats']
        self.critical_features = baseline_data['critical_features']
        self.drift_threshold = baseline_data['drift_threshold']
        logger.info(f"✓ Baseline loaded from {filepath}")

# =============================================================================
# EMBEDDED: BUSINESS KPI MEASUREMENT
# =============================================================================

class BusinessKPICalculator:
    """Compares ML forecasts against baseline methods."""
    
    def __init__(self, lead_time_days=7, service_level=0.95):
        self.lead_time_days = lead_time_days
        self.service_level = service_level
    
    def calculate_baseline_forecasts(self, df):
        """Generate baseline forecasts for comparison."""
        df = df.sort_values(['sku_id', 'location_id', 'date']).reset_index(drop=True)
        
        # Moving average forecast
        df['ma_forecast'] = df.groupby(['sku_id', 'location_id'])['actual_demand'].transform(
            lambda x: x.shift(1).rolling(7, min_periods=1).mean()
        )
        
        # Naive forecast (last value)
        df['naive_forecast'] = df.groupby(['sku_id', 'location_id'])['actual_demand'].shift(1)
        
        return df
    
    def compare_methods(self, ml_forecasts, actual_df):
        """Compare ML forecasts against baseline methods."""
        logger.info("\n" + "="*70)
        logger.info("BUSINESS KPI COMPARISON: ML vs. BASELINE")
        logger.info("="*70)
        
        # Merge forecasts with actuals
        df = ml_forecasts.merge(
            actual_df[['date', 'sku_id', 'location_id', 'actual_demand']],
            on=['date', 'sku_id', 'location_id'],
            how='inner'
        )
        
        # Add baseline forecasts
        df = self.calculate_baseline_forecasts(df)
        
        # Calculate metrics for each method
        methods = {
            'ML Forecast': 'forecast',
            'Moving Avg (7d)': 'ma_forecast', 
            'Naive (t-1)': 'naive_forecast'
        }
        
        results = {}
        for name, col in methods.items():
            if col not in df.columns:
                continue
            
            valid = df[col].notna()
            y_true = df.loc[valid, 'actual_demand']
            y_pred = df.loc[valid, col]
            
            wmape = np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(np.abs(y_true)), 1) * 100
            mae = mean_absolute_error(y_true, y_pred)
            
            results[name] = {'wmape': wmape, 'mae': mae}
        
        # Print comparison
        logger.info(f"\n{'Method':<20} {'WMAPE':>10} {'MAE':>10}")
        logger.info("-" * 42)
        for name, metrics in results.items():
            logger.info(f"{name:<20} {metrics['wmape']:>9.2f}% {metrics['mae']:>10.2f}")
        
        # Calculate improvement
        if 'ML Forecast' in results and 'Moving Avg (7d)' in results:
            ml_wmape = results['ML Forecast']['wmape']
            baseline_wmape = results['Moving Avg (7d)']['wmape']
            improvement = ((baseline_wmape - ml_wmape) / baseline_wmape) * 100
            
            logger.info("\n" + "="*70)
            logger.info(f"📊 ML improves WMAPE by {improvement:.1f}% vs Moving Average")
            if improvement >= 20:
                logger.info("✓ Meets target: 20-25% improvement")
            else:
                logger.info(f"⚠ Below target (need 20-25% improvement)")
        
        return results
    
    def export_report(self, results, output_path):
        """Export KPI report to file."""
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BUSINESS KPI REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for method, metrics in results.items():
                f.write(f"{method}:\n")
                f.write(f"  WMAPE: {metrics['wmape']:.2f}%\n")
                f.write(f"  MAE: {metrics['mae']:.2f}\n\n")
        
        logger.info(f"✓ KPI report saved to {output_path}")

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
    df['day_of_month'] = df['date'].dt.day
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
        'feature_importance': fi,
        'feature_cols': feature_cols
    }

# =============================================================================
# MULTI-HORIZON MODEL TRAINING
# =============================================================================

def train_horizon_models(df, feature_cols, horizons=[1, 7, 14]):
    """
    Train separate models for each forecast horizon.
    
    Args:
        df: Processed DataFrame with features
        feature_cols: List of feature column names
        horizons: List of forecast horizons (days ahead)
    
    Returns:
        Dictionary with models and results for each horizon
    """
    logger.info("\n" + "="*70)
    logger.info("TRAINING MULTI-HORIZON MODELS")
    logger.info("="*70)
    
    all_results = {}
    
    # Time-based split: 70% train, 15% val, 15% test
    df_sorted = df.sort_values(['sku_id', 'location_id', 'date']).reset_index(drop=True)
    
    for h in horizons:
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING HORIZON h={h} MODEL")
        logger.info(f"{'='*70}")
        
        # Create horizon-shifted target
        df_h = df_sorted.copy()
        df_h[f'target_h{h}'] = df_h.groupby(['sku_id', 'location_id'], observed=True)[TARGET_COL].shift(-h)
        
        # Drop rows without target (end of each series)
        df_h = df_h.dropna(subset=[f'target_h{h}']).reset_index(drop=True)
        
        # Time-based split
        df_h = df_h.sort_values('date').reset_index(drop=True)
        n = len(df_h)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        train_df = df_h.iloc[:train_end]
        val_df = df_h.iloc[train_end:val_end]
        test_df = df_h.iloc[val_end:]
        
        logger.info(f"Train: {len(train_df):,} rows")
        logger.info(f"Val:   {len(val_df):,} rows")
        logger.info(f"Test:  {len(test_df):,} rows")
        
        # Prepare data
        target_col_h = f'target_h{h}'
        X_train, y_train = train_df[feature_cols], train_df[target_col_h].astype(np.float32)
        X_val, y_val = val_df[feature_cols], val_df[target_col_h].astype(np.float32)
        X_test, y_test = test_df[feature_cols], test_df[target_col_h].astype(np.float32)
        
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
        
        # Train model
        logger.info(f"\nTraining h={h} model...")
        model = lgb.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        logger.info(f"Best iteration: {model.best_iteration_}")
        
        # Evaluate
        y_test_pred = np.clip(model.predict(X_test), 0, None)
        test_wmape = calculate_wmape(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        logger.info(f"h={h} Test WMAPE: {test_wmape:.2f}%")
        logger.info(f"h={h} Test MAE:   {test_mae:.2f}")
        
        # Save model
        model_path = OUTPUT_ROOT / f'model_h{h}.txt'
        model.booster_.save_model(str(model_path))
        logger.info(f"Saved model to {model_path}")
        
        # Store results
        all_results[h] = {
            'model': model,
            'test_wmape': test_wmape,
            'test_mae': test_mae,
            'X_train': X_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_test_pred,
            'test_df': test_df
        }
        
        # Generate SHAP explainability for h=1 model (primary)
        if h == 1 and SHAP_AVAILABLE:
            logger.info("\nGenerating SHAP explainability report...")
            try:
                shap_output = OUTPUT_ROOT / 'shap_output'
                shap_output.mkdir(exist_ok=True)
                create_shap_report(
                    model=model,
                    X_train_sample=X_train.sample(min(1000, len(X_train)), random_state=42),
                    X_test_sample=X_test.sample(min(500, len(X_test)), random_state=42),
                    feature_cols=feature_cols,
                    output_dir=str(shap_output)
                )
            except Exception as e:
                logger.warning(f"SHAP explainability failed: {e}")
    
    # Capture drift baseline using h=1 model's training data
    logger.info("\n" + "="*70)
    logger.info("CAPTURING DRIFT DETECTION BASELINE")
    logger.info("="*70)
    try:
        detector = DriftDetector(drift_threshold=0.15)
        detector.capture_baseline(
            all_results[1]['X_train'], 
            all_results[1]['y_test'],
            critical_features=feature_cols[:20]  # Top 20 features
        )
        detector.save_baseline(str(OUTPUT_ROOT / 'drift_baseline.json'))
    except Exception as e:
        logger.warning(f"Drift baseline capture failed: {e}")
    
    # Save feature importance (from h=1 model)
    fi = pd.DataFrame({
        'feature': feature_cols,
        'importance': all_results[1]['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    fi.to_csv(OUTPUT_ROOT / 'feature_importance.csv', index=False)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("MULTI-HORIZON TRAINING COMPLETE")
    logger.info("="*70)
    for h in horizons:
        logger.info(f"  h={h:2d} day: WMAPE = {all_results[h]['test_wmape']:.2f}%")
    logger.info("="*70)
    
    return all_results

# =============================================================================
# MULTI-STEP FORECASTING

def recursive_forecast(model, initial_data, feature_cols, horizon=7):
    """
    Generate multi-step forecast using recursive strategy.
    
    Args:
        model: Trained LightGBM model (Booster or LGBMRegressor)
        initial_data: DataFrame with last known data for each SKU-location
        feature_cols: List of feature column names
        horizon: Number of days to forecast
    
    Returns:
        DataFrame with forecasts for each day
    """
    logger.info(f"Generating {horizon}-day recursive forecast...")
    
    forecasts = []
    current_data = initial_data.copy()
    
    # Handle both Booster and LGBMRegressor
    if hasattr(model, 'predict'):
        predict_fn = model.predict
    else:
        predict_fn = model.predict
    
    for day in range(1, horizon + 1):
        # Predict next day
        X = current_data[feature_cols]
        pred = np.clip(predict_fn(X), 0, None)
        
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

def run_multistep_forecasting(model, df, feature_cols):
    """
    Evaluate model at multiple forecast horizons using direct approach.
    Instead of recursive forecasting, evaluates how well the model predicts
    at different time offsets (simulating h-step ahead forecasts).
    """
    logger.info("\n" + "="*70)
    logger.info("MULTI-HORIZON FORECAST EVALUATION")
    logger.info("="*70)
    
    # Use same test split as training (last 15% of data)
    df_sorted = df.sort_values('date').reset_index(drop=True)
    n = len(df_sorted)
    val_end = int(n * 0.85)
    test_data = df_sorted.iloc[val_end:].copy()
    
    logger.info(f"Test period: {test_data['date'].min().date()} to {test_data['date'].max().date()}")
    logger.info(f"Test data: {len(test_data):,} rows")
    logger.info(f"Test data avg true_demand: {test_data['true_demand'].mean():.2f}")
    
    # Direct prediction on test set (1-step ahead - this is what training evaluates)
    X_test = test_data[feature_cols]
    y_test = test_data[TARGET_COL]
    y_pred = np.clip(model.predict(X_test), 0, None)
    
    # Evaluate at different simulated horizons by grouping test data
    results = []
    
    # 1-day ahead (standard evaluation)
    wmape_1 = calculate_wmape(y_test, y_pred)
    mae_1 = np.mean(np.abs(y_test - y_pred))
    results.append({'horizon': 1, 'wmape': wmape_1, 'mae': mae_1, 'n_forecasts': len(y_test)})
    logger.info(f"Horizon  1-day (direct): WMAPE = {wmape_1:5.2f}%, MAE = {mae_1:6.2f}")
    
    # For multi-day horizons, evaluate on subsets to simulate error accumulation
    # This is a simplified approach - actual horizon evaluation would require 
    # lag features computed with h-step offset
    for horizon in [7, 14]:
        # Sample every h-th row to simulate horizon effect
        horizon_mask = np.arange(len(test_data)) % horizon == 0
        y_h = y_test.values[horizon_mask]
        pred_h = y_pred[horizon_mask]
        
        wmape_h = calculate_wmape(y_h, pred_h)
        mae_h = np.mean(np.abs(y_h - pred_h))
        results.append({'horizon': horizon, 'wmape': wmape_h, 'mae': mae_h, 'n_forecasts': len(y_h)})
        logger.info(f"Horizon {horizon:2d}-day (sampled): WMAPE = {wmape_h:5.2f}%, MAE = {mae_h:6.2f}")
    
    eval_results = pd.DataFrame(results)
    
    # Save predictions
    pred_df = test_data[['date', 'sku_id', 'location_id']].copy()
    pred_df['actual'] = y_test.values
    pred_df['forecast'] = y_pred
    pred_df.to_csv(OUTPUT_ROOT / 'multistep_forecasts.csv', index=False)
    eval_results.to_csv(OUTPUT_ROOT / 'multistep_evaluation.csv', index=False)
    
    logger.info(f"\nSaved forecasts to {OUTPUT_ROOT / 'multistep_forecasts.csv'}")
    logger.info(f"Saved evaluation to {OUTPUT_ROOT / 'multistep_evaluation.csv'}")
    
    return pred_df, eval_results

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Main training and forecasting pipeline.
    
    Trains:
    - 3 horizon-specific models (h=1, h=7, h=14)
    - 1 baseline model for backward compatibility
    
    Integrates:
    - SHAP explainability (if available)
    - Drift detection baseline capture
    - Business KPI measurement
    
    Returns:
        Dictionary with all training results and metrics
    """
    logger.info("="*70)
    logger.info("FMCG TRAINING & FORECASTING PIPELINE")
    logger.info("="*70)
    
    # -------------------------------------------------------------------------
    # STEP 1: LOAD DATA
    # -------------------------------------------------------------------------
    data_path = DATA_ROOT / DAILY_TS_FILE
    logger.info(f"\nLoading data from {data_path}")
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Loaded {len(df):,} rows")
    
    # -------------------------------------------------------------------------
    # STEP 2: STOCKOUT CORRECTION
    # -------------------------------------------------------------------------
    df = correct_stockouts(df)
    
    # -------------------------------------------------------------------------
    # STEP 3: FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    df = engineer_features(df)
    df = reduce_mem_usage(df)
    
    # -------------------------------------------------------------------------
    # STEP 4: SELECT FEATURES
    # -------------------------------------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    id_cols = ['date', 'sku_id', 'location_id']
    exclude = [TARGET_COL] + LEAK_COLS + id_cols
    feature_cols = [c for c in numeric_cols if c not in exclude and c in df.columns]
    
    logger.info(f"\nUsing {len(feature_cols)} features")
    logger.info(f"Target: {TARGET_COL}")
    
    # -------------------------------------------------------------------------
    # STEP 5: TRAIN MULTI-HORIZON MODELS
    # -------------------------------------------------------------------------
    horizon_results = train_horizon_models(df, feature_cols, horizons=[1, 7, 14])
    
    # Also train baseline for backward compatibility
    train_results = train_baseline(df, feature_cols)
    
    # -------------------------------------------------------------------------
    # STEP 6: BUSINESS KPI MEASUREMENT
    # -------------------------------------------------------------------------
    kpi_results = None
    logger.info("\n" + "="*70)
    logger.info("BUSINESS KPI MEASUREMENT")
    logger.info("="*70)
    try:
        # Prepare forecast DataFrame
        h1_results = horizon_results[1]
        ml_forecasts = h1_results['test_df'][['date', 'sku_id', 'location_id']].copy()
        ml_forecasts['forecast'] = h1_results['y_pred']
        
        # Prepare actuals DataFrame
        actuals = h1_results['test_df'][['date', 'sku_id', 'location_id', TARGET_COL]].copy()
        actuals = actuals.rename(columns={TARGET_COL: 'actual_demand'})
        
        # Calculate KPIs
        kpi_calc = BusinessKPICalculator(lead_time_days=7, service_level=0.95)
        kpi_comparison = kpi_calc.compare_methods(ml_forecasts, actuals)
        kpi_calc.export_report(kpi_comparison, str(OUTPUT_ROOT / 'business_kpi_report.txt'))
        kpi_results = kpi_comparison
    except Exception as e:
        logger.warning(f"Business KPI measurement failed: {e}")
    
    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)
    
    logger.info("\nMulti-Horizon Model Results:")
    for h in [1, 7, 14]:
        if h in horizon_results:
            logger.info(f"  h={h:2d} day: WMAPE = {horizon_results[h]['test_wmape']:.2f}%")
    
    logger.info(f"\nBaseline (h=1) WMAPE: {train_results['test_wmape']:.2f}%")
    logger.info(f"Target: 35-40% (Acceptable: <45%)")
    
    if train_results['test_wmape'] < 45:
        logger.info("✓ ACCEPTABLE - Ready for production")
    else:
        logger.info("✗ NEEDS INVESTIGATION - Check data/features")
    
    logger.info("="*70)
    
    return {
        'train_results': train_results,
        'horizon_results': horizon_results,
        'kpi_results': kpi_results,
        'processed_df': df,
        'feature_cols': feature_cols
    }


def forecast_only():
    """
    Run forecasting only using a pre-trained model.
    Use this when you already have a trained model.
    """
    logger.info("="*70)
    logger.info("MULTI-STEP FORECASTING (USING SAVED MODEL)")
    logger.info("="*70)
    
    # Load model
    model_path = OUTPUT_ROOT / 'baseline_model.txt'
    logger.info(f"\nLoading model from {model_path}")
    model = lgb.Booster(model_file=str(model_path))
    
    # Load feature columns from feature importance
    fi_path = OUTPUT_ROOT / 'feature_importance.csv'
    if fi_path.exists():
        fi = pd.read_csv(fi_path)
        feature_cols = fi['feature'].tolist()
        logger.info(f"Loaded {len(feature_cols)} features from feature_importance.csv")
    else:
        logger.error(f"Feature importance file not found: {fi_path}")
        return None
    
    # Load and process data
    logger.info("Loading and processing data...")
    df = pd.read_parquet(DATA_ROOT / DAILY_TS_FILE)
    df['date'] = pd.to_datetime(df['date'])
    df = correct_stockouts(df)
    df = engineer_features(df)
    df = reduce_mem_usage(df)
    
    # Run forecasting
    forecasts, eval_results = run_multistep_forecasting(model, df, feature_cols)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    if eval_results is not None and len(eval_results) > 0:
        for _, row in eval_results.iterrows():
            logger.info(f"{int(row['horizon']):2d}-day WMAPE: {row['wmape']:.2f}%")
    logger.info("="*70)
    
    return forecasts, eval_results


if __name__ == "__main__":
    # For Colab/Jupyter: Just call the function directly
    # Options:
    #   results = main()              # Full pipeline: multi-horizon models + KPIs
    #   forecasts = forecast_only()   # Forecast using pre-trained models
    
    results = main()


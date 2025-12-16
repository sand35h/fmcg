# -*- coding: utf-8 -*-
"""
FMCG Demand Forecasting Trainer
================================
A production-ready XGBoost trainer for FMCG demand forecasting.

Key Improvements over notebook version:
- Modular, class-based architecture
- Configurable via dataclasses
- Automatic device detection (GPU/CPU fallback)
- Proper logging instead of print statements
- Cross-validation support
- Hyperparameter tuning with Optuna (optional)
- Early stopping with configurable patience
- Model versioning and experiment tracking
- Robust error handling
- Type hints throughout
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import gc
import psutil
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Optional imports for ensemble models
try:
    import lightgbm
except ImportError:
    lightgbm = None

try:
    import catboost
except ImportError:
    catboost = None

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None

# Optional imports for optimization
try:
    import scipy
except ImportError:
    scipy = None

try:
    import optuna
except ImportError:
    optuna = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 ** 3

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        # Skip objects, categories, and datetimes
        if str(col_type) == 'category' or str(col_type) == 'object' or 'datetime' in str(col_type):
            continue

        c_min = df[col].min()
        c_max = df[col].max()
            
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                # float16 has lower precision, using float32 is safer for training
                df[col] = df[col].astype(np.float32)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(f"Memory usage after optimization is: {end_mem:.2f} MB")
        logger.info(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")
    
    return df


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DataConfig:
    """Configuration for data paths and columns."""

    data_root: Path = Path("/kaggle/input/fmcg-dataset")
    output_root: Path = Path("./output")

    # File names
    sku_file: str = "sku_master (1).csv"
    location_file: str = "location_master (1).csv"
    festival_file: str = "festival_calendar (1).csv"
    macro_file: str = "monthly_macro (1).csv"
    daily_ts_file: str = "daily_timeseries (1).csv"
    shipment_file: str = "shipment_data.csv"
    competitor_file: str = "competitor_activity.csv"
    weather_file: str = "weather_data.csv"

    # Column definitions
    target_col: str = "actual_demand"
    date_col: str = "date"
    id_cols: List[str] = field(
        default_factory=lambda: [
            "date",
            "sku_id",
            "location_id",
            "sku_name",
            "city",
            "region",
        ]
    )
    leak_cols: List[str] = field(
        default_factory=lambda: [
            "expected_demand",
            "unfulfilled_demand",
            "incoming_stock",
            "closing_stock",
            "reorder_point",
            "safety_stock",
        ]
    )
    lag_cols: List[str] = field(
        default_factory=lambda: [
            "loc_lag_1",
            "loc_lag_7",
            "loc_lag_14",
            "loc_lag_30",
            "loc_rolling_7_mean",
            "loc_rolling_30_mean",
            "chan_lag_1",
            "chan_rolling_7_mean",
        ]
    )


@dataclass
class ModelConfig:
    """Configuration for XGBoost model parameters."""

    n_estimators: int = 500
    max_depth: int = 8
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: int = 3
    gamma: float = 0.0
    random_state: int = 42

    # Training settings
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"

    # Device settings (auto-detect if None)
    use_gpu: Optional[bool] = None


@dataclass
class TrainingConfig:
    """Configuration for training process."""

    test_days: int = 365  # Days to hold out for testing
    val_days: int = 90  # Days for validation (within train)
    cv_folds: int = 5  # For cross-validation
    use_cv: bool = False  # Whether to use cross-validation

    # Experiment tracking
    experiment_name: str = "fmcg_xgb"
    save_predictions: bool = True
    save_feature_importance: bool = True


# =============================================================================
# METRICS
# =============================================================================


class Metrics:
    """Calculate and store forecasting metrics."""

    @staticmethod
    def calculate(
        y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> Dict[str, float]:
        """
        Calculate comprehensive forecasting metrics.

        Args:
            y_true: Actual values
            y_pred: Predicted values
            prefix: Prefix for metric names (e.g., 'train_', 'test_')

        Returns:
            Dictionary of metric names to values
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE with protection against division by zero
        denom = np.maximum(np.abs(y_true), 1)
        mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100

        # Symmetric MAPE (more robust)
        smape = (
            100
            * 2
            * np.mean(
                np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
            )
        )

        # Weighted MAPE (weighted by actual values)
        wmape = np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(np.abs(y_true)), 1) * 100

        # Bias (positive = over-forecasting)
        bias = np.mean(y_pred - y_true)
        bias_pct = bias / np.maximum(np.mean(y_true), 1) * 100

        metrics = {
            f"{prefix}mae": mae,
            f"{prefix}rmse": rmse,
            f"{prefix}r2": r2,
            f"{prefix}mape": mape,
            f"{prefix}smape": smape,
            f"{prefix}wmape": wmape,
            f"{prefix}bias": bias,
            f"{prefix}bias_pct": bias_pct,
        }

        return metrics

    @staticmethod
    def log_metrics(metrics: Dict[str, float], name: str = "") -> None:
        """Log metrics in a formatted way."""
        logger.info(f"\n{'=' * 50}")
        logger.info(f"{name} METRICS")
        logger.info(f"{'=' * 50}")
        logger.info(f"MAE   : {metrics.get(f'{name.lower()}_mae', 0):,.3f}")
        logger.info(f"RMSE  : {metrics.get(f'{name.lower()}_rmse', 0):,.3f}")
        logger.info(f"R²    : {metrics.get(f'{name.lower()}_r2', 0):.4f}")
        logger.info(f"MAPE  : {metrics.get(f'{name.lower()}_mape', 0):.2f}%")
        logger.info(f"SMAPE : {metrics.get(f'{name.lower()}_smape', 0):.2f}%")
        logger.info(f"WMAPE : {metrics.get(f'{name.lower()}_wmape', 0):.2f}%")
        logger.info(f"Bias  : {metrics.get(f'{name.lower()}_bias_pct', 0):+.2f}%")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================


class FeatureEngineer:
    """Feature engineering pipeline for FMCG forecasting."""

    def __init__(self, data_config: DataConfig):
        self.config = data_config
        self._sku_df: Optional[pd.DataFrame] = None
        self._location_df: Optional[pd.DataFrame] = None
        self._festival_df: Optional[pd.DataFrame] = None
        self._macro_df: Optional[pd.DataFrame] = None

    def load_reference_data(self) -> None:
        """Load reference/lookup tables."""
        root = self.config.data_root

        try:
            self._sku_df = pd.read_csv(root / self.config.sku_file)
            logger.info(f"Loaded SKU master: {len(self._sku_df)} records")
        except FileNotFoundError:
            logger.warning("SKU master file not found, skipping SKU features")

        try:
            self._location_df = pd.read_csv(root / self.config.location_file)
            logger.info(f"Loaded Location master: {len(self._location_df)} records")
        except FileNotFoundError:
            logger.warning("Location master file not found, skipping location features")

        try:
            self._festival_df = pd.read_csv(
                root / self.config.festival_file, parse_dates=["date"]
            )
            logger.info(f"Loaded Festival calendar: {len(self._festival_df)} records")
        except FileNotFoundError:
            logger.warning("Festival calendar not found, skipping festival features")

        try:
            self._macro_df = pd.read_csv(root / self.config.macro_file)
            logger.info(f"Loaded Macro data: {len(self._macro_df)} records")
        except FileNotFoundError:
            logger.warning("Macro data not found, skipping macro features")

    def add_sku_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SKU master features."""
        if self._sku_df is None or "sku_id" not in df.columns:
            return df
        return df.merge(self._sku_df, on="sku_id", how="left", suffixes=("", "_sku"))

    def add_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location master features."""
        if self._location_df is None or "location_id" not in df.columns:
            return df
        return df.merge(
            self._location_df, on="location_id", how="left", suffixes=("", "_loc")
        )

    def add_festival_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add festival indicator flags using fast vectorized operations."""
        if self._festival_df is None or "date" not in df.columns:
            return df

        fest = self._festival_df[["date"]].copy()
        # Optimize memory usage for festival features
        fest["is_festival"] = np.int8(1)
        out = df.merge(fest, on="date", how="left")
        out["is_festival"] = out["is_festival"].fillna(0).astype(np.int8)

        # Clean up
        del fest
        gc.collect()

        # Fast vectorized approach for festival proximity
        # Create lookup sets for dates near festivals
        if len(self._festival_df) > 0:
            festival_dates = set(pd.to_datetime(self._festival_df["date"]))
            
            for days in [1, 3, 7]:
                # Create set of dates that are within 'days' before a festival
                near_festival_dates = set()
                for fd in festival_dates:
                    for d in range(1, days + 1):
                        near_festival_dates.add(fd - pd.Timedelta(days=d))
                
                # Vectorized lookup
                out[f"festival_in_{days}d"] = out["date"].isin(near_festival_dates).astype(np.int8)

        return out

    def add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add macroeconomic features."""
        if self._macro_df is None or "date" not in df.columns:
            return df

        tmp = df.copy()
        tmp["month_key"] = tmp["date"].dt.to_period("M").dt.to_timestamp()

        macro_tmp = self._macro_df.copy()
        macro_tmp["month_key"] = (
            pd.to_datetime(macro_tmp["month"]).dt.to_period("M").dt.to_timestamp()
        )
        macro_tmp = macro_tmp.drop_duplicates(subset=["month_key"])

        # Select available macro columns
        macro_cols = ["month_key"]
        for col in ["gdp_growth", "cpi_index", "consumer_confidence"]:
            if col in macro_tmp.columns:
                macro_cols.append(col)

        if len(macro_cols) > 1:
            out = tmp.merge(macro_tmp[macro_cols], on="month_key", how="left")
            return out.drop(columns=["month_key"])
        return df

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar/time-based features."""
        if "date" not in df.columns:
            return df

        tmp = df.copy()
        tmp["day_of_week"] = tmp["date"].dt.weekday
        tmp["week_of_year"] = tmp["date"].dt.isocalendar().week.astype(int)
        tmp["day_of_month"] = tmp["date"].dt.day
        tmp["month"] = tmp["date"].dt.month
        tmp["year"] = tmp["date"].dt.year
        tmp["quarter"] = tmp["date"].dt.quarter
        tmp["is_weekend"] = tmp["day_of_week"].isin([5, 6]).astype(int)
        tmp["is_month_start"] = tmp["date"].dt.is_month_start.astype(int)
        tmp["is_month_end"] = tmp["date"].dt.is_month_end.astype(int)
        tmp["days_in_month"] = tmp["date"].dt.days_in_month

        # Cyclical encoding for temporal features
        tmp["day_of_week_sin"] = np.sin(2 * np.pi * tmp["day_of_week"] / 7)
        tmp["day_of_week_cos"] = np.cos(2 * np.pi * tmp["day_of_week"] / 7)
        tmp["month_sin"] = np.sin(2 * np.pi * tmp["month"] / 12)
        tmp["month_cos"] = np.cos(2 * np.pi * tmp["month"] / 12)

        return tmp

    def add_price_promo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price and promotion features."""
        tmp = df.copy()

        if {"price", "base_price"}.issubset(tmp.columns):
            tmp["price_to_base"] = tmp["price"] / tmp["base_price"].replace(0, np.nan)
            tmp["price_to_base"] = tmp["price_to_base"].fillna(1.0)

        if {"mrp", "price"}.issubset(tmp.columns):
            tmp["discount_depth"] = (tmp["mrp"] - tmp["price"]) / tmp["mrp"].replace(0, np.nan)
            tmp["discount_depth"] = tmp["discount_depth"].fillna(0).clip(-1, 1)
            tmp["is_on_promo"] = (tmp["discount_depth"] > 0.05).astype(int)

        return tmp

    def add_lag_features(self, df: pd.DataFrame, create_new: bool = False) -> pd.DataFrame:
        """
        Add lag and rolling features with minimal memory usage.
        
        Args:
            df: Input dataframe
            create_new: If True, create new lag features; if False, just validate existing ones
        """
        if not create_new:
            return df

        target = self.config.target_col
        if target not in df.columns:
            return df
            
        logger.info(f"Generating lag features for target: {target}")

        # Ensure sorted for correct shifts
        df = df.sort_values(["sku_id", "location_id", "date"])
        
        # Calculate lags one by one to keep memory peak low
        for lag in [1, 7, 14, 30]:
            col_name = f"demand_lag_{lag}"
            logger.info(f"Creating {col_name}...")
            # Assign directly to avoid intermediate series copy if possible
            df[col_name] = df.groupby(["sku_id", "location_id"])[target].shift(lag)
            
            # Immediate downcast
            if df[col_name].dtype == 'float64':
                df[col_name] = df[col_name].astype(np.float32)
                
            gc.collect()

        # Calculate rolling features one by one
        for window in [7, 14, 30]:
            # Rolling Mean
            col_mean = f"demand_rolling_{window}_mean"
            logger.info(f"Creating {col_mean}...")
            
            # Using transform to keep dimensions aligned
            df[col_mean] = (
                df.groupby(["sku_id", "location_id"])[target]
                .shift(1)
                .rolling(window, min_periods=1)
                .mean()
                .astype(np.float32) # Cast immediately
            )
            gc.collect()
            
            # Rolling Std
            col_std = f"demand_rolling_{window}_std"
            logger.info(f"Creating {col_std}...")
            
            df[col_std] = (
                df.groupby(["sku_id", "location_id"])[target]
                .shift(1)
                .rolling(window, min_periods=1)
                .std()
                .astype(np.float32) # Cast immediately
            )
            gc.collect()

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full feature engineering pipeline with aggressive memory optimization."""
        logger.info("Starting feature engineering...")

        # 1. Base SKU Features
        logger.info("Adding SKU features...")
        df = self.add_sku_features(df)
        df = reduce_mem_usage(df)
        gc.collect()

        # 2. Location Features
        logger.info("Adding Location features...")
        df = self.add_location_features(df)
        df = reduce_mem_usage(df)
        gc.collect()

        # 3. Festival Features
        logger.info("Adding Festival features...")
        df = self.add_festival_flags(df)
        df = reduce_mem_usage(df)
        gc.collect()

        # 4. Macro Features
        logger.info("Adding Macro features...")
        df = self.add_macro_features(df)
        df = reduce_mem_usage(df)
        gc.collect()

        # 5. Calendar Features
        logger.info("Adding Calendar features...")
        df = self.add_calendar_features(df)
        df = reduce_mem_usage(df)
        gc.collect()

        # 6. Price/Promo Features
        logger.info("Adding Price/Promo features...")
        df = self.add_price_promo_features(df)
        df = reduce_mem_usage(df)
        gc.collect()

        # 7. Lag Features (Most memory intensive)
        logger.info("Adding Lag features...")
        df = self.add_lag_features(df, create_new=True)
        # Note: add_lag_features already does internal GC/optimization now
        df = reduce_mem_usage(df)
        gc.collect()

        # Clean up duplicate columns from merges
        result = self._clean_merge_columns(df)
        
        # Explicitly delete intermediate df reference
        del df
        gc.collect()

        # Drop rows with missing lag features
        lag_cols = [c for c in self.config.lag_cols if c in result.columns]
        if lag_cols:
            before = len(result)
            result = result.dropna(subset=lag_cols)
            logger.info(f"Dropped {before - len(result)} rows with missing lag features")

        logger.info(f"Feature engineering complete. Shape: {result.shape}")
        
        # Final memory cleanup
        result = reduce_mem_usage(result)
        gc.collect()
        
        return result

    def _clean_merge_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up duplicate columns from merges."""
        # Drop _y columns
        y_cols = [c for c in df.columns if c.endswith("_y")]
        if y_cols:
            df = df.drop(columns=y_cols)

        # Rename _x columns back to base names
        x_cols = {c: c.replace("_x", "") for c in df.columns if c.endswith("_x")}
        if x_cols:
            df = df.rename(columns=x_cols)

        return df


# =============================================================================
# TRAINER
# =============================================================================


class FMCGTrainer:
    """
    XGBoost trainer for FMCG demand forecasting.

    Features:
    - Automatic GPU/CPU detection
    - Early stopping with validation set
    - Cross-validation support
    - Comprehensive metrics and logging
    - Model persistence and experiment tracking
    """

    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ):
        self.data_config = data_config or DataConfig()
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()

        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_cols: List[str] = []
        self.feature_importance: Optional[pd.DataFrame] = None
        self.metrics_history: Dict[str, Dict[str, float]] = {}

        # Create output directory
        self.data_config.output_root.mkdir(parents=True, exist_ok=True)

        # Feature engineer
        self.feature_engineer = FeatureEngineer(self.data_config)

    def _detect_device(self) -> Tuple[str, str]:
        """Detect best available device (GPU or CPU)."""
        if self.model_config.use_gpu is False:
            logger.info("GPU disabled by config, using CPU")
            return "hist", "cpu"

        try:
            # Try to detect CUDA availability
            import subprocess

            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                logger.info("GPU detected, using CUDA acceleration")
                return "hist", "cuda"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        logger.info("No GPU detected, using CPU")
        return "hist", "cpu"

    def _get_model_params(self) -> Dict[str, Any]:
        """Build XGBoost parameters dict."""
        tree_method, device = self._detect_device()

        params = {
            "n_estimators": self.model_config.n_estimators,
            "max_depth": self.model_config.max_depth,
            "learning_rate": self.model_config.learning_rate,
            "subsample": self.model_config.subsample,
            "colsample_bytree": self.model_config.colsample_bytree,
            "reg_lambda": self.model_config.reg_lambda,
            "reg_alpha": self.model_config.reg_alpha,
            "min_child_weight": self.model_config.min_child_weight,
            "gamma": self.model_config.gamma,
            "objective": "reg:squarederror",
            "random_state": self.model_config.random_state,
            "n_jobs": -1,
            "tree_method": tree_method,
            "device": device,
        }

        return params

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """Prepare feature columns for modeling."""
        # Get numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target, IDs, and leaky columns
        exclude = (
            [self.data_config.target_col]
            + self.data_config.leak_cols
            + [c for c in self.data_config.id_cols if c in num_cols]
        )

        feature_cols = [c for c in num_cols if c not in exclude]
        logger.info(f"Selected {len(feature_cols)} features for modeling")

        return feature_cols, df

    def _time_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by time into train, validation, and test sets.
        Optimized to reduce memory spikes by avoiding holding 3 copies simultaneously.
        """
        date_col = self.data_config.date_col
        max_date = df[date_col].max()

        test_start = max_date - pd.Timedelta(days=self.training_config.test_days)
        val_start = test_start - pd.Timedelta(days=self.training_config.val_days)

        logger.info(f"Split dates: Val start={val_start}, Test start={test_start}")

        # Create masks
        train_mask = df[date_col] < val_start
        val_mask = (df[date_col] >= val_start) & (df[date_col] < test_start)
        test_mask = df[date_col] >= test_start
        
        # Create splits sequentially and force GC
        logger.info("Creating Test set...")
        test_df = df.loc[test_mask].copy()
        test_df = reduce_mem_usage(test_df)
        
        logger.info("Creating Validation set...")
        val_df = df.loc[val_mask].copy()
        val_df = reduce_mem_usage(val_df)
        
        logger.info("Creating Train set...")
        # For training set, we can assign without copy if possible, or just copy and delete df immediately
        train_df = df.loc[train_mask].copy()
        train_df = reduce_mem_usage(train_df)
        
        logger.info(f"Train: {len(train_df):,} rows")
        logger.info(f"Val:   {len(val_df):,} rows")
        logger.info(f"Test:  {len(test_df):,} rows")
        
        # Crucial: Delete the original dataframe to free memory
        del df
        gc.collect()

        return train_df, val_df, test_df

    def load_and_prepare_data(
        self, df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Load and prepare data for training."""
        if df is None:
            # Load from file
            data_path = self.data_config.data_root / self.data_config.daily_ts_file
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path, parse_dates=[self.data_config.date_col])
            
            # Initial memory reduction
            df = reduce_mem_usage(df)

        logger.info(f"Loaded data: {len(df):,} rows")

        # Load reference data for feature engineering
        self.feature_engineer.load_reference_data()

        # Apply feature engineering
        df = self.feature_engineer.transform(df)

        # Remove rows with missing target
        target = self.data_config.target_col
        df = df[~df[target].isna()].reset_index(drop=True)

        return df

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Train the XGBoost model.

        Args:
            df: Prepared dataframe with features
            feature_cols: Optional list of feature columns to use

        Returns:
            Dictionary of training and test metrics
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)

        # Prepare features
        if feature_cols is None:
            self.feature_cols, df = self._prepare_features(df)
        else:
            self.feature_cols = feature_cols

        # Time-based split
        train_df, val_df, test_df = self._time_split(df)

        target = self.data_config.target_col
        X_train = train_df[self.feature_cols]
        y_train = train_df[target]
        X_val = val_df[self.feature_cols]
        y_val = val_df[target]
        X_test = test_df[self.feature_cols]
        y_test = test_df[target]

        # Clear memory of original parts
        del df
        gc.collect()

        # Initialize model
        params = self._get_model_params()
        self.model = xgb.XGBRegressor(**params)

        # Train with early stopping on validation set
        logger.info("Training XGBoost model...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,  # Log every 100 rounds
        )

        # Get best iteration
        best_iter = self.model.best_iteration
        if best_iter:
            logger.info(f"Best iteration: {best_iter}")

        # Generate predictions (clip to non-negative)
        y_train_pred = np.maximum(self.model.predict(X_train), 0)
        y_val_pred = np.maximum(self.model.predict(X_val), 0)
        y_test_pred = np.maximum(self.model.predict(X_test), 0)

        # Calculate metrics
        train_metrics = Metrics.calculate(y_train, y_train_pred, "train_")
        val_metrics = Metrics.calculate(y_val, y_val_pred, "val_")
        test_metrics = Metrics.calculate(y_test, y_test_pred, "test_")

        Metrics.log_metrics(train_metrics, "TRAIN")
        Metrics.log_metrics(val_metrics, "VAL")
        Metrics.log_metrics(test_metrics, "TEST")

        # Store all metrics
        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        self.metrics_history["latest"] = all_metrics

        # Store feature importance
        self.feature_importance = pd.DataFrame(
            {"feature": self.feature_cols, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Save artifacts
        self._save_artifacts(test_df, y_test, y_test_pred, all_metrics)

        return all_metrics

    def train_cv(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        n_splits: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Train with time-series cross-validation.

        Args:
            df: Prepared dataframe with features
            feature_cols: Optional list of feature columns
            n_splits: Number of CV folds (default from config)

        Returns:
            Average metrics across folds
        """
        logger.info("=" * 60)
        logger.info("STARTING CROSS-VALIDATION TRAINING")
        logger.info("=" * 60)

        if feature_cols is None:
            self.feature_cols, df = self._prepare_features(df)
        else:
            self.feature_cols = feature_cols

        n_splits = n_splits or self.training_config.cv_folds
        target = self.data_config.target_col

        # Sort by date for proper time series CV
        df = df.sort_values(self.data_config.date_col).reset_index(drop=True)

        X = df[self.feature_cols]
        y = df[target]
        
        del df
        gc.collect()

        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_metrics = []
        params = self._get_model_params()

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            logger.info(f"\n--- Fold {fold}/{n_splits} ---")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)

            y_pred = np.maximum(model.predict(X_test), 0)
            metrics = Metrics.calculate(y_test, y_pred, f"fold{fold}_")
            fold_metrics.append(metrics)

            logger.info(
                f"Fold {fold}: MAE={metrics[f'fold{fold}_mae']:.2f}, "
                f"RMSE={metrics[f'fold{fold}_rmse']:.2f}, "
                f"MAPE={metrics[f'fold{fold}_mape']:.2f}%"
            )

        # Average metrics across folds
        avg_metrics = {}
        for key in ["mae", "rmse", "r2", "mape", "smape", "wmape"]:
            values = [m[f"fold{i+1}_{key}"] for i, m in enumerate(fold_metrics)]
            avg_metrics[f"cv_avg_{key}"] = np.mean(values)
            avg_metrics[f"cv_std_{key}"] = np.std(values)

        logger.info("\n=== CROSS-VALIDATION SUMMARY ===")
        logger.info(f"MAE:  {avg_metrics['cv_avg_mae']:.2f} ± {avg_metrics['cv_std_mae']:.2f}")
        logger.info(f"RMSE: {avg_metrics['cv_avg_rmse']:.2f} ± {avg_metrics['cv_std_rmse']:.2f}")
        logger.info(f"MAPE: {avg_metrics['cv_avg_mape']:.2f}% ± {avg_metrics['cv_std_mape']:.2f}%")

        # Train final model on all data (except holdout)
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X, y, verbose=False)

        self.feature_importance = pd.DataFrame(
            {"feature": self.feature_cols, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        return avg_metrics

    def predict(
        self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Generate predictions for new data.

        Args:
            df: Dataframe with features
            feature_cols: Feature columns (uses training features if None)

        Returns:
            Array of predictions (clipped to non-negative)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        features = feature_cols or self.feature_cols
        X = df[features]

        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)

    def _save_artifacts(
        self,
        test_df: pd.DataFrame,
        y_test: pd.Series,
        y_test_pred: np.ndarray,
        metrics: Dict[str, float],
    ) -> None:
        """Save model, metrics, and predictions."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.training_config.experiment_name
        out_dir = self.data_config.output_root

        # Save model
        model_path = out_dir / f"{exp_name}_model_{timestamp}.json"
        self.model.save_model(model_path)
        logger.info(f"Saved model to {model_path}")

        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_path = out_dir / f"{exp_name}_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics to {metrics_path}")

        # Save feature importance
        if self.training_config.save_feature_importance and self.feature_importance is not None:
            fi_path = out_dir / f"{exp_name}_feature_importance_{timestamp}.csv"
            self.feature_importance.to_csv(fi_path, index=False)
            logger.info(f"Saved feature importance to {fi_path}")

        # Save predictions
        if self.training_config.save_predictions:
            id_cols = [c for c in self.data_config.id_cols if c in test_df.columns]
            pred_df = test_df[id_cols].copy()
            pred_df["actual"] = y_test.values
            pred_df["predicted"] = y_test_pred

            pred_path = out_dir / f"{exp_name}_predictions_{timestamp}.csv"
            pred_df.to_csv(pred_path, index=False)
            logger.info(f"Saved predictions to {pred_path}")

    def save_model(self, path: Union[str, Path]) -> None:
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        self.model.save_model(str(path))
        logger.info(f"Model saved to {path}")

    def load_model(self, path: Union[str, Path]) -> None:
        """Load model from file."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(path))
        logger.info(f"Model loaded from {path}")


# =============================================================================
# VISUALIZATION
# =============================================================================


class ForecastVisualizer:
    """
    Visualization utilities for FMCG demand forecasting.
    
    Provides methods to visualize:
    - Feature importance
    - Actual vs Predicted comparisons
    - Residual analysis
    - Time series decomposition
    - Error distribution
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./output/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use("seaborn-v0_8")
            sns.set_palette("husl")
        except ImportError:
            logger.warning("matplotlib/seaborn not installed. Plots will not be available.")
        except Exception:
            pass

    def plot_feature_importance(
        self,
        feature_importance: pd.DataFrame,
        top_n: int = 20,
        save: bool = True,
        show: bool = True,
    ) -> None:
        """
        Plot horizontal bar chart of feature importances.
        
        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to show
            save: Whether to save the plot
            show: Whether to display the plot
        """
        if plt is None:
            logger.warning("matplotlib not installed. Cannot generate plots.")
            return
        import matplotlib.pyplot as plt
        
        top_features = feature_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
        
        bars = ax.barh(
            range(len(top_features)),
            top_features["importance"],
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, top_features["importance"]):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
            )
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
            logger.info(f"Saved feature importance plot to {self.output_dir}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_actual_vs_predicted(
        self,
        y_actual: np.ndarray,
        y_predicted: np.ndarray,
        dates: Optional[pd.Series] = None,
        aggregate: bool = True,
        save: bool = True,
        show: bool = True,
    ) -> None:
        """
        Plot actual vs predicted demand over time.
        
        Args:
            y_actual: Actual demand values
            y_predicted: Predicted demand values
            dates: Date index for time series
            aggregate: Whether to aggregate by date (for multiple SKUs)
            save: Whether to save the plot
            show: Whether to display the plot
        """
        if plt is None:
            logger.warning("matplotlib not installed. Cannot generate plots.")
            return
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
        
        if dates is not None and aggregate:
            # Aggregate by date
            df = pd.DataFrame({
                "date": dates,
                "actual": y_actual,
                "predicted": y_predicted,
            })
            daily = df.groupby("date").sum().reset_index()
            x = daily["date"]
            actual = daily["actual"]
            predicted = daily["predicted"]
        else:
            x = range(len(y_actual))
            actual = y_actual
            predicted = y_predicted
        
        # Time series plot
        axes[0].plot(x, actual, label="Actual", alpha=0.8, linewidth=1.5, color="#2E86AB")
        axes[0].plot(x, predicted, label="Predicted", alpha=0.8, linewidth=1.5, color="#E94F37")
        axes[0].fill_between(x, actual, predicted, alpha=0.2, color="gray")
        axes[0].set_title("Actual vs Predicted Demand", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Demand")
        axes[0].legend(loc="upper right")
        axes[0].grid(alpha=0.3)
        
        # Scatter plot with regression line
        axes[1].scatter(actual, predicted, alpha=0.3, s=10, c="#2E86AB")
        
        # Perfect prediction line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Fit")
        
        axes[1].set_xlabel("Actual Demand")
        axes[1].set_ylabel("Predicted Demand")
        axes[1].set_title("Prediction Accuracy", fontsize=12)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "actual_vs_predicted.png", dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_residual_analysis(
        self,
        y_actual: np.ndarray,
        y_predicted: np.ndarray,
        save: bool = True,
        show: bool = True,
    ) -> None:
        """
        Plot residual analysis - distribution and patterns.
        
        Args:
            y_actual: Actual values
            y_predicted: Predicted values
            save: Whether to save
            show: Whether to display
        """
        if plt is None or sns is None or scipy is None:
            logger.warning("matplotlib/seaborn/scipy not installed. Cannot generate plots.")
            return
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        residuals = y_actual - y_predicted
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residual distribution
        sns.histplot(residuals, kde=True, ax=axes[0, 0], color="#2E86AB", bins=50)
        axes[0, 0].axvline(0, color="red", linestyle="--", linewidth=2)
        axes[0, 0].set_title("Residual Distribution", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Residual (Actual - Predicted)")
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Q-Q Plot", fontsize=12, fontweight="bold")
        
        # Residuals vs Predicted
        axes[1, 0].scatter(y_predicted, residuals, alpha=0.3, s=10, c="#E94F37")
        axes[1, 0].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[1, 0].set_xlabel("Predicted Value")
        axes[1, 0].set_ylabel("Residual")
        axes[1, 0].set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")
        
        # Residuals over time (index)
        axes[1, 1].plot(residuals, alpha=0.5, linewidth=0.5, color="#2E86AB")
        axes[1, 1].axhline(0, color="red", linestyle="--", linewidth=1)
        axes[1, 1].set_xlabel("Observation Index")
        axes[1, 1].set_ylabel("Residual")
        axes[1, 1].set_title("Residuals Over Time", fontsize=12, fontweight="bold")
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "residual_analysis.png", dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_error_by_segment(
        self,
        df: pd.DataFrame,
        y_actual: np.ndarray,
        y_predicted: np.ndarray,
        segment_col: str = "sku_id",
        metric: str = "mape",
        top_n: int = 20,
        save: bool = True,
        show: bool = True,
    ) -> None:
        """
        Plot error metrics by segment (SKU, location, etc).
        
        Args:
            df: DataFrame with segment column
            y_actual: Actual values
            y_predicted: Predicted values
            segment_col: Column to segment by
            metric: Metric to calculate ('mape', 'mae', 'rmse')
            top_n: Number of segments to show
            save: Whether to save
            show: Whether to display
        """
        if plt is None:
            logger.warning("matplotlib not installed. Cannot generate plots.")
            return
        import matplotlib.pyplot as plt
        
        if segment_col not in df.columns:
            logger.warning(f"Segment column '{segment_col}' not found in dataframe")
            return
        
        temp_df = df[[segment_col]].copy()
        temp_df["actual"] = y_actual
        temp_df["predicted"] = y_predicted
        temp_df["error"] = np.abs(y_actual - y_predicted)
        
        if metric == "mape":
            temp_df["metric"] = temp_df["error"] / np.maximum(temp_df["actual"], 1) * 100
        elif metric == "mae":
            temp_df["metric"] = temp_df["error"]
        else:  # rmse
            temp_df["metric"] = temp_df["error"] ** 2
        
        segment_errors = temp_df.groupby(segment_col)["metric"].mean()
        
        if metric == "rmse":
            segment_errors = np.sqrt(segment_errors)
        
        segment_errors = segment_errors.sort_values(ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(segment_errors)))
        
        ax.barh(range(len(segment_errors)), segment_errors.values, color=colors)
        ax.set_yticks(range(len(segment_errors)))
        ax.set_yticklabels(segment_errors.index)
        ax.invert_yaxis()
        ax.set_xlabel(f"{metric.upper()}")
        ax.set_title(f"Top {top_n} {segment_col} by {metric.upper()}", fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f"error_by_{segment_col}.png", dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_forecast_horizon(
        self,
        df: pd.DataFrame,
        y_actual: np.ndarray,
        y_predicted: np.ndarray,
        date_col: str = "date",
        save: bool = True,
        show: bool = True,
    ) -> None:
        """
        Plot error metrics by forecast horizon (days from start of test).
        
        Args:
            df: DataFrame with date column
            y_actual: Actual values
            y_predicted: Predicted values
            date_col: Date column name
            save: Whether to save
            show: Whether to display
        """
        if plt is None:
            logger.warning("matplotlib not installed. Cannot generate plots.")
            return
        import matplotlib.pyplot as plt
        
        if date_col not in df.columns:
            logger.warning(f"Date column '{date_col}' not found")
            return
        
        temp_df = df[[date_col]].copy()
        temp_df["actual"] = y_actual
        temp_df["predicted"] = y_predicted
        temp_df["ae"] = np.abs(y_actual - y_predicted)
        temp_df["ape"] = temp_df["ae"] / np.maximum(temp_df["actual"], 1) * 100
        
        # Calculate days from start
        min_date = temp_df[date_col].min()
        temp_df["horizon"] = (temp_df[date_col] - min_date).dt.days
        
        # Aggregate by horizon
        horizon_metrics = temp_df.groupby("horizon").agg({
            "ae": "mean",
            "ape": "mean",
        }).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # MAE by horizon
        axes[0].plot(horizon_metrics["horizon"], horizon_metrics["ae"], 
                     color="#2E86AB", linewidth=1.5)
        axes[0].fill_between(horizon_metrics["horizon"], 0, horizon_metrics["ae"], alpha=0.3)
        axes[0].set_xlabel("Days from Test Start")
        axes[0].set_ylabel("Mean Absolute Error")
        axes[0].set_title("MAE by Forecast Horizon", fontsize=12, fontweight="bold")
        axes[0].grid(alpha=0.3)
        
        # MAPE by horizon
        axes[1].plot(horizon_metrics["horizon"], horizon_metrics["ape"],
                     color="#E94F37", linewidth=1.5)
        axes[1].fill_between(horizon_metrics["horizon"], 0, horizon_metrics["ape"], alpha=0.3)
        axes[1].set_xlabel("Days from Test Start")
        axes[1].set_ylabel("Mean Absolute Percentage Error (%)")
        axes[1].set_title("MAPE by Forecast Horizon", fontsize=12, fontweight="bold")
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "error_by_horizon.png", dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_ensemble_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save: bool = True,
        show: bool = True,
    ) -> None:
        """
        Compare performance across ensemble models.
        
        Args:
            metrics_dict: Dict mapping model names to their metrics
            save: Whether to save
            show: Whether to display
        """
        if plt is None:
            logger.warning("matplotlib not installed. Cannot generate plots.")
            return
        import matplotlib.pyplot as plt
        
        models = list(metrics_dict.keys())
        metrics_to_plot = ["mae", "rmse", "mape"]
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        colors = ["#2E86AB", "#E94F37", "#A23B72", "#F18F01", "#C73E1D"]
        
        for idx, metric in enumerate(metrics_to_plot):
            values = []
            for model in models:
                # Try different key formats
                for key in [f"test_{metric}", metric, f"cv_avg_{metric}"]:
                    if key in metrics_dict[model]:
                        values.append(metrics_dict[model][key])
                        break
                else:
                    values.append(0)
            
            bars = axes[idx].bar(models, values, color=colors[:len(models)], edgecolor="white")
            axes[idx].set_ylabel(metric.upper())
            axes[idx].set_title(f"{metric.upper()} Comparison", fontsize=12, fontweight="bold")
            axes[idx].tick_params(axis="x", rotation=45)
            axes[idx].grid(axis="y", alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, values):
                axes[idx].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "ensemble_comparison.png", dpi=150, bbox_inches="tight")
        
        if show:
            plt.show()
        else:
            plt.close()


# =============================================================================
# ENSEMBLE TRAINER
# =============================================================================


class EnsembleTrainer:
    """
    Ensemble trainer combining XGBoost, LightGBM, and CatBoost.
    
    Supports:
    - Individual model training
    - Weighted averaging
    - Stacking (optional)
    - Model comparison
    """

    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        use_xgb: bool = True,
        use_lgb: bool = True,
        use_catboost: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.data_config = data_config or DataConfig()
        self.use_xgb = use_xgb
        self.use_lgb = use_lgb
        self.use_catboost = use_catboost
        
        # Model weights for ensemble (default: equal)
        self.weights = weights or {"xgb": 1.0, "lgb": 1.0, "catboost": 1.0}
        
        self.models: Dict[str, Any] = {}
        self.feature_cols: List[str] = []
        self.metrics_by_model: Dict[str, Dict[str, float]] = {}
        self.visualizer = ForecastVisualizer(self.data_config.output_root / "plots")
        
        # Check library availability
        self._check_libraries()

    def _check_libraries(self) -> None:
        """Check which libraries are available."""
        if self.use_lgb:
            if lightgbm is None:
                logger.warning("LightGBM not installed. Run: pip install lightgbm")
                self.use_lgb = False
            else:
                logger.info("LightGBM available")
        
        if self.use_catboost:
            if catboost is None:
                logger.warning("CatBoost not installed. Run: pip install catboost")
                self.use_catboost = False
            else:
                logger.info("CatBoost available")

    def _get_xgb_model(self, use_gpu: bool = False) -> Any:
        """Get XGBoost model with optimal parameters."""
        tree_method = "hist"
        device = "cuda" if use_gpu else "cpu"
        
        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05, 
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1,
            tree_method=tree_method,
            device=device,
            early_stopping_rounds=50,
        )

    def _get_lgb_model(self, use_gpu: bool = False) -> Any:
        """Get LightGBM model with optimal parameters."""
        if lightgbm is None:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        import lightgbm as lgb
        
        return lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_samples=20,
            random_state=42,
            n_jobs=-1,
            device="gpu" if use_gpu else "cpu",
            verbose=-1,
            force_row_wise=True,
        )

    def _get_catboost_model(self, use_gpu: bool = False) -> Any:
        """Get CatBoost model with optimal parameters."""
        if catboost is None:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        from catboost import CatBoostRegressor
        
        return CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_state=42,
            task_type="GPU" if use_gpu else "CPU",
            verbose=False,
            allow_writing_files=False,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        use_gpu: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all ensemble models with RMSLE strategy.
        """
        logger.info("=" * 60)
        logger.info("TRAINING ENSEMBLE MODELS (RMSLE Strategy)")
        logger.info("=" * 60)
        
        self.feature_cols = list(X_train.columns)
        
        # Log-transform targets for RMSLE (Optimize memory with float32)
        y_train_log = np.log1p(y_train).astype(np.float32)
        # Delete original y_train immediately to free memory (it's huge)
        del y_train 
        
        y_val_log = np.log1p(y_val).astype(np.float32)
        del y_val
        gc.collect()
        
        # Train XGBoost
        if self.use_xgb:
            logger.info("\n--- Training XGBoost ---")
            self.models["xgb"] = self._get_xgb_model(use_gpu)
            self.models["xgb"].fit(
                X_train, y_train_log,
                eval_set=[(X_val, y_val_log)],
                verbose=False,
            )
            # Predict and inverse transform
            y_pred_log = self.models["xgb"].predict(X_test)
            y_pred = np.maximum(np.expm1(y_pred_log), 0)
            self.metrics_by_model["xgb"] = Metrics.calculate(y_test, y_pred, "test_")
            Metrics.log_metrics(self.metrics_by_model["xgb"], "TEST")
            
            # OPTIONAL: Save model to disk and clear from memory if tight? 
            # For now, let's keep it but force GC
            gc.collect()
        
        # Train LightGBM
        if self.use_lgb:
            logger.info("\n--- Training LightGBM ---")
            self.models["lgb"] = self._get_lgb_model(use_gpu)
            callbacks = None
            if lightgbm is not None:
                callbacks = [lightgbm.early_stopping(50, verbose=False)]
            self.models["lgb"].fit(
                X_train, y_train_log,
                eval_set=[(X_val, y_val_log)],
                callbacks=callbacks
            )
            y_pred_log = self.models["lgb"].predict(X_test)
            y_pred = np.maximum(np.expm1(y_pred_log), 0)
            self.metrics_by_model["lgb"] = Metrics.calculate(y_test, y_pred, "test_")
            Metrics.log_metrics(self.metrics_by_model["lgb"], "TEST")
        
        # Train CatBoost
        if self.use_catboost:
            logger.info("\n--- Training CatBoost ---")
            self.models["catboost"] = self._get_catboost_model(use_gpu)
            self.models["catboost"].fit(
                X_train, y_train_log,
                eval_set=(X_val, y_val_log),
                early_stopping_rounds=50,
                verbose=False
            )
            y_pred_log = self.models["catboost"].predict(X_test)
            y_pred = np.maximum(np.expm1(y_pred_log), 0)
            self.metrics_by_model["catboost"] = Metrics.calculate(y_test, y_pred, "test_")
            Metrics.log_metrics(self.metrics_by_model["catboost"], "TEST")
        
        # Ensemble prediction
        logger.info("\n--- Ensemble Prediction ---")
        y_ensemble_log = self.predict_ensemble(X_test)
        y_ensemble = np.maximum(np.expm1(y_ensemble_log), 0)
        self.metrics_by_model["ensemble"] = Metrics.calculate(y_test, y_ensemble, "test_")
        Metrics.log_metrics(self.metrics_by_model["ensemble"], "TEST")
        
        return self.metrics_by_model

    def predict_model(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """Predict using a specific model (returns LOG scale)."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained")
        
        return self.models[model_name].predict(X)

    # Alias for sklearn compatibility / pipeline integration
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the ensemble (returns LOG scale)."""
        return self.predict_ensemble(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return avg feature importance from XGB/LGB models (CatBoost often differs)."""
        importances = []
        if self.use_xgb and "xgb" in self.models:
            importances.append(self.models["xgb"].feature_importances_)
        if self.use_lgb and "lgb" in self.models:
            importances.append(self.models["lgb"].feature_importances_)
            
        if not importances:
            return np.zeros(len(self.feature_cols))
            
        # Average available importances
        return np.mean(importances, axis=0)

    def predict_ensemble(
        self,
        X: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Generate weighted ensemble predictions (LOG scale).
        """
        weights = weights or self.weights
        
        predictions = []
        total_weight = 0
        
        for name, model in self.models.items():
            if name in weights:
                # Predict (returns log scale)
                pred_log = model.predict(X)
                predictions.append(pred_log * weights[name])
                total_weight += weights[name]
        
        if total_weight == 0:
            raise ValueError("No models available for prediction")
        
        # Average on log scale
        ensemble_log = np.sum(predictions, axis=0) / total_weight
        
        # Return LOG scale (pipeline handles inverse transform)
        return ensemble_log

    def optimize_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = "rmse",
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights using validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation target
            metric: Metric to optimize ('rmse', 'mae', 'mape')
        
        Returns:
            Optimized weights
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.error("scipy not installed. Run: pip install scipy")
            raise
        
        model_names = list(self.models.keys())
        n_models = len(model_names)
        
        if n_models == 0:
            raise ValueError("No models trained")
        
        # Get predictions from all models
        all_preds = {name: self.models[name].predict(X_val) for name in model_names}
        
        def objective(weights):
            weights = np.abs(weights)  # Ensure positive
            weights = weights / weights.sum()  # Normalize
            
            ensemble_pred = np.zeros(len(y_val))
            for i, name in enumerate(model_names):
                ensemble_pred += all_preds[name] * weights[i]
            
            ensemble_pred = np.maximum(ensemble_pred, 0)
            
            if metric == "rmse":
                return np.sqrt(mean_squared_error(y_val, ensemble_pred))
            elif metric == "mae":
                return mean_absolute_error(y_val, ensemble_pred)
            else:  # mape
                denom = np.maximum(np.abs(y_val), 1)
                return np.mean(np.abs((y_val - ensemble_pred) / denom)) * 100
        
        # Initialize equal weights
        x0 = np.ones(n_models) / n_models
        
        result = minimize(objective, x0, method="Nelder-Mead")
        
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        self.weights = {name: w for name, w in zip(model_names, optimal_weights)}
        
        logger.info(f"Optimized weights: {self.weights}")
        logger.info(f"Optimized {metric}: {result.fun:.4f}")
        
        return self.weights

    def get_feature_importance(self, aggregation: str = "mean") -> pd.DataFrame:
        """
        Get aggregated feature importance across all models.
        
        Args:
            aggregation: How to aggregate ('mean', 'max', 'weighted')
        
        Returns:
            DataFrame with feature and aggregated importance
        """
        importance_dfs = []
        
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                fi = pd.DataFrame({
                    "feature": self.feature_cols,
                    name: model.feature_importances_,
                })
                importance_dfs.append(fi.set_index("feature"))
        
        if not importance_dfs:
            return pd.DataFrame()
        
        combined = pd.concat(importance_dfs, axis=1)
        
        if aggregation == "mean":
            combined["importance"] = combined.mean(axis=1)
        elif aggregation == "max":
            combined["importance"] = combined.max(axis=1)
        else:  # weighted
            weights = [self.weights.get(name, 1.0) for name in combined.columns if name != "importance"]
            combined["importance"] = np.average(combined.values, axis=1, weights=weights)
        
        result = (
            combined[["importance"]]
            .reset_index()
            .sort_values("importance", ascending=False)
        )
        
        return result

    def compare_models(self, show_plot: bool = True) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Args:
            show_plot: Whether to show comparison plot
        
        Returns:
            DataFrame comparing model metrics
        """
        if not self.metrics_by_model:
            raise ValueError("No models trained yet")
        
        comparison_data = []
        for model_name, metrics in self.metrics_by_model.items():
            row = {"model": model_name}
            for key, value in metrics.items():
                # Clean up key names
                clean_key = key.replace("test_", "")
                row[clean_key] = value
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        logger.info("\n=== MODEL COMPARISON ===")
        logger.info(comparison_df.to_string(index=False))
        
        if show_plot:
            self.visualizer.plot_ensemble_comparison(self.metrics_by_model)
        
        return comparison_df


# =============================================================================
# MULTI-STEP FORECASTING
# =============================================================================


class MultiStepForecaster:
    """
    Multi-step demand forecaster for predicting multiple horizons.
    
    Strategies:
    - RECURSIVE: Predict day 1, use as input for day 2, etc.
    - DIRECT: Train separate model for each horizon
    - MULTI_OUTPUT: Single model predicts all horizons at once
    
    Use cases for FMCG:
    - Weekly shipment planning (7-day forecast)
    - Monthly production planning (30-day forecast)
    - Promotional planning (14-day forecast)
    """

    STRATEGY_RECURSIVE = "recursive"
    STRATEGY_DIRECT = "direct"
    STRATEGY_MULTI_OUTPUT = "multi_output"

    def __init__(
        self,
        horizons: List[int] = None,
        strategy: str = "direct",
        base_model: str = "xgboost",
        data_config: Optional[DataConfig] = None,
    ):
        """
        Initialize multi-step forecaster.
        
        Args:
            horizons: Forecast horizons in days (e.g., [1, 7, 14, 30])
            strategy: Forecasting strategy ('recursive', 'direct', 'multi_output')
            base_model: Base model type ('xgboost', 'lightgbm', 'catboost')
            data_config: Data configuration
        """
        self.horizons = horizons or [1, 7, 14, 30]
        self.strategy = strategy
        self.base_model = base_model
        self.data_config = data_config or DataConfig()
        
        self.models: Dict[int, Any] = {}  # horizon -> model
        self.feature_cols: List[str] = []
        self.metrics_by_horizon: Dict[int, Dict[str, float]] = {}
        
        logger.info(f"MultiStepForecaster initialized:")
        logger.info(f"  Horizons: {self.horizons}")
        logger.info(f"  Strategy: {self.strategy}")
        logger.info(f"  Base model: {self.base_model}")

    def _get_base_model(self, use_gpu: bool = False) -> Any:
        """Get base model instance."""
        if self.base_model == "xgboost":
            tree_method = "gpu_hist" if use_gpu else "hist"
            device = "cuda" if use_gpu else "cpu"
            return xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                tree_method=tree_method,
                device=device,
            )
        elif self.base_model == "lightgbm":
            if lightgbm is not None:
                import lightgbm as lgb
                return lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                )
            else:
                logger.warning("LightGBM not available, falling back to XGBoost")
                return self._get_base_model(use_gpu)
        elif self.base_model == "catboost":
            if catboost is not None:
                from catboost import CatBoostRegressor
                return CatBoostRegressor(
                    iterations=300,
                    depth=6,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=False,
                )
            else:
                logger.warning("CatBoost not available, falling back to XGBoost")
                return self._get_base_model(use_gpu)
        else:
            raise ValueError(f"Unknown base model: {self.base_model}")

    def _create_horizon_target(
        self,
        df: pd.DataFrame,
        target_col: str,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Create target variable for a specific horizon.
        
        Shifts the target backward so current features predict future demand.
        """
        result = df.copy()
        result[f"target_h{horizon}"] = (
            result.groupby(["sku_id", "location_id"])[target_col]
            .shift(-horizon)
        )
        return result

    def _create_multi_horizon_targets(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> pd.DataFrame:
        """Create target columns for all horizons."""
        result = df.copy()
        for h in self.horizons:
            result[f"target_h{h}"] = (
                result.groupby(["sku_id", "location_id"])[target_col]
                .shift(-h)
            )
        return result

    def train_direct(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "actual_demand",
        val_ratio: float = 0.2,
        use_gpu: bool = False,
    ) -> Dict[int, Dict[str, float]]:
        """
        Train using DIRECT strategy: one model per horizon.
        
        Args:
            df: Dataframe with features and target
            feature_cols: Feature column names
            target_col: Target column name
            val_ratio: Validation set ratio
            use_gpu: Whether to use GPU
        
        Returns:
            Metrics for each horizon
        """
        logger.info("=" * 60)
        logger.info("TRAINING MULTI-STEP FORECASTER (DIRECT STRATEGY)")
        logger.info("=" * 60)
        
        self.feature_cols = feature_cols
        
        # Sort by date
        df = df.sort_values([self.data_config.date_col]).reset_index(drop=True)
        
        # Split train/val by time
        split_idx = int(len(df) * (1 - val_ratio))
        
        for horizon in self.horizons:
            logger.info(f"\n--- Training model for horizon {horizon} days ---")
            
            # Create horizon-specific target
            df_h = self._create_horizon_target(df, target_col, horizon)
            target_h = f"target_h{horizon}"
            
            # Drop rows where target is NaN (end of series)
            df_h = df_h.dropna(subset=[target_h])
            
            # Split
            train_df = df_h.iloc[:split_idx]
            val_df = df_h.iloc[split_idx:]
            
            X_train = train_df[feature_cols]
            y_train = train_df[target_h]
            X_val = val_df[feature_cols]
            y_val = val_df[target_h]
            
            # Train model
            model = self._get_base_model(use_gpu)
            
            if hasattr(model, "fit"):
                if self.base_model == "xgboost":
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train)
            
            self.models[horizon] = model
            
            # Evaluate
            y_pred = np.maximum(model.predict(X_val), 0)
            metrics = Metrics.calculate(y_val, y_pred, f"h{horizon}_")
            self.metrics_by_horizon[horizon] = metrics
            
            logger.info(
                f"Horizon {horizon}: MAE={metrics[f'h{horizon}_mae']:.2f}, "
                f"RMSE={metrics[f'h{horizon}_rmse']:.2f}, "
                f"MAPE={metrics[f'h{horizon}_mape']:.2f}%"
            )
        
        return self.metrics_by_horizon

    def train_recursive(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "actual_demand",
        val_ratio: float = 0.2,
        use_gpu: bool = False,
    ) -> Dict[int, Dict[str, float]]:
        """
        Train using RECURSIVE strategy: single model, iterative predictions.
        
        Note: Only trains a 1-step model, then applies recursively.
        """
        logger.info("=" * 60)
        logger.info("TRAINING MULTI-STEP FORECASTER (RECURSIVE STRATEGY)")
        logger.info("=" * 60)
        
        self.feature_cols = feature_cols
        
        # Sort by date
        df = df.sort_values([self.data_config.date_col]).reset_index(drop=True)
        
        # Split
        split_idx = int(len(df) * (1 - val_ratio))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[feature_cols]
        y_val = val_df[target_col]
        
        # Train single model for 1-step prediction
        logger.info("\n--- Training 1-step model ---")
        model = self._get_base_model(use_gpu)
        
        if self.base_model == "xgboost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        
        self.models[1] = model
        
        # For recursive, we store the same model for all horizons
        for h in self.horizons:
            self.models[h] = model
        
        logger.info("Recursive strategy uses the same model for all horizons")
        logger.info("Predictions are made iteratively, feeding predictions as inputs")
        
        # Evaluate on 1-step
        y_pred = np.maximum(model.predict(X_val), 0)
        metrics = Metrics.calculate(y_val, y_pred, "h1_")
        self.metrics_by_horizon[1] = metrics
        
        logger.info(
            f"1-step: MAE={metrics['h1_mae']:.2f}, "
            f"RMSE={metrics['h1_rmse']:.2f}, "
            f"MAPE={metrics['h1_mape']:.2f}%"
        )
        
        return self.metrics_by_horizon

    def train_multi_output(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "actual_demand",
        val_ratio: float = 0.2,
        use_gpu: bool = False,
    ) -> Dict[int, Dict[str, float]]:
        """
        Train using MULTI-OUTPUT strategy: single model predicts all horizons.
        
        Uses MultiOutputRegressor wrapper.
        """
        from sklearn.multioutput import MultiOutputRegressor
        
        logger.info("=" * 60)
        logger.info("TRAINING MULTI-STEP FORECASTER (MULTI-OUTPUT STRATEGY)")
        logger.info("=" * 60)
        
        self.feature_cols = feature_cols
        
        # Sort by date
        df = df.sort_values([self.data_config.date_col]).reset_index(drop=True)
        
        # Create all horizon targets
        df = self._create_multi_horizon_targets(df, target_col)
        
        # Drop rows where any target is NaN
        target_cols = [f"target_h{h}" for h in self.horizons]
        df = df.dropna(subset=target_cols)
        
        # Split
        split_idx = int(len(df) * (1 - val_ratio))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        X_train = train_df[feature_cols]
        y_train = train_df[target_cols]
        X_val = val_df[feature_cols]
        y_val = val_df[target_cols]
        
        logger.info(f"\n--- Training multi-output model for horizons {self.horizons} ---")
        
        # Wrap base model in MultiOutputRegressor
        base = self._get_base_model(use_gpu)
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        
        self.models["multi_output"] = model
        
        # Evaluate each horizon
        y_pred = model.predict(X_val)
        y_pred = np.maximum(y_pred, 0)
        
        for i, horizon in enumerate(self.horizons):
            metrics = Metrics.calculate(y_val.iloc[:, i], y_pred[:, i], f"h{horizon}_")
            self.metrics_by_horizon[horizon] = metrics
            
            logger.info(
                f"Horizon {horizon}: MAE={metrics[f'h{horizon}_mae']:.2f}, "
                f"RMSE={metrics[f'h{horizon}_rmse']:.2f}, "
                f"MAPE={metrics[f'h{horizon}_mape']:.2f}%"
            )
        
        return self.metrics_by_horizon

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "actual_demand",
        val_ratio: float = 0.2,
        use_gpu: bool = False,
    ) -> Dict[int, Dict[str, float]]:
        """
        Train using the configured strategy.
        
        Args:
            df: Dataframe with features and target
            feature_cols: Feature column names
            target_col: Target column name
            val_ratio: Validation set ratio
            use_gpu: Whether to use GPU
        
        Returns:
            Metrics for each horizon
        """
        if self.strategy == self.STRATEGY_RECURSIVE:
            return self.train_recursive(df, feature_cols, target_col, val_ratio, use_gpu)
        elif self.strategy == self.STRATEGY_DIRECT:
            return self.train_direct(df, feature_cols, target_col, val_ratio, use_gpu)
        elif self.strategy == self.STRATEGY_MULTI_OUTPUT:
            return self.train_multi_output(df, feature_cols, target_col, val_ratio, use_gpu)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def predict(
        self,
        X: pd.DataFrame,
        horizon: Optional[int] = None,
    ) -> Union[np.ndarray, Dict[int, np.ndarray]]:
        """
        Generate predictions for specified horizon(s).
        
        Args:
            X: Feature dataframe
            horizon: Specific horizon to predict (None = all horizons)
        
        Returns:
            Predictions array or dict of horizon -> predictions
        """
        if self.strategy == self.STRATEGY_MULTI_OUTPUT:
            if "multi_output" not in self.models:
                raise ValueError("Model not trained")
            
            all_preds = self.models["multi_output"].predict(X)
            all_preds = np.maximum(all_preds, 0)
            
            if horizon is not None:
                idx = self.horizons.index(horizon)
                return all_preds[:, idx]
            
            return {h: all_preds[:, i] for i, h in enumerate(self.horizons)}
        
        else:  # direct or recursive
            if horizon is not None:
                if horizon not in self.models:
                    raise ValueError(f"No model for horizon {horizon}")
                pred = self.models[horizon].predict(X)
                return np.maximum(pred, 0)
            
            return {
                h: np.maximum(self.models[h].predict(X), 0)
                for h in self.horizons
                if h in self.models
            }

    def predict_recursive_sequence(
        self,
        X_start: pd.DataFrame,
        n_steps: int,
        update_features_fn: Optional[callable] = None,
    ) -> np.ndarray:
        """
        Generate recursive multi-step predictions.
        
        Args:
            X_start: Starting features (single row or batch)
            n_steps: Number of steps to predict forward
            update_features_fn: Function to update features with new prediction
                                Should take (features, prediction, step) -> new_features
        
        Returns:
            Array of shape (n_samples, n_steps) with predictions
        """
        if 1 not in self.models:
            raise ValueError("Recursive prediction requires a trained 1-step model")
        
        model = self.models[1]
        n_samples = len(X_start)
        predictions = np.zeros((n_samples, n_steps))
        
        current_X = X_start.copy()
        
        for step in range(n_steps):
            pred = model.predict(current_X)
            pred = np.maximum(pred, 0)
            predictions[:, step] = pred
            
            if update_features_fn is not None:
                current_X = update_features_fn(current_X, pred, step)
            else:
                # Default: update lag features if they exist
                current_X = self._default_feature_update(current_X, pred)
        
        return predictions

    def _default_feature_update(
        self,
        X: pd.DataFrame,
        prediction: np.ndarray,
    ) -> pd.DataFrame:
        """Default feature update for recursive forecasting."""
        X_new = X.copy()
        
        # Update lag_1 feature if it exists
        lag_cols = [c for c in X.columns if "lag_1" in c.lower()]
        for col in lag_cols:
            X_new[col] = prediction
        
        # Shift other lag features
        for lag in [7, 14, 30]:
            lag_cols = [c for c in X.columns if f"lag_{lag}" in c.lower()]
            for col in lag_cols:
                # This is a simplified update - real implementation would need
                # to track history of predictions
                pass
        
        return X_new

    def get_forecast_summary(self) -> pd.DataFrame:
        """
        Get summary of forecast performance across horizons.
        
        Returns:
            DataFrame with metrics for each horizon
        """
        if not self.metrics_by_horizon:
            raise ValueError("No metrics available. Train first.")
        
        rows = []
        for horizon, metrics in self.metrics_by_horizon.items():
            row = {"horizon": horizon}
            for key, value in metrics.items():
                # Clean up key names
                clean_key = key.split("_", 1)[-1] if "_" in key else key
                row[clean_key] = value
            rows.append(row)
        
        summary = pd.DataFrame(rows).sort_values("horizon")
        
        logger.info("\n=== MULTI-STEP FORECAST SUMMARY ===")
        logger.info(summary.to_string(index=False))
        
        return summary

    def plot_horizon_performance(
        self,
        save_path: Optional[Path] = None,
        show: bool = True,
    ) -> None:
        """
        Plot performance metrics across forecast horizons.
        
        Args:
            save_path: Path to save plot
            show: Whether to display plot
        """
        if plt is None:
            logger.warning("matplotlib not installed. Cannot generate plots.")
            return
        import matplotlib.pyplot as plt
        
        if not self.metrics_by_horizon:
            logger.warning("No metrics to plot")
            return
        
        horizons = sorted(self.metrics_by_horizon.keys())
        maes = [self.metrics_by_horizon[h].get(f"h{h}_mae", 0) for h in horizons]
        mapes = [self.metrics_by_horizon[h].get(f"h{h}_mape", 0) for h in horizons]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # MAE by horizon
        axes[0].bar(
            [str(h) for h in horizons],
            maes,
            color="#2E86AB",
            edgecolor="white",
        )
        axes[0].set_xlabel("Forecast Horizon (days)")
        axes[0].set_ylabel("Mean Absolute Error")
        axes[0].set_title("MAE by Forecast Horizon", fontsize=12, fontweight="bold")
        axes[0].grid(axis="y", alpha=0.3)
        
        # Add trend line
        if len(horizons) > 2:
            z = np.polyfit(horizons, maes, 1)
            p = np.poly1d(z)
            axes[0].plot(
                range(len(horizons)),
                p(horizons),
                "r--",
                linewidth=2,
                label=f"Trend (+{z[0]:.2f}/day)",
            )
            axes[0].legend()
        
        # MAPE by horizon
        axes[1].bar(
            [str(h) for h in horizons],
            mapes,
            color="#E94F37",
            edgecolor="white",
        )
        axes[1].set_xlabel("Forecast Horizon (days)")
        axes[1].set_ylabel("MAPE (%)")
        axes[1].set_title("MAPE by Forecast Horizon", fontsize=12, fontweight="bold")
        axes[1].grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved horizon performance plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


# =============================================================================
# TRAINING PIPELINE (Best Practice Workflow)
# =============================================================================


class TrainingPipeline:
    """
    End-to-end training pipeline implementing ML best practices.
    
    Workflow:
    1. Split Data - Train/Val/Test with time order preserved
    2. Establish Baseline - Train with default params for benchmark
    3. Hyperparameter Search - Bayesian optimization on validation set
    4. Early Stopping - Prevent overfitting during training
    5. Final Training - Retrain on train+val with best params
    6. Evaluate - Assess on unseen test set
    
    This ensures proper model selection without data leakage.
    """

    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ):
        """
        Initialize training pipeline.
        
        Args:
            data_config: Data configuration
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation (hyperparameter tuning)
            test_ratio: Proportion of data for final testing
            random_state: Random seed for reproducibility
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        self.data_config = data_config or DataConfig()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        # Data splits
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        
        # Model and results
        self.baseline_model: Optional[xgb.XGBRegressor] = None
        self.best_model: Optional[xgb.XGBRegressor] = None
        self.final_model: Optional[xgb.XGBRegressor] = None
        self.ensemble_trainer: Optional[EnsembleTrainer] = None
        
        self.feature_cols: List[str] = []
        # Metrics
        self.baseline_metrics: Dict[str, float] = {}
        self.best_params: Dict[str, Any] = {}
        self.tuning_metrics: Dict[str, float] = {}
        self.final_metrics: Dict[str, float] = {}
        
        # Visualizer
        self.visualizer = ForecastVisualizer(self.data_config.output_root / "plots")
        
        logger.info("TrainingPipeline initialized")
        logger.info(f"  Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%}")

    def _detect_device(self) -> Tuple[str, str]:
        """Detect GPU availability."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return "hist", "cuda"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "hist", "cpu"

    # =========================================================================
    # STEP 1: SPLIT DATA
    # =========================================================================

    def split_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets preserving time order.
        
        Args:
            df: Full dataset with features and target
            feature_cols: List of feature column names
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("=" * 60)
        logger.info("STEP 1: SPLITTING DATA (Time-Ordered)")
        logger.info("=" * 60)
        
        self.feature_cols = feature_cols
        date_col = self.data_config.date_col
        
        # Sort by date to ensure time order
        df = df.sort_values(date_col).reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        self.train_df = df.iloc[:train_end].copy()
        self.val_df = df.iloc[train_end:val_end].copy()
        self.test_df = df.iloc[val_end:].copy()
        
        logger.info(f"\nData split complete:")
        logger.info(f"  Train: {len(self.train_df):,} rows "
                   f"({self.train_df[date_col].min()} → {self.train_df[date_col].max()})")
        logger.info(f"  Val:   {len(self.val_df):,} rows "
                   f"({self.val_df[date_col].min()} → {self.val_df[date_col].max()})")
        logger.info(f"  Test:  {len(self.test_df):,} rows "
                   f"({self.test_df[date_col].min()} → {self.test_df[date_col].max()})")
        
        return self.train_df, self.val_df, self.test_df

    # =========================================================================
    # STEP 2: ESTABLISH BASELINE
    # =========================================================================

    def train_baseline(
        self,
        early_stopping_rounds: int = 50,
    ) -> Dict[str, float]:
        """
        Train baseline model with default parameters.
        
        This establishes a performance benchmark to beat.
        
        Args:
            early_stopping_rounds: Rounds for early stopping
        
        Returns:
            Baseline metrics on validation set
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: ESTABLISHING BASELINE")
        logger.info("=" * 60)
        
        if self.train_df is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        target = self.data_config.target_col
        tree_method, device = self._detect_device()
        
        # Default XGBoost parameters
        default_params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "min_child_weight": 1,
            "random_state": self.random_state,
            "n_jobs": -1,
            "tree_method": tree_method,
            "device": device,
            "early_stopping_rounds": early_stopping_rounds,
        }
        
        logger.info("\nDefault parameters:")
        for k, v in default_params.items():
            if k not in ["n_jobs", "random_state", "tree_method", "device"]:
                logger.info(f"  {k}: {v}")
        
        # Prepare data
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df[target]
        X_val = self.val_df[self.feature_cols]
        y_val = self.val_df[target]
        
        # Train baseline
        logger.info("\nTraining baseline model (Target transformed with log1p)...")
        self.baseline_model = xgb.XGBRegressor(**default_params)
        
        # Log-transform targets for RMSLE optimization
        y_train_log = np.log1p(y_train).astype(np.float32)
        y_val_log = np.log1p(y_val).astype(np.float32)
        
        self.baseline_model.fit(
            X_train, y_train_log,
            eval_set=[(X_val, y_val_log)],
            verbose=False,
        )
        
        # Evaluate on validation (inverse transform prediction)
        y_val_pred_log = self.baseline_model.predict(X_val)
        y_val_pred = np.maximum(np.expm1(y_val_pred_log), 0)
        
        self.baseline_metrics = Metrics.calculate(y_val, y_val_pred, "baseline_")
        
        logger.info("\n--- BASELINE PERFORMANCE (Validation Set) ---")
        logger.info(f"MAE:  {self.baseline_metrics['baseline_mae']:.2f}")
        logger.info(f"RMSE: {self.baseline_metrics['baseline_rmse']:.2f}")
        logger.info(f"MAPE: {self.baseline_metrics['baseline_mape']:.2f}%")
        logger.info(f"R²:   {self.baseline_metrics['baseline_r2']:.4f}")
        
        if self.baseline_model.best_iteration:
            logger.info(f"\nEarly stopping at iteration: {self.baseline_model.best_iteration}")
        
        return self.baseline_metrics

    # =========================================================================
    # STEP 3: HYPERPARAMETER SEARCH
    # =========================================================================

    def tune_hyperparameters(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        early_stopping_rounds: int = 50,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform Bayesian hyperparameter optimization using Optuna.
        
        Optimizes on validation set to find best parameters.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Optional timeout in seconds
            early_stopping_rounds: Rounds for early stopping within each trial
            show_progress: Whether to show progress bar
        
        Returns:
            Best hyperparameters found
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: HYPERPARAMETER SEARCH (Optuna)")
        logger.info("=" * 60)
        
        try:
            import optuna
        except ImportError:
            logger.error("Optuna not installed. Run: pip install optuna")
            raise
        
        from optuna.samplers import TPESampler
        
        if self.train_df is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        target = self.data_config.target_col
        tree_method, device = self._detect_device()
        
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df[target]
        X_val = self.val_df[self.feature_cols]
        y_val = self.val_df[target]
        
        # Prepare data with log variables
        # Use log targets for training and validation to optimize RMSLE
        # OPTIMIZATION: Cast to float32 and delete originals immediately
        y_train_log = np.log1p(y_train).astype(np.float32)
        del y_train
        y_val_log = np.log1p(y_val).astype(np.float32)
        del y_val
        gc.collect()
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function."""
            params = {
                # Core hyperparameters
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                
                # Regularization
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                
                # Fixed parameters
                "random_state": self.random_state,
                "n_jobs": -1,
                "tree_method": tree_method,
                "device": device,
                "early_stopping_rounds": early_stopping_rounds,
            }
            
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train_log,
                eval_set=[(X_val, y_val_log)],
                verbose=False,
            )
            
            # Predict on log scale
            y_pred_log = model.predict(X_val)
            
            # RMSE on log scale is RMSLE
            rmsle = np.sqrt(mean_squared_error(y_val_log, y_pred_log))
            
            return rmsle
        
        # Create study with TPE sampler (Bayesian optimization)
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name="fmcg_xgb_tuning",
        )
        
        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        logger.info(f"\nRunning {n_trials} optimization trials...")
        logger.info("Optimizing RMSLE (log-RMSE) on validation set")
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress,
            gc_after_trial=True,
        )
        
        # Store best parameters
        self.best_params = study.best_trial.params
        
        logger.info("\n--- BEST HYPERPARAMETERS FOUND ---")
        for k, v in self.best_params.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"  {k}: {v}")
        
        # Train model with best params to get metrics
        best_full_params = {
            **self.best_params,
            "random_state": self.random_state,
            "n_jobs": -1,
            "tree_method": tree_method,
            "device": device,
            "early_stopping_rounds": early_stopping_rounds,
        }
        
        self.best_model = xgb.XGBRegressor(**best_full_params)
        self.best_model.fit(
            X_train, y_train_log,
            eval_set=[(X_val, y_val_log)],
            verbose=False,
        )
        
        y_val_pred_log = self.best_model.predict(X_val)
        y_val_pred = np.maximum(np.expm1(y_val_pred_log), 0)
        
        # Re-fetch y_val since we deleted it earlier for memory
        y_val = self.val_df[target]
        self.tuning_metrics = Metrics.calculate(y_val, y_val_pred, "tuned_")
        
        logger.info("\n--- TUNED MODEL PERFORMANCE (Validation Set) ---")
        logger.info(f"MAE:  {self.tuning_metrics['tuned_mae']:.2f}")
        logger.info(f"RMSE: {self.tuning_metrics['tuned_rmse']:.2f}")
        logger.info(f"MAPE: {self.tuning_metrics['tuned_mape']:.2f}%")
        logger.info(f"R²:   {self.tuning_metrics['tuned_r2']:.4f}")
        
        # Compare with baseline
        improvement_mae = (
            (self.baseline_metrics['baseline_mae'] - self.tuning_metrics['tuned_mae'])
            / self.baseline_metrics['baseline_mae'] * 100
        )
        improvement_rmse = (
            (self.baseline_metrics['baseline_rmse'] - self.tuning_metrics['tuned_rmse'])
            / self.baseline_metrics['baseline_rmse'] * 100
        )
        
        logger.info(f"\n--- IMPROVEMENT OVER BASELINE ---")
        logger.info(f"MAE:  {improvement_mae:+.2f}%")
        logger.info(f"RMSE: {improvement_rmse:+.2f}%")
        
        return self.best_params

    # =========================================================================
    # STEP 4 & 5: FINAL TRAINING (Train + Val with Best Params)
    # =========================================================================

    def train_final_model(
        self,
        early_stopping_rounds: int = 50,
    ) -> xgb.XGBRegressor:
        """
        Train final model on combined train+val data with best hyperparameters.
        
        Uses a small holdout from train+val for early stopping.
        
        Args:
            early_stopping_rounds: Rounds for early stopping
        
        Returns:
            Final trained model
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4 & 5: FINAL TRAINING (Train + Val)")
        logger.info("=" * 60)
        
        if not self.best_params:
            raise ValueError("No best params. Call tune_hyperparameters() first.")
        
        target = self.data_config.target_col
        tree_method, device = self._detect_device()
        
        # Combine train and validation data
        combined_df = pd.concat([self.train_df, self.val_df], ignore_index=True)
        
        # Use a small holdout for early stopping (5% of combined)
        n = len(combined_df)
        holdout_size = int(n * 0.05)
        
        train_combined = combined_df.iloc[:-holdout_size]
        holdout = combined_df.iloc[-holdout_size:]
        
        logger.info(f"\nCombined train+val: {len(combined_df):,} rows")
        logger.info(f"  Training on: {len(train_combined):,} rows")
        logger.info(f"  Early stopping holdout: {len(holdout):,} rows")
        
        X_train = train_combined[self.feature_cols]
        y_train = train_combined[target]
        X_holdout = holdout[self.feature_cols]
        y_holdout = holdout[target]
        
        # Log-transform for RMSLE (Optimize memory with float32)
        y_train_log = np.log1p(y_train).astype(np.float32)
        y_holdout_log = np.log1p(y_holdout).astype(np.float32)
        
        # Build final params
        final_params = {
            **self.best_params,
            "random_state": self.random_state,
            "n_jobs": -1,
            "tree_method": tree_method,
            "device": device,
            "early_stopping_rounds": early_stopping_rounds,
        }
        
        logger.info("\nTraining final model with optimized hyperparameters (Target transformed)...")
        self.final_model = xgb.XGBRegressor(**final_params)
        self.final_model.fit(
            X_train, y_train_log,
            eval_set=[(X_holdout, y_holdout_log)],
            verbose=100,
        )
        
        if self.final_model.best_iteration:
            logger.info(f"\nEarly stopping at iteration: {self.final_model.best_iteration}")
        
        return self.final_model

    def train_ensemble(self, use_gpu: bool = True) -> None:
        """Train ensemble models (XGB+LGB+CatBoost)."""
        if self.train_df is None:
            raise ValueError("Data not split. Call split_data() first.")
            
        target = self.data_config.target_col
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df[target]
        
        # AGGRESSIVE MEMORY CLEANUP: Delete original df immediately
        del self.train_df
        gc.collect()
        
        X_val = self.val_df[self.feature_cols]
        y_val = self.val_df[target]
        
        # AGGRESSIVE MEMORY CLEANUP
        del self.val_df
        gc.collect()
        
        X_test = self.test_df[self.feature_cols]
        y_test = self.test_df[target]
        
        self.ensemble_trainer = EnsembleTrainer(
            data_config=self.data_config,
            use_xgb=True,
            use_lgb=True,
            use_catboost=True,
        )
        
        # Train and get metrics
        metrics = self.ensemble_trainer.train(
            X_train, y_train, 
            X_val, y_val, 
            X_test, y_test, 
            use_gpu=use_gpu
        )
        
        # Set final model to ensemble for evaluation/saving
        self.final_model = self.ensemble_trainer
        
        # Use ensemble metrics
        if "ensemble" in metrics:
            self.final_metrics = metrics["ensemble"]
            
        logger.info("\nEnsemble training complete. Final model set to Ensemble.")

    # =========================================================================
    # STEP 6: FINAL EVALUATION ON TEST SET
    # =========================================================================

    def evaluate_on_test(
        self,
        generate_plots: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate final model on completely unseen test set.
        
        This gives the true estimate of generalization performance.
        
        Args:
            generate_plots: Whether to generate visualization plots
        
        Returns:
            Final test metrics
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: FINAL EVALUATION (Test Set)")
        logger.info("=" * 60)
        
        if self.final_model is None:
            raise ValueError("No final model. Call train_final_model() first.")
        
        target = self.data_config.target_col
        
        X_test = self.test_df[self.feature_cols]
        y_test = self.test_df[target]
        
        # Generate predictions (log scale)
        y_test_pred_log = self.final_model.predict(X_test)
        
        # Inverse transform to get actual demand
        y_test_pred = np.maximum(np.expm1(y_test_pred_log), 0)
        
        # Calculate metrics on original scale
        self.final_metrics = Metrics.calculate(y_test, y_test_pred, "test_")
        
        logger.info("\n" + "=" * 50)
        logger.info("FINAL MODEL PERFORMANCE (UNSEEN TEST SET)")
        logger.info("=" * 50)
        logger.info(f"MAE:   {self.final_metrics['test_mae']:,.2f}")
        logger.info(f"RMSE:  {self.final_metrics['test_rmse']:,.2f}")
        logger.info(f"MAPE:  {self.final_metrics['test_mape']:.2f}%")
        logger.info(f"SMAPE: {self.final_metrics['test_smape']:.2f}%")
        logger.info(f"WMAPE: {self.final_metrics['test_wmape']:.2f}%")
        logger.info(f"R²:    {self.final_metrics['test_r2']:.4f}")
        logger.info(f"Bias:  {self.final_metrics['test_bias_pct']:+.2f}%")
        logger.info("=" * 50)
        
        # Generate visualizations
        if generate_plots:
            logger.info("\nGenerating evaluation plots...")
            
            # Feature importance
            fi = pd.DataFrame({
                "feature": self.feature_cols,
                "importance": self.final_model.feature_importances_,
            }).sort_values("importance", ascending=False)
            
            self.visualizer.plot_feature_importance(fi, save=True, show=True)
            
            # Actual vs Predicted
            dates = self.test_df[self.data_config.date_col] if self.data_config.date_col in self.test_df.columns else None
            self.visualizer.plot_actual_vs_predicted(
                y_test.values, y_test_pred, dates=dates, save=True, show=True
            )
            
            # Residual analysis
            self.visualizer.plot_residual_analysis(
                y_test.values, y_test_pred, save=True, show=True
            )
            
            logger.info(f"Plots saved to {self.visualizer.output_dir}")
        
        return self.final_metrics

    # =========================================================================
    # RUN FULL PIPELINE
    # =========================================================================

    def run(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        n_trials: int = 50,
        early_stopping_rounds: int = 50,
        generate_plots: bool = True,
        use_ensemble: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            df: Full dataset with features and target
            feature_cols: List of feature column names
            n_trials: Number of hyperparameter tuning trials
            early_stopping_rounds: Rounds for early stopping
            generate_plots: Whether to generate visualization plots
            use_ensemble: Whether to use Ensemble (XGB+LGB+CatBoost) strategy
        
        Returns:
            Dictionary with all results
        """
        logger.info("\n" + "#" * 60)
        logger.info("STARTING COMPLETE TRAINING PIPELINE")
        logger.info("#" * 60)
        
        # Step 1: Split data
        self.split_data(df, feature_cols)
        
        # Step 2: Baseline
        self.train_baseline(early_stopping_rounds)
        
        if use_ensemble:
            # OPTION 1: Ensemble Strategy (Skips Optuna for speed/defaults)
            logger.info("\nSkipping Optuna tuning (using Ensemble defaults)...")
            self.train_ensemble(use_gpu=True)
            
            # Mock empty tuning metrics for summary
            self.tuning_metrics = {"tuned_mae": 0, "tuned_rmse": 0, "tuned_mape": 0}
            
        else:
            # OPTION 2: Single XGBoost with Hyperparameter Tuning
            # Step 3: Hyperparameter tuning
            self.tune_hyperparameters(n_trials, early_stopping_rounds=early_stopping_rounds)
            
            # Step 4 & 5: Final training
            self.train_final_model(early_stopping_rounds)
        
        # Step 6: Final evaluation
        # Note: evaluate_on_test uses self.final_model (which is set to ensemble if used)
        self.evaluate_on_test(generate_plots)
        
        # Summary
        logger.info("\n" + "#" * 60)
        logger.info("PIPELINE COMPLETE - SUMMARY")
        logger.info("#" * 60)
        
        logger.info("\nMetrics Comparison:")
        logger.info(f"{'Stage':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
        logger.info("-" * 52)
        logger.info(
            f"{'Baseline (val)':<20} "
            f"{self.baseline_metrics['baseline_mae']:>10.2f} "
            f"{self.baseline_metrics['baseline_rmse']:>10.2f} "
            f"{self.baseline_metrics['baseline_mape']:>9.2f}%"
        )
        logger.info(
            f"{'Tuned (val)':<20} "
            f"{self.tuning_metrics['tuned_mae']:>10.2f} "
            f"{self.tuning_metrics['tuned_rmse']:>10.2f} "
            f"{self.tuning_metrics['tuned_mape']:>9.2f}%"
        )
        logger.info(
            f"{'Final (test)':<20} "
            f"{self.final_metrics['test_mae']:>10.2f} "
            f"{self.final_metrics['test_rmse']:>10.2f} "
            f"{self.final_metrics['test_mape']:>9.2f}%"
        )
        
        return {
            "baseline_metrics": self.baseline_metrics,
            "tuning_metrics": self.tuning_metrics,
            "test_metrics": self.final_metrics,
            "best_params": self.best_params,
            "final_model": self.final_model,
        }

    def save_results(
        self,
        output_dir: Optional[Path] = None,
    ) -> None:
        """
        Save all pipeline results and artifacts.
        
        Args:
            output_dir: Directory to save results (uses config default if None)
        """
        output_dir = output_dir or self.data_config.output_root
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save final model
        if self.final_model:
            model_path = output_dir / f"final_model_{timestamp}.json"
            self.final_model.save_model(model_path)
            logger.info(f"Saved model to {model_path}")
        
        # Save best params
        if self.best_params:
            import json
            params_path = output_dir / f"best_params_{timestamp}.json"
            with open(params_path, "w") as f:
                json.dump(self.best_params, f, indent=2)
            logger.info(f"Saved params to {params_path}")
        
        # Save metrics comparison
        metrics_data = {
            "stage": ["baseline_val", "tuned_val", "final_test"],
            "mae": [
                self.baseline_metrics.get("baseline_mae", 0),
                self.tuning_metrics.get("tuned_mae", 0),
                self.final_metrics.get("test_mae", 0),
            ],
            "rmse": [
                self.baseline_metrics.get("baseline_rmse", 0),
                self.tuning_metrics.get("tuned_rmse", 0),
                self.final_metrics.get("test_rmse", 0),
            ],
            "mape": [
                self.baseline_metrics.get("baseline_mape", 0),
                self.tuning_metrics.get("tuned_mape", 0),
                self.final_metrics.get("test_mape", 0),
            ],
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = output_dir / f"pipeline_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save test predictions
        if self.final_model and self.test_df is not None:
            target = self.data_config.target_col
            X_test = self.test_df[self.feature_cols]
            y_test = self.test_df[target]
            
            # Generate predictions (log scale) & inverse transform
            y_pred_log = self.final_model.predict(X_test)
            y_pred = np.maximum(np.expm1(y_pred_log), 0)
            
            id_cols = [c for c in self.data_config.id_cols if c in self.test_df.columns]
            pred_df = self.test_df[id_cols].copy()
            pred_df["actual"] = y_test.values
            pred_df["predicted"] = y_pred
            pred_df["error"] = pred_df["actual"] - pred_df["predicted"]
            pred_df["abs_pct_error"] = np.abs(pred_df["error"]) / np.maximum(pred_df["actual"], 1) * 100
            
            pred_path = output_dir / f"test_predictions_{timestamp}.csv"
            pred_df.to_csv(pred_path, index=False)
            logger.info(f"Saved predictions to {pred_path}")


# =============================================================================
# HYPERPARAMETER TUNING (Optional - requires optuna)
# =============================================================================


def tune_hyperparameters(
    df: pd.DataFrame,
    data_config: DataConfig,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """
    Tune XGBoost hyperparameters using Optuna.

    Args:
        df: Prepared dataframe with features
        data_config: Data configuration
        n_trials: Number of Optuna trials

    Returns:
        Best hyperparameters found
    """
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        raise

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }

        model_config = ModelConfig(**params)
        trainer = FMCGTrainer(data_config=data_config, model_config=model_config)
        metrics = trainer.train_cv(df.copy(), n_splits=3)

        return metrics["cv_avg_rmse"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial RMSE: {study.best_trial.value:.4f}")
    logger.info(f"Best parameters: {study.best_trial.params}")

    return study.best_trial.params


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main():
    """
    Main training pipeline demonstrating the complete workflow.
    
    This uses the TrainingPipeline class which implements best practices:
    1. Time-ordered train/val/test split
    2. Baseline model for benchmarking
    3. Hyperparameter tuning with Optuna
    4. Early stopping to prevent overfitting
    5. Final training on train+val with best params
    6. Evaluation on unseen test set
    """
    import os
    
    # Auto-detect data path (kagglehub cache or local)
    kaggle_input_path = Path("/kaggle/input/fmcg-dataset")
    kagglehub_path = Path.home() / ".cache" / "kagglehub" / "datasets" / "sand35h44jsd" / "fmcg-dataset" / "versions" / "1"
    local_path = Path("./data")
    
    if kaggle_input_path.exists():
        data_root = kaggle_input_path
        logger.info(f"Using Kaggle input: {data_root}")
    elif kagglehub_path.exists():
        data_root = kagglehub_path
        logger.info(f"Using kagglehub cache: {data_root}")
    elif local_path.exists():
        data_root = local_path
        logger.info(f"Using local data: {data_root}")
    else:
        # Try to download using kagglehub
        try:
            import kagglehub
            data_root = Path(kagglehub.dataset_download('sand35h44jsd/fmcg-dataset'))
            logger.info(f"Downloaded data to: {data_root}")
        except ImportError:
            logger.error("kagglehub not installed. Run: pip install kagglehub")
            logger.error("Or manually download the FMCG dataset from Kaggle")
            return None, None
        except Exception as e:
            logger.error(f"Could not download data: {e}")
            logger.error("Please manually download the FMCG dataset from Kaggle")
            return None, None
    
    # Initialize configurations
    data_config = DataConfig(
        data_root=data_root,
        output_root=Path("./output"),
    )
    
    # Create output directory
    data_config.output_root.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # OPTION 1: Use TrainingPipeline (RECOMMENDED - Best Practice Workflow)
    # =========================================================================
    
    logger.info("\n" + "=" * 70)
    logger.info("FMCG DEMAND FORECASTING - TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # First, load and prepare data using FeatureEngineer
    logger.info("\nLoading and preparing data...")
    
    # Load raw data
    daily_ts = pd.read_csv(data_config.data_root / data_config.daily_ts_file, parse_dates=["date"])
    logger.info(f"Loaded {len(daily_ts):,} rows from {data_config.daily_ts_file}")
    
    # Initialize feature engineer and load reference data
    fe = FeatureEngineer(data_config)
    fe.load_reference_data()
    
    # Transform features
    df = fe.transform(daily_ts)
    
    # Remove rows with missing target
    target = data_config.target_col
    df = df[~df[target].isna()].reset_index(drop=True)
    logger.info(f"Prepared dataset: {len(df):,} rows")
    
    # Prepare feature columns (exclude IDs, target, leaky columns)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = [target] + data_config.leak_cols + [c for c in data_config.id_cols if c in num_cols]
    feature_cols = [c for c in num_cols if c not in exclude]
    logger.info(f"Using {len(feature_cols)} features")
    
    # Initialize and run the training pipeline
    pipeline = TrainingPipeline(
        data_config=data_config,
        train_ratio=0.7,   # 70% for training
        val_ratio=0.15,    # 15% for validation (hyperparameter tuning)
        test_ratio=0.15,   # 15% for final testing
    )
    
    # Run the complete pipeline (Ensemble Strategy enabled)
    results = pipeline.run(
        df=df,
        feature_cols=feature_cols,
        n_trials=20,     # Ignored if use_ensemble=True
        early_stopping_rounds=50,
        generate_plots=True,
        use_ensemble=True, # <--- ENABLE ENSEMBLE
    )
    
    # Save all results
    pipeline.save_results()
    
    # Print final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Test Metrics:")
    print(f"  MAE:  {results['test_metrics']['test_mae']:,.2f}")
    print(f"  RMSE: {results['test_metrics']['test_rmse']:,.2f}")
    print(f"  MAPE: {results['test_metrics']['test_mape']:.2f}%")
    print(f"  R²:   {results['test_metrics']['test_r2']:.4f}")
    print(f"\nBest hyperparameters saved to: {data_config.output_root}")
    print(f"Plots saved to: {data_config.output_root / 'plots'}")
    
    return pipeline, results


if __name__ == "__main__":
    pipeline, results = main()


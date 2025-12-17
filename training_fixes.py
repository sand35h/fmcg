# Critical Fixes for FMCG Forecasting Model
# Apply these changes to training.py

# ============================================================================
# FIX 1: Replace Price Elasticity with Signal-Rich Features
# ============================================================================

def add_price_features_v2(self, df: pd.DataFrame) -> pd.DataFrame:
    """Replace noisy elasticity with actionable price features."""
    if not {"price", "base_price"}.issubset(df.columns):
        return df
    
    df = df.sort_values(["sku_id", "location_id", "date"])
    g = df.groupby(["sku_id", "location_id"], observed=True)
    
    # Discount depth (direct signal)
    df["discount_depth"] = (df["base_price"] - df["price"]) / df["base_price"].replace(0, np.nan)
    df["discount_depth"] = df["discount_depth"].fillna(0).clip(0, 1)
    
    # Price change flag (binary)
    df["price_changed"] = (g["price"].shift(7) != df["price"]).astype(np.int8)
    
    # Promo intensity (rolling)
    if "promo_flag" in df.columns:
        df["promo_intensity_7d"] = g["promo_flag"].rolling(7, min_periods=1).mean().astype(np.float32)
    
    # Price volatility (30-day std)
    df["price_volatility_30d"] = g["price"].rolling(30, min_periods=5).std().fillna(0).astype(np.float32)
    
    return df


# ============================================================================
# FIX 2: Tweedie Objective + WMAPE Early Stopping
# ============================================================================

def _get_model_params_v2(self) -> Dict[str, Any]:
    """Optimized LightGBM params with Tweedie loss."""
    return {
        "objective": "tweedie",
        "tweedie_variance_power": 1.3,  # Between Poisson (1) and Gamma (2)
        "metric": "None",  # Use custom WMAPE
        "learning_rate": 0.03,
        "num_leaves": 256,
        "max_depth": -1,  # No limit (controlled by num_leaves)
        "min_data_in_leaf": 500,
        "max_bin": 512,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
        "n_estimators": 5000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


def wmape_lgb(y_pred, y_true):
    """Custom WMAPE metric for LightGBM early stopping."""
    y_true = y_true.get_label()
    y_pred = np.clip(y_pred, 0, None)
    wmape = np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(np.abs(y_true)), 1) * 100
    return "wmape", wmape, False  # Lower is better


# ============================================================================
# FIX 3: Rolling-Origin Validation
# ============================================================================

def rolling_origin_split(df: pd.DataFrame, n_origins: int = 3) -> List[Tuple]:
    """Create multiple train/val splits for robust hyperparameter tuning."""
    date_col = "date"
    max_date = df[date_col].max()
    
    splits = []
    for i in range(n_origins):
        val_end = max_date - pd.Timedelta(days=365 * i)
        val_start = val_end - pd.Timedelta(days=90)
        train_end = val_start - pd.Timedelta(days=1)
        
        train_mask = df[date_col] < train_end
        val_mask = (df[date_col] >= val_start) & (df[date_col] < val_end)
        
        splits.append((train_mask, val_mask))
    
    return splits


# ============================================================================
# FIX 4: Categorical Features (Native LightGBM Encoding)
# ============================================================================

def train_v2(self, df: pd.DataFrame) -> Dict[str, float]:
    """Train with categorical features and custom metric."""
    import lightgbm as lgb
    
    # Identify categorical columns
    cat_cols = ["sku_id", "location_id", "category", "channel", "region"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    # Convert to category dtype
    for col in cat_cols:
        df[col] = df[col].astype("category")
    
    # Prepare features
    self.feature_cols, df = self._prepare_features(df)
    train_df, val_df, test_df = self._time_split(df)
    
    X_train = train_df[self.feature_cols]
    y_train = train_df[self.data_config.target_col]
    X_val = val_df[self.feature_cols]
    y_val = val_df[self.data_config.target_col]
    
    # Train with custom metric
    params = self._get_model_params_v2()
    self.model = lgb.LGBMRegressor(**params)
    
    self.model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=wmape_lgb,
        callbacks=[lgb.early_stopping(100, verbose=False)],
        categorical_feature=cat_cols,
    )
    
    # Predict with clipping
    y_test_pred = np.clip(self.model.predict(test_df[self.feature_cols]), 0, None)
    
    # Calculate metrics (NO MAPE)
    test_metrics = {
        "wmape": np.sum(np.abs(test_df[self.data_config.target_col] - y_test_pred)) / 
                 np.sum(np.abs(test_df[self.data_config.target_col])) * 100,
        "bias_pct": (np.mean(y_test_pred) - np.mean(test_df[self.data_config.target_col])) / 
                    np.mean(test_df[self.data_config.target_col]) * 100,
        "service_level": np.mean(y_test_pred >= test_df[self.data_config.target_col]) * 100,
    }
    
    return test_metrics


# ============================================================================
# FIX 5: Per-Segment Evaluation
# ============================================================================

def evaluate_by_segment(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray):
    """Evaluate WMAPE by ABC class, category, and time period."""
    df = df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["abs_error"] = np.abs(y_true - y_pred)
    
    # By ABC class
    print("\n=== WMAPE by ABC Class ===")
    for abc in ["A", "B", "C"]:
        mask = df["abc_class"] == abc
        wmape = df.loc[mask, "abs_error"].sum() / df.loc[mask, "y_true"].sum() * 100
        print(f"{abc}-class: {wmape:.2f}%")
    
    # By category
    print("\n=== WMAPE by Category ===")
    for cat in df["category"].unique():
        mask = df["category"] == cat
        wmape = df.loc[mask, "abs_error"].sum() / df.loc[mask, "y_true"].sum() * 100
        print(f"{cat}: {wmape:.2f}%")
    
    # By year
    print("\n=== WMAPE by Year ===")
    for year in df["date"].dt.year.unique():
        mask = df["date"].dt.year == year
        wmape = df.loc[mask, "abs_error"].sum() / df.loc[mask, "y_true"].sum() * 100
        print(f"{year}: {wmape:.2f}%")


# ============================================================================
# FIX 6: Feature Importance (SHAP, not gain)
# ============================================================================

def analyze_feature_importance(model, X_sample):
    """Use SHAP for accurate feature importance."""
    import shap
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample.sample(min(1000, len(X_sample))))
    
    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "shap_importance": np.abs(shap_values).mean(axis=0)
    }).sort_values("shap_importance", ascending=False)
    
    print("\n=== Top 20 Features (SHAP) ===")
    print(importance_df.head(20))
    
    return importance_df

# Dependency Fixes Applied to training.py

## Summary
Fixed all potential import errors by adding proper optional dependency handling throughout the codebase.

## Fixed Dependencies

### 1. **LightGBM** ✅
- Added module-level import with try-except
- Fixed callback usage in ensemble training
- Added checks in `_get_lgb_model()` and `_check_libraries()`
- Added fallback to XGBoost in MultiStepForecaster

### 2. **CatBoost** ✅
- Added module-level import with try-except
- Added checks in `_get_catboost_model()` and `_check_libraries()`
- Added fallback to XGBoost in MultiStepForecaster

### 3. **Matplotlib** ✅
- Added module-level import with try-except
- Added checks at the start of all plot methods:
  - `plot_feature_importance()`
  - `plot_actual_vs_predicted()`
  - `plot_residual_analysis()`
  - `plot_error_by_segment()`
  - `plot_forecast_horizon()`
  - `plot_ensemble_comparison()`
  - `plot_horizon_performance()`

### 4. **Seaborn** ✅
- Added module-level import with try-except
- Added check in `plot_residual_analysis()`
- Added check in ForecastVisualizer `__init__()`

### 5. **SciPy** ✅
- Added module-level import with try-except
- Added check in `plot_residual_analysis()` (for stats.probplot)
- Added try-except in `optimize_weights()` method

### 6. **Optuna** ✅
- Added module-level import with try-except
- Improved error handling in `tune_hyperparameters()`
- Separated TPESampler import after main optuna check

### 7. **Kagglehub** ✅
- Added better error handling in main() function
- Separated ImportError from other exceptions
- Provides clear installation instructions

## Installation Commands

If you encounter missing dependencies, install them with:

```bash
# Core ML libraries (required)
pip install numpy pandas scikit-learn xgboost

# Optional ensemble models
pip install lightgbm catboost

# Optional visualization
pip install matplotlib seaborn

# Optional optimization
pip install scipy optuna

# Optional data download
pip install kagglehub
```

## Behavior

- **Without optional dependencies**: Code will run but skip features requiring those libraries
- **With all dependencies**: Full functionality including ensemble models, hyperparameter tuning, and visualizations
- **Graceful degradation**: Missing libraries log warnings but don't crash the program

## Testing

The code now handles these scenarios:
1. ✅ Running with only XGBoost (minimal setup)
2. ✅ Running with XGBoost + LightGBM
3. ✅ Running with full ensemble (XGBoost + LightGBM + CatBoost)
4. ✅ Running without visualization libraries
5. ✅ Running without Optuna (skips hyperparameter tuning)

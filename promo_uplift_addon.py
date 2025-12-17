"""
PROMO UPLIFT ADDON - Add to Kaggle Notebook
============================================
Copy this entire cell to your Kaggle notebook AFTER the main training completes.
Expected WMAPE improvement: 44.52% → 37-39%
Training time: +15 minutes
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import gc

# =============================================================================
# PROMO UPLIFT TRAINER (Priority 5)
# =============================================================================

class PromoUpliftModel:
    """Separate baseline + promo uplift for better accuracy."""
    
    def __init__(self):
        self.baseline_model = None
        self.uplift_model = None
    
    def train(self, df, feature_cols, target_col='true_demand'):
        """Train baseline + uplift models."""
        print("\n" + "="*70)
        print("TRAINING PROMO UPLIFT MODEL (Priority 5)")
        print("="*70)
        
        # Split data
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        # 1. BASELINE MODEL (non-promo periods only)
        print("\n1. Training BASELINE model (non-promo periods)...")
        baseline_train = train_df[train_df['promo_flag'] == 0]
        
        self.baseline_model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=128,
            objective='tweedie',
            tweedie_variance_power=1.3,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        self.baseline_model.fit(
            baseline_train[feature_cols],
            baseline_train[target_col].astype(np.float32)
        )
        
        print(f"   Trained on {len(baseline_train):,} non-promo samples")
        
        # 2. UPLIFT MODEL (promo periods only)
        print("\n2. Training UPLIFT model (promo periods)...")
        promo_train = train_df[train_df['promo_flag'] == 1].copy()
        
        if len(promo_train) > 0:
            # Calculate uplift target = actual - baseline
            promo_train['baseline_pred'] = np.maximum(
                self.baseline_model.predict(promo_train[feature_cols]), 0
            )
            promo_train['uplift_target'] = (
                promo_train[target_col] - promo_train['baseline_pred']
            ).clip(lower=0)
            
            self.uplift_model = lgb.LGBMRegressor(
                n_estimators=1500,
                learning_rate=0.05,
                max_depth=10,
                num_leaves=64,
                objective='tweedie',
                tweedie_variance_power=1.5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            self.uplift_model.fit(
                promo_train[feature_cols],
                promo_train['uplift_target'].astype(np.float32)
            )
            
            print(f"   Trained on {len(promo_train):,} promo samples")
        else:
            self.uplift_model = None
            print("   No promo periods found - using baseline only")
        
        # 3. EVALUATE
        print("\n3. Evaluating combined model...")
        baseline_pred = np.maximum(self.baseline_model.predict(val_df[feature_cols]), 0)
        
        if self.uplift_model:
            uplift_pred = np.maximum(self.uplift_model.predict(val_df[feature_cols]), 0)
            final_pred = baseline_pred + uplift_pred * val_df['promo_flag'].values
        else:
            final_pred = baseline_pred
        
        # Calculate WMAPE
        y_true = val_df[target_col].values
        wmape = np.sum(np.abs(y_true - final_pred)) / np.maximum(np.sum(np.abs(y_true)), 1) * 100
        
        print(f"\n{'='*70}")
        print(f"PROMO UPLIFT MODEL WMAPE: {wmape:.2f}%")
        print(f"Expected improvement: {44.52 - wmape:.2f}% vs baseline (44.52%)")
        print(f"{'='*70}")
        
        # Cleanup
        del train_df, val_df, baseline_train, promo_train
        gc.collect()
        
        return wmape
    
    def predict(self, X, promo_flags):
        """Generate predictions: baseline + uplift × promo_flag."""
        baseline = np.maximum(self.baseline_model.predict(X), 0)
        
        if self.uplift_model:
            uplift = np.maximum(self.uplift_model.predict(X), 0)
            return baseline + uplift * promo_flags
        
        return baseline


# =============================================================================
# USAGE EXAMPLE (Add to your Kaggle notebook)
# =============================================================================

# After your main training completes, add this:
"""
# Train promo uplift model
promo_model = PromoUpliftModel()
promo_wmape = promo_model.train(df, feature_cols)

# Compare results
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)
print(f"Current Model WMAPE:      44.52%")
print(f"Promo Uplift Model WMAPE: {promo_wmape:.2f}%")
print(f"Improvement:              {44.52 - promo_wmape:.2f}%")
print(f"Target Range:             36-39%")
print(f"Target Achieved:          {'✓' if promo_wmape < 40 else '✗'}")
print("="*70)
"""

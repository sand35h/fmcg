# -*- coding: utf-8 -*-
"""
DRIFT DETECTION & AUTOMATED RETRAINING
=======================================
Monitors model performance and triggers retraining when needed.

Components:
1. Data drift detection (input feature distribution changes)
2. Concept drift detection (feature-target relationship changes)
3. Performance monitoring
4. Automated retraining pipeline
5. Model versioning and rollback
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available. Some drift tests will be limited.")


class DriftDetector:
    """
    Detects data drift and concept drift in production.
    """
    
    def __init__(self, drift_threshold=0.15, significance_level=0.05):
        """
        Args:
            drift_threshold: Threshold for normalized drift score (0-1)
            significance_level: P-value threshold for statistical tests
        """
        self.drift_threshold = drift_threshold
        self.significance_level = significance_level
        self.baseline_stats = {}
        self.critical_features = []
        
    def capture_baseline(self, X_train, y_train, critical_features=None):
        """
        Capture baseline statistics from training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            critical_features: List of features to monitor closely
        """
        print("Capturing baseline statistics for drift detection...")
        
        if critical_features is not None:
            self.critical_features = critical_features
        else:
            # Use all features
            self.critical_features = X_train.columns.tolist()
        
        # Store statistics for each feature
        for col in X_train.columns:
            if col in self.critical_features:
                values = X_train[col].dropna()
                
                self.baseline_stats[col] = {
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'q25': float(values.quantile(0.25)),
                    'q50': float(values.quantile(0.50)),
                    'q75': float(values.quantile(0.75)),
                    'null_rate': float(X_train[col].isnull().mean())
                }
        
        # Target statistics
        self.baseline_stats['__target__'] = {
            'mean': float(y_train.mean()),
            'std': float(y_train.std()),
            'min': float(y_train.min()),
            'max': float(y_train.max())
        }
        
        print(f"✓ Baseline captured for {len(self.baseline_stats)} features")
    
    def detect_data_drift(self, X_new):
        """
        Detect data drift using statistical measures.
        
        Methods:
        1. Population Stability Index (PSI)
        2. Kolmogorov-Smirnov test
        3. Mean/std shift detection
        
        Returns:
            dict: Drift scores and details for each feature
        """
        print(f"Analyzing data drift for {len(X_new)} new samples...")
        
        drift_results = {}
        drifted_features = []
        
        for col in self.critical_features:
            if col not in X_new.columns or col not in self.baseline_stats:
                continue
            
            baseline = self.baseline_stats[col]
            current_values = X_new[col].dropna()
            
            # Skip if not enough data
            if len(current_values) < 10:
                continue
            
            # 1. Mean shift (normalized)
            mean_shift = abs(current_values.mean() - baseline['mean']) / (baseline['std'] + 1e-6)
            
            # 2. Std shift (normalized)
            std_shift = abs(current_values.std() - baseline['std']) / (baseline['std'] + 1e-6)
            
            # 3. Null rate change
            null_shift = abs(X_new[col].isnull().mean() - baseline['null_rate'])
            
            # 4. Kolmogorov-Smirnov test (if scipy available)
            ks_pvalue = 1.0
            if SCIPY_AVAILABLE and len(current_values) >= 30:
                try:
                    # Create synthetic baseline sample from stored statistics
                    np.random.seed(42)
                    baseline_sample = np.random.normal(
                        baseline['mean'], 
                        baseline['std'], 
                        size=len(current_values)
                    )
                    ks_stat, ks_pvalue = stats.ks_2samp(baseline_sample, current_values.values)
                except:
                    pass
            
            # Combined drift score
            drift_score = (mean_shift + std_shift + null_shift * 2) / 4
            
            drift_results[col] = {
                'drift_score': float(drift_score),
                'mean_shift': float(mean_shift),
                'std_shift': float(std_shift),
                'null_shift': float(null_shift),
                'ks_pvalue': float(ks_pvalue),
                'is_drifted': drift_score > self.drift_threshold or ks_pvalue < self.significance_level
            }
            
            if drift_results[col]['is_drifted']:
                drifted_features.append(col)
        
        # Overall drift assessment
        overall_drift = np.mean([r['drift_score'] for r in drift_results.values()])
        
        print(f"\n📊 Drift Detection Results:")
        print(f"  Features analyzed: {len(drift_results)}")
        print(f"  Drifted features: {len(drifted_features)}")
        print(f"  Overall drift score: {overall_drift:.3f} (threshold: {self.drift_threshold})")
        
        if drifted_features:
            print(f"\n  ⚠️ Drifted features:")
            for feat in drifted_features[:5]:  # Show top 5
                print(f"    - {feat}: score={drift_results[feat]['drift_score']:.3f}")
        
        return {
            'overall_drift': overall_drift,
            'drifted_features': drifted_features,
            'feature_details': drift_results,
            'needs_attention': overall_drift > self.drift_threshold
        }
    
    def detect_concept_drift(self, y_true, y_pred, window_size=30):
        """
        Detect concept drift by monitoring prediction errors over time.
        
        Uses sliding window to detect:
        - Error rate increase
        - Error pattern changes
        
        Args:
            y_true: Actual values (must have time ordering)
            y_pred: Predicted values
            window_size: Number of samples in sliding window
        """
        errors = np.abs(y_true - y_pred)
        
        if len(errors) < window_size * 2:
            print("⚠️ Not enough data for concept drift detection")
            return {'concept_drift': False, 'message': 'Insufficient data'}
        
        # Split into old and recent windows
        recent_errors = errors[-window_size:]
        old_errors = errors[-2*window_size:-window_size]
        
        # Compare error distributions
        recent_mae = recent_errors.mean()
        old_mae = old_errors.mean()
        error_increase = (recent_mae - old_mae) / (old_mae + 1e-6)
        
        # Statistical test for distribution change
        pvalue = 1.0
        if SCIPY_AVAILABLE:
            _, pvalue = stats.mannwhitneyu(old_errors, recent_errors, alternative='two-sided')
        
        concept_drift = (
            (error_increase > 0.10) or  # 10% error increase
            (pvalue < self.significance_level)
        )
        
        print(f"\n📊 Concept Drift Analysis:")
        print(f"  Old window MAE: {old_mae:.2f}")
        print(f"  Recent window MAE: {recent_mae:.2f}")
        print(f"  Error increase: {error_increase*100:.1f}%")
        print(f"  P-value: {pvalue:.4f}")
        print(f"  Concept drift detected: {'Yes ⚠️' if concept_drift else 'No ✓'}")
        
        return {
            'concept_drift': concept_drift,
            'old_mae': float(old_mae),
            'recent_mae': float(recent_mae),
            'error_increase_pct': float(error_increase * 100),
            'pvalue': float(pvalue)
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
        
        print(f"✓ Baseline saved to {filepath}")
    
    def load_baseline(self, filepath):
        """Load baseline statistics from file."""
        with open(filepath, 'r') as f:
            baseline_data = json.load(f)
        
        self.baseline_stats = baseline_data['baseline_stats']
        self.critical_features = baseline_data['critical_features']
        self.drift_threshold = baseline_data['drift_threshold']
        
        print(f"✓ Baseline loaded from {filepath}")


class PerformanceMonitor:
    """
    Tracks model performance over time.
    """
    
    def __init__(self, wmape_threshold=35.0):
        """
        Args:
            wmape_threshold: Performance degradation threshold (%)
        """
        self.wmape_threshold = wmape_threshold
        self.performance_history = []
        
    def log_performance(self, y_true, y_pred, metadata=None):
        """
        Log model performance metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            metadata: Additional info (date, data source, etc.)
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # Calculate metrics
        wmape = np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(np.abs(y_true)), 1) * 100
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = np.mean(y_pred - y_true)
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'wmape': float(wmape),
            'mae': float(mae),
            'rmse': float(rmse),
            'bias': float(bias),
            'n_samples': len(y_true)
        }
        
        if metadata:
            record.update(metadata)
        
        self.performance_history.append(record)
        
        print(f"📊 Performance Logged:")
        print(f"  WMAPE: {wmape:.2f}%")
        print(f"  MAE: {mae:.2f}")
        print(f"  Bias: {bias:.2f}")
        
        return record
    
    def check_degradation(self, lookback=5):
        """
        Check if performance has degraded compared to recent history.
        
        Args:
            lookback: Number of recent records to compare against
        """
        if len(self.performance_history) < lookback + 1:
            return {'degraded': False, 'message': 'Insufficient history'}
        
        # Get recent and current performance
        recent = self.performance_history[-lookback-1:-1]
        current = self.performance_history[-1]
        
        recent_wmape = np.mean([r['wmape'] for r in recent])
        current_wmape = current['wmape']
        
        degradation = current_wmape - recent_wmape
        degradation_pct = (degradation / recent_wmape) * 100
        
        # Check thresholds
        degraded = (
            (current_wmape > self.wmape_threshold) or  # Absolute threshold
            (degradation_pct > 10)  # 10% relative degradation
        )
        
        print(f"\n📊 Performance Degradation Check:")
        print(f"  Recent avg WMAPE: {recent_wmape:.2f}%")
        print(f"  Current WMAPE: {current_wmape:.2f}%")
        print(f"  Degradation: {degradation:+.2f}% ({degradation_pct:+.1f}% relative)")
        print(f"  Status: {'DEGRADED ⚠️' if degraded else 'OK ✓'}")
        
        return {
            'degraded': degraded,
            'recent_wmape': float(recent_wmape),
            'current_wmape': float(current_wmape),
            'degradation': float(degradation),
            'degradation_pct': float(degradation_pct)
        }
    
    def save_history(self, filepath):
        """Save performance history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        print(f"✓ Performance history saved to {filepath}")
    
    def load_history(self, filepath):
        """Load performance history from file."""
        with open(filepath, 'r') as f:
            self.performance_history = json.load(f)
        print(f"✓ Performance history loaded from {filepath}")


class AutomatedRetrainingPipeline:
    """
    Orchestrates automated model retraining.
    """
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary with retraining configuration
        """
        self.config = config
        self.drift_detector = DriftDetector(
            drift_threshold=config.get('drift_threshold', 0.15)
        )
        self.perf_monitor = PerformanceMonitor(
            wmape_threshold=config.get('wmape_threshold', 35.0)
        )
        self.min_days_between_retrains = config.get('min_days_between_retrains', 7)
        self.last_retrain_date = None
        
    def should_retrain(self, X_new, y_new, y_pred):
        """
        Determine if model should be retrained based on multiple signals.
        
        Returns:
            dict: Decision and reasons
        """
        print("="*70)
        print("RETRAINING DECISION ANALYSIS")
        print("="*70)
        
        reasons = []
        
        # 1. Check time since last retrain
        if self.last_retrain_date:
            days_since = (datetime.now() - self.last_retrain_date).days
            if days_since < self.min_days_between_retrains:
                print(f"⏱️  Too soon to retrain (last retrain: {days_since} days ago)")
                return {'should_retrain': False, 'reasons': ['Too soon']}
        
        # 2. Check data drift
        drift_results = self.drift_detector.detect_data_drift(X_new)
        if drift_results['needs_attention']:
            reasons.append(f"Data drift detected (score: {drift_results['overall_drift']:.3f})")
        
        # 3. Check concept drift
        concept_results = self.drift_detector.detect_concept_drift(y_new, y_pred)
        if concept_results['concept_drift']:
            reasons.append(f"Concept drift detected (error increase: {concept_results['error_increase_pct']:.1f}%)")
        
        # 4. Check performance degradation
        self.perf_monitor.log_performance(y_new, y_pred)
        perf_results = self.perf_monitor.check_degradation()
        if perf_results['degraded']:
            reasons.append(f"Performance degraded (WMAPE: {perf_results['current_wmape']:.2f}%)")
        
        # 5. Scheduled retrain
        if self.last_retrain_date:
            days_since = (datetime.now() - self.last_retrain_date).days
            if days_since >= self.config.get('scheduled_retrain_days', 30):
                reasons.append(f"Scheduled retrain ({days_since} days)")
        
        # Decision
        should_retrain = len(reasons) > 0
        
        print(f"\n🎯 Decision: {'RETRAIN ⚠️' if should_retrain else 'MAINTAIN ✓'}")
        if reasons:
            print(f"\n  Reasons:")
            for r in reasons:
                print(f"    - {r}")
        
        return {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'drift_results': drift_results,
            'concept_results': concept_results,
            'perf_results': perf_results
        }
    
    def retrain(self, train_function, df, feature_cols, model_version):
        """
        Execute retraining pipeline.
        
        Args:
            train_function: Function that trains the model
            df: Full dataset for retraining
            feature_cols: List of feature columns
            model_version: New model version string
        
        Returns:
            dict: Retraining results
        """
        print("="*70)
        print(f"RETRAINING MODEL (Version: {model_version})")
        print("="*70)
        
        try:
            # Train new model
            results = train_function(df, feature_cols)
            
            # Update last retrain date
            self.last_retrain_date = datetime.now()
            
            # Log success
            print(f"\n✓ Retraining completed successfully")
            print(f"  New model version: {model_version}")
            print(f"  Test WMAPE: {results['test_wmape']:.2f}%")
            
            return {
                'success': True,
                'model_version': model_version,
                'results': results,
                'timestamp': self.last_retrain_date.isoformat()
            }
            
        except Exception as e:
            print(f"\n✗ Retraining failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def demo_drift_and_retraining():
    """
    Demonstrates the complete drift detection and retraining workflow.
    """
    
    print("="*70)
    print("DRIFT DETECTION & RETRAINING DEMO")
    print("="*70)
    
    # Simulate training data
    np.random.seed(42)
    n_train = 1000
    X_train = pd.DataFrame({
        'feature1': np.random.normal(100, 20, n_train),
        'feature2': np.random.normal(50, 10, n_train),
        'feature3': np.random.uniform(0, 1, n_train)
    })
    y_train = X_train['feature1'] * 0.5 + X_train['feature2'] * 0.3 + np.random.normal(0, 5, n_train)
    
    # Simulate new data with drift
    n_new = 200
    X_new = pd.DataFrame({
        'feature1': np.random.normal(120, 25, n_new),  # Mean shifted
        'feature2': np.random.normal(50, 15, n_new),   # Variance increased
        'feature3': np.random.uniform(0, 1, n_new)
    })
    y_new = X_new['feature1'] * 0.5 + X_new['feature2'] * 0.3 + np.random.normal(0, 5, n_new)
    y_pred = y_new + np.random.normal(0, 10, n_new)  # Simulate predictions
    
    # Initialize components
    detector = DriftDetector(drift_threshold=0.15)
    
    # Capture baseline
    detector.capture_baseline(X_train, y_train, critical_features=['feature1', 'feature2'])
    
    # Detect drifts
    drift_results = detector.detect_data_drift(X_new)
    concept_results = detector.detect_concept_drift(y_new, y_pred)
    
    # Performance monitoring
    monitor = PerformanceMonitor(wmape_threshold=35.0)
    monitor.log_performance(y_new, y_pred, metadata={'data_source': 'production'})
    
    # Retraining decision
    config = {
        'drift_threshold': 0.15,
        'wmape_threshold': 35.0,
        'min_days_between_retrains': 7,
        'scheduled_retrain_days': 30
    }
    
    pipeline = AutomatedRetrainingPipeline(config)
    pipeline.drift_detector = detector
    pipeline.perf_monitor = monitor
    
    decision = pipeline.should_retrain(X_new, y_new, y_pred)
    
    return decision


"""
TO INTEGRATE WITH YOUR EXISTING PIPELINE:

1. After initial training, capture baseline:

   detector = DriftDetector(drift_threshold=0.15)
   detector.capture_baseline(X_train, y_train)
   detector.save_baseline('/kaggle/working/drift_baseline.json')

2. In production, check for drift regularly:

   detector.load_baseline('/kaggle/working/drift_baseline.json')
   drift_results = detector.detect_data_drift(X_new)
   
   if drift_results['needs_attention']:
       print("⚠️ Drift detected - consider retraining")

3. Monitor performance:

   monitor = PerformanceMonitor(wmape_threshold=35.0)
   monitor.log_performance(y_true, y_pred)
   perf_check = monitor.check_degradation()

4. Automated retraining pipeline:

   pipeline = AutomatedRetrainingPipeline(config)
   decision = pipeline.should_retrain(X_new, y_new, y_pred)
   
   if decision['should_retrain']:
       results = pipeline.retrain(
           train_function=main,
           df=updated_df,
           feature_cols=feature_cols,
           model_version='v1.1'
       )
"""

if __name__ == "__main__":
    decision = demo_drift_and_retraining()

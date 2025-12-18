# -*- coding: utf-8 -*-
"""
SHAP EXPLAINABILITY MODULE
==========================
Provides model interpretability using SHAP (SHapley Additive exPlanations).

Features:
1. Global feature importance (which features matter most overall)
2. Local explanations (why did the model predict X for this specific instance)
3. Feature interaction analysis
4. Visualization exports for reports
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️ SHAP not installed. Install with: pip install shap")
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """
    Provides model explanations using SHAP values.
    """
    
    def __init__(self, model, feature_cols):
        """
        Args:
            model: Trained model (LightGBM)
            feature_cols: List of feature column names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required. Install with: pip install shap")
        
        self.model = model
        self.feature_cols = feature_cols
        self.explainer = None
        self.shap_values = None
        
    def fit_explainer(self, X_background):
        """
        Initialize SHAP explainer with background dataset.
        
        Args:
            X_background: Sample of training data (typically 100-1000 rows)
                         for TreeExplainer background
        """
        print("Initializing SHAP explainer...")
        
        # For tree models (LightGBM), use TreeExplainer
        self.explainer = shap.TreeExplainer(
            self.model,
            data=X_background,
            feature_perturbation="tree_path_dependent"
        )
        
        print(f"✓ SHAP explainer ready (background size: {len(X_background)})")
    
    def explain(self, X, check_additivity=False):
        """
        Calculate SHAP values for given data.
        
        Args:
            X: Data to explain (DataFrame or numpy array)
            check_additivity: Whether to verify SHAP values sum to prediction
        
        Returns:
            SHAP values (same shape as X)
        """
        if self.explainer is None:
            raise ValueError("Call fit_explainer() first")
        
        print(f"Calculating SHAP values for {len(X)} instances...")
        
        shap_values = self.explainer.shap_values(
            X, 
            check_additivity=check_additivity
        )
        
        self.shap_values = shap_values
        print("✓ SHAP values calculated")
        
        return shap_values
    
    def get_global_importance(self, shap_values=None, top_n=15):
        """
        Calculate global feature importance from SHAP values.
        
        Returns:
            DataFrame with features ranked by importance
        """
        if shap_values is None:
            if self.shap_values is None:
                raise ValueError("No SHAP values available. Call explain() first.")
            shap_values = self.shap_values
        
        # Calculate mean absolute SHAP value for each feature
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': np.abs(shap_values).mean(axis=0)
        })
        
        importance = importance.sort_values('importance', ascending=False).reset_index(drop=True)
        
        print(f"\n📊 Top {top_n} Most Important Features (Global SHAP):")
        print(importance.head(top_n).to_string(index=False))
        
        return importance
    
    def explain_prediction(self, X_instance, instance_idx=0):
        """
        Explain a single prediction.
        
        Args:
            X_instance: Single instance or DataFrame with multiple instances
            instance_idx: Which instance to explain (if X_instance has multiple rows)
        
        Returns:
            DataFrame with feature contributions for this prediction
        """
        if self.explainer is None:
            raise ValueError("Call fit_explainer() first")
        
        # Ensure X_instance is 2D
        if isinstance(X_instance, pd.Series):
            X_instance = X_instance.to_frame().T
        elif len(X_instance.shape) == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # Calculate SHAP values for this instance
        shap_vals = self.explainer.shap_values(X_instance)
        
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[instance_idx]
        
        # Get base value (expected value)
        base_value = self.explainer.expected_value
        
        # Create explanation DataFrame
        explanation = pd.DataFrame({
            'feature': self.feature_cols,
            'feature_value': X_instance.iloc[instance_idx] if isinstance(X_instance, pd.DataFrame) else X_instance[instance_idx],
            'shap_value': shap_vals,
            'abs_shap': np.abs(shap_vals)
        })
        
        explanation = explanation.sort_values('abs_shap', ascending=False).reset_index(drop=True)
        
        # Calculate prediction
        prediction = base_value + shap_vals.sum()
        
        print(f"\n🔍 Prediction Explanation:")
        print(f"  Base value (average): {base_value:.2f}")
        print(f"  SHAP contributions: {shap_vals.sum():.2f}")
        print(f"  Final prediction: {prediction:.2f}")
        print(f"\n  Top 5 Contributing Features:")
        print(explanation[['feature', 'feature_value', 'shap_value']].head(5).to_string(index=False))
        
        return explanation, base_value, prediction
    
    def plot_summary(self, X, shap_values=None, max_display=15, 
                     output_path='shap_summary.png'):
        """
        Create SHAP summary plot (beeswarm plot).
        Shows feature importance and feature effects.
        """
        if shap_values is None:
            if self.shap_values is None:
                print("Calculating SHAP values for plotting...")
                shap_values = self.explain(X)
            else:
                shap_values = self.shap_values
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X, 
            feature_names=self.feature_cols,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Summary plot saved to {output_path}")
    
    def plot_waterfall(self, X_instance, instance_idx=0, 
                       output_path='shap_waterfall.png'):
        """
        Create waterfall plot for a single prediction.
        Shows how each feature contributes to pushing the prediction
        from base value to final value.
        """
        if self.explainer is None:
            raise ValueError("Call fit_explainer() first")
        
        # Ensure X_instance is 2D
        if isinstance(X_instance, pd.Series):
            X_instance = X_instance.to_frame().T
        
        # Calculate SHAP values
        shap_vals = self.explainer.shap_values(X_instance)
        
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[instance_idx]
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=self.explainer.expected_value,
            data=X_instance.iloc[instance_idx].values if isinstance(X_instance, pd.DataFrame) else X_instance[instance_idx],
            feature_names=self.feature_cols
        )
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(explanation, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Waterfall plot saved to {output_path}")
    
    def plot_force(self, X_instance, instance_idx=0, 
                   output_path='shap_force.png'):
        """
        Create force plot showing positive and negative contributions.
        """
        if self.explainer is None:
            raise ValueError("Call fit_explainer() first")
        
        # Ensure X_instance is 2D
        if isinstance(X_instance, pd.Series):
            X_instance = X_instance.to_frame().T
        
        # Calculate SHAP values
        shap_vals = self.explainer.shap_values(X_instance)
        
        if len(shap_vals.shape) > 1:
            shap_vals = shap_vals[instance_idx]
        
        # Create force plot
        shap.force_plot(
            self.explainer.expected_value,
            shap_vals,
            X_instance.iloc[instance_idx] if isinstance(X_instance, pd.DataFrame) else X_instance[instance_idx],
            feature_names=self.feature_cols,
            matplotlib=True,
            show=False
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Force plot saved to {output_path}")
    
    def plot_dependence(self, feature_name, X, shap_values=None, 
                       interaction_feature=None, output_path=None):
        """
        Plot how a single feature affects predictions.
        Optionally show interaction with another feature.
        """
        if shap_values is None:
            if self.shap_values is None:
                print("Calculating SHAP values for plotting...")
                shap_values = self.explain(X)
            else:
                shap_values = self.shap_values
        
        if feature_name not in self.feature_cols:
            raise ValueError(f"Feature '{feature_name}' not found")
        
        feature_idx = self.feature_cols.index(feature_name)
        
        if output_path is None:
            output_path = f'shap_dependence_{feature_name}.png'
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            X,
            feature_names=self.feature_cols,
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Dependence plot saved to {output_path}")


# =============================================================================
# INTEGRATION HELPER FUNCTIONS
# =============================================================================

def create_explainability_report(model, X_train, X_test, y_test, 
                                feature_cols, output_dir='./shap_output'):
    """
    Generate a complete explainability report with all visualizations.
    
    This is the main function to call after training your model.
    """
    
    print("="*70)
    print("GENERATING EXPLAINABILITY REPORT")
    print("="*70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize explainer
    explainer = SHAPExplainer(model, feature_cols)
    
    # Use a sample of training data as background (for efficiency)
    n_background = min(100, len(X_train))
    X_background = X_train.sample(n=n_background, random_state=42)
    
    explainer.fit_explainer(X_background)
    
    # Calculate SHAP values for test set (use sample for large datasets)
    n_explain = min(1000, len(X_test))
    X_explain = X_test.sample(n=n_explain, random_state=42)
    
    shap_values = explainer.explain(X_explain)
    
    # 1. Global feature importance
    importance_df = explainer.get_global_importance(shap_values)
    importance_df.to_csv(output_path / 'feature_importance_shap.csv', index=False)
    
    # 2. Summary plot (global overview)
    explainer.plot_summary(
        X_explain, 
        shap_values, 
        output_path=output_path / 'shap_summary.png'
    )
    
    # 3. Explain specific predictions (best, worst, median)
    predictions = model.predict(X_explain)
    errors = np.abs(predictions - y_test.iloc[:n_explain].values)
    
    # Best prediction (lowest error)
    best_idx = errors.argmin()
    explainer.plot_waterfall(
        X_explain.iloc[best_idx:best_idx+1],
        output_path=output_path / 'shap_waterfall_best.png'
    )
    
    # Worst prediction (highest error)
    worst_idx = errors.argmax()
    explainer.plot_waterfall(
        X_explain.iloc[worst_idx:worst_idx+1],
        output_path=output_path / 'shap_waterfall_worst.png'
    )
    
    # 4. Dependence plots for top 3 features
    top_features = importance_df.head(3)['feature'].tolist()
    for feat in top_features:
        if feat in X_explain.columns:
            explainer.plot_dependence(
                feat,
                X_explain,
                shap_values,
                output_path=output_path / f'shap_dependence_{feat}.png'
            )
    
    # 5. Create summary text report
    with open(output_path / 'explainability_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("SHAP EXPLAINABILITY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("Top 15 Most Important Features:\n")
        f.write(importance_df.head(15).to_string(index=False))
        f.write("\n\n")
        
        f.write("Best Prediction Explanation:\n")
        explanation, base, pred = explainer.explain_prediction(X_explain, best_idx)
        f.write(f"  Actual: {y_test.iloc[best_idx]:.2f}\n")
        f.write(f"  Predicted: {pred:.2f}\n")
        f.write(f"  Error: {errors[best_idx]:.2f}\n")
        f.write("\n  Top 5 Features:\n")
        f.write(explanation[['feature', 'feature_value', 'shap_value']].head(5).to_string(index=False))
        f.write("\n\n")
        
        f.write("Worst Prediction Explanation:\n")
        explanation, base, pred = explainer.explain_prediction(X_explain, worst_idx)
        f.write(f"  Actual: {y_test.iloc[worst_idx]:.2f}\n")
        f.write(f"  Predicted: {pred:.2f}\n")
        f.write(f"  Error: {errors[worst_idx]:.2f}\n")
        f.write("\n  Top 5 Features:\n")
        f.write(explanation[['feature', 'feature_value', 'shap_value']].head(5).to_string(index=False))
    
    print(f"\n✓ Complete explainability report saved to {output_path}")
    print(f"  - Feature importance CSV")
    print(f"  - Summary plot (beeswarm)")
    print(f"  - Waterfall plots (best/worst predictions)")
    print(f"  - Dependence plots (top 3 features)")
    print(f"  - Text summary report")
    
    return explainer, importance_df


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
TO ADD TO YOUR EXISTING PIPELINE:

After training your model, add this:

# Generate explainability report
explainer, importance_df = create_explainability_report(
    model=train_results['model'],
    X_train=X_train,
    X_test=X_test,
    y_test=y_test,
    feature_cols=feature_cols,
    output_dir='/kaggle/working/shap_output'
)

# For real-time explanations in production:
# 1. Save the explainer
import joblib
joblib.dump(explainer, '/kaggle/working/shap_explainer.pkl')

# 2. Later, load and explain new predictions
explainer = joblib.load('/kaggle/working/shap_explainer.pkl')
explanation, base, pred = explainer.explain_prediction(new_instance)
"""

if __name__ == "__main__":
    print("SHAP Explainability Module")
    print("Install SHAP with: pip install shap")
    print("\nSee usage example at the end of this file for integration.")

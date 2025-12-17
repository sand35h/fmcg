"""
FMCG Dataset Exploratory Data Analysis (EDA) Script
====================================================

This script performs comprehensive analysis on FMCG forecasting datasets to:
1. Identify ABC classification (if not present)
2. Check data availability and quality
3. Assess memory requirements for Kaggle
4. Recommend optimal metrics and training strategy

Usage:
    python eda_analysis.py --data_path /path/to/data

Outputs:
    - eda_report.txt: Comprehensive analysis report
    - eda_summary.json: Machine-readable summary for pipeline configuration
"""

import pandas as pd
import numpy as np
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class EDAConfig:
    """Configuration for EDA analysis"""
    # Memory thresholds (Kaggle has ~13GB, leave buffer)
    KAGGLE_MEMORY_LIMIT_GB = 10.0  # Safe limit
    
    # ABC Classification thresholds (Pareto principle)
    ABC_THRESHOLDS = {
        'A': 0.80,  # Top 80% of revenue
        'B': 0.95,  # Next 15% of revenue
        'C': 1.00   # Remaining 5%
    }
    
    # Data quality thresholds
    MIN_HIST_DAYS = 365  # Minimum history for training
    MAX_MISSING_PCT = 20  # Max acceptable missing data %
    
    # Feature importance threshold
    MIN_FEATURE_IMPORTANCE = 0.001


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def sizeof_fmt(num, suffix='B'):
    """Human-readable file size"""
    for unit in ['', 'K', 'M', 'G']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}T{suffix}"


def memory_usage_mb(df):
    """Calculate DataFrame memory usage in MB"""
    return df.memory_usage(deep=True).sum() / 1024**2


# ==============================================================================
# DATA LOADING
# ==============================================================================

def discover_data_files(data_path):
    """Discover all available data files in the directory"""
    print(f"\n{'='*70}")
    print("STEP 1: DISCOVERING DATA FILES")
    print('='*70)
    
    data_path = Path(data_path)
    discovered = {
        'timeseries': None,
        'sku_master': None,
        'location_master': None,
        'festival_calendar': None,
        'external_shocks': None,
        'macro': None
    }
    
    # Search for files
    for file in data_path.glob('**/*'):
        if file.is_file():
            name_lower = file.name.lower()
            
            # Timeseries data
            if any(x in name_lower for x in ['daily', 'timeseries', 'sales']):
                discovered['timeseries'] = file
            
            # Master data
            elif 'sku' in name_lower and 'master' in name_lower:
                discovered['sku_master'] = file
            elif 'location' in name_lower and 'master' in name_lower:
                discovered['location_master'] = file
            
            # External data
            elif 'festival' in name_lower or 'holiday' in name_lower:
                discovered['festival_calendar'] = file
            elif 'shock' in name_lower or 'event' in name_lower:
                discovered['external_shocks'] = file
            elif 'macro' in name_lower or 'economic' in name_lower:
                discovered['macro'] = file
    
    # Report findings
    print("\n📁 Discovered Files:")
    for key, filepath in discovered.items():
        status = "✅ FOUND" if filepath else "❌ MISSING"
        filepath_str = str(filepath) if filepath else "N/A"
        print(f"  {key:20s} : {status:10s} | {filepath_str}")
    
    return discovered


def load_data_smart(filepath):
    """Load data from CSV or Parquet with memory optimization"""
    if filepath is None:
        return None
    
    ext = filepath.suffix.lower()
    
    try:
        if ext == '.csv':
            # Load with dtype inference and date parsing
            df = pd.read_csv(filepath, parse_dates=['date'], low_memory=False)
        elif ext == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            print(f"⚠️  Unknown file type: {ext}")
            return None
        
        print(f"  Loaded: {filepath.name} ({sizeof_fmt(filepath.stat().st_size)}) → {memory_usage_mb(df):.1f} MB in memory")
        return df
    
    except Exception as e:
        print(f"  ❌ Error loading {filepath.name}: {e}")
        return None


# ==============================================================================
# ABC CLASSIFICATION
# ==============================================================================

def calculate_abc_classification(df, sku_col='sku_id', demand_col='actual_demand', price_col='price'):
    """
    Calculate ABC classification for SKUs based on revenue (Pareto principle)
    
    A-class: Top 20% SKUs contributing to 80% revenue
    B-class: Next 30% SKUs contributing to 15% revenue
    C-class: Remaining 50% SKUs contributing to 5% revenue
    """
    print(f"\n{'='*70}")
    print("STEP 2: ABC CLASSIFICATION ANALYSIS")
    print('='*70)
    
    # Calculate revenue per SKU
    if price_col in df.columns:
        df['revenue'] = df[demand_col] * df[price_col]
        revenue_col = 'revenue'
    else:
        print("⚠️  Price column not found. Using demand as proxy for revenue.")
        revenue_col = demand_col
    
    sku_revenue = df.groupby(sku_col)[revenue_col].sum().sort_values(ascending=False).reset_index()
    sku_revenue['cumulative_revenue'] = sku_revenue[revenue_col].cumsum()
    sku_revenue['revenue_pct'] = sku_revenue['cumulative_revenue'] / sku_revenue[revenue_col].sum()
    sku_revenue['rank_pct'] = (sku_revenue.index + 1) / len(sku_revenue)
    
    # Assign ABC class
    sku_revenue['abc_class'] = 'C'
    sku_revenue.loc[sku_revenue['revenue_pct'] <= EDAConfig.ABC_THRESHOLDS['A'], 'abc_class'] = 'A'
    sku_revenue.loc[
        (sku_revenue['revenue_pct'] > EDAConfig.ABC_THRESHOLDS['A']) &
        (sku_revenue['revenue_pct'] <= EDAConfig.ABC_THRESHOLDS['B']),
        'abc_class'
    ] = 'B'
    
    # Summary
    abc_summary = sku_revenue.groupby('abc_class').agg({
        sku_col: 'count',
        revenue_col: 'sum'
    }).rename(columns={sku_col: 'sku_count', revenue_col: 'total_revenue'})
    abc_summary['sku_pct'] = abc_summary['sku_count'] / abc_summary['sku_count'].sum() * 100
    abc_summary['revenue_pct'] = abc_summary['total_revenue'] / abc_summary['total_revenue'].sum() * 100
    
    print("\n📊 ABC Classification Results:")
    print(abc_summary.to_string())
    
    # Merge back to original dataframe
    sku_abc = sku_revenue[[sku_col, 'abc_class']]
    
    return sku_abc, abc_summary


# ==============================================================================
# DATA QUALITY ASSESSMENT
# ==============================================================================

def assess_data_quality(df):
    """Comprehensive data quality assessment"""
    print(f"\n{'='*70}")
    print("STEP 3: DATA QUALITY ASSESSMENT")
    print('='*70)
    
    report = {}
    
    # Basic info
    report['total_rows'] = len(df)
    report['total_columns'] = len(df.columns)
    report['memory_mb'] = memory_usage_mb(df)
    
    # Date range
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        report['date_range'] = {
            'start': df['date'].min().strftime('%Y-%m-%d'),
            'end': df['date'].max().strftime('%Y-%m-%d'),
            'days': (df['date'].max() - df['date'].min()).days
        }
        
        # Check for gaps
        date_range_full = pd.date_range(df['date'].min(), df['date'].max(), freq='D')
        missing_dates = set(date_range_full) - set(df['date'].unique())
        report['missing_dates_count'] = len(missing_dates)
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report['missing_data'] = missing_pct[missing_pct > 0].to_dict()
    
    # Duplicates
    duplicates = df.duplicated().sum()
    report['duplicate_rows'] = int(duplicates)
    report['duplicate_pct'] = round(duplicates / len(df) * 100, 2)
    
    # Key columns check
    critical_cols = ['sku_id', 'location_id', 'date', 'actual_demand']
    report['critical_columns_present'] = [col for col in critical_cols if col in df.columns]
    report['critical_columns_missing'] = [col for col in critical_cols if col not in df.columns]
    
    # Print report
    print(f"\n📋 Basic Info:")
    print(f"  Total Rows: {report['total_rows']:,}")
    print(f"  Total Columns: {report['total_columns']}")
    print(f"  Memory Usage: {report['memory_mb']:.2f} MB")
    
    if 'date_range' in report:
        print(f"\n📅 Date Range:")
        print(f"  Start: {report['date_range']['start']}")
        print(f"  End: {report['date_range']['end']}")
        print(f"  Days: {report['date_range']['days']}")
        print(f"  Missing Dates: {report['missing_dates_count']}")
    
    print(f"\n❗ Missing Data (>0%):")
    if report['missing_data']:
        for col, pct in report['missing_data'].items():
            status = "⚠️" if pct > EDAConfig.MAX_MISSING_PCT else "✓"
            print(f"  {status} {col:30s}: {pct:6.2f}%")
    else:
        print("  ✅ No missing data!")
    
    print(f"\n🔁 Duplicates:")
    print(f"  Count: {report['duplicate_rows']:,} ({report['duplicate_pct']:.2f}%)")
    
    print(f"\n🔑 Critical Columns:")
    print(f"  Present: {', '.join(report['critical_columns_present'])}")
    if report['critical_columns_missing']:
        print(f"  ⚠️ Missing: {', '.join(report['critical_columns_missing'])}")
    
    return report


# ==============================================================================
# KAGGLE MEMORY ASSESSMENT
# ==============================================================================

def assess_kaggle_compatibility(df, sku_abc=None):
    """Assess whether dataset fits Kaggle memory constraints"""
    print(f"\n{'='*70}")
    print("STEP 4: KAGGLE MEMORY COMPATIBILITY")
    print('='*70)
    
    current_memory_gb = memory_usage_mb(df) / 1024
    
    # Estimate training memory requirements
    # Rule of thumb: XGBoost needs ~3-4x data size, ensemble needs ~6-8x
    estimated_training_memory = {
        'xgboost_only': current_memory_gb * 4,
        'ensemble_3models': current_memory_gb * 7,
        'with_tuning': current_memory_gb * 10
    }
    
    print(f"\n💾 Current Dataset Memory: {current_memory_gb:.2f} GB")
    print(f"\n📊 Estimated Training Memory:")
    for scenario, mem_gb in estimated_training_memory.items():
        fits = "✅ FITS" if mem_gb < EDAConfig.KAGGLE_MEMORY_LIMIT_GB else "❌ EXCEEDS"
        print(f"  {scenario:25s}: {mem_gb:6.2f} GB  [{fits}]")
    
    # Recommendations
    print(f"\n💡 Memory Optimization Recommendations:")
    
    if estimated_training_memory['ensemble_3models'] > EDAConfig.KAGGLE_MEMORY_LIMIT_GB:
        print("  ⚠️  Full ensemble may cause OOM on Kaggle!")
        print("  Recommendations:")
        print("    1. Use stratified sampling (train on A+B class only)")
        print("    2. Sequential training (train models one at a time, save to disk)")
        print("    3. Reduce ensemble size (XGBoost + LightGBM only)")
        print("    4. Feature selection (drop low-importance features)")
    else:
        print("  ✅ Dataset should fit comfortably on Kaggle with ensemble!")
    
    if sku_abc is not None:
        # Estimate stratified approach
        a_b_only = df[df['sku_id'].isin(sku_abc[sku_abc['abc_class'].isin(['A', 'B'])]['sku_id'])]
        stratified_memory = memory_usage_mb(a_b_only) / 1024
        print(f"\n  📉 Stratified Approach (A+B class only): {stratified_memory:.2f} GB")
        print(f"     Training estimate: {stratified_memory * 7:.2f} GB")
    
    compatibility = {
        'current_memory_gb': current_memory_gb,
        'estimated_training_memory': estimated_training_memory,
        'fits_kaggle': estimated_training_memory['ensemble_3models'] < EDAConfig.KAGGLE_MEMORY_LIMIT_GB
    }
    
    return compatibility


# ==============================================================================
# METRIC RECOMMENDATION
# ==============================================================================

def analyze_demand_characteristics(df):
    """Analyze demand patterns to recommend best metrics"""
    print(f"\n{'='*70}")
    print("STEP 5: DEMAND CHARACTERISTICS & METRIC RECOMMENDATION")
    print('='*70)
    
    demand_col = 'actual_demand'
    
    # Statistical characteristics
    demand_stats = df[demand_col].describe()
    
    # Zero demand frequency
    zero_demand_pct = (df[demand_col] == 0).sum() / len(df) * 100
    
    # Coefficient of variation (volatility)
    cv = demand_stats['std'] / demand_stats['mean']
    
    # Intermittent demand (ADI - Average Demand Interval)
    # How often do we see non-zero demand?
    nonzero_mask = df[demand_col] > 0
    if 'date' in df.columns and 'sku_id' in df.columns:
        # Calculate for sample SKU
        sample_sku = df['sku_id'].iloc[0]
        sku_data = df[df['sku_id'] == sample_sku].sort_values('date')
        nonzero_dates = sku_data[sku_data[demand_col] > 0]
        if len(nonzero_dates) > 1:
            intervals = nonzero_dates['date'].diff().dt.days.dropna()
            adi = intervals.mean()
        else:
            adi = None
    else:
        adi = None
    
    print(f"\n📊 Demand Statistics:")
    print(f"  Mean: {demand_stats['mean']:.2f}")
    print(f"  Median: {demand_stats['50%']:.2f}")
    print(f"  Std Dev: {demand_stats['std']:.2f}")
    print(f"  Coefficient of Variation: {cv:.2f}")
    print(f"  Zero Demand %: {zero_demand_pct:.2f}%")
    if adi:
        print(f"  Avg Demand Interval (days): {adi:.1f}")
    
    # Metric recommendations
    print(f"\n🎯 METRIC RECOMMENDATIONS:")
    print(f"\n  Based on your data characteristics:")
    
    metrics_recommended = []
    
    # Primary metric
    if zero_demand_pct > 5:
        print(f"  ⚠️  High zero-demand frequency ({zero_demand_pct:.1f}%)")
        print(f"      → PRIMARY: WMAPE (handles zeros, volume-weighted)")
        metrics_recommended.append(('PRIMARY', 'WMAPE'))
    else:
        print(f"  ✓ Low zero-demand frequency")
        print(f"      → PRIMARY: MAPE (industry standard)")
        metrics_recommended.append(('PRIMARY', 'MAPE'))
    
    # Secondary metrics
    print(f"\n  SECONDARY METRICS (use all for comprehensive evaluation):")
    
    if cv > 1.0:
        print(f"  📈 High volatility (CV={cv:.2f})")
        print(f"      → MAE (robust to outliers)")
        metrics_recommended.append(('SECONDARY', 'MAE'))
    else:
        print(f"  → MAE (absolute error magnitude)")
        metrics_recommended.append(('SECONDARY', 'MAE'))
    
    print(f"      → RMSE (penalizes large errors)")
    metrics_recommended.append(('SECONDARY', 'RMSE'))
    
    print(f"      → Bias % (detect systematic over/under prediction)")
    metrics_recommended.append(('SECONDARY', 'Bias'))
    
    print(f"      → Service Level (operational KPI)")
    metrics_recommended.append(('OPERATIONAL', 'ServiceLevel'))
    
    # Explain each metric
    print(f"\n  📖 METRIC EXPLANATIONS:")
    print(f"\n  1. WMAPE (Weighted Mean Absolute Percentage Error)")
    print(f"     Formula: (Σ|actual - pred|) / (Σ|actual|) × 100%")
    print(f"     Why: Volume-weighted, handles zeros, directly related to inventory cost")
    print(f"     Target: < 20% (excellent), < 25% (good), < 30% (acceptable)")
    
    print(f"\n  2. MAPE (Mean Absolute Percentage Error)")
    print(f"     Formula: mean(|actual - pred| / |actual|) × 100%")
    print(f"     Why: Industry standard, easy to interpret")
    print(f"     Target: < 15% (excellent), < 20% (good), < 30% (acceptable)")
    print(f"     ⚠️  Limitation: Undefined for zero demand, biased toward low-volume SKUs")
    
    print(f"\n  3. MAE (Mean Absolute Error)")
    print(f"     Formula: mean(|actual - pred|)")
    print(f"     Why: Same scale as demand, robust to outliers")
    print(f"     Target: Domain-specific (depends on demand scale)")
    
    print(f"\n  4. RMSE (Root Mean Squared Error)")
    print(f"     Formula: sqrt(mean((actual - pred)²))")
    print(f"     Why: Penalizes large errors more heavily")
    print(f"     Target: Should be close to MAE (if far apart, many outliers)")
    
    print(f"\n  5. Bias %")
    print(f"     Formula: (Σpred - Σactual) / Σactual × 100%")
    print(f"     Why: Detects systematic over/under-prediction")
    print(f"     Target: -3% to +3% (balanced)")
    print(f"     Impact: Negative = stockouts, Positive = excess inventory")
    
    print(f"\n  6. Service Level")
    print(f"     Formula: % of periods where prediction ≥ actual × threshold")
    print(f"     Why: Direct operational impact on stockouts")
    print(f"     Target: 95% for A-class, 90% for B-class, 85% for C-class")
    
    characteristics = {
        'mean': float(demand_stats['mean']),
        'std': float(demand_stats['std']),
        'cv': float(cv),
        'zero_demand_pct': float(zero_demand_pct),
        'metrics_recommended': metrics_recommended
    }
    
    return characteristics


# ==============================================================================
# MAIN EDA ORCHESTRATION
# ==============================================================================

def run_eda(data_path, output_dir='.'):
    """Run complete EDA analysis"""
    print(f"\n" + "="*70)
    print("FMCG FORECASTING DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Discover files
    discovered_files = discover_data_files(data_path)
    
    # Load timeseries data (main dataset)
    if discovered_files['timeseries'] is None:
        print("\n❌ ERROR: No timeseries data found!")
        print("   Please ensure you have a file with 'daily', 'timeseries', or 'sales' in the name.")
        return None
    
    print(f"\n{'='*70}")
    print("LOADING DATASETS")
    print('='*70)
    df = load_data_smart(discovered_files['timeseries'])
    
    if df is None:
        print("❌ Failed to load timeseries data!")
        return None
    
    # Load supplementary data
    sku_master = load_data_smart(discovered_files['sku_master'])
    loc_master = load_data_smart(discovered_files['location_master'])
    
    # Run analyses
    quality_report = assess_data_quality(df)
    
    # ABC Classification
    if 'abc_class' in df.columns:
        print(f"\n✅ ABC classification already present in data!")
        sku_abc = df[['sku_id', 'abc_class']].drop_duplicates()
        abc_summary = None
    else:
        sku_abc, abc_summary = calculate_abc_classification(df)
        # Merge ABC back to main df
        df = df.merge(sku_abc, on='sku_id', how='left')
    
    # Kaggle compatibility
    kaggle_compat = assess_kaggle_compatibility(df, sku_abc)
    
    # Demand characteristics & metrics
    demand_chars = analyze_demand_characteristics(df)
    
    # ====================
    # GENERATE SUMMARY
    # ====================
    
    summary = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'dataset_path': str(data_path),
        },
        'files_discovered': {k: str(v) if v else None for k, v in discovered_files.items()},
        'data_quality': quality_report,
        'abc_classification': abc_summary.to_dict() if abc_summary is not None else None,
        'kaggle_compatibility': kaggle_compat,
        'demand_characteristics': demand_chars,
    }
    
    # Save summary
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    summary_path = output_dir / 'eda_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("EDA COMPLETE!")
    print('='*70)
    print(f"📄 Summary saved to: {summary_path}")
    
    # Print final recommendations
    print(f"\n{'='*70}")
    print("FINAL RECOMMENDATIONS FOR KAGGLE TRAINING")
    print('='*70)
    
    print(f"\n🎯 METRICS TO USE:")
    for priority, metric in demand_chars['metrics_recommended']:
        print(f"  [{priority:12s}] {metric}")
    
    print(f"\n💾 MEMORY STRATEGY:")
    if kaggle_compat['fits_kaggle']:
        print(f"  ✅ Full ensemble possible on Kaggle")
        print(f"  Recommended: XGBoost + LightGBM + CatBoost")
    else:
        print(f"  ⚠️  Memory constraints detected")
        print(f"  Recommended:")
        print(f"    Option 1: Sequential training (save/load models)")
        print(f"    Option 2: Stratified sampling (A+B class only)")
        print(f"    Option 3: Reduce to 2-model ensemble (XGB + LGB)")
    
    print(f"\n🔧 HYPERPARAMETER TUNING:")
    print(f"  n_trials: 50-100 (balance time vs accuracy)")
    print(f"  early_stopping: 100 (patient for seasonality)")
    print(f"  stratify_by_abc: {True if abc_summary is not None else False}")
    
    return summary


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    parser = argparse.ArgumentParser(description='FMCG Dataset EDA for Kaggle Training')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./output',
        help='Path to data directory (default: ./output)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Output directory for reports (default: current directory)'
    )
    
    # Check if running within a Jupyter/Colab notebook
    # If so, process empty arguments to use defaults and avoid kernel flag errors
    try:
        get_ipython()
        print("ℹ️  Running in Notebook/Colab detected. Using default arguments.")
        args = parser.parse_args([])
    except NameError:
        # Standard script execution
        args = parser.parse_args()
    
    # Run EDA
    summary = run_eda(args.data_path, args.output_dir)
    
    if summary:
        print(f"\n✨ Analysis complete! Review eda_summary.json for detailed findings.")
    else:
        print(f"\n❌ Analysis failed. Please check error messages above.")

# -*- coding: utf-8 -*-
"""
BUSINESS KPI MEASUREMENT
=========================
Simulates business impact of ML forecasts vs. baseline methods.

KPIs Measured (from project brief):
1. Stockout Reduction: Reduce stockouts by 25-30%
2. Excess Inventory Reduction: Reduce dead stock by ~15%
3. Service Level (OTIF): Increase On-Time In-Full delivery
4. Inventory Turnover: Improve inventory turns
5. Forecast Accuracy: Improve MAPE by 20-25%

This module runs counterfactual simulations to demonstrate business value.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class InventorySimulator:
    """
    Simulates inventory operations under different forecasting methods.
    """
    
    def __init__(self, lead_time_days=7, service_level=0.95):
        """
        Args:
            lead_time_days: Replenishment lead time
            service_level: Target service level (e.g., 0.95 = 95%)
        """
        self.lead_time_days = lead_time_days
        self.service_level = service_level
        
    def simulate_policy(self, forecast_df, actual_df, policy_name="ml_forecast"):
        """
        Simulate inventory operations under a specific policy.
        
        Args:
            forecast_df: DataFrame with forecasts (date, sku_id, location_id, forecast)
            actual_df: DataFrame with actuals (date, sku_id, location_id, actual_demand)
            policy_name: One of ["ml_forecast", "moving_average", "last_year", "naive"]
        
        Returns:
            DataFrame with simulation results
        """
        print(f"\n{'='*70}")
        print(f"SIMULATING: {policy_name.upper()} POLICY")
        print(f"{'='*70}")
        
        # Merge forecasts with actuals
        sim_df = forecast_df.merge(
            actual_df[['date', 'sku_id', 'location_id', 'actual_demand']],
            on=['date', 'sku_id', 'location_id'],
            how='inner'
        )
        
        # Sort by SKU, location, date
        sim_df = sim_df.sort_values(['sku_id', 'location_id', 'date']).reset_index(drop=True)
        
        # Initialize tracking columns
        sim_df['inventory'] = 0.0
        sim_df['replenishment'] = 0.0
        sim_df['stockout_units'] = 0.0
        sim_df['stockout_flag'] = 0
        sim_df['excess_inventory'] = 0.0
        sim_df['holding_cost'] = 0.0
        
        # Simulate for each SKU-location combination
        for (sku, loc), group in sim_df.groupby(['sku_id', 'location_id']):
            inventory = 100.0  # Starting inventory
            
            for idx in group.index:
                row_position = group.index.get_loc(idx)
                
                # Calculate replenishment need based on policy
                if policy_name == "ml_forecast":
                    # Use ML forecast * safety factor
                    expected_demand = sim_df.loc[idx, 'forecast'] * self.lead_time_days * 1.2
                elif policy_name == "moving_average":
                    # Use 7-day moving average
                    if row_position >= 7:
                        recent_demand = sim_df.loc[group.index[max(0, row_position-7):row_position], 'actual_demand'].mean()
                        expected_demand = recent_demand * self.lead_time_days * 1.2
                    else:
                        expected_demand = inventory * 0.5
                elif policy_name == "last_year":
                    # Use last year's demand for same date
                    expected_demand = sim_df.loc[idx, 'actual_demand'] * 1.1  # Simplified
                else:  # naive
                    expected_demand = inventory * 0.5
                
                # Replenish if inventory below threshold
                reorder_point = expected_demand * 0.7
                if inventory < reorder_point:
                    replenish = max(0, expected_demand - inventory)
                else:
                    replenish = 0
                
                inventory += replenish
                sim_df.loc[idx, 'replenishment'] = replenish
                
                # Fulfill demand
                actual_demand = sim_df.loc[idx, 'actual_demand']
                fulfilled = min(inventory, actual_demand)
                stockout = actual_demand - fulfilled
                
                inventory -= fulfilled
                
                # Calculate costs
                excess = max(0, inventory - 30)  # Units above 30 days supply
                holding = inventory * 0.02  # $0.02 per unit per day holding cost
                
                # Record results
                sim_df.loc[idx, 'inventory'] = inventory
                sim_df.loc[idx, 'stockout_units'] = stockout
                sim_df.loc[idx, 'stockout_flag'] = 1 if stockout > 0 else 0
                sim_df.loc[idx, 'excess_inventory'] = excess
                sim_df.loc[idx, 'holding_cost'] = holding
        
        return sim_df
    
    def calculate_kpis(self, sim_df):
        """Calculate business KPIs from simulation results."""
        
        total_days = len(sim_df)
        total_demand = sim_df['actual_demand'].sum()
        
        # 1. Stockout Rate
        stockout_rate = sim_df['stockout_flag'].mean() * 100
        stockout_units = sim_df['stockout_units'].sum()
        lost_sales_pct = (stockout_units / total_demand) * 100 if total_demand > 0 else 0
        
        # 2. Service Level (OTIF)
        service_level = (1 - sim_df['stockout_flag'].mean()) * 100
        
        # 3. Inventory Metrics
        avg_inventory = sim_df['inventory'].mean()
        avg_excess_rate = (sim_df['excess_inventory'] > 0).mean() * 100
        total_holding_cost = sim_df['holding_cost'].sum()
        
        # 4. Inventory Turnover
        # Turnover = COGS / Average Inventory
        # COGS ≈ Total Demand Fulfilled
        total_fulfilled = total_demand - stockout_units
        if avg_inventory > 0:
            inventory_turnover = total_fulfilled / avg_inventory
        else:
            inventory_turnover = 0
        
        # 5. Forecast Accuracy
        mae = np.mean(np.abs(sim_df['forecast'] - sim_df['actual_demand']))
        mape = np.mean(np.abs(sim_df['forecast'] - sim_df['actual_demand']) / 
                      (sim_df['actual_demand'] + 1)) * 100
        
        kpis = {
            'stockout_rate_pct': stockout_rate,
            'stockout_units': stockout_units,
            'lost_sales_pct': lost_sales_pct,
            'service_level_pct': service_level,
            'avg_inventory_units': avg_inventory,
            'excess_inventory_rate_pct': avg_excess_rate,
            'total_holding_cost_usd': total_holding_cost,
            'inventory_turnover': inventory_turnover,
            'forecast_mae': mae,
            'forecast_mape_pct': mape
        }
        
        return kpis


class BusinessKPICalculator:
    """
    Compares ML forecasts against baseline methods and calculates improvements.
    """
    
    def __init__(self, lead_time_days=7, service_level=0.95):
        self.simulator = InventorySimulator(lead_time_days, service_level)
        
    def compare_methods(self, ml_forecasts, actual_df, baseline_methods=None):
        """
        Compare ML forecasts against baseline methods.
        
        Args:
            ml_forecasts: DataFrame with ML predictions (date, sku_id, location_id, forecast)
            actual_df: DataFrame with actual demand
            baseline_methods: List of baseline methods to compare
        
        Returns:
            dict: Comparison results
        """
        if baseline_methods is None:
            baseline_methods = ["moving_average", "naive"]
        
        print("\n" + "="*70)
        print("BUSINESS KPI COMPARISON: ML vs. BASELINE METHODS")
        print("="*70)
        
        results = {}
        
        # 1. Simulate ML forecasts
        ml_sim = self.simulator.simulate_policy(ml_forecasts, actual_df, "ml_forecast")
        results['ml_forecast'] = {
            'simulation': ml_sim,
            'kpis': self.simulator.calculate_kpis(ml_sim)
        }
        
        # 2. Simulate baseline methods
        for method in baseline_methods:
            baseline_sim = self.simulator.simulate_policy(ml_forecasts, actual_df, method)
            results[method] = {
                'simulation': baseline_sim,
                'kpis': self.simulator.calculate_kpis(baseline_sim)
            }
        
        # 3. Calculate improvements
        improvements = self._calculate_improvements(results, baseline_methods[0])
        
        # 4. Print summary
        self._print_comparison_summary(results, improvements)
        
        return {
            'results': results,
            'improvements': improvements
        }
    
    def _calculate_improvements(self, results, baseline_name):
        """Calculate percentage improvements over baseline."""
        
        ml_kpis = results['ml_forecast']['kpis']
        baseline_kpis = results[baseline_name]['kpis']
        
        improvements = {}
        
        # Stockout reduction (lower is better)
        improvements['stockout_reduction_pct'] = (
            (baseline_kpis['stockout_rate_pct'] - ml_kpis['stockout_rate_pct']) /
            baseline_kpis['stockout_rate_pct'] * 100
            if baseline_kpis['stockout_rate_pct'] > 0 else 0
        )
        
        # Service level improvement
        improvements['service_level_improvement_pct'] = (
            ml_kpis['service_level_pct'] - baseline_kpis['service_level_pct']
        )
        
        # Excess inventory reduction (lower is better)
        improvements['excess_inventory_reduction_pct'] = (
            (baseline_kpis['excess_inventory_rate_pct'] - ml_kpis['excess_inventory_rate_pct']) /
            baseline_kpis['excess_inventory_rate_pct'] * 100
            if baseline_kpis['excess_inventory_rate_pct'] > 0 else 0
        )
        
        # Inventory turnover improvement
        improvements['inventory_turnover_improvement_pct'] = (
            (ml_kpis['inventory_turnover'] - baseline_kpis['inventory_turnover']) /
            baseline_kpis['inventory_turnover'] * 100
            if baseline_kpis['inventory_turnover'] > 0 else 0
        )
        
        # Holding cost reduction
        improvements['holding_cost_reduction_pct'] = (
            (baseline_kpis['total_holding_cost_usd'] - ml_kpis['total_holding_cost_usd']) /
            baseline_kpis['total_holding_cost_usd'] * 100
            if baseline_kpis['total_holding_cost_usd'] > 0 else 0
        )
        
        # Forecast accuracy improvement (lower MAPE is better)
        improvements['mape_improvement_pct'] = (
            (baseline_kpis['forecast_mape_pct'] - ml_kpis['forecast_mape_pct']) /
            baseline_kpis['forecast_mape_pct'] * 100
            if baseline_kpis['forecast_mape_pct'] > 0 else 0
        )
        
        return improvements
    
    def _print_comparison_summary(self, results, improvements):
        """Print formatted comparison summary."""
        
        print("\n" + "="*70)
        print("📊 KPI COMPARISON SUMMARY")
        print("="*70)
        
        ml_kpis = results['ml_forecast']['kpis']
        baseline_kpis = results[list(results.keys())[1]]['kpis']  # First baseline
        
        print(f"\n{'Metric':<35} {'Baseline':<15} {'ML Forecast':<15} {'Improvement'}")
        print("-" * 75)
        
        metrics = [
            ('Stockout Rate', 'stockout_rate_pct', '%', 'stockout_reduction_pct'),
            ('Service Level (OTIF)', 'service_level_pct', '%', 'service_level_improvement_pct'),
            ('Avg Inventory', 'avg_inventory_units', 'units', None),
            ('Excess Inventory Rate', 'excess_inventory_rate_pct', '%', 'excess_inventory_reduction_pct'),
            ('Inventory Turnover', 'inventory_turnover', 'x', 'inventory_turnover_improvement_pct'),
            ('Forecast MAPE', 'forecast_mape_pct', '%', 'mape_improvement_pct'),
            ('Holding Cost', 'total_holding_cost_usd', ', 'holding_cost_reduction_pct')
        ]
        
        for metric_name, key, unit, imp_key in metrics:
            baseline_val = baseline_kpis[key]
            ml_val = ml_kpis[key]
            
            if unit == ':
                baseline_str = f"${baseline_val:,.0f}"
                ml_str = f"${ml_val:,.0f}"
            elif unit == 'units':
                baseline_str = f"{baseline_val:,.1f} {unit}"
                ml_str = f"{ml_val:,.1f} {unit}"
            else:
                baseline_str = f"{baseline_val:.2f}{unit}"
                ml_str = f"{ml_val:.2f}{unit}"
            
            if imp_key and imp_key in improvements:
                imp = improvements[imp_key]
                imp_str = f"{imp:+.1f}%"
                if imp > 0:
                    imp_str = f"✓ {imp_str}"
                elif imp < 0:
                    imp_str = f"✗ {imp_str}"
            else:
                imp_str = "-"
            
            print(f"{metric_name:<35} {baseline_str:<15} {ml_str:<15} {imp_str}")
        
        print("\n" + "="*70)
        print("🎯 KEY FINDINGS:")
        print("="*70)
        
        # Check against project targets
        if improvements['stockout_reduction_pct'] >= 25:
            print(f"✓ Stockout reduction: {improvements['stockout_reduction_pct']:.1f}% (Target: 25-30%)")
        else:
            print(f"⚠️ Stockout reduction: {improvements['stockout_reduction_pct']:.1f}% (Target: 25-30%)")
        
        if improvements['excess_inventory_reduction_pct'] >= 15:
            print(f"✓ Excess inventory reduction: {improvements['excess_inventory_reduction_pct']:.1f}% (Target: ~15%)")
        else:
            print(f"⚠️ Excess inventory reduction: {improvements['excess_inventory_reduction_pct']:.1f}% (Target: ~15%)")
        
        if improvements['mape_improvement_pct'] >= 20:
            print(f"✓ Forecast accuracy improvement: {improvements['mape_improvement_pct']:.1f}% (Target: 20-25%)")
        else:
            print(f"⚠️ Forecast accuracy improvement: {improvements['mape_improvement_pct']:.1f}% (Target: 20-25%)")
        
        print("="*70)
    
    def export_report(self, comparison_results, output_path='business_kpi_report.txt'):
        """Export detailed report to file."""
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BUSINESS KPI REPORT\n")
            f.write("="*70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            results = comparison_results['results']
            improvements = comparison_results['improvements']
            
            f.write("SUMMARY OF IMPROVEMENTS\n")
            f.write("-"*70 + "\n")
            for key, value in improvements.items():
                f.write(f"{key}: {value:.2f}%\n")
            
            f.write("\n\nDETAILED KPI BREAKDOWN\n")
            f.write("-"*70 + "\n")
            for method, data in results.items():
                f.write(f"\n{method.upper()}:\n")
                for kpi, value in data['kpis'].items():
                    f.write(f"  {kpi}: {value:.2f}\n")
        
        print(f"\n✓ Detailed report saved to {output_path}")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
TO INTEGRATE WITH YOUR EXISTING PIPELINE:

1. After generating forecasts on test set:

   # Prepare ML forecast DataFrame
   ml_forecasts = test_df[['date', 'sku_id', 'location_id']].copy()
   ml_forecasts['forecast'] = your_predictions  # From model
   
   # Prepare actual demand DataFrame
   actuals = test_df[['date', 'sku_id', 'location_id', 'true_demand']].copy()
   actuals = actuals.rename(columns={'true_demand': 'actual_demand'})

2. Calculate business KPIs:

   kpi_calc = BusinessKPICalculator(lead_time_days=7, service_level=0.95)
   
   comparison = kpi_calc.compare_methods(
       ml_forecasts=ml_forecasts,
       actual_df=actuals,
       baseline_methods=["moving_average", "naive"]
   )

3. Export report:

   kpi_calc.export_report(comparison, '/kaggle/working/business_kpi_report.txt')

4. Save simulation results:

   for method, data in comparison['results'].items():
       data['simulation'].to_csv(
           f'/kaggle/working/simulation_{method}.csv',
           index=False
       )
"""

if __name__ == "__main__":
    print("Business KPI Measurement Module")
    print("\nThis module simulates inventory operations to measure business impact.")
    print("See usage example at the end of this file for integration."
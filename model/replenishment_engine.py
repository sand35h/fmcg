# -*- coding: utf-8 -*-
"""
REPLENISHMENT RECOMMENDATION ENGINE
====================================
Converts forecasts into actionable inventory recommendations.

Implements:
1. Safety stock calculation
2. Reorder point calculation
3. Order quantity recommendations
4. Business constraint enforcement (MOQ, capacity, shelf life)
"""

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta


class ReplenishmentEngine:
    """
    Generates inventory replenishment recommendations.
    """
    
    def __init__(self, lead_time_days=7, service_level=0.95):
        """
        Args:
            lead_time_days: Default lead time between order and delivery
            service_level: Target service level (e.g., 0.95 = 95% in-stock)
        """
        self.lead_time_days = lead_time_days
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)
        
    def calculate_safety_stock(self, avg_demand, demand_std):
        """
        Calculate safety stock using service level approach.
        
        Formula: SS = Z * σ * sqrt(L)
        where:
            Z = z-score for service level
            σ = demand standard deviation
            L = lead time
        """
        safety_stock = self.z_score * demand_std * np.sqrt(self.lead_time_days)
        return np.maximum(safety_stock, 0)
    
    def calculate_reorder_point(self, avg_demand, demand_std):
        """
        Calculate reorder point (ROP).
        
        ROP = (Average demand × Lead time) + Safety stock
        """
        lead_time_demand = avg_demand * self.lead_time_days
        safety_stock = self.calculate_safety_stock(avg_demand, demand_std)
        
        return lead_time_demand + safety_stock
    
    def generate_recommendations(self, forecast_df, current_inventory_df, 
                                 constraints_df=None):
        """
        Generate complete replenishment recommendations.
        
        Args:
            forecast_df: DataFrame with columns:
                - sku_id, location_id, date
                - mean_forecast: Expected demand
                - std_forecast: Demand uncertainty (from quantile models)
            
            current_inventory_df: Current stock levels:
                - sku_id, location_id
                - current_stock: Units in stock
                - last_updated: Timestamp
            
            constraints_df (optional): Business constraints:
                - sku_id
                - moq: Minimum order quantity
                - max_capacity: Max storage capacity
                - shelf_life_days: Product shelf life
        
        Returns:
            DataFrame with recommendations for each SKU-location
        """
        print("Generating replenishment recommendations...")
        
        # Aggregate forecast for next N days (lead time period)
        forecast_agg = forecast_df.groupby(['sku_id', 'location_id']).agg({
            'mean_forecast': 'mean',  # Average daily demand
            'std_forecast': 'mean'    # Average daily std
        }).reset_index()
        
        # Merge with current inventory
        rec_df = forecast_agg.merge(
            current_inventory_df,
            on=['sku_id', 'location_id'],
            how='left'
        )
        
        # Handle missing inventory (assume 0)
        rec_df['current_stock'] = rec_df['current_stock'].fillna(0)
        
        # Calculate lead time demand
        rec_df['avg_daily_demand'] = rec_df['mean_forecast']
        rec_df['daily_demand_std'] = rec_df['std_forecast']
        rec_df['lead_time_demand'] = rec_df['avg_daily_demand'] * self.lead_time_days
        
        # Calculate safety stock
        rec_df['safety_stock'] = self.calculate_safety_stock(
            rec_df['avg_daily_demand'],
            rec_df['daily_demand_std']
        )
        
        # Calculate reorder point
        rec_df['reorder_point'] = rec_df['lead_time_demand'] + rec_df['safety_stock']
        
        # Determine if reorder is needed
        rec_df['stock_position'] = rec_df['current_stock']
        rec_df['stock_gap'] = rec_df['reorder_point'] - rec_df['stock_position']
        rec_df['needs_reorder'] = (rec_df['stock_gap'] > 0).astype(int)
        
        # Calculate recommended order quantity
        # Use Economic Order Quantity concept: order enough for ~2 lead times
        rec_df['recommended_order'] = np.where(
            rec_df['needs_reorder'] == 1,
            np.maximum(rec_df['stock_gap'], rec_df['avg_daily_demand'] * self.lead_time_days * 2),
            0
        )
        
        # Apply business constraints if provided
        if constraints_df is not None:
            rec_df = self._apply_constraints(rec_df, constraints_df)
        
        # Calculate risk flags
        rec_df['uncertainty_ratio'] = (
            rec_df['daily_demand_std'] / 
            rec_df['avg_daily_demand'].replace(0, 1)
        )
        rec_df['high_uncertainty'] = (rec_df['uncertainty_ratio'] > 0.5).astype(int)
        rec_df['stockout_risk'] = (rec_df['current_stock'] < rec_df['safety_stock']).astype(int)
        
        # Priority scoring (1=highest, 3=lowest)
        rec_df['priority'] = 3  # Default: low priority
        rec_df.loc[rec_df['stockout_risk'] == 1, 'priority'] = 1  # High: stockout risk
        rec_df.loc[
            (rec_df['needs_reorder'] == 1) & (rec_df['stockout_risk'] == 0),
            'priority'
        ] = 2  # Medium: needs reorder but not urgent
        
        # Calculate expected days until stockout
        rec_df['days_until_stockout'] = np.where(
            rec_df['avg_daily_demand'] > 0,
            rec_df['current_stock'] / rec_df['avg_daily_demand'],
            999  # Very high for items with no demand
        )
        
        # Add timestamp
        rec_df['recommendation_date'] = datetime.now()
        
        # Select and order columns
        output_cols = [
            'sku_id', 'location_id', 'recommendation_date',
            'current_stock', 'avg_daily_demand', 'daily_demand_std',
            'safety_stock', 'reorder_point', 'lead_time_demand',
            'needs_reorder', 'recommended_order', 'priority',
            'stockout_risk', 'high_uncertainty', 'days_until_stockout'
        ]
        
        if constraints_df is not None:
            output_cols.extend(['moq', 'max_capacity', 'shelf_life_days'])
        
        rec_df = rec_df[[c for c in output_cols if c in rec_df.columns]]
        
        # Summary statistics
        total_recommendations = len(rec_df)
        needs_reorder = rec_df['needs_reorder'].sum()
        high_priority = (rec_df['priority'] == 1).sum()
        total_order_value = rec_df['recommended_order'].sum()
        
        print(f"\n📊 Recommendation Summary:")
        print(f"  Total SKU-locations: {total_recommendations:,}")
        print(f"  Needs reorder: {needs_reorder:,} ({needs_reorder/total_recommendations*100:.1f}%)")
        print(f"  High priority (stockout risk): {high_priority:,}")
        print(f"  Total recommended order units: {total_order_value:,.0f}")
        
        return rec_df
    
    def _apply_constraints(self, rec_df, constraints_df):
        """Apply business constraints to recommendations."""
        print("Applying business constraints...")
        
        # Merge constraints
        rec_df = rec_df.merge(
            constraints_df[['sku_id', 'moq', 'max_capacity', 'shelf_life_days']],
            on='sku_id',
            how='left'
        )
        
        # Fill missing constraints with defaults
        rec_df['moq'] = rec_df['moq'].fillna(1)
        rec_df['max_capacity'] = rec_df['max_capacity'].fillna(9999)
        rec_df['shelf_life_days'] = rec_df['shelf_life_days'].fillna(365)
        
        # Apply Minimum Order Quantity (round up to MOQ)
        rec_df['recommended_order'] = np.where(
            rec_df['recommended_order'] > 0,
            np.ceil(rec_df['recommended_order'] / rec_df['moq']) * rec_df['moq'],
            0
        )
        
        # Apply Maximum Capacity constraint
        rec_df['recommended_order'] = np.minimum(
            rec_df['recommended_order'],
            rec_df['max_capacity'] - rec_df['current_stock']
        )
        
        # Apply Shelf Life constraint (don't order more than can be sold before expiry)
        max_sellable = rec_df['avg_daily_demand'] * rec_df['shelf_life_days'] * 0.8  # 80% buffer
        rec_df['recommended_order'] = np.minimum(
            rec_df['recommended_order'],
            max_sellable
        )
        
        # Ensure non-negative
        rec_df['recommended_order'] = np.maximum(rec_df['recommended_order'], 0)
        
        return rec_df
    
    def export_to_erp_format(self, recommendations_df, output_path):
        """
        Export recommendations in ERP-friendly format.
        """
        # Filter to only items that need reorder
        reorder_df = recommendations_df[
            recommendations_df['needs_reorder'] == 1
        ].copy()
        
        # Create ERP export format
        erp_export = pd.DataFrame({
            'Order_ID': [f"ORD_{datetime.now().strftime('%Y%m%d')}_{i:06d}" 
                        for i in range(len(reorder_df))],
            'SKU_ID': reorder_df['sku_id'],
            'Location_ID': reorder_df['location_id'],
            'Order_Quantity': reorder_df['recommended_order'].astype(int),
            'Priority': reorder_df['priority'],
            'Expected_Delivery_Date': (
                datetime.now() + timedelta(days=self.lead_time_days)
            ).strftime('%Y-%m-%d'),
            'Reason': np.where(
                reorder_df['stockout_risk'] == 1,
                'STOCKOUT_RISK',
                'REPLENISHMENT'
            )
        })
        
        erp_export.to_csv(output_path, index=False)
        print(f"✓ Exported {len(erp_export)} orders to {output_path}")
        
        return erp_export


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def demo_replenishment_engine():
    """
    Demonstrates how to use the replenishment engine with your forecasts.
    """
    
    print("="*70)
    print("REPLENISHMENT ENGINE DEMO")
    print("="*70)
    
    # 1. Create sample forecast data (replace with your actual forecasts)
    forecast_df = pd.DataFrame({
        'sku_id': ['SKU001', 'SKU001', 'SKU002', 'SKU002'],
        'location_id': ['LOC001', 'LOC002', 'LOC001', 'LOC002'],
        'date': pd.date_range('2024-01-01', periods=4),
        'mean_forecast': [100, 150, 80, 120],
        'std_forecast': [15, 20, 10, 18]
    })
    
    # 2. Create current inventory snapshot
    current_inventory = pd.DataFrame({
        'sku_id': ['SKU001', 'SKU001', 'SKU002', 'SKU002'],
        'location_id': ['LOC001', 'LOC002', 'LOC001', 'LOC002'],
        'current_stock': [50, 200, 30, 150],
        'last_updated': datetime.now()
    })
    
    # 3. Optional: Define business constraints
    constraints = pd.DataFrame({
        'sku_id': ['SKU001', 'SKU002'],
        'moq': [50, 25],  # Minimum order quantity
        'max_capacity': [1000, 500],  # Max storage
        'shelf_life_days': [90, 180]  # Shelf life
    })
    
    # 4. Initialize engine
    engine = ReplenishmentEngine(
        lead_time_days=7,
        service_level=0.95
    )
    
    # 5. Generate recommendations
    recommendations = engine.generate_recommendations(
        forecast_df,
        current_inventory,
        constraints
    )
    
    # 6. Display recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print(recommendations.to_string(index=False))
    
    # 7. Export for ERP
    engine.export_to_erp_format(recommendations, 'replenishment_orders.csv')
    
    return recommendations


if __name__ == "__main__":
    # Run demo
    recommendations = demo_replenishment_engine()
    
    """
    TO INTEGRATE WITH YOUR EXISTING PIPELINE:
    
    1. After generating forecasts with uncertainty intervals:
       
       engine = ReplenishmentEngine(lead_time_days=7, service_level=0.95)
       
       recommendations = engine.generate_recommendations(
           forecast_df=your_forecasts_with_uncertainty,
           current_inventory_df=your_inventory_snapshot,
           constraints_df=your_constraints  # Optional
       )
       
       recommendations.to_csv('/kaggle/working/replenishment_recommendations.csv', index=False)
    
    2. Export for ERP integration:
       
       engine.export_to_erp_format(
           recommendations, 
           '/kaggle/working/erp_orders.csv'
       )
    """

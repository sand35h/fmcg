#!/usr/bin/env python3
"""
Test script for price elasticity feature
"""

import pandas as pd
import numpy as np
from pathlib import Path
from training import DataConfig, FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_price_elasticity():
    """Test the price elasticity calculation"""
    
    # Load a small sample of data
    data_config = DataConfig()
    
    # Check if data files exist
    data_path = Path("./data/daily_timeseries.parquet")
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Load sample data
    logger.info("Loading sample data...")
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Take a small sample for testing
    sample_df = df.head(1000).copy()
    
    # Initialize feature engineer
    fe = FeatureEngineer(data_config)
    
    # Test price elasticity calculation
    logger.info("Testing price elasticity calculation...")
    result_df = fe.add_price_elasticity_features(sample_df)
    
    # Check results
    if 'price_elasticity' in result_df.columns:
        logger.info("✅ Price elasticity feature added successfully!")
        
        # Show some statistics
        elasticity_stats = result_df['price_elasticity'].describe()
        logger.info(f"Elasticity statistics:\n{elasticity_stats}")
        
        # Show elasticity categories
        if 'elasticity_category' in result_df.columns:
            category_counts = result_df['elasticity_category'].value_counts()
            logger.info(f"Elasticity categories:\n{category_counts}")
        
        # Show sample results
        sample_results = result_df[['sku_id', 'location_id', 'date', 'price', 'actual_demand', 'price_elasticity', 'elasticity_category']].head(10)
        logger.info(f"Sample results:\n{sample_results}")
        
    else:
        logger.error("❌ Price elasticity feature not found!")

if __name__ == "__main__":
    test_price_elasticity()
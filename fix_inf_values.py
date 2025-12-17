"""
Quick fix: Add this to training.py in the _prepare_features method
"""

def clean_data_for_training(df, feature_cols):
    """Remove inf and extreme values from features"""
    import numpy as np
    
    for col in feature_cols:
        if col in df.columns:
            # Replace inf with nan
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill nan with median (or 0 for safety)
            if df[col].isna().any():
                median_val = df[col].median()
                if np.isnan(median_val):
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(median_val)
    
    return df

# Add this line in split_data method after creating train/val/test:
# self.train_df = clean_data_for_training(self.train_df, self.feature_cols)
# self.val_df = clean_data_for_training(self.val_df, self.feature_cols)  
# self.test_df = clean_data_for_training(self.test_df, self.feature_cols)

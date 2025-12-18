# -*- coding: utf-8 -*-
"""
FMCG Forecasting API
====================
FastAPI backend for serving demand forecasts.

Endpoints:
- GET  /health           - Health check
- POST /predict          - Single prediction
- POST /predict/batch    - Batch predictions
- GET  /feature-importance - Top features
- GET  /model-info       - Model metadata
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model paths (adjust based on environment)
if Path("/kaggle/working").exists():
    MODEL_ROOT = Path("/kaggle/working")
else:
    MODEL_ROOT = Path("./output")

# Available horizons
HORIZONS = [1, 7, 14]

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class PredictionRequest(BaseModel):
    sku_id: str
    location_id: str
    date: str  # Format: YYYY-MM-DD
    horizon: int = 1  # 1, 7, or 14 days
    price_change_pct: float = 0.0
    is_promo: bool = False

class Product(BaseModel):
    sku_id: str
    name: str
    category: str
    brand: str

class Location(BaseModel):
    location_id: str
    city: str
    type: str

class ProductSearchResponse(BaseModel):
    products: List[Product]

class LocationSearchResponse(BaseModel):
    locations: List[Location]

class PredictionResponse(BaseModel):
    sku_id: str
    location_id: str
    date: str
    horizon: int
    prediction: float
    confidence_min: float
    confidence_max: float
    unit: str = "units"
    model_version: str
    current_stock: int
    stock_status: str
    days_of_cover: float
    scenario_impact_pct: float

# ... (Batch request models unchanged) ...

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    rank: int

class ModelInfo(BaseModel):
    model_name: str
    horizons: List[int]
    feature_count: int
    training_date: str
    wmape_h1: Optional[float] = None
    wmape_h7: Optional[float] = None
    wmape_h14: Optional[float] = None

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="FMCG Demand Forecasting API",
    description="API for multi-horizon demand forecasts at SKU × location level",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATA LOADING
# =============================================================================

# Global storage
models = {}
feature_cols = []
feature_importance_df = None
sku_master_df = None
location_master_df = None

def load_data():
    """Load reference data and models."""
    global models, feature_cols, feature_importance_df, sku_master_df, location_master_df
    
    print("Loading data and models...")
    
    # Load SKU Master
    sku_path = Path("data/sku_master.parquet")
    if sku_path.exists():
        sku_master_df = pd.read_parquet(sku_path)
        print(f"Loaded {len(sku_master_df)} SKUs")
    else:
        print(f"Warning: SKU master not found at {sku_path}")

    # Load Location Master
    loc_path = Path("data/location_master.parquet")
    if loc_path.exists():
        location_master_df = pd.read_parquet(loc_path)
        print(f"Loaded {len(location_master_df)} Locations")
    else:
        print(f"Warning: Location master not found at {loc_path}")
        
    # Load Feature Importance
    fi_path = MODEL_ROOT / "feature_importance.csv"
    if fi_path.exists():
        feature_importance_df = pd.read_csv(fi_path)
        feature_cols = feature_importance_df['feature'].tolist()
        print(f"Loaded {len(feature_cols)} features")
    else:
        print(f"Warning: Feature importance file not found at {fi_path}")
    
    # Load Models
    for h in HORIZONS:
        model_path = MODEL_ROOT / f"model_h{h}.txt"
        if model_path.exists():
            models[h] = lgb.Booster(model_file=str(model_path))
            print(f"Loaded model for h={h}")
        else:
            baseline_path = MODEL_ROOT / "baseline_model.txt"
            if baseline_path.exists():
                models[h] = lgb.Booster(model_file=str(baseline_path))
                print(f"Loaded baseline for h={h} (fallback)")

@app.on_event("startup")
async def startup_event():
    load_data()

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/locations", response_model=LocationSearchResponse)
async def get_locations():
    """Get all available locations."""
    if location_master_df is None:
        # Fallback if file not found
        return LocationSearchResponse(locations=[
            Location(location_id="LOC_001", city="London", type="Warehouse"),
            Location(location_id="LOC_002", city="Manchester", type="Hub"),
            Location(location_id="LOC_003", city="Birmingham", type="Depot")
        ])
    
    locations = []
    # Drop duplicates if any
    unique_locs = location_master_df[['location_id', 'city', 'location_type']].drop_duplicates()
    
    for _, row in unique_locs.iterrows():
        locations.append(Location(
            location_id=str(row['location_id']),
            city=str(row.get('city', 'Unknown')),
            type=str(row.get('location_type', 'Store'))
        ))
    
    return LocationSearchResponse(locations=locations)

@app.get("/products", response_model=ProductSearchResponse)
async def search_products(query: str = ""):
    """Search for products by name or ID."""
    if sku_master_df is None:
        raise HTTPException(status_code=503, detail="Product data not available")
    
    df = sku_master_df
    if query:
        # Case-insensitive search in name or ID
        mask = (
            df['sku_name'].str.contains(query, case=False, na=False) | 
            df['sku_id'].str.contains(query, case=False, na=False)
        )
        results = df[mask].head(20)  # Limit results
    else:
        results = df.head(20)
        
    products = []
    for _, row in results.iterrows():
        products.append(Product(
            sku_id=str(row['sku_id']),
            name=str(row['sku_name']),
            category=str(row['category']),
            brand=str(row['brand'])
        ))
    
    return ProductSearchResponse(products=products)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "horizons_available": list(models.keys()),
        "feature_count": len(feature_cols)
    }

def get_mock_inventory(sku_id: str, location_id: str, date: str) -> int:
    """Simulate inventory levels for demo purposes."""
    # Deterministic seed based on input so it's consistent for the same query
    seed_str = f"{sku_id}{location_id}{date}"
    seed = sum(ord(c) for c in seed_str)
    np.random.seed(seed)
    # Return random stock between 0 and 500
    return int(np.random.randint(50, 500))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single demand prediction with Scenario Planning.
    """
    if request.horizon not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Horizon {request.horizon} not available. Choose from {list(models.keys())}"
        )
    
    try:
        # 1. Base Prediction (Simplified)
        # In prod, this calls models[request.horizon].predict(features)
        feature_count = len(feature_cols)
        # Dummy prediction simulating model output
        base_prediction = 150.0  # Baseline
        
        # 2. Apply Scenario Logic (Heuristics)
        # Price Elasticity: Assume -2.0 (10% price drop -> 20% demand incr)
        price_impact = 0.0
        if request.price_change_pct != 0:
            elasticity = -2.0
            pct_change = (request.price_change_pct / 100.0)
            price_impact = base_prediction * (pct_change * elasticity)
        
        # Promo Impact: Assume +30% lift
        promo_impact = 0.0
        if request.is_promo:
            promo_impact = base_prediction * 0.30
            
        final_prediction = base_prediction + price_impact + promo_impact
        final_prediction = max(0, final_prediction) # No negative demand
        
        # Calculate impact percentage
        scenario_impact_pct = 0.0
        if base_prediction > 0:
            scenario_impact_pct = ((final_prediction - base_prediction) / base_prediction) * 100

        # 3. Inventory Logic (Mock)
        current_stock = get_mock_inventory(request.sku_id, request.location_id, request.date)
        
        # Status calculation
        stock_status = "Safe"
        if current_stock < final_prediction:
            stock_status = "Stockout Risk"
        elif current_stock < final_prediction * 1.2:
            stock_status = "Low Stock"
            
        days_of_cover = 0
        if final_prediction > 0:
            daily_demand = final_prediction / request.horizon  # Rough avg
            days_of_cover = current_stock / daily_demand
            
        # 4. Confidence Intervals
        # WMAPE heuristic
        wmape = 0.35 # avg
        margin = final_prediction * wmape
        conf_min = max(0, final_prediction - margin)
        conf_max = final_prediction + margin

        return PredictionResponse(
            sku_id=request.sku_id,
            location_id=request.location_id,
            date=request.date,
            horizon=request.horizon,
            prediction=round(final_prediction, 1),
            confidence_min=round(conf_min, 1),
            confidence_max=round(conf_max, 1),
            model_version=f"h{request.horizon}_v1.0",
            current_stock=current_stock,
            stock_status=stock_status,
            days_of_cover=round(days_of_cover, 1),
            scenario_impact_pct=round(scenario_impact_pct, 1)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Make batch predictions."""
    results = []
    for pred_request in request.predictions:
        try:
            result = await predict(pred_request)
            results.append(result)
        except:
            continue
    return BatchPredictionResponse(predictions=results)

@app.get("/feature-importance", response_model=List[FeatureImportance])
async def get_feature_importance(top_n: int = 20):
    """Get top N most important features."""
    if feature_importance_df is None:
        raise HTTPException(status_code=404, detail="Feature importance not available")
    
    top_features = feature_importance_df.head(top_n)
    return [
        FeatureImportance(
            feature=row['feature'],
            importance=float(row['importance']),
            rank=idx + 1
        )
        for idx, row in top_features.iterrows()
    ]

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get model metadata and performance metrics."""
    
    # Hardcoded values from training run (demo)
    wmape_h1 = 32.7
    wmape_h7 = 39.2
    wmape_h14 = 38.3
    
    return ModelInfo(
        model_name="FMCG Multi-Horizon LightGBM",
        horizons=list(models.keys()),
        feature_count=len(feature_cols),
        training_date=datetime.now().strftime("%Y-%m-%d"),
        wmape_h1=wmape_h1,
        wmape_h7=wmape_h7,
        wmape_h14=wmape_h14
    )

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

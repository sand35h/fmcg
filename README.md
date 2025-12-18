MCG Forecasting System - Gap Analysis & Implementation Plan
Current Implementation Status
What's Working ✅
Requirement	Current Status
Demand forecasting at SKU×location	✅ LightGBM model predicting true_demand at SKU×location level
Multi-horizon forecasts	✅ 1-day, 7-day, 14-day evaluation implemented
Forecast accuracy < 45% WMAPE	✅ Achieving 30.38% WMAPE on test set
Stockout correction	✅ 
correct_stockouts()
 handles censored demand
Feature engineering	✅ Lag features, rolling stats, calendar, weather, festivals, price/promo
External data integration	✅ Weather, festival calendar merged into features
Time-based train/val/test split	✅ 70/15/15 split preventing data leakage
Model saving/loading	✅ Saves to baseline_model.txt, feature importance CSV
Gaps to Address ⚠️
Requirement	Gap	Priority
Explainability (SHAP/feature importance)	Feature importance saved but no SHAP values	Medium
Confidence intervals/uncertainty	Not implemented	Medium
Replenishment recommendations	Not implemented	Low (out of scope for MVP)
What-if scenario planning	Not implemented	Low
REST API for forecasts	Missing - need FastAPI backend	High
Web UI dashboard	Missing - need frontend	High
Model retraining UI	Not implemented	Low
Drift detection/monitoring	Not implemented	Low
Audit trails	Not implemented	Low
Recommendations
For MVP: Focus on API + Frontend with core prediction capability
Future iterations: Add SHAP, uncertainty bands, scenario planning
Production: Add MLOps monitoring, retraining pipeline, audit logs
Implementation Plan
1. FastAPI Backend
[NEW] 
api.py
Create FastAPI application with:

GET /health - Health check endpoint
POST /predict - Single prediction for SKU×location×date
POST /predict/batch - Batch predictions
GET /feature-importance - Return top features with importance scores
GET /model-info - Return model metadata (training date, WMAPE, etc.)
2. HTML/JS/CSS Frontend
[NEW] 
frontend/index.html
Main dashboard page with:

Header with logo and navigation
Forecast input form (SKU, location, date)
Prediction results display
Feature importance chart
Model performance metrics
[NEW] 
frontend/styles.css
Modern dark theme styling with:

CSS variables for theming
Responsive grid layout
Form styling
Card components
Chart container styling
[NEW] 
frontend/app.js
JavaScript for:

API calls to FastAPI backend
Form submission handling
Dynamic result rendering
Chart visualization (using Chart.js)
Error handling
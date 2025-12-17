# Deep Dive AIML: Mathematics & Mechanics

This document explains the **mathematical and algorithmic principles** powering the FMCG demand forecasting engine.

---

## 1. Feature Engineering: The Mathematical Transformations

Raw data is transformed into $X_{features}$ using mathematical operations to expose patterns.

### 1.1 Cyclical Encoding of Time
Time is cyclical (December is close to January), but integers are linear ($12$ is far from $1$). We transform date parts into (Sine, Cosine) pairs.

**Formula**:
For a month $m \in [1, 12]$:
$$
Month_{sin} = \sin\left(\frac{2 \pi m}{12}\right), \quad Month_{cos} = \cos\left(\frac{2 \pi m}{12}\right)
$$

**Why?**: This places months on a unit circle. Jan ($m=1$) and Dec ($m=12$) become neighbors in vector space. Similarly applied to day_of_week.

### 1.2 Lag Features & Rolling Statistics
We capture temporal dependencies using lag features and rolling window statistics.

**Lag Features**: For time $t$, we include past demand values:
$$ X_{lag\_k} = y_{t-k}, \quad k \in \{1, 7, 14, 30\} $$

**Rolling Mean**: Captures trend over window $W$ of size $k$:
$$ \mu_t = \frac{1}{k} \sum_{i=1}^{k} y_{t-i}, \quad k \in \{14, 28\} $$

**Coefficient of Variation**: Measures demand volatility:
$$ CV_t = \frac{\sigma_t}{\mu_t + \epsilon}, \quad \epsilon = 10^{-6} $$

### 1.3 Price Features (Elasticity Replaced)
**Issue**: Rolling correlation elasticity was 99.7% noise (mean≈0, only 0.03% elastic products).

**Replacement Features**:
1. **Discount Depth**: $\frac{base\_price - price}{base\_price}$
2. **Price Change Flag**: Binary indicator of price change vs previous week
3. **Promo Intensity Rolling**: $\frac{\sum_{t-7}^{t} promo\_flag}{7}$
4. **Price Volatility**: $\text{std}_{30d}(price)$

**Why?**: These capture promotional impact without requiring sufficient price variance per SKU.

---

## 2. Algorithm Deep Dive

### 2.1 LightGBM with Tweedie Loss
Our primary algorithm for tabular forecasting. It's an ensemble of **decision trees** trained sequentially.

#### **The Objective Function**
We use **Tweedie loss** (compound Poisson-Gamma) optimized for count data with overdispersion:
$$ \mathcal{L}(\phi) = \sum_{i} l_{tweedie}(\hat{y}_i, y_i) + \sum_{k} \Omega(f_k) $$
Where:
*   $l_{tweedie}$: Tweedie deviance with variance power $p \in [1, 2]$
    $$ l = \begin{cases} 
    -y\log(\hat{y}) + \hat{y} & p=1 \text{ (Poisson)} \\
    \frac{y^{2-p}}{(1-p)(2-p)} - \frac{y\hat{y}^{1-p}}{1-p} + \frac{\hat{y}^{2-p}}{2-p} & 1<p<2 \\
    \frac{(y-\hat{y})^2}{y} & p=2 \text{ (Gamma)}
    \end{cases} $$
*   **Why Tweedie?**: Naturally handles zeros, overdispersion, and right-skewed demand
*   $\Omega$: Regularization term prevents overfitting

#### **How it Learns (Gradient Boosting)**
Trees are trained sequentially, each correcting errors:
$$ \hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i) $$
*   $\eta$: Learning Rate (`learning_rate=0.03`). Lower = better generalization
*   Early stopping monitors validation WMAPE (not RMSE)

#### **Optimized Hyperparameters**
```python
objective='tweedie'
tweedie_variance_power=1.3  # Between Poisson (1) and Gamma (2)
learning_rate=0.03
num_leaves=256
min_data_in_leaf=500
max_bin=512
feature_fraction=0.7
bagging_fraction=0.7
bagging_freq=1
```

---

## 3. Training Pipeline Architecture

The training workflow follows a structured pipeline:

### Step 1: Data Loading & Preparation
*   **Input**: Parquet files from `dataset2.py`:
    - `daily_timeseries.parquet`: Main sales data
    - `sku_master.parquet`: Product metadata
    - `location_master.parquet`: Store demographics
    - `festival_calendar.parquet`: UK holidays (Easter, Christmas, Black Friday)
    - `weather_data.parquet`: Temperature, precipitation, humidity
    - `macro_indicators.parquet`: GDP growth, CPI, consumer confidence
    - `external_shocks.parquet`: Financial Crisis (2008), Brexit (2016), COVID-19 (2020)
    - `competitor_activity.parquet`: Competitor promotions and pricing

### Step 2: Feature Engineering Pipeline
*   **Memory Optimization**: `reduce_mem_usage()` converts float64→float32, int64→int16
*   **Feature Categories**:
    1. **Calendar**: Cyclical encoding (sin/cos), weekend flags, month_start/end
    2. **Lag Features**: demand_lag_{1,7,14,30}
    3. **Rolling Stats**: demand_rolling_{14,28}_mean, demand_rolling_{14,28}_cv
    4. **Price Features**: price_to_base, discount_depth, price_elasticity
    5. **External**: Festival flags, shock events, weather, macro indicators
    6. **Lifecycle**: days_since_launch (product maturity)

### Step 3: Rolling-Origin Validation
**Issue**: Single time split (2018-2020 val, 2021-2023 test) showed distribution shift (RMSE 36→113).

**Solution**: Rolling-origin cross-validation
```
Origin 1: Train [2004-2019] → Val [2020]
Origin 2: Train [2004-2020] → Val [2021]
Origin 3: Train [2004-2021] → Val [2022]
Test: [2023]
```
*   **Hyperparameters**: Tuned on average validation WMAPE across origins
*   **Final Model**: Retrained on [2004-2022], evaluated on [2023]
*   **Per-Segment Metrics**: Reported by ABC class, category, and year

### Step 4: Model Training
*   **Algorithm**: LightGBM with Tweedie objective
*   **Categorical Features**: SKU, location, category (native encoding)
*   **Early Stopping**: Monitors validation WMAPE (custom metric); patience=100 rounds
*   **Prediction Post-Processing**:
    ```python
    pred = model.predict(X)
    pred = np.clip(pred, 0, None)  # No negatives
    # No log transform - trained on raw demand
    ```
*   **Output**: Model + SHAP feature importance (gain-based is misleading)

---

## 4. Evaluation Metrics

### Business Objective
**Primary Goal**: Minimize aggregate WMAPE across all SKU-Location pairs while maintaining 95% service level.

### Metric Alignment
**Training Objective**: Tweedie loss (optimizes for count data distribution)
**Evaluation Metric**: WMAPE (business-aligned)
**Early Stopping**: Validation WMAPE (not RMSE)

### Primary Metrics
1.  **WMAPE (Weighted Mean Absolute Percentage Error)**:
    $$ \text{WMAPE} = \frac{\sum |y_t - \hat{y}_t|}{\max(\sum |y_t|, 1)} \times 100\% $$
    *   **Why Primary?**: Handles zeros, weights by volume, aligns with business KPI
    *   **Target**: <15% for A-class SKUs, <25% for B/C-class

2.  **Service Level (Inventory Planning)**:
    $$ \text{Service Level} = \frac{\sum \mathbb{1}[\hat{y}_t \geq y_t]}{n} \times 100\% $$
    *   **Target**: ≥95% (avoid stockouts)

3.  **Bias %**: Detects systematic over/under-forecasting
    $$ \text{Bias\%} = \frac{\sum(\hat{y} - y)}{\max(\sum y, 1)} \times 100\% $$
    *   **Target**: [-5%, +5%] (slight over-forecast acceptable for safety stock)

### Metrics We DON'T Use
- ❌ **MAPE**: Undefined for zeros, explodes on low-volume SKUs
- ❌ **RMSLE**: Misaligned with business objective
- ⚠️ **RMSE**: Reported but not optimized (dominated by high-volume outliers)

---

## 5. Design Rationale

1.  **Why LightGBM over ARIMA?**
    *   ARIMA requires separate models for each SKU-Location pair (50 SKUs × 20 Locations = 1,000 models)
    *   LightGBM is a **Global Model**: learns cross-SKU patterns ("All dairy products spike before Christmas")
    *   Handles exogenous features (weather, promotions) natively

2.  **Why Tweedie Loss Instead of Log-Transform?**
    *   **Problem**: Log-transform + RMSE training ≠ WMAPE evaluation (misaligned objectives)
    *   **Solution**: Tweedie loss directly optimizes for count data with zeros
    *   **Benefit**: No inverse transform errors, no negative predictions

3.  **Why Remove Price Elasticity?**
    *   **Empirical Finding**: 99.7% of elasticity values were noise (mean≈0, std=0.18)
    *   **Root Cause**: Insufficient price variance per SKU-location (most prices static)
    *   **Replacement**: Discount depth, promo flags, price volatility (higher signal)

4.  **Why Realistic Dataset (dataset2.py)?**
    *   **Lifecycle**: Products launch/retire (mimics real portfolio management)
    *   **External Shocks**: COVID-19, Brexit, Financial Crisis (tests model robustness)
    *   **Cross-SKU Effects**: Substitution/cannibalization (e.g., Coke vs Pepsi)
    *   **Strategic Promotions**: Aligned with holidays (not random noise)

5.  **Why Single Model (No Ensemble)?**
    *   **Empirical**: Ensemble increased test RMSE (36→113)
    *   **Root Cause**: Models trained on different objectives (log vs raw)
    *   **Decision**: One strong LightGBM > poorly combined ensemble

6.  **Forecast Horizon Strategy**
    *   **Short-term (1-7 days)**: Lag features dominate
    *   **Mid-term (14-30 days)**: Rolling stats + seasonality
    *   **Long-term (30+ days)**: External features (weather, macro)
    *   **Implementation**: Single model with `forecast_horizon` as feature OR separate models per horizon

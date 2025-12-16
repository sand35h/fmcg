# Deep Dive AIML: Mathematics & Mechanics

This document peels back the layers of abstraction to explain the **mathematical and algorithmic principles** powering the forecasting engine.

---

## 1. Feature Engineering: The Mathematical Transformations

Raw data is never enough. We transform $X_{raw}$ into $X_{features}$ using specific mathematical operations to expose patterns to the model.

### 1.1 Cyclical Encoding of Time
Time is cyclical (December is close to January), but integers are linear ($12$ is far from $1$). To help the model understand this "closeness", we transform date parts into (Sine, Cosine) pairs.

**Formula**:
For a month $m \in [1, 12]$:
$$
Month_{sin} = \sin\left(\frac{2 \pi m}{12}\right), \quad Month_{cos} = \cos\left(\frac{2 \pi m}{12}\right)
$$

**Why?**: This places months on a unit circle. Jan ($m=1$) and Dec ($m=12$) become neighbors in vector space.

### 1.2 Target Transformation (Log-Scaling)
FMCG sales data is often **heteroscedastic** (variance increases with mean). A promotion might spike sales from 100 to 1,000.
To stabilize variance, we train on the Logarithm of sales:

**Formula**:
$$ y' = \log(y + 1) $$
*(We add +1 to handle days with 0 sales, as $\log(0)$ is undefined).*

**Inverse Transform**:
When predicting, we must revert this to get real units:
$$ \hat{y} = \exp(\hat{y}') - 1 $$

### 1.3 Rolling Window Statistics
We capture "Trend" by calculating statistics over a sliding window $W$ of size $k$.
For a time $t$, the rolling mean is:
$$ \mu_t = \frac{1}{k} \sum_{i=1}^{k} y_{t-i} $$

We generate these for multiple windows: $k \in \{7, 14, 30, 90\}$.

---

## 2. Algorithm Deep Dive

### 2.1 XGBoost (Extreme Gradient Boosting)
This is our primary workhorse for tabular data. It is an ensemble of **weak learners** (Decision Trees).

#### **The Objective Function**
XGBoost minimizes a regularized objective function:
$$ \mathcal{L}(\phi) = \sum_{i} l(\hat{y}_i, y_i) + \sum_{k} \Omega(f_k) $$
Where:
*   $l$: Loss function (e.g., Squared Error $(y-\hat{y})^2$).
*   $\Omega$: Regularization term controls complexity (prevents overfitting).
    $$ \Omega(f) = \gamma T + \frac{1}{2} \lambda ||w||^2 $$
    *   $T$: Number of leaves in the tree.
    *   $w$: Leaf weights.

#### **How it Learns (Gradient Boosting)**
It doesn't train all trees at once. It trains text_tree $_{t}$ to fix the errors (residuals) of text_tree $_{t-1}$.
$$ \hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(x_i) $$
*   $\eta$: Learning Rate (shrinkage). A lower $\eta$ (e.g., 0.01) means we need more trees but get a more generalized model.

### 2.2 Prophet (Bayesian Structural Time Series)
Used for our "Baseline" models. It decomposes time series into 3 additive components:

**Formula**:
$$ y(t) = g(t) + s(t) + h(t) + \epsilon_t $$

1.  **$g(t)$ (Trend)**: Modeled as a piece-wise linear growth curve. It detects "changepoints" where the trend shifts.
2.  **$s(t)$ (Seasonality)**: Modeled using Fourier Series (a sum of Sines and Cosines with different frequencies).
3.  **$h(t)$ (Holidays)**: Modeled as indicator functions for specific dates (e.g., "Is diwali?" = 1).

---

## 3. SageMaker Pipeline Details

We implement MLOps using **SageMaker Pipelines**, which treats the training workflow as a Directed Acyclic Graph (DAG).

### Step 1: `ProcessingStep` (Data Prep)
*   **Instance**: `ml.m5.xlarge` (General Purpose).
*   **Logic**: Runs a Scikit-Learn script.
    1.  Downloads raw Parquet from S3.
    2.  Merges `Sales` with `Macro_Economics` and `Weather`.
    3.  Performs `log(y+1)` transform.
    4.  Splits data: Train (Jan 2004 - Dec 2022), Validation (Jan 2023 - Dec 2023), Test (Jan 2024 - Present).
    5.  Saves artifacts to `/opt/ml/processing/output`.

### Step 2: `TrainingStep` (Model Fitting)
*   **Instance**: `ml.c5.2xlarge` (Compute Optimized).
*   **Algorithm**: AWS Built-in XGBoost Container.
*   **Hyperparameters**:
    *   `num_round`: 1000 (Number of trees).
    *   `early_stopping_rounds`: 50 (Stop if validation loss doesn't improve).
    *   `max_depth`: 6 (Tree depth).
    *   `subsample`: 0.8 (Train on random 80% of data to prevent overfitting).

### Step 3: `EvolutionStep` (Quality Gate)
*   **Logic**:
    1.  Loads the trained model.
    2.  Predicts on the `Test` set.
    3.  Calculates **MAPE**:
        $$ \text{MAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right| $$
    4.  **Condition**: If $MAPE < MAPE_{baseline}$ (e.g., 15%), proceed to register. Else, fail.

---

## 4. Why these choices? (Design Rationale)

1.  **Why XGBoost and not ARIMA?**
    *   ARIMA creates a new model distinct for *every single SKU*. If we have 5,000 SKUs, we manage 5,000 models.
    *   XGBoost is a **Global Model**. It learns patterns *across* SKUs (e.g., "All dairy products dip in winter"). We train 1 big model for 5,000 SKUs. This is more powerful and easier to manage.

2.  **Why Log-Transform?**
    *   Without it, the model obsess over high-volume items (Coke) and ignores low-volume items (Specialty Tea). Log-transform puts them on a similar scale so the model learns both equally well.

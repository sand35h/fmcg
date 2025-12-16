# FMCG Forecasting: Study Guide & Implementation Roadmap

You asked for **detail** and **how to achieve high accuracy**. This document is your curriculum. It breaks the project down into **Learning Phases**.

If you master these topics, you will not only finish the project but also build a system that rivals professional enterprise solutions.

---

## 🏗️ Phase 1: The Foundation (Data Engineering)
**Goal**: Get comfortable manipulating time-series data. 80% of "Accuracy" comes from clean data, not the model.

### 1.1 Key Concepts to Study
*   **Pandas for Time Series**:
    *   **Resampling**: Converting daily data to weekly (`df.resample('W').sum()`).
    *   **Imputation**: Filling missing values effectively (`df.interpolate()`, `ffill()`, `bfill()`). *Do not just fill with 0!*
    *   **Time Deltas**: Handling simple date math.
*   **Stationarity**:
    *   Study the **Augmented Dickey-Fuller (ADF) Test**.
    *   Learn why ML models hate "Non-Stationary" data (data where the mean changes over time).
    *   Learn **Differencing** (`df.diff()`) to fix it.

### 1.2 "High Accuracy" Secrets
*   **Handling Outliers**: Study "Interquartile Range (IQR)" method. If you leave a massive outlier (e.g., a data entry error of 1,000,000 sales), it will ruin your model's accuracy.
*   **Feature Engineering**: This is the single biggest driver of accuracy.
    *   **Lags**: What happened 7 days ago? (`t-7`)
    *   **Rolling Statistics**: `rolling_mean_7d`, `rolling_std_30d`.
    *   **Date Parts**: `is_weekend`, `month_sin`, `month_cos` (Cyclical encoding).

### 📚 Recommended Resources
*   *Pandas Documentation: Time Series / Date functionality*
*   *Online Course / Article*: "Feature Engineering for Time Series Forecasting"

---

## 🧠 Phase 2: The Brain (Machine Learning)
**Goal**: Build a forecasting model that beats a simple average.

### 2.1 Key Concepts to Study
*   **Supervised Learning for Time Series**:
    *   Understand how to transform a list of numbers `[1, 2, 3, 4]` into a supervised problem `X=[1, 2, 3], y=[4]`.
*   **Gradient Boosting (XGBoost / LightGBM)**:
    *   This is the industry standard for tabular forecasting.
    *   Study **Hyperparameters**: `learning_rate`, `max_depth`, `subsample`.
*   **Evaluation Metrics**:
    *   **MAPE** (Mean Absolute Percentage Error): Good for business explainability ("We are off by 5%").
    *   **RMSE** (Root Mean Squared Error): Penalizes big mistakes heavily.
    *   **MASE** (Mean Absolute Scaled Error): The "Pro" metric. Checks if you are better than a naive forecast.

### 2.2 "High Accuracy" Secrets
*   **Time Series Cross-Validation**:
    *   **NEVER** use random `train_test_split`. You cannot predict the past using the future.
    *   Study **Rolling Origin Validation** (Walk-Forward Validation).
*   **Target transformation**:
    *   Predicting `log(sales)` instead of `sales` can often improve accuracy for volatile products.
*   **Ensembling**:
    *   Train one ARIMA model and one XGBoost model. Average their predictions. This almost always increases accuracy.

### 📚 Recommended Resources
*   *Library*: **Scikit-Learn** (for metrics and utility functions).
*   *Book*: "Forecasting: Principles and Practice" by Rob J Hyndman (The Bible of forecasting - free online).

---

## ☁️ Phase 3: The Cloud (AWS Serverless)
**Goal**: Move your code from a laptop to the cloud without managing servers.

### 3.1 Key Concepts to Study
*   **Infrastructure as Code (IaC)**:
    *   Don't click around the console. Learn **AWS SAM** (Serverless Application Model) or **Terraform**.
    *   It defines your API and Lambda in a simple YAML file.
*   **AWS Lambda Lifecycles**:
    *   **Cold Starts**: The delay when a function runs for the first time.
    *   **Timeouts**: Lambda dies after 15 mins. Don't train models here! Trigger SageMaker instead.
*   **API Gateway Integration**:
    *   Study **Proxy Integration**. It passes the raw request to Lambda, giving you full control.

### 3.2 "High Accuracy" Secrets (Reliability)
*   **Dead Letter Queues (DLQ)**: Where do failed events go? (e.g., if Kinesis fails to write).
*   **Idempotency**: Ensuring that if a message is sent twice, you don't process it twice (e.g., don't deduct stock twice).

### 📚 Recommended Resources
*   *Tool*: **AWS SAM CLI** (Install this first).
*   *Concept*: **Event-Driven Architecture** (Udemy / Coursera).

---

## 🖥️ Phase 4: The Interface (React Frontend)
**Goal**: Show the data in a way that Planners actually trust.

### 4.1 Key Concepts to Study
*   **React Hooks**: `useEffect`, `useState` (The basics).
*   **Data Fetching**:
    *   Study **React Query** (TanStack Query). It handles caching, loading states, and refetching automatically. Much better than standard `fetch()`.
*   **Visualization**:
    *   Study **Recharts** or **Chart.js**. You need to plot:
        *   Line 1: Historical Sales (Solid)
        *   Line 2: Forecast (Dotted)
        *   Area: Confidence Interval (Shaded region).

### 4.2 "High Accuracy" Secrets (UX)
*   **Optimistic UI**: Update the chart instantaneously when a user changes a parameter, even before the backend confirms.
*   **Polling vs WebSockets**:
    *   For the "Streaming Stock" feature, study **Long Polling** (easier) vs **WebSockets** (harder but real-time).

---

## ✅ Implementation Checklist (Sprint Plan)

### Sprint 1: Data & Baseline
- [ ] Learn Pandas Time Series.
- [ ] Clean the "Dataset Generator" output.
- [ ] Train a simple **Prophet** model locally.
- [ ] Measure MAPE.

### Sprint 2: The Cloud Backend
- [ ] Install AWS SAM.
- [ ] Deploy a "Hello World" Lambda behind API Gateway.
- [ ] Create an S3 bucket and upload your cleaned data.
- [ ] Write a Lambda to read that S3 file and return JSON.

### Sprint 3: Advanced ML
- [ ] Study XGBoost.
- [ ] Engineer "Lag" and "Rolling" features.
- [ ] Train XGBoost model. Compare MAPE vs Prophet.
- [ ] (Bonus) Combine them (Ensemble).

### Sprint 4: Frontend & Connectivity
- [ ] Create a generic React App (`npx create-react-app`).
- [ ] Install **Recharts**.
- [ ] Fetch data from your API Gateway URL.
- [ ] Plot the chart.

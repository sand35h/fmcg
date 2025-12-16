# Strategic Execution Plan: FMCG Supply & Demand Forecasting

This document is the **Strategic Implementation Guide**. While the other documents detail the *Structure* (Architecture), *Function* (System Overview), and *Science* (AIML), this document details the **Execution**.

---

## 1. Core Strategic Pillars

A Final Year Project succeeds not by features, but by **Robustness**. We prioritize:
1.  **Safety**: The system must never crash. We use Serverless (Lambda) which auto-scales.
2.  **Explainability**: The system must not be a "Black Box". Every forecast explains *why* (via SHAP values).
3.  **Realism**: The simulation must "feel" real to the external examiner. We achieve this by streaming POS data that mimics actual retail patterns.

---

## 2. Implementation Phases (The Gantt Chart)

### Phase 1: The "Digital Twin" (Weeks 1-2)
*   **Goal**: Create a realistic synthetic world.
*   **Key Action**: Enhance `dataset generator.py` to produce data that *looks* real (e.g., sales drop on rainy days).
*   **Deliverable**: `s3://fmcg-datalake-raw/sales_history_2004_2024.parquet`.

### Phase 2: The "Pipeline" (Weeks 3-5)
*   **Goal**: Automate the data flow.
*   **Key Action**: Write infrastructure as code (SAM) to deploy Kinesis, Firehose, and Glue.
*   **Deliverable**: A script `deploy.sh` that spins up the backend in 10 minutes.

### Phase 3: The "Brain" (Weeks 6-8)
*   **Goal**: Train the XGBoost model.
*   **Key Action**:
    1.  Feature Engineering in SageMaker Processing.
    2.  Training in SageMaker Training.
    3.  Tuning Hyperparameters to beat the "Naive Forecast" (Baseline).
*   **Deliverable**: A saved model artifact `model.tar.gz` with MAPE < 15%.

### Phase 4: The "Face" (Weeks 9-12)
*   **Goal**: Build the React Dashboard.
*   **Key Action**:
    1.  Connect React Query to API Gateway.
    2.  Implement the WebSocket listener for real-time alerts.
*   **Deliverable**: A hosted URL (CloudFront) where users can login.

---

## 3. Streaming Strategy: The "Virtual Retailer"

To satisfy the "Data Streaming" requirement, we do not just play back a CSV file. We simulate a **Store Network**.

### The Simulation Architecture
1.  **The "Director" Script**: A Python script running on EC2 (or your laptop).
2.  **Logic**:
    *   It wakes up every 1 second (represents 1 "simulation hour").
    *   It picks a random "Store" (e.g., London Central).
    *   It generates a "basket" of goods (e.g., 2x Coke, 1x Chips).
    *   It injects "Noise" (e.g., 10% chance of a double-scan error).
3.  **The Payload**:
    It sends this to Kinesis. The backend handles it exactly as it would handle real data from Wal-Mart.

---

## 4. Key Success Metrics (Project KPIs)

To define success for this project, we track:

1.  **Forecast Accuracy (MAPE)**: Target < 15%.
    *   *Why?* Less than 15% is considered "Good" in industry.
2.  **System Latency**: Limit P99 < 500ms.
    *   *Why?* UI performance is the first thing an examiner notices.
3.  **Cost**: Monthly AWS Bill < $50.
    *   *Strategy*: Use Spot Instances for training and aggressive DynamoDB Auto-Scaling.

---

## 5. Risk Management

| Risk | Probability | Mitigation Strategy |
| :--- | :--- | :--- |
| **AWS Bill Shock** | Medium | Set AWS Budget Alarm at $20. use `t3.micro`. |
| **Model Underfitting** | High | If XGBoost fails, fallback to simple "Prophet" model which is easier to tune. |
| **Complexity Overload** | High | Cut features. Drop "Scenario Planning" if needed. Focus on "Forecast View". |

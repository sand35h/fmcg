# Comprehensive System Overview & End-to-End Flow

This document serves as the **Functional Specification** for the FMCG Supply & Demand Forecasting System. It bridges the gap between high-level requirements and technical implementation, detailed specifically for developers, architects, and business stakeholders.

---

## 1. User Personas & Permissions
The system is designed for three distinct user roles, each with specific permissions and workflows.

| Persona | Role ID | Access Level | Responsibilities |
| :--- | :--- | :--- | :--- |
| **Demand Planner** | `ROLE_PLANNER` | **Read/Write** | Daily optimization of stock, reviewing "High Confidence" vs "Low Confidence" forecasts, triggering replenishment orders. |
| **Supply Chain Manager** | `ROLE_MANAGER` | **Read/Write** | Approving high-value orders (>$50k), viewing strategic dashboards, running long-term "What-If" scenarios (e.g., "What if we open a new DC?"). |
| **System Admin** | `ROLE_ADMIN` | **Full Admin** | User management (Cognito), adjusting system configurations (SLA thresholds, Risk tolerances), monitoring pipeline health. |

---

## 2. Detailed User Stories & Data Flow

This section traces the **exact lifecycle** of a user interaction, including the specific **API Contracts** (JSON payloads) used.

### Story A: The Planner Reviews a Low-Stock Alert
**Scenario**: It is 10:00 AM. The system detects that "Dairy Milk 200ml" is selling faster than expected in the "London" region.

#### Step 1: Real-time Notification (The Trigger)
*   **Mechanism**: The frontend maintains a **WebSocket** connection (via API Gateway WebSocket API) or **Long Polling** interval (15s) to the `/notifications` endpoint.
*   **Event Payload (Server -> Client)**:
    ```json
    {
      "event_id": "evt_882391",
      "type": "STOCK_ALERT",
      "severity": "CRITICAL",
      "timestamp": "2023-10-27T10:00:05Z",
      "data": {
        "sku_id": "SKU_055",
        "location_id": "LOC_LON_01",
        "current_stock": 140,
        "burn_rate": 55.2,
        "estimated_depletion": "2023-10-27T14:30:00Z"
      }
    }
    ```
*   **UI Behavior**: A "Toast" notification appears: *"Critical: Dairy Milk 200ml will stick out by 2:30 PM."*

#### Step 2: Investigation (The Read)
*   **User Action**: Planner clicks the notification. The app navigates to `/forecasts/SKU_055`.
*   **API Call**: `GET /api/v1/forecasts/SKU_055?location=LOC_LON_01`
*   **Backend Logic**:
    1.  API Gateway verifies `Authorization: Bearer <JWT>`.
    2.  Lambda queries **DynamoDB** (Table: `Forecasts`) for the partition key `SKU_055#LOC_LON_01`.
*   **Response Payload**:
    ```json
    {
      "sku_metadata": { "name": "Dairy Milk 200ml", "category": "Dairy" },
      "forecast_data": [
        { "date": "2023-10-27", "predicted": 500, "actual_so_far": 360, "lower_bound": 450, "upper_bound": 550 },
        { "date": "2023-10-28", "predicted": 520, "actual_so_far": 0, "lower_bound": 480, "upper_bound": 560 }
      ],
      "ml_explainability": {
        "primary_driver": "Competitor Promo",
        "driver_impact": "+15%"
      }
    }
    ```

#### Step 3: Action (The Write)
*   **User Action**: Planner clicks "Emergency Replenish".
*   **API Call**: `POST /api/v1/orders/replenish`
*   **Request Payload**:
    ```json
    {
      "sku_id": "SKU_055",
      "source_dc_id": "DC_NORTH",
      "target_location_id": "LOC_LON_01",
      "quantity": 500,
      "priority": "URGENT"
    }
    ```
*   **Backend Logic**:
    1.  Lambda writes an "Order" item to **DynamoDB**.
    2.  Lambda publishes an event to **EventBridge**: `com.fmcg.orders.placed`.
    3.  A subscriber rule triggers the ERP integration Lambda to sync with SAP.

---

## 3. Frontend Architecture

The user interface is a sophisticated React application designed for high performance and state consistency.

### 3.1 Tech Stack
*   **Core**: React 18 (Functional Components, Hooks).
*   **State Management**: **React Query (TanStack Query)**.
    *   *Why?* We deal with Server State (data on AWS), not Client State. React Query handles caching, background refetching (stale-while-revalidate), and optimistic updates effectively.
*   **Global State**: **Zustand** (for lightweight UI state like "isSidebarOpen" or "currentUserRole").
*   **Visualization**: **Recharts** for D3-based responsive time-series charts.
*   **Styling**: **Tailwind CSS** for utility-first, consistent design tokens.

### 3.2 Key UX Components
1.  **The Forecasting Chart (Complex Component)**:
    *   **Logic**: Renders three synchronized lines (Historical, Forecast, Upper/Lower Bounds).
    *   **Interactivity**: "Brush" tool to zoom in on specific weeks.
    *   **Tooltip**: Custom tooltip showing exact values and "Reason Codes" (e.g., "Mothers Day Uplift").
2.  **The Scenario Simulator**:
    *   **Logic**: A form with sliders (Price, Weather, Marketing Spend).
    *   **Behavior**: Debounced inputs (500ms) trigger a re-calculation API call. The chart updates optimistically to show a "Ghost Line" of the new scenario.

### 3.3 Folder Structure (Proposed)
```
/src
  /api          # Axios instances and API endpoint definitions
  /components
    /charts     # Reusable Recharts wrappers
    /layout     # Sidebar, Header, ProtectedRoute
  /hooks        # Custom React Query hooks (useForecast, useSKUMetadata)
  /pages        # Dashboard, ForecastView, Settings
  /types        # TypeScript interfaces (SKU, SalesRecord, ForecastResponse)
  /utils        # Date formatters (date-fns), Currency formatters
```

---

## 4. Operational Workflows

### 4.1 "Day in the Life" of the System
1.  **02:00 AM**: **AWS Glue** Jobs wake up. They process yesterday's raw sales CSVs from S3.
2.  **03:00 AM**: **Batch Transform Job** runs. The trained XGBoost model predicts demand for the next 14 days.
3.  **04:00 AM**: Results are loaded into **DynamoDB** for fast access.
4.  **09:00 AM**: Planners log in. They see fresh data instantly (served from DynamoDB Cache).
5.  **09:00 AM - 05:00 PM**:
    *   **Streaming**: Virtual sales flow in via Kinesis.
    *   **Alerting**: CloudWatch Alarms trigger SNS if deviations > 20% occur.
    *   **Drift Detection**: SageMaker Model Monitor checks if today's data distribution matches the training data. If not, it flags "Concept Drift".

---

## 5. Security & Compliance
*   **Data Encryption**:
    *   **At Rest**: S3 Buckets and DynamoDB tables encrypted with **AWS KMS** (AES-256).
    *   **In Transit**: All API calls use **TLS 1.2+** via HTTPS.
*   **Audit Logging**:
    *   Every "Write" action (e.g., placing an order, changing a forecast override) is logged to **CloudTrail** and a separate immutable **DynamoDB Audit Table** (`Audit_Log`).
    *   *Payload*: `{ actor: "user_123", action: "OVERRIDE_FORECAST", old_val: 500, new_val: 600, reason: "Local Festival" }`

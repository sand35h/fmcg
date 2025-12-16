# Deep Dive Technical Architecture

This document provides the **Implementation Details** for the FMCG Forecasting System. It goes beyond the "what" (Services) to the "how" (Configuration, Networking, Security).

---

## 1. Networking & VPC Architecture (The Hidden Layer)

Although "Serverless" implies no servers, our system must operate within a secure network boundary (VPC) to access private resources like RDS (if used) or internal APIs.

### 1.1 Virtual Private Cloud (VPC) Design
*   **CIDR Block**: `10.0.0.0/16` (65,536 IPs)
*   **Subnets**:
    1.  **Public Subnets (x2)**: `10.0.1.0/24`, `10.0.2.0/24`. Hosts: NAT Gateway, Application Load Balancer.
    2.  **Private App Subnets (x2)**: `10.0.3.0/24`, `10.0.4.0/24`. Hosts: **Lambda Interfaces**, Glue Jobs.
    3.  **Private Data Subnets (x2)**: `10.0.5.0/24`, `10.0.6.0/24`. Hosts: **VPC Endpoints** (Interface/Gateway) for S3 and DynamoDB.

### 1.2 Security Groups (The Firewalls)
*   `sg-lambda`: Allow Outbound TCP 443 (APS calls). Allow Outbound to `sg-db`.
*   `sg-db`: Allow Inbound TCP 5432 **only** from `sg-lambda`. **Deny All** from Public Internet.

---

## 2. Infrastructure as Code (IaC) Structure

We use **AWS SAM (Serverless Application Model)** to define the infrastructure. The `template.yaml` is the source of truth.

### 2.1 Directory Structure
```
root/
├── template.yaml            # The Master Blueprint
├── events/                  # Sample JSON events for local testing
├── src/
│   ├── backend/
│   │   ├── get_forecast/    # Lambda Function Code
│   │   │   ├── app.py
│   │   │   └── requirements.txt
│   │   └── ingest_stream/
│   ├── layers/              # Shared Code (Pandas, Utils)
│   │   └── python/
│   │       └── lib/
```

### 2.2 Key SAM Resource Definitions
#### **The API Gateway**
```yaml
FmcgApi:
  Type: AWS::Serverless::Api
  Properties:
    StageName: v1
    Auth:
      DefaultAuthorizer: CognitoAuthorizer
      Authorizers:
        CognitoAuthorizer:
          UserPoolArn: !GetAtt UserPool.Arn
```

#### **The Core Lambda Function**
```yaml
GetForecastFunction:
  Type: AWS::Serverless::Function
  Properties:
    CodeUri: src/backend/get_forecast/
    Handler: app.lambda_handler
    Runtime: python3.9
    MemorySize: 2048          # High memory for Pandas operations
    Timeout: 30               # Latency SLA is <5s, but 30s for safety
    Policies:
      - DynamoDBReadPolicy:   # "Least Privilege" Access
          TableName: !Ref ForecastTable
      - S3ReadPolicy:
          BucketName: !Ref DataLakeCurated
```

---

## 3. Data Storage Schema Design

### 3.1 DynamoDB Table: `Forecasts`
Designed for single-digit millisecond retrieval by specific SKU+Location.

*   **Partition Key (PK)**: `SKU#{sku_id}` (e.g., `SKU#SKU_055`)
*   **Sort Key (SK)**: `LOC#{location_id}#DATE#{date}` (e.g., `LOC#LON_01#DATE#2023-10-27`)
*   **Attributes**:
    *   `forecast_val` (Number)
    *   `confidence_lower` (Number)
    *   `confidence_upper` (Number)
    *   `ttl` (Number): Epoch timestamp for auto-deletion after 90 days.

### 3.2 S3 Data Lake Structure
S3 allows us to partition data for performant Athena/Glue queries.
*   **Format**: `s3://bucket/table/year/month/day/file.parquet`
*   **Raw Zone**: `s3://fmcg-raw/sales/2023/10/27/stream-shard-01.json`
*   **Curated Zone**: `s3://fmcg-curated/sales/year=2023/month=10/day=27/data.parquet` (Partitioned by Hive style)

---

## 4. Security & IAM (Identity and Access Management)

We follow the **Principle of Least Privilege**. No `AdministratorAccess` policies allowed.

### 4.1 Lambda Execution Roles
Each Lambda function has its own specialized role.
*   **Role**: `GetForecastLambdaRole`
    *   **Allowed**: `dynamodb:GetItem`, `dynamodb:Query` on `resource: table/Forecasts`.
    *   **Allowed**: `logs:CreateLogGroup`, `logs:PutLogEvents`.
    *   **Denied**: `s3:*` (It should not access S3 directly if it only reads DynamoDB).

### 4.2 Cognito User Groups
*   `AdminGroup`: Can call `/admin/*` and `/users/*`.
*   `PlannerGroup`: Can call `GET /forecasts` and `POST /orders`.
*   `ViewerGroup`: Can call `GET /forecasts` **only**.

---

## 5. Observability & Monitoring Strategy

We don't just "deploy and pray". We monitor every heartbeat.

### 5.1 CloudWatch Metrics
Custom Metrics we publish from Lambda/Kinesis:
1.  `FMCG/Business`: `StockoutCount` (Count of SKUs with stock = 0).
2.  `FMCG/Business`: `ForecastBias` (Percentage difference between Forecast vs Actual).
3.  `FMCG/System`: `StreamLagMilliseconds` (How far behind is the storage from the realtime stream?).

### 5.2 X-Ray Tracing
Enable **AWS X-Ray** on API Gateway and Lambda. This visualizes the request chain:
*   *Trace Graph*: `Client -> API GW (15ms) -> Lambda (200ms) -> DynamoDB (5ms)`.
*   *Why?*: Helps identify if the bottleneck is Python code or Database latency.

### 5.3 Alarms
*   **Critical**: If `StockoutCount > 50` for 15 minutes -> PagerDuty/SNS (Simulated Email).
*   **Warning**: If `ForecastBias > 25%` -> Email to Data Science Team (Model is drifting).

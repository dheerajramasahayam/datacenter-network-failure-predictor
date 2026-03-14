# Architecture

## Data Flow
The architecture simulates the flow of metrics from networking hardware to a predictive alerting system.

**Routers → NetFlow → Telemetry Collector → ML Predictor → Alert Engine**

1. **Routers & Switches (Simulated)**: Generate raw data regarding network traffic, interface statuses, etc.
2. **NetFlow/sFlow Exporter**: Formats and exports this data as flow records.
3. **Telemetry Collector**: Aggregates the flow data into a central data store (represented by `telemetry/sample_netflow.csv`). Features tracked:
   - Packet loss (%)
   - Latency (ms)
   - Interface errors (count)
4. **ML Predictor (`src/predictor.py`)**: 
   - A Random Forest machine learning model ingests this historical data alongside labels.
   - It performs anomaly detection (`src/anomaly_detection.py`) to automatically identify spikes.
   - It trains on these features to continuously predict the likelihood of an `outage`.
5. **Alert Engine (Simulated via script output)**: Acts on predictions from the ML model to warn administrators of impending failures, essentially automating the proactive mitigation of network disruptions.

## Key Test Scenario
The core testing logic (`tests/test_prediction.py`) explicitly validates the system's ability to trigger the Alert Engine when specific conditions are met:
- **Scenario**: Interface errors increase significantly.
- **Expected Outcome**: The ML model correctly outputs a prediction of `1` (potential outage).

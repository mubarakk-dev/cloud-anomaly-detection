# Inference architecture

The deployment layer is deliberately small so that it remains traceable to the evaluated research pipeline.

1. `POST /events` validates a single raw telemetry observation.
2. `StreamingWindowDetector` assigns it to a `(window_start, service)` buffer.
3. Arrival of a later timestamp closes eligible buffers.
4. `aggregate_windows` calculates the same behavioural features used offline.
5. The persisted scikit-learn pipeline applies one-hot encoding and Random Forest inference.
6. The saved validation threshold converts the anomaly probability to a decision.
7. The API returns the window, service, score, threshold, prediction, and processing time.

The API does not accept ground truth. The dashboard can show simulator labels beside predictions because it evaluates a controlled replay, but those fields do not enter the model.

## State and ordering

State is process-local and held in memory. Events are expected in non-decreasing timestamp order. `/reset` clears state and `/flush` closes remaining buffers for finite tests. A production implementation would need durable state, explicit late-event policy, authentication, monitoring, back-pressure, and horizontal coordination.

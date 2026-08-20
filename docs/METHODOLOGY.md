# Methodology summary

The final study generates one observation per service per simulated second for six services. Normal behaviour contains Gaussian noise, periodic load variation, occasional warnings, errors, and high status codes. Labelled latency, resource-pressure, error-rate, and partial-degradation incidents modify observable telemetry for one service over 30–90 seconds.

Raw observations are represented either individually or as fixed, non-overlapping behavioural service windows. Window features comprise service identity; average and maximum response time, CPU, and memory; server-error, warning, and error-log rates; and observation count. Ground-truth metadata is excluded from all model features.

Random Forest and Isolation Forest are evaluated at 15, 30, and 60 seconds. Each run uses chronological 60% training, 20% validation, and 20% test partitions. Preprocessing and models are fitted on training data, thresholds are selected using validation anomaly F1, and the test set is used once for final evaluation. Results are repeated across seeds 42, 123, 456, 789, and 2026.

The bundled deployment artifact is the final 30-second Random Forest pipeline. This duration offers a useful demonstration balance; it is not presented as universally optimal. The API and dashboard reuse the exact `aggregate_windows` function and persisted preprocessing/model artifact used offline.

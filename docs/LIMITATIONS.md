# Limitations

- Telemetry and incident labels are synthetic and reflect manually specified generation assumptions.
- Six simulated services do not represent the diversity, dependency structure, traffic, missingness, or concept drift of a production estate.
- Five seeds expose some stochastic variation but provide limited statistical power.
- Fixed non-overlapping windows simplify interpretation but make decisions boundary-sensitive.
- Validation-labelled data calibrates the Isolation Forest threshold, so it is not a completely label-free deployed system.
- The API and dashboard are local prototypes. No claim is made about production throughput, resilience, security, or scalability.
- The bundled artifact is intended to make the demonstration reproducible; operational systems would require model registry, schema versioning, monitoring, and retraining policy.

These constraints bound the conclusions: the results support behavioural aggregation within the implemented environment and motivate evaluation on real operational telemetry.

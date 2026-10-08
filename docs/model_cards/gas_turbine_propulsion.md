# Model card: gas-turbine propulsion health and diagnosis (`naval_gas_turbine`)

**Intended use:** early warning of compressor / turbine decay on a marine gas turbine, and
which component it is, across all load levels. Decision support for the engineering
watch and the maintenance planner; it does not trip or control the engine.

**Models:** load-aware residuals (per-lever-position expected values) → Mahalanobis
health score fitted on healthy states only, alarm at the 99th percentile; LightGBM
fault classifier (healthy / compressor / turbine / combined) and decay-severity
regressors. LightGBM and XGBoost tie in grouped CV (macro-F1 0.947).

**Data:** UCI naval propulsion plant, a validated **simulator** of a frigate CODLAG gas
turbine; 1,326 decay states x 9 load levels. Split by decay state (no state in both
train and test).

**Performance (held-out states):** detection 95% (95% CI 94-97) at 3% false alarms
(0-7); exhaust-temperature alarm 0%; diagnosis accuracy 97.8%, macro-F1 0.94;
decay within 1.1-1.7% of range. Offline slice check by lever position in the dashboard.

**Limits:** simulator data, 99 healthy test states (wide false-alarm interval); the
trained model does not transfer to a real engine - the method does, with that
engine's own healthy baseline.

**Monitoring and retraining:** PSI drift on normal-operation residuals > 0.25 or > 5%
of samples outside the trained load envelope → retrain; precision of acknowledged
alerts below 50% (upper CI) → recalibrate.

**Safety mapping:** compressor / turbine decay → loss of propulsion (severity 3),
combined → severity 4; two turbines installed, one required.

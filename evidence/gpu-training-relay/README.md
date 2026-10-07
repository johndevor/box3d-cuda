# Historical training telemetry, not a promotion gate

`leg6-metrics.jsonl` was published by the incoming `duck-grid-walk` branch in
commit `5eb06a452274e6be306be43d24d75c0f675f72d4`. Preserve the raw rows.
They are not results produced by this integration review.

The trainer computes `steps_per_s` from rollout time only; it excludes PPO
update time. Do not label that field end-to-end training throughput. Compare
the increment in `env_steps` against `wall_update_s` for a per-update training
rate, and include setup/evaluation/export time for a whole-run rate.

The rows alone do not pin the executed library, model, GPU or source bundle,
nor provide independent gait/contact acceptance. `faults_total=7348` is a
cumulative recorded counter, not evidence that this segment introduced that
many faults. The integration adds fast-policy fault quarantine; these older
rows cannot validate that fix or establish the validity of earlier samples.

No walking, 200k-active-body, independent-physics, or current-head CUDA
performance claim follows from this telemetry. A new exact-artifact GPU
admission and separate gait evaluation are still required.

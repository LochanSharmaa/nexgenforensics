# configs/

Runtime configuration comes from environment variables, not from files here.
See [`.env.example`](../../.env.example) for the full list and
[`imatch_api/core/config.py`](../imatch_api/core/config.py) for validation and
defaults.

This directory previously held `model_config.yaml`, `inference_config.yaml`,
`security_config.yaml`, and `deployment_config.yaml`. They were removed rather
than updated: nothing read them, and they described an eight-backbone ensemble
architecture that did not exist. A configuration file that no code loads is
worse than none — it looks authoritative while drifting arbitrarily far from
what the system actually does.

`training_config.yaml` is kept as a reference for fine-tuning runs, which are
driven by `nexgen_engine/training/` rather than by the service.

| Setting | Where it lives now |
|---|---|
| Model pack, device, engine mode | `NEXGEN_MODEL_PACK`, `NEXGEN_ENGINE_DEVICE`, `NEXGEN_ENGINE_MODE` |
| Thresholds | `NEXGEN_MATCH_THRESHOLD`, `NEXGEN_REVIEW_THRESHOLD`, `NEXGEN_VERIFY_THRESHOLD` |
| Quality gates | `NEXGEN_MIN_QUALITY`, `NEXGEN_MIN_DETECTION_CONFIDENCE` |
| Security | `NEXGEN_JWT_SECRET`, `NEXGEN_TEMPLATE_KEY`, `NEXGEN_RATE_LIMIT_PER_MINUTE` |
| Deployment | `deployment/k8s/`, `deployment/Dockerfile`, `docker-compose.yml` |

# Fine-Tuning Guide

The local engine uses deterministic embeddings so it can run without GPU weights. Production fine-tuning should replace the deterministic backbones with trained PyTorch modules that preserve the same interface.

1. Prepare consent-approved client data with workspace isolation.
2. Run quality filtering and keep the highest-scoring approved samples.
3. Train backbone adapters or full backbones with the four-stage curriculum.
4. Validate on held-out verification and identification sets.
5. Generate a benchmark report and compare against the active model.
6. Deploy only with a rollback checkpoint and audit entry.

Do not publish accuracy claims until results are independently validated.

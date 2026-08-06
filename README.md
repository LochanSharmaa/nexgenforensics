<div align="center">

# NexGen iMATCH

**Forensic face recognition — auditable 1:1 verification and 1:N gallery search, with every accuracy claim measured and published with its protocol.**

[![CI](https://github.com/LochanSharmaa/nexgenforensics/actions/workflows/ci.yml/badge.svg)](https://github.com/LochanSharmaa/nexgenforensics/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

</div>

Working system, locally verified — not deployed, not independently validated. Runs stock InsightFace ArcFace (`buffalo_l` / `w600k_r50`) at a calibrated verification threshold of 0.2871 (FMR ≈ 0.1%). Results ship with plain-language explanations, a hash-chained audit trail, encrypted templates at rest, and forensic report export.

## Quick start

Python 3.11+ and Node 18+. An NVIDIA GPU with CUDA 12.x is optional — CPU works, more slowly.

```bash
git clone https://github.com/LochanSharmaa/nexgenforensics.git
cd nexgenforensics
python scripts/setup_gpu.py        # GPU hosts; CPU-only: pip install -r backend/requirements.txt -r backend/requirements-engine.txt -r backend/requirements-cpu.txt
cp .env.example .env               # then set NEXGEN_JWT_SECRET and NEXGEN_TEMPLATE_KEY (generation commands are in the file)
python backend/scripts/bootstrap_admin.py --email you@example.com --password '<strong password>' --role admin
python -m uvicorn imatch_api.main:app --host 127.0.0.1 --port 8443 --app-dir backend
```

Frontend: `cd frontend && npm install && npm run dev`. API docs at `http://127.0.0.1:8443/docs`.

## Benchmarks

1:1 verification accuracy, 10-fold cross-validation on published protocols: **LFW 99.78%** · AgeDB-30 96.68% · CPLFW 94.47% · **TinyFace 82.45%** (surveillance resolution). At the deployed threshold, TAR@FAR=0.1% is 96.03% on clean imagery and 33.13% on TinyFace.

Full protocols, per-fold results, demographic breakdown and negative results: [BENCHMARKS.md](BENCHMARKS.md).

## Limitations

- **Degraded surveillance footage is weak** — at FAR=0.1%, roughly one genuine match in five on TinyFace, and on QMUL-SurvFace's open-set split the system cannot reject strangers at all. See [docs/MEASUREMENT_RECORD.md](docs/MEASUREMENT_RECORD.md).
- **Not independently validated** — no NIST FRTE submission; every figure here is internal.
- **Error rates are not uniform across demographics** — see [BENCHMARKS.md](BENCHMARKS.md) §5.
- **Liveness and synthetic-media checks are heuristics**, not certified anti-spoofing, and say so in their own output.

The complete issue register is [SCORECARD.md](SCORECARD.md); every product claim maps to its backing test in [CLAIMS.md](CLAIMS.md).

## Responsible use

This is biometric identification software. Outputs are investigative leads, not identifications — a qualified examiner must verify any candidate before it is relied upon, and every response carries that notice. A lawful basis is a required, audited field on every search. Compliance with biometric-data law (GDPR, BIPA and equivalents) is the deployer's responsibility.

## Security

No credentials, API keys or tokens are committed to this repository — tracked `.env.example` files contain placeholders only, and this has been verified against the full git history. Real secrets (`NEXGEN_JWT_SECRET`, `NEXGEN_TEMPLATE_KEY`) live in a git-ignored `.env` and are generated per install. Biometric templates are encrypted at rest with AES-256-GCM. Please report suspected vulnerabilities privately via GitHub security advisories rather than public issues.

## Documentation

[BENCHMARKS.md](BENCHMARKS.md) · [CLAIMS.md](CLAIMS.md) · [SCORECARD.md](SCORECARD.md) · [ROADMAP.md](ROADMAP.md) · [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) · review package in [delivery/](delivery/) · operator guides in [docs/](docs/)

## Testing

```bash
pytest backend/tests_engine/ backend/tests/     # full suite
python backend/scripts/regression_check.py      # accuracy + configuration regression gate
python scripts/verify_gpu.py                    # verified GPU binding
```

## License

[MIT](LICENSE). The InsightFace model packs downloaded at runtime and several evaluation datasets carry their own non-MIT, research-use-only licences — check both before commercial use.

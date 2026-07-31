# Contributing

Issues and pull requests are welcome.

## The one rule that matters

**No accuracy claim without a measurement.**

If a change touches the model pack, decision thresholds, fusion method, or the
embedding pipeline, run the regression gate and include its output in the PR:

```bash
python backend/scripts/regression_check.py
```

It compares measured accuracy against a recorded baseline *and* asserts
configuration invariants. It exits non-zero on regression.

If the gate fails, do not refresh the baseline to make it pass. Establish why
the number moved first. A baseline updated to silence a failure defeats the
gate.

## Before opening a PR

```bash
pytest backend/tests_engine/ backend/tests/     # 161 tests, all must pass
python backend/scripts/regression_check.py      # must exit 0
python scripts/verify_gpu.py                    # if you changed anything GPU-related
```

Run both test directories **together**, not separately. A threshold change once
passed `tests_engine/` while breaking `tests/`, and only the combined run caught
it.

## House conventions

**Heuristics must say they are heuristics.** The liveness and synthetic-media
screens are not trained models and are not certified detection. Every layer that
surfaces their scores — API response, PDF report, UI — carries
`certified: false` and the method name alongside. If you add a component of this
kind, label it the same way, and do not let product copy imply otherwise.

**Thresholds live in exactly one place**: `nexgen_engine/config.py::ThresholdConfig`.
Everything else derives from it. This project previously shipped four copies
that drifted apart, and the UI displayed a decision rule the engine had stopped
applying. The regression gate now fails if a second copy reappears.

**Prefer a loud failure to a silent fallback.** Silent CPU fallback, silent
threshold drift, and unhandled exceptions surfacing as 500s have each caused
real defects here. A typed error the API can map to a 4xx is better than an
exception that escapes.

**State limitations in the same place as the claim.** If you add a capability,
add its constraints to [CLAIMS.md](CLAIMS.md) and link the test that backs it.
Anything that cannot be backed that way should be reworded or removed.

## Reporting a security issue

Do not open a public issue for a vulnerability affecting biometric data
handling, authentication, or template storage. Contact the maintainers
privately first.

## Licence

Contributions are accepted under the [MIT licence](LICENSE). Note that the
InsightFace model packs downloaded at runtime carry their own licences, which
are not MIT.

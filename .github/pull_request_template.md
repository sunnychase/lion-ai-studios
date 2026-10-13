## What and why

<!-- What does this change, and what problem does it solve? Link the issue: "Fixes #123". -->

## How it was tested

<!-- Only what you actually ran, with the result. -->

- [ ] `bash scripts/check.sh` passes
- On a DGX Spark (if it touches rendering): <!-- builder, preset, length, model, and what you saw; or "not run" and why -->

## Checklist

- [ ] One focused change.
- [ ] Machine-specific values come from `studio.env` through `app/config.py`; new settings are in `studio.env.example` with a comment.
- [ ] No personal paths, host names, addresses, tokens or e-mail addresses in code, docs or commit messages.
- [ ] CHANGELOG.md has a dated entry, and the README or `docs/` are updated if behaviour, settings or requirements changed.
- [ ] Numbers in docs come from a real run (hardware and number of runs stated).
- [ ] Anything not tested is said above.

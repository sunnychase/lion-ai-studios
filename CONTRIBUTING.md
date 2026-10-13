# Contributing to Lion AI Studios

Thanks for helping. Bug reports, fixes, documentation and focused features are all welcome.

## Before you start

- **Bugs and ideas:** open an [issue](https://github.com/sunnychase/lion-ai-studios/issues) with the bug or feature form.
  Search first; one problem or idea per issue.
- **Security problems:** do not open a public issue; see [SECURITY.md](SECURITY.md).
- **Bigger changes** (a new builder, a new engine, a change to how jobs run): open an issue first so the approach can be
  agreed before you spend time on it.

## Running it locally

The studio is meant for an NVIDIA DGX Spark; follow [Quick start](README.md#quick-start) for the full setup. For development,
run a second copy beside your normal one so your library is not touched:

```bash
cp studio.env.example studio.env      # once; edit paths
set -a; . ./studio.env; set +a
STUDIO_HOME=$HOME/lion-studio-dev STUDIO_PORT=8492 python3 app/studio_server.py
```

The web server uses only the Python standard library. `app/index.html` is read on every request, so page changes show up on
reload; Python changes need a restart.

Much of the code can be exercised without a GPU: importing `app/studio_server.py` starts nothing, and the queue, parameters
and helpers can be called directly with a throw-away `STUDIO_HOME`.

## House rules

- **Machine settings live in `studio.env`, read through `app/config.py`.** Paths, ports, URLs, service names and commands
  never go into the code. A new setting gets a safe default in `config.py` and a commented line in `studio.env.example`.
- **No personal or machine-specific data**: no home-directory paths, host names, private addresses, user names, tokens or
  e-mail addresses, in code, docs, comments or commit messages.
- **The web server stays standard-library only.** Anything that needs packages runs as a helper in its own environment
  (see `voice.py`, `midi_tool.py`, `ming_runner.py`), and helpers import only `config.py`.
- **Optional features stay out of the way**: a model appears only once it is installed, and an engine that needs extra
  software is off until it is configured.
- **Keep the guard rails**: consent checkboxes, localhost by default, the memory and thermal guards.
- **Say what changed**: add a dated entry to [CHANGELOG.md](CHANGELOG.md) and update the README or `docs/` when behaviour,
  settings or requirements change. Numbers in docs must come from a real run; say on what hardware and how many runs.

## Checks

Run before every commit. CI (`.github/workflows/ci.yml`) runs the same script on every push and pull request:

```bash
bash scripts/check.sh
```

It compiles every Python file (`python3 -m py_compile`), runs `node --check` on each inline `<script>` of `app/index.html`,
and fails on home-directory paths, private network addresses or token-shaped strings in tracked files.

If your change touches rendering, say in the pull request what you ran on a Spark (preset, length, model) and what you saw.

## Pull requests

- One focused change per pull request, with a short imperative title ("Video: keep the chosen length").
- Fill in the template: what changed, why, how you tested it, and anything you could not test.
- Do not commit `studio.env`, models, outputs, `__pycache__` or anything under your `STUDIO_HOME`.

By contributing you agree that your contribution is licensed under the [MIT licence](LICENSE).

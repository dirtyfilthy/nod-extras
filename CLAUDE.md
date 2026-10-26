# CLAUDE.md

Sibling of the `nod` repo. Nod is a deliberately thin agent core; everything
richer than its five built-in tools lives here. Keep the same style: plain
over clever, stdlib plus what `nod` already depends on, no new runtime
dependencies without a reason.

## Commands

```sh
uv run --extra dev pytest -q      # needs ../nod checked out (path dependency)
```

Tests never call a model: `tests/conftest.py` has a scripted `FakeProvider`.

## Layout

- `src/nod_extras/jobs.py`: the `Jobs` manager (ids, deadlines, limits, the
  reaper thread, the on-disk job record), the `shell` and `agent` job kinds,
  and the job tools. It depends on `nod.inbox` for delivery and on
  `nod.agent.current_session` to know which session a tool call belongs to.
  The design it implements is `docs/background-jobs.md` in the `nod` repo.
- `skills/`: SKILL.md files, scanned by Nod's default `skills_dirs`.

Nod's `cli.py:build_agent` is the only place that wires this package in, and
it does so with a soft import, so Nod runs without it.

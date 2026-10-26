# nod-extras

The parts of [Nod](https://github.com/dirtyfilthy/nod) that don't belong in
the thin core. The core stays small and readable; anything richer lives here
and is wired in through `nod/cli.py:build_agent`.

What's here:

- `nod_extras.jobs`: background jobs. Shell commands and subagents that run
  off the main loop, with deadlines, cancellation and a `<job-finished>`
  event posted to the agent's inbox when each one ends.
- `skills/`: SKILL.md files. Nod's default config scans `../nod-extras/skills`.

## Using it

Check this repo out next to `nod` and run Nod with it on the path:

```sh
cd ../nod
uv run --with-editable ../nod-extras nod
```

`build_agent` imports `nod_extras.jobs` if it can and registers the job tools
(`start_shell_job`, `spawn_agent`, `list_jobs`, `kill_job`, `wait_jobs`).
Without it, Nod runs with the five built-in tools and nothing else.

## Tests

```sh
uv run --extra dev pytest -q
```

Tests use a scripted provider and short real shell commands; nothing calls a
model.

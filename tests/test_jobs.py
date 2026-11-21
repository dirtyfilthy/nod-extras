from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest
from conftest import FakeProvider

from nod.agent import Agent
from nod.inbox import InMemoryInbox
from nod.memory import InMemoryMemory
from nod.messages import ToolCall, assistant
from nod.tools import Tool, ToolRegistry
from nod_extras.jobs import AgentJobKind, JobError, Jobs, ShellJobKind, job_tools


def _wait(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _wait_event(inbox, session="s", timeout=5.0):
    assert _wait(lambda: inbox.pending(session), timeout), "no event arrived"
    return inbox.pending(session)[0].text


@pytest.fixture
def inbox():
    return InMemoryInbox()


@pytest.fixture
def jobs(inbox, tmp_path):
    j = Jobs(inbox, [ShellJobKind(term_grace=0.3)], jobs_dir=tmp_path, poll_interval=0.02)
    yield j
    j.shutdown()


def _call(tools, name, **arguments):
    return ToolRegistry(tools).dispatch(ToolCall(id="c", name=name, arguments=arguments))


# --- shell kind --------------------------------------------------------------


def test_start_shell_job_returns_at_once_and_reports_when_done(jobs, inbox, tmp_path):
    tools = job_tools(jobs)
    t0 = time.monotonic()
    out = _call(tools, "start_shell_job", command="sleep 0.2; echo hello; exit 3")
    assert time.monotonic() - t0 < 0.1
    assert out.startswith("started job 1 (shell), log: ")
    assert "still running" in _call(tools, "list_jobs") or "running for" in _call(tools, "list_jobs")

    text = _wait_event(inbox, "default")
    assert text.startswith("<job-finished id=1 kind=shell status=done duration=")
    assert "command: sleep 0.2; echo hello; exit 3" in text
    assert "exit code 3" in text
    assert "--- last 50 lines (full log: " in text
    assert "\nhello\n</job-finished>" in text
    assert (tmp_path / "default" / "1.log").read_text() == "hello\n"
    assert "#1 shell done after" in _call(tools, "list_jobs")


def test_event_is_delivered_to_the_agent_before_its_next_provider_call(jobs, inbox):
    memory = InMemoryMemory()
    agent = Agent(provider=FakeProvider([]), registry=ToolRegistry(job_tools(jobs)), memory=memory, inbox=inbox)

    def start(_messages):
        return assistant(tool_calls=[ToolCall(id="c1", name="start_shell_job", arguments={"command": "echo hi"})])

    def check(messages):
        # Block until the reaper has posted, as a model "thinking" for a while would.
        assert _wait(lambda: any(m.kind == "event" for m in messages) or inbox.pending("s"))
        return assistant("waiting")

    def after(messages):
        events = [m for m in messages if m.kind == "event"]
        assert len(events) == 1 and events[0].role == "user"
        assert "<job-finished id=1 kind=shell status=done" in events[0].content
        return assistant("seen")

    agent.provider = FakeProvider([start, check, after])
    agent.send("run it", session_id="s")
    # The event went into the inbox during `check`; the next send delivers it.
    _wait(lambda: inbox.pending("s"))
    assert agent.send("and?", session_id="s") == "seen"
    stored = memory.load("s")
    assert sum(1 for m in stored if m.kind == "event") == 1


def test_wait_jobs_returns_the_event_and_it_is_not_injected_again(jobs, inbox):
    memory = InMemoryMemory()
    agent = Agent(provider=FakeProvider([]), registry=ToolRegistry(job_tools(jobs)), memory=memory, inbox=inbox)
    seen: dict[str, str] = {}

    def start(_messages):
        return assistant(tool_calls=[ToolCall(id="c1", name="start_shell_job", arguments={"command": "echo done"})])

    def wait(messages):
        return assistant(tool_calls=[ToolCall(id="c2", name="wait_jobs", arguments={"ids": [1], "max_seconds": 5})])

    def check(messages):
        seen["wait_result"] = [m for m in messages if m.role == "tool"][-1].content
        assert not [m for m in messages if m.kind == "event"]
        return assistant("ok")

    agent.provider = FakeProvider([start, wait, check])
    assert agent.send("go", session_id="s") == "ok"
    assert "<job-finished id=1 kind=shell status=done" in seen["wait_result"]
    assert inbox.pending("s") == []
    assert not [m for m in memory.load("s") if m.kind == "event"]

    # Asking again reports it as already handed out.
    out = _call_in_session(jobs, "s", "wait_jobs", ids=[1], max_seconds=1)
    assert out == "job 1 (shell) finished with status done; already reported"


def _call_in_session(jobs, session, name, **arguments):
    from nod.agent import _current_session

    token = _current_session.set(session)
    try:
        return _call(job_tools(jobs), name, **arguments)
    finally:
        _current_session.reset(token)


def test_wait_jobs_times_out_and_reports_running(jobs):
    _call_in_session(jobs, "s", "start_shell_job", command="sleep 5")
    t0 = time.monotonic()
    out = _call_in_session(jobs, "s", "wait_jobs", ids=[1], max_seconds=0)
    assert time.monotonic() - t0 < 1
    assert out.startswith("job 1 (shell) still running after")
    assert _call_in_session(jobs, "s", "wait_jobs", ids=[9]).startswith("error: ")


def test_shell_timeout_kills_the_whole_process_group(jobs, inbox, tmp_path):
    pidfile = tmp_path / "child.pid"
    job = jobs.start(
        "shell", "s", description="command: x", timeout=1,
        command=f"sleep 30 & echo $! > {pidfile}; wait",
    )
    # The deadline is at least 1s by design; shorten it for the test.
    job.deadline = time.time() + 0.2
    text = _wait_event(inbox)
    assert "status=timed_out" in text
    assert "timed out after 1 s; process group killed" in text
    child_pid = int(pidfile.read_text())
    assert _wait(lambda: not _alive(child_pid), 3), "backgrounded child survived the kill"


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie counts as dead for our purposes; reap it if it's ours.
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


def test_sigterm_ignoring_job_is_killed_after_grace(jobs, inbox):
    job = jobs.start("shell", "s", description="command: x", timeout=1, command="trap '' TERM; sleep 30")
    job.deadline = time.time() + 0.1
    text = _wait_event(inbox, timeout=5)
    assert "status=timed_out" in text
    assert job.handle.proc.returncode == -9


def test_kill_job_cancels_and_the_event_still_arrives(jobs, inbox):
    tools = job_tools(jobs)
    _call_in_session(jobs, "s", "start_shell_job", command="sleep 30")
    out = _call_in_session(jobs, "s", "kill_job", id=1)
    assert out == "job 1 (shell) cancel requested; its <job-finished> message will follow"
    text = _wait_event(inbox)
    assert "status=cancelled" in text and "cancelled; process group killed" in text
    assert _call_in_session(jobs, "s", "kill_job", id=1) == "job 1 already finished (cancelled)"
    assert _call_in_session(jobs, "s", "kill_job", id=7).startswith("error: ")


def test_running_limit_per_kind_and_session(inbox, tmp_path):
    jobs = Jobs(inbox, [ShellJobKind(max_running=1)], jobs_dir=tmp_path, poll_interval=0.02)
    try:
        _call_in_session(jobs, "s", "start_shell_job", command="sleep 30")
        out = _call_in_session(jobs, "s", "start_shell_job", command="sleep 30")
        assert "limit of 1 running shell jobs per session reached (running: 1)" in out
        # Another session has its own budget.
        assert _call_in_session(jobs, "t", "start_shell_job", command="echo").startswith("started job 1")
    finally:
        jobs.shutdown()


def test_timeout_is_clamped_to_the_kind_maximum(inbox, tmp_path):
    jobs = Jobs(inbox, [ShellJobKind(max_timeout=2)], jobs_dir=tmp_path, poll_interval=0.02)
    try:
        job = jobs.start("shell", "s", description="d", timeout=99999, command="true")
        assert job.timeout == 2
    finally:
        jobs.shutdown()


def test_log_cap_fails_the_job(inbox, tmp_path):
    jobs = Jobs(inbox, [ShellJobKind(max_log_bytes=1000, term_grace=0.3)], jobs_dir=tmp_path, poll_interval=0.02)
    try:
        jobs.start("shell", "s", description="d", command="yes | head -c 100000; sleep 30")
        text = _wait_event(inbox)
        assert "status=failed" in text and "log exceeded 0 MB" in text
    finally:
        jobs.shutdown()


def test_shutdown_kills_running_jobs(inbox, tmp_path):
    jobs = Jobs(inbox, [ShellJobKind(term_grace=0.3)], jobs_dir=tmp_path, poll_interval=0.02)
    job = jobs.start("shell", "s", description="d", command="sleep 30")
    jobs.shutdown()
    assert job.handle.proc.poll() is not None
    with pytest.raises(JobError):
        jobs.start("shell", "s", description="d", command="true")


# --- the record ---------------------------------------------------------------


def test_lost_jobs_are_reported_on_startup_and_ids_continue(tmp_path):
    record = tmp_path / "s" / "jobs.jsonl"
    record.parent.mkdir()
    dead_pid = _dead_pid()
    lines = [
        {"id": 1, "kind": "shell", "session": "s", "description": "command: a", "started": 1.0,
         "deadline": 2.0, "nod_pid": dead_pid, "ref": {"pid": 424242}},
        {"id": 1, "status": "done"},
        {"id": 2, "kind": "shell", "session": "s", "description": "command: b", "started": 1.0,
         "deadline": 2.0, "nod_pid": dead_pid, "ref": {"pid": 424243}},
        {"id": 3, "kind": "agent", "session": "s", "description": "task: c", "started": 1.0,
         "deadline": 2.0, "nod_pid": dead_pid, "ref": {"session": "s/job-3"}},
        {"id": 4, "kind": "shell", "session": "s", "description": "command: ours", "started": 1.0,
         "deadline": 2.0, "nod_pid": os.getpid(), "ref": {"pid": 1}},
    ]
    record.write_text("".join(json.dumps(line) + "\n" for line in lines))

    inbox = InMemoryInbox()
    jobs = Jobs(inbox, [ShellJobKind()], jobs_dir=tmp_path, poll_interval=0.02)
    try:
        events = [e.text for e in inbox.pending("s")]
        assert len(events) == 2
        assert "<job-finished id=2 kind=shell status=lost" in events[0]
        assert f"lost: Nod (pid {dead_pid}) exited while this job was running. The process (pid 424243) may still be running." in events[0]
        assert "<job-finished id=3 kind=agent status=lost" in events[1]
        assert "Its transcript is in session s/job-3." in events[1]
        # Reported once: the record now says so, and a second start-up stays quiet.
        finished = [json.loads(l) for l in record.read_text().splitlines() if '"status"' in l]
        assert [(f["id"], f["status"]) for f in finished] == [(1, "done"), (2, "lost"), (3, "lost")]
        # Ids continue past everything in the record.
        job = jobs.start("shell", "s", description="d", command="true")
        assert job.id == 5
    finally:
        jobs.shutdown()

    other = InMemoryInbox()
    again = Jobs(other, [ShellJobKind()], jobs_dir=tmp_path, poll_interval=0.02)
    try:
        assert [e.text for e in other.pending("s")] == []
    finally:
        again.shutdown()


def test_recycled_pid_does_not_hide_a_lost_job(tmp_path):
    import fcntl

    owners = tmp_path / "owners"
    owners.mkdir()
    # A record whose owner is dead and cleaned up, one whose owner is dead but
    # left its lock file, and one whose owner is alive and holding its lock.
    # Every one names our own pid, as if the OS had recycled it.
    (owners / "stale").touch()
    held = (owners / "alive").open("w")
    fcntl.flock(held, fcntl.LOCK_EX)
    record = tmp_path / "s" / "jobs.jsonl"
    record.parent.mkdir()
    lines = [
        {"id": 1, "kind": "shell", "session": "s", "description": "command: a", "started": 1.0,
         "deadline": 2.0, "nod_pid": os.getpid(), "nod_lock": "gone", "ref": {"pid": 424242}},
        {"id": 2, "kind": "shell", "session": "s", "description": "command: b", "started": 1.0,
         "deadline": 2.0, "nod_pid": os.getpid(), "nod_lock": "stale", "ref": {"pid": 424243}},
        {"id": 3, "kind": "shell", "session": "s", "description": "command: c", "started": 1.0,
         "deadline": 2.0, "nod_pid": os.getpid(), "nod_lock": "alive", "ref": {"pid": 424244}},
    ]
    record.write_text("".join(json.dumps(line) + "\n" for line in lines))

    inbox = InMemoryInbox()
    jobs = Jobs(inbox, [ShellJobKind()], jobs_dir=tmp_path, poll_interval=0.02)
    try:
        events = [e.text for e in inbox.pending("s")]
        assert [t.split("\n")[0] for t in events] == [
            "<job-finished id=1 kind=shell status=lost duration=unknown>",
            "<job-finished id=2 kind=shell status=lost duration=unknown>",
        ]
        assert f"lost: Nod (pid {os.getpid()}) exited" in events[0]
        assert not (owners / "stale").exists()  # a dead owner's lock file is cleaned up
        assert (owners / "alive").exists()
        assert (owners / jobs._owner_name).exists()
    finally:
        held.close()
        jobs.shutdown()
    assert not (owners / jobs._owner_name).exists()  # a clean exit removes ours


def test_two_jobs_in_one_process_do_not_wait_on_each_other(tmp_path):
    # Each takes a lock file of its own. They used to share a name when built
    # in the same second, and the second then hung on the first one's lock.
    first = Jobs(InMemoryInbox(), [ShellJobKind()], jobs_dir=tmp_path, poll_interval=0.02)
    second = Jobs(InMemoryInbox(), [ShellJobKind()], jobs_dir=tmp_path, poll_interval=0.02)
    try:
        assert first._owner_name != second._owner_name
    finally:
        first.shutdown()
        second.shutdown()
    assert list((tmp_path / "owners").iterdir()) == []


def test_a_job_cancelled_by_shutdown_is_not_reported_lost(tmp_path):
    jobs = Jobs(InMemoryInbox(), [ShellJobKind()], jobs_dir=tmp_path, poll_interval=0.02)
    jobs.start("shell", "s", description="d", command="sleep 30")
    jobs.shutdown()
    lines = [json.loads(l) for l in (tmp_path / "s" / "jobs.jsonl").read_text().splitlines()]
    assert lines[1:] == [{"id": 1, "status": "cancelled"}]

    inbox = InMemoryInbox()
    again = Jobs(inbox, [ShellJobKind()], jobs_dir=tmp_path, poll_interval=0.02)
    try:
        assert inbox.pending("s") == []
    finally:
        again.shutdown()


def _dead_pid():
    import subprocess

    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid  # reaped, so no longer alive


def test_record_has_start_and_finish_lines(jobs, inbox, tmp_path):
    jobs.start("shell", "s", description="command: echo", command="echo")
    _wait_event(inbox)
    lines = [json.loads(l) for l in (tmp_path / "s" / "jobs.jsonl").read_text().splitlines()]
    assert lines[0]["id"] == 1 and lines[0]["kind"] == "shell" and lines[0]["nod_pid"] == os.getpid()
    # The record names our lock file, and we hold it.
    import fcntl

    assert lines[0]["nod_lock"] == jobs._owner_name
    with (tmp_path / "owners" / jobs._owner_name).open() as f:
        with pytest.raises(BlockingIOError):
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert "pid" in lines[0]["ref"] and lines[0]["ref"]["log"].endswith("/s/1.log")
    assert lines[1] == {"id": 1, "status": "done"}


# --- agent kind ------------------------------------------------------------------


def _agent_jobs(inbox, tmp_path, child_replies, child_tools=None, max_turns=20, parent_provider=None):
    memory = InMemoryMemory()
    child_provider = FakeProvider(child_replies)

    def factory(provider):
        return Agent(
            provider=provider,
            registry=ToolRegistry(child_tools or []),
            memory=memory,
            soul="you are a subagent",
            max_turns=max_turns,
        )

    kind = AgentJobKind(child_provider, factory, max_running=2)
    return Jobs(inbox, [ShellJobKind(), kind], jobs_dir=tmp_path, poll_interval=0.02), memory, child_provider


def test_spawn_agent_returns_at_once_and_reports_the_final_reply(inbox, tmp_path):
    jobs, memory, child_provider = _agent_jobs(inbox, tmp_path, [assistant("I did the thing")])
    try:
        out = _call_in_session(jobs, "p", "spawn_agent", task="do the thing\nwith care")
        assert out.startswith("started job 1 (agent), session: p/job-1.")
        text = _wait_event(inbox, "p")
        assert text.startswith("<job-finished id=1 kind=agent status=done")
        assert "task: do the thing" in text
        assert "final reply\n--- reply (full transcript: session p/job-1) ---\nI did the thing\n</job-finished>" in text
        # The child's own transcript survives in the shared memory, under its session.
        assert [m.content for m in memory.load("p/job-1")] == ["do the thing\nwith care", "I did the thing"]
        # The cancelling wrapper passes the parent's provider through, with the child's soul.
        assert child_provider.calls[0][0].content == "you are a subagent"
    finally:
        jobs.shutdown()


def test_child_hitting_max_turns_is_done_but_says_so(inbox, tmp_path):
    def always_tool(_m):
        return assistant(tool_calls=[ToolCall(id="x", name="noop", arguments={})])

    noop = Tool(name="noop", description="", parameters={"type": "object", "properties": {}}, func=lambda: "ok")
    jobs, _, _ = _agent_jobs(inbox, tmp_path, [always_tool] * 5, child_tools=[noop], max_turns=2)
    try:
        jobs.start("agent", "p", description="task: loop", task="loop")
        text = _wait_event(inbox, "p")
        assert "status=done" in text and "hit max_turns before finishing" in text
        assert "(stopped: hit the max_turns limit before finishing)" in text
    finally:
        jobs.shutdown()


def test_cancelling_an_agent_job_stops_it_before_its_next_model_call(inbox, tmp_path):
    release = threading.Event()
    entered = threading.Event()

    def block() -> str:
        entered.set()
        release.wait(5)
        return "tool done"

    blocker = Tool(name="block", description="", parameters={"type": "object", "properties": {}}, func=block)
    first = assistant(tool_calls=[ToolCall(id="c1", name="block", arguments={})])
    jobs, memory, child_provider = _agent_jobs(inbox, tmp_path, [first, assistant("should not happen")], child_tools=[blocker])
    try:
        jobs.start("agent", "p", description="task: t", task="t")
        assert entered.wait(5)
        _call_in_session(jobs, "p", "kill_job", id=1)
        release.set()
        text = _wait_event(inbox, "p")
        assert "status=cancelled" in text and "cancelled before its next model call" in text
        # One model call happened; the second was refused by the wrapper.
        assert len(child_provider.calls) == 1
        # Everything up to the last tool round is in the child's transcript.
        roles = [m.role for m in memory.load("p/job-1")]
        assert roles == ["user", "assistant", "tool"]
    finally:
        jobs.shutdown()


def test_agent_job_timeout_uses_the_same_flag(inbox, tmp_path):
    release = threading.Event()
    blocker = Tool(name="block", description="", parameters={"type": "object", "properties": {}}, func=lambda: (release.wait(5), "x")[1])
    first = assistant(tool_calls=[ToolCall(id="c1", name="block", arguments={})])
    jobs, _, _ = _agent_jobs(inbox, tmp_path, [first, assistant("nope")], child_tools=[blocker])
    try:
        job = jobs.start("agent", "p", description="task: t", task="t", timeout=1)
        job.deadline = time.time() + 0.1
        assert _wait(lambda: job.cancel_requested)
        release.set()
        text = _wait_event(inbox, "p")
        assert "status=timed_out" in text and "timed out after 1 s; stopped before its next model call" in text
    finally:
        jobs.shutdown()


def test_child_agent_may_not_have_spawn_agent(inbox, tmp_path):
    def factory(provider):
        fake_spawn = Tool(name="spawn_agent", description="", parameters={"type": "object", "properties": {}}, func=lambda: "")
        return Agent(provider=provider, registry=ToolRegistry([fake_spawn]), memory=InMemoryMemory())

    jobs = Jobs(inbox, [AgentJobKind(FakeProvider([]), factory)], jobs_dir=tmp_path, poll_interval=0.02)
    try:
        out = _call_in_session(jobs, "p", "spawn_agent", task="t")
        assert out == "error: tool 'spawn_agent' failed: a child agent must not have the spawn_agent tool"
        assert jobs.list("p") == []  # nothing left in the table, nothing recorded
        assert not (tmp_path / "p" / "jobs.jsonl").exists()
    finally:
        jobs.shutdown()


def test_child_failure_is_reported_as_failed(inbox, tmp_path):
    def boom(_m):
        raise RuntimeError("brain offline")

    jobs, _, _ = _agent_jobs(inbox, tmp_path, [boom])
    try:
        jobs.start("agent", "p", description="task: t", task="t")
        text = _wait_event(inbox, "p")
        assert "status=failed" in text and "RuntimeError: brain offline" in text
    finally:
        jobs.shutdown()


def test_job_tools_follow_the_registered_kinds(inbox, tmp_path):
    only_shell = Jobs(inbox, [ShellJobKind()], jobs_dir=tmp_path, poll_interval=0.02)
    try:
        names = [t.name for t in job_tools(only_shell)]
        assert names == ["start_shell_job", "list_jobs", "kill_job", "wait_jobs"]
    finally:
        only_shell.shutdown()

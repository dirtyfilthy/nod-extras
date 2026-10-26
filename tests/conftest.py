"""Shared test helpers."""

from __future__ import annotations

from typing import Any, Callable

from nod.messages import Message
from nod.provider import OnToken


class FakeProvider:
    """A scripted brain.

    Pass a list of replies; each `complete` call pops the next one. Replies may be
    `Message` objects or callables that receive the current message list and return
    a `Message`, which lets a test assert on what the loop fed back in.
    """

    def __init__(self, replies: list[Message | Callable[[list[Message]], Message]]):
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        on_token: OnToken | None = None,
    ) -> Message:
        self.calls.append(list(messages))
        if not self._replies:
            return Message(role="assistant", content="(out of scripted replies)")
        nxt = self._replies.pop(0)
        reply = nxt(messages) if callable(nxt) else nxt
        if on_token is not None and reply.content:
            on_token(reply.content)
        return reply

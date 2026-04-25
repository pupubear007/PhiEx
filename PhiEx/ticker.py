"""
PhiEx.ticker — reasoning-ticker logger.

The ticker is the prototype's most distinctive UI element:  a scrolling log
of every pipeline event tagged by framework symbol (ϕ / ∃ / s / tϕ / i).
This module is the producer side.  The FastAPI SSE endpoint subscribes to
these events and forwards them to the browser, which renders them in the
bottom strip of the page in the prototype's style.

Tagging convention (matches the prototype's CSS classes):
    "phi"   ϕ → ∃     deductive theory step (force calc, ESMFold, P2Rank)
    "exist" ∃         a concrete outcome (pose found, trajectory finished)
    "s"     s∃        sampled existence — a single trajectory event
    "t"     tϕ        theory refinement — surrogate fit, AL update
    "i"     i         iteration boundary — "iteration i = 3"
    "sys"   ···       system messages (device load, errors)

Every ML adapter MUST log on load with its device:

    log("t", "loaded MACE-OFF23 on device=mps, backend=Metal Performance Shaders")

Every ML prediction MUST log uncertainty:

    log("phi", f"ESMFold predicted structure  pLDDT={mean_plddt:.1f} ± {sd:.1f}")
"""

from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from typing import AsyncIterator, Literal

Tag = Literal["phi", "exist", "s", "t", "i", "sys"]


@dataclass
class Event:
    tag: Tag
    msg: str
    t: float = field(default_factory=time.time)
    extras: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"tag": self.tag, "msg": self.msg,
                           "t": self.t, "extras": self.extras})


class Ticker:
    """In-process pub-sub for reasoning events.

    Producers call `log(tag, msg)`.  The FastAPI SSE handler calls
    `subscribe()` to get an async iterator.  The ticker also keeps the last
    N events in a ring buffer so a fresh page load can replay history.
    """

    def __init__(self, history: int = 500) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._history: list[Event] = []
        self._history_max = history

    # ──────────────────────────────────────────────────────────────
    # producer side
    # ──────────────────────────────────────────────────────────────
    def log(self, tag: Tag, msg: str, **extras) -> Event:
        ev = Event(tag=tag, msg=msg, extras=extras)
        self._history.append(ev)
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max:]
        for q in list(self._subscribers):
            try:
                q.put_nowait(ev)
            except asyncio.QueueFull:
                pass
        # also surface to stderr-style logging so headless runs (test, CLI)
        # see the same trace
        print(f"[{tag:>5}] {msg}", flush=True)
        return ev

    # ──────────────────────────────────────────────────────────────
    # consumer side (SSE)
    # ──────────────────────────────────────────────────────────────
    async def subscribe(self,
                        replay: bool = True) -> AsyncIterator[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=1024)
        self._subscribers.append(q)
        try:
            if replay:
                for ev in list(self._history):
                    yield ev
            while True:
                yield await q.get()
        finally:
            self._subscribers.remove(q)

    def history(self) -> list[dict]:
        return [asdict(ev) for ev in self._history]


# Module-level singleton — there is exactly one ticker per process.
TICKER = Ticker()


def log(tag: Tag, msg: str, **extras) -> Event:
    """Convenience wrapper for the module-level singleton."""
    return TICKER.log(tag, msg, **extras)

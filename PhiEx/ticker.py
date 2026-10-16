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
import threading
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
        # each subscriber is (queue, the event loop that owns it).  Producers
        # may run on worker threads (FastAPI threadpool, run_in_executor), and
        # asyncio.Queue is not thread-safe, so off-loop puts are marshalled
        # onto the owning loop with call_soon_threadsafe.
        self._subscribers: list[tuple[asyncio.Queue[Event], asyncio.AbstractEventLoop]] = []
        self._lock = threading.Lock()
        self._history: list[Event] = []
        self._history_max = history

    # ──────────────────────────────────────────────────────────────
    # producer side
    # ──────────────────────────────────────────────────────────────
    def log(self, tag: Tag, msg: str, **extras) -> Event:
        ev = Event(tag=tag, msg=msg, extras=extras)
        with self._lock:
            self._history.append(ev)
            if len(self._history) > self._history_max:
                self._history = self._history[-self._history_max:]
            subscribers = list(self._subscribers)
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            current = None
        for q, loop in subscribers:
            if loop is current:
                _put_dropping(q, ev)
            elif not loop.is_closed():
                loop.call_soon_threadsafe(_put_dropping, q, ev)
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
        entry = (q, asyncio.get_running_loop())
        with self._lock:
            self._subscribers.append(entry)
            backlog = list(self._history) if replay else []
        try:
            for ev in backlog:
                yield ev
            while True:
                yield await q.get()
        finally:
            with self._lock:
                self._subscribers.remove(entry)

    def history(self) -> list[dict]:
        with self._lock:
            return [asdict(ev) for ev in self._history]

    def clear_history(self) -> None:
        """Wipe the replay buffer so a fresh page load shows nothing.
        Live subscribers are unaffected."""
        with self._lock:
            self._history.clear()


def _put_dropping(q: "asyncio.Queue[Event]", ev: Event) -> None:
    """Deliver to a subscriber queue, dropping the event if a slow client
    has let the queue fill up."""
    try:
        q.put_nowait(ev)
    except asyncio.QueueFull:
        pass


# Module-level singleton — there is exactly one ticker per process.
TICKER = Ticker()


def log(tag: Tag, msg: str, **extras) -> Event:
    """Convenience wrapper for the module-level singleton."""
    return TICKER.log(tag, msg, **extras)

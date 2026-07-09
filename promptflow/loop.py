"""Static-unrolled loop primitive for promptflow.

A :class:`Loop` is a single unrolled iteration *stage*. A full loop is built by
composing ``max_iters`` identical stages into an acyclic chain via
:func:`build_loop`. Every object flows through all stages, but once an object
satisfies the stopping condition (or reaches the cap) it is flagged ``done`` and
subsequent stages simply propagate it without running the iteration body again.

Because the pipeline is a normal acyclic DAG, the framework's existing STOP
propagation (each process stops its output only after its input stops) shuts the
whole chain down without any special bookkeeping.
"""

from __future__ import annotations

import inspect
import operator
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Awaitable, Callable, Union

from promptflow.actor import try_to_convert_to_input
from promptflow.process import Filter, Junction, Map, MetaMap
from promptflow.workflow import WorkFlow

__all__ = ["LoopInstance", "Loop", "build_loop", "build_stream_loop", "WhileLoop"]


Body = Callable[[Any], Union[Any, Awaitable[Any]]]
Check = Callable[[Any], bool]


@dataclass
class LoopInstance:
    """Per-object loop state that rides through the unrolled chain.

    ``payload`` is the user's object, ``count`` is how many times the iteration
    body has run on it, and ``done`` marks that the stopping condition (or the
    iteration cap) has been reached.
    """

    payload: Any
    count: int = 0
    done: bool = False


class Loop(MetaMap):
    """One unrolled iteration stage.

    On first sight an object is wrapped into a :class:`LoopInstance`. If the
    instance is already ``done`` it is propagated untouched; otherwise the
    iteration ``body`` runs once, ``count`` is incremented, and the stopping
    ``check`` (or the ``max_iters`` cap) decides whether it becomes ``done``.
    """

    def __init__(
        self,
        body: Body,
        check: Check,
        max_iters: int,
        name: Union[str, None] = None,
    ) -> None:
        if max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        self.body = body
        self.check = check
        self.max_iters = max_iters
        super().__init__(
            func=self._stage,
            many=False,
            name=name or f"Loop(max_iters={max_iters})",
        )

    async def _stage(self, obj: Any) -> LoopInstance:
        if not isinstance(obj, LoopInstance):
            obj = LoopInstance(payload=obj)

        if obj.done:
            return obj

        result = self.body(obj.payload)
        if inspect.isawaitable(result):
            result = await result
        obj.payload = result
        obj.count += 1

        if self.check(obj.payload) or obj.count >= self.max_iters:
            obj.done = True

        return obj


class _LoopWorkflow(WorkFlow):
    """Workflow wrapping the composed chain of :class:`Loop` stages."""

    def __init__(self, pipeline, unwrap: bool, name: str = "loop") -> None:
        super().__init__(name=name)
        self.pipeline = pipeline
        self.unwrap = unwrap

    def forward(self, inputs):
        node = try_to_convert_to_input(inputs)
        out = self.pipeline(node)
        if self.unwrap:
            out = Map(func=lambda instance: instance.payload, name="loop_unwrap")(out)
        return out


def build_loop(
    body: Body,
    check: Check,
    max_iters: int = 8,
    unwrap: bool = True,
) -> WorkFlow:
    """Build a loop by statically unrolling ``max_iters`` identical stages.

    Args:
        body: What happens in one iteration - a sync or async callable
            ``payload -> payload``.
        check: Stopping condition ``payload -> bool`` evaluated after each body
            run; once true the object stops being processed and is propagated.
        max_iters: Number of unrolled stages (and the per-object iteration cap).
        unwrap: If true, the workflow output is the raw ``payload`` of each
            object; otherwise the :class:`LoopInstance` wrappers are returned.

    Returns:
        A :class:`WorkFlow` with a stream in and a stream out.
    """
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")

    stages = [
        Loop(body=body, check=check, max_iters=max_iters, name=f"Loop[{i}]")
        for i in range(max_iters)
    ]
    pipeline = reduce(operator.or_, stages)
    return _LoopWorkflow(pipeline=pipeline, unwrap=unwrap)

# ---------------------------------------------------------------------------
# Stream-level loop: the iteration body is a whole sub-workflow (stream in /
# stream out) rather than a scalar callable. This is what you want when one
# iteration is itself a promptflow pipeline of Map objects (e.g. an LLM-call
# map followed by a code-execution map). State is carried inside each item
# (assumed to be a dict) under two reserved keys so the ordinary dict-passing
# Map style keeps working.
# ---------------------------------------------------------------------------


class _Concat(Junction):
    """Merge several actor streams into one, forwarding every item as-is."""

    async def execute(self, *inputs, output):
        for inp in inputs:
            async for id, data in inp.iterable(output):
                await output.commit(id, data)
        await output.stop()


def build_stream_loop(
    body: Union[WorkFlow, Any],
    check: Check,
    max_iters: int = 8,
    *,
    count_key: str = "__loop_count",
    done_key: str = "__loop_done",
    name: str = "stream_loop",
) -> WorkFlow:
    """Statically unroll a loop whose *body is a stream workflow*.

    Each iteration is one unrolled stage that:

    1. splits the stream into *active* (``not done``) and *done* items;
    2. runs ``body`` on the active items only (a WorkFlow applied via
       ``forward`` or any Process applied via ``__call__``, mapping item->item);
    3. increments ``count_key`` and sets ``done_key`` once ``check`` passes or
       the ``max_iters`` cap is hit;
    4. merges the advanced items back with the propagated done items.

    Args:
        body: The per-iteration workflow (stream in / stream out), mapping each
            item (a dict) to an item. May be a ``WorkFlow`` or a ``Process``.
        check: Stopping condition ``item -> bool`` evaluated after each body run.
        max_iters: Number of unrolled stages and the per-item iteration cap.
        count_key/done_key: Reserved item keys used to carry loop state.

    Returns:
        A :class:`WorkFlow` with a stream in and a stream out.
    """
    if max_iters < 1:
        raise ValueError(f"max_iters must be >= 1, got {max_iters}")

    def _init(x: dict) -> dict:
        if count_key in x and done_key in x:
            return x
        y = dict(x)
        y.setdefault(count_key, 0)
        y.setdefault(done_key, False)
        return y

    def _advance(x: dict) -> dict:
        y = dict(x)
        y[count_key] = y.get(count_key, 0) + 1
        if check(y) or y[count_key] >= max_iters:
            y[done_key] = True
        return y

    def _apply_body(node):
        if isinstance(body, WorkFlow):
            return body.forward(node)
        return body(node)

    def _is_active(x: dict) -> bool:
        return not x.get(done_key, False)

    def _is_done(x: dict) -> bool:
        return bool(x.get(done_key, False))

    class _StreamLoopWorkflow(WorkFlow):
        def forward(self, inputs):
            node = try_to_convert_to_input(inputs)
            node = Map(func=_init, name=f"{name}_init")(node)
            for i in range(max_iters):
                active = Filter(_is_active, name=f"{name}_active[{i}]")(node)
                done = Filter(_is_done, name=f"{name}_done[{i}]")(node)
                processed = _apply_body(active)
                advanced = Map(func=_advance, name=f"{name}_advance[{i}]")(processed)
                node = _Concat(name=f"{name}_merge[{i}]")(advanced, done)
            return node

    return _StreamLoopWorkflow(name=name)


def WhileLoop(body: WorkFlow, check: Check, max_iters: int = 8) -> WorkFlow:
    """A while loop whose *body is a stream workflow*, unrolled up to ``max_iters``.

    Public alias for :func:`build_stream_loop`: on each iteration the ``body``
    (a ``WorkFlow`` / ``Process`` mapping item->item) runs only on the items that
    have not finished yet; an item stops once ``check`` returns true or the
    ``max_iters`` cap is hit, and finished items are propagated unchanged.
    """
    return build_stream_loop(body, check, max_iters=max_iters)

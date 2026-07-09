"""Tests for the static-unrolled ``Loop`` primitive (promptflow/loop.py).

These use trivial deterministic bodies over plain dict payloads - no model,
gateway, or code server - to verify the loop mechanics:
stop-at-target, done-then-propagate, per-item targets, the max_iters cap,
empty input, completeness/identity, state continuity, and status-based checks.
"""

from promptflow.loop import LoopInstance, Loop, build_loop


# --- trivial iteration bodies -------------------------------------------------

async def inc(p):
    """One iteration: bump the counter."""
    p["n"] += 1
    return p


def inc_sync(p):
    """Sync body variant (Loop must accept sync callables too)."""
    p["n"] += 1
    return p


async def accumulate(p):
    """One iteration: bump the counter and record it in a trace."""
    p["n"] += 1
    p["trace"].append(p["n"])
    return p


async def step_status(p):
    """One iteration: bump k, flip status to done once it reaches stop_at."""
    p["k"] += 1
    if p["k"] >= p["stop_at"]:
        p["status"] = "done"
    return p


def reached_target(p):
    return p["n"] >= p["target"]


def by_id(results):
    return {inst.payload["id"]: inst for inst in results}


# --- tests --------------------------------------------------------------------

def test_uniform_target_below_cap():
    inputs = [{"id": i, "n": 0, "target": 3} for i in range(5)]
    wf = build_loop(inc, reached_target, max_iters=8, unwrap=False)
    results = wf(inputs)

    assert len(results) == 5
    for inst in results:
        assert isinstance(inst, LoopInstance)
        assert inst.done is True
        assert inst.payload["n"] == 3  # stopped at target, not driven to 8
        assert inst.count == 3


def test_done_then_propagate():
    # A single item that finishes at iteration 2 must NOT keep advancing through
    # the remaining 6 stages: count freezes at 2 and n stays at the target.
    inputs = [{"id": 0, "n": 0, "target": 2}]
    wf = build_loop(inc, reached_target, max_iters=8, unwrap=False)
    (inst,) = wf(inputs)

    assert inst.done is True
    assert inst.count == 2
    assert inst.payload["n"] == 2


def test_per_item_targets():
    inputs = [{"id": "a", "n": 0, "target": 1},
              {"id": "b", "n": 0, "target": 2},
              {"id": "c", "n": 0, "target": 5}]
    wf = build_loop(inc, reached_target, max_iters=8, unwrap=False)
    results = by_id(wf(inputs))

    for key, target in (("a", 1), ("b", 2), ("c", 5)):
        assert results[key].payload["n"] == target  # none overshoots
        assert results[key].count == target
        assert results[key].done is True


def test_max_iters_cap_below_target():
    inputs = [{"id": 0, "n": 0, "target": 100}]
    wf = build_loop(inc, reached_target, max_iters=8, unwrap=False)
    (inst,) = wf(inputs)

    assert inst.count == 8  # capped
    assert inst.payload["n"] == 8
    assert inst.done is True


def test_empty_input():
    wf = build_loop(inc, reached_target, max_iters=8, unwrap=False)
    assert wf([]) == []


def test_completeness_and_identity():
    inputs = [{"id": i, "n": 0, "target": (i % 5) + 1} for i in range(12)]
    wf = build_loop(inc, reached_target, max_iters=8, unwrap=False)
    results = wf(inputs)

    assert len(results) == len(inputs)
    ids = sorted(inst.payload["id"] for inst in results)
    assert ids == list(range(12))  # each input appears exactly once
    for inst in results:
        assert inst.payload["n"] == inst.payload["target"]


def test_accumulator_state_continuity():
    # The SAME LoopInstance/payload must ride through the chain, so the trace
    # must have exactly one entry per body invocation (== count).
    inputs = [{"id": i, "n": 0, "target": t, "trace": []}
              for i, t in enumerate((1, 3, 6))]
    wf = build_loop(accumulate, reached_target, max_iters=8, unwrap=False)
    for inst in wf(inputs):
        assert len(inst.payload["trace"]) == inst.count
        assert inst.payload["trace"] == list(range(1, inst.count + 1))


def test_status_based_check():
    inputs = [{"id": 0, "k": 0, "stop_at": 3, "status": "running"}]
    wf = build_loop(step_status, lambda p: p["status"] == "done",
                    max_iters=8, unwrap=False)
    (inst,) = wf(inputs)

    assert inst.payload["status"] == "done"
    assert inst.count == 3
    assert inst.done is True


def test_sync_body_supported():
    inputs = [{"id": 0, "n": 0, "target": 4}]
    wf = build_loop(inc_sync, reached_target, max_iters=8, unwrap=False)
    (inst,) = wf(inputs)

    assert inst.payload["n"] == 4
    assert inst.count == 4


def test_unwrap_returns_payload():
    inputs = [{"id": 0, "n": 0, "target": 3}]
    wf = build_loop(inc, reached_target, max_iters=8, unwrap=True)
    (payload,) = wf(inputs)

    assert isinstance(payload, dict)
    assert payload["n"] == 3


def test_invalid_max_iters():
    import pytest

    with pytest.raises(ValueError):
        build_loop(inc, reached_target, max_iters=0)


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))

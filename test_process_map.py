"""Tests for process-pool-backed map stages."""

import asyncio
import os

import pytest

from promptflow import ProcessFlatMap, ProcessMap, WorkFlow
from promptflow.asynchronous import (
    get_process_pool,
    process_wrap,
    shutdown_process_pool,
)


def add_one(value):
    return value + 1


def split_with_next(value):
    return [value, value + 1]


def worker_pid(_value):
    return os.getpid()


def fail(value):
    raise ValueError(f"bad value: {value}")


def multiplier(factor):
    return lambda value: value * factor


class SingleStageWorkflow(WorkFlow):
    def __init__(self, process):
        super().__init__(name="single-stage")
        self.process = process

    def forward(self, values):
        return self.process(values)


@pytest.fixture(autouse=True)
def close_process_pool():
    yield
    shutdown_process_pool()


def test_process_pool_is_shared():
    assert get_process_pool() is get_process_pool()


def test_process_map_matches_native_map_semantics():
    workflow = SingleStageWorkflow(ProcessMap(add_one))

    assert workflow([1, 2, 3]) == [2, 3, 4]


def test_process_flat_map_preserves_fan_out():
    workflow = SingleStageWorkflow(ProcessFlatMap(split_with_next))

    assert workflow([1, 2]) == [1, 2, 2, 3]


def test_process_map_supports_closures():
    workflow = SingleStageWorkflow(ProcessMap(multiplier(3)))

    assert workflow([2, 4]) == [6, 12]


def test_process_map_runs_in_a_worker_process():
    workflow = SingleStageWorkflow(ProcessMap(worker_pid))

    assert all(pid != os.getpid() for pid in workflow([1, 2]))


def test_process_wrap_propagates_worker_exceptions():
    async def invoke():
        return await process_wrap(fail)(7)

    with pytest.raises(ValueError, match="bad value: 7"):
        asyncio.run(invoke())

"""Compare thread-backed Map and process-backed ProcessMap for CPU work.

Run from this directory:
    python benchmark_process_map.py

Increase ``--iterations`` for a longer, less overhead-sensitive benchmark.
"""

import argparse
import os
from time import perf_counter

from promptflow import Map, ProcessMap, WorkFlow
from promptflow.asynchronous import shutdown_process_pool


def cpu_bound_work(item):
    """Pure Python work that cannot run in parallel across threads."""
    value, iterations = item
    accumulator = value
    for index in range(iterations):
        accumulator = (accumulator * 1_103_515_245 + index) & 0xFFFFFFFF
    return accumulator


class BenchmarkWorkflow(WorkFlow):
    def __init__(self, process):
        super().__init__(name="process-map-benchmark")
        self.process = process

    def forward(self, values):
        return self.process(values)


def run_benchmark(process, inputs):
    start = perf_counter()
    result = BenchmarkWorkflow(process)(inputs)
    return perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--items",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help="number of independent CPU-bound inputs (default: up to 8)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500_000,
        help="loop iterations per input (default: 500000)",
    )
    args = parser.parse_args()

    if args.items < 1 or args.iterations < 1:
        parser.error("--items and --iterations must be positive")

    # Promptflow treats a list of tuples as explicit (key, value) records.
    # Keep the CPU-work arguments inside the value payload.
    inputs = [
        (value, (value, args.iterations)) for value in range(args.items)
    ]
    try:
        thread_seconds, thread_result = run_benchmark(Map(cpu_bound_work), inputs)
        process_seconds, process_result = run_benchmark(
            ProcessMap(cpu_bound_work), inputs
        )
    finally:
        shutdown_process_pool()

    if thread_result != process_result:
        raise RuntimeError("Map and ProcessMap produced different results")

    speedup = thread_seconds / process_seconds if process_seconds else float("inf")
    print(f"Inputs: {args.items}; iterations per input: {args.iterations:,}")
    print(f"Map (threads):          {thread_seconds:.3f}s")
    print(f"ProcessMap (processes): {process_seconds:.3f}s")
    print(f"ProcessMap speedup:     {speedup:.2f}x")


if __name__ == "__main__":
    main()

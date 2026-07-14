import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial, wraps
from threading import Lock

import cloudpickle


_process_pool = None
_process_pool_lock = Lock()


def _run_cloudpickled_call(payload):
    """Execute a cloudpickle-serialized callable in a worker process."""
    func, args, kwargs = cloudpickle.loads(payload)
    return func(*args, **kwargs)


def get_process_pool():
    """Return the shared, lazily-created process pool for CPU map stages.

    The pool uses Python's default ``ProcessPoolExecutor`` worker count and the
    ``spawn`` start method so workers do not inherit event-loop or network state.
    Call :func:`shutdown_process_pool` during application teardown when a
    long-lived host no longer needs CPU map stages.
    """
    global _process_pool
    with _process_pool_lock:
        if _process_pool is None:
            _process_pool = ProcessPoolExecutor(
                mp_context=multiprocessing.get_context("spawn")
            )
        return _process_pool


def shutdown_process_pool(wait=True):
    """Shut down and clear the shared process pool.

    Primarily useful for application teardown and tests. A later CPU map use
    creates a fresh pool.
    """
    global _process_pool
    with _process_pool_lock:
        pool, _process_pool = _process_pool, None
    if pool is not None:
        pool.shutdown(wait=wait)


def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)

    return run


def process_wrap(func):
    """Make a synchronous CPU-bound callable awaitable in the shared process pool.

    ``cloudpickle`` allows lambdas and closures to be used as callbacks. Inputs
    and return values must still be transferable through
    ``ProcessPoolExecutor``'s normal IPC serialization.
    """

    @wraps(func)
    async def run(*args, **kwargs):
        payload = cloudpickle.dumps((func, args, kwargs))
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            get_process_pool(), _run_cloudpickled_call, payload
        )

    return run


def _graceful_termination(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task cancellation should not be logged as an error.
    except Exception:  # pylint: disable=broad-except
        logging.exception("Exception raised by task = %r", task)
        asyncio.get_event_loop().stop()


def create_task(future: asyncio.Future) -> asyncio.Task:

    task = asyncio.create_task(future)
    task.add_done_callback(_graceful_termination)
    return task


Queue = asyncio.Queue

gather = asyncio.gather

import asyncio
import logging
from functools import partial, wraps


def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)

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

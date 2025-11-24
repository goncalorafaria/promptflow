# external imports
import asyncio
import logging
import random
from typing import Any, Callable, Dict, List, Tuple

import aiohttp
from aiohttp import ClientSession

HttpSession = ClientSession


async def on_request_start(session, context, params):
    logging.getLogger("aiohttp.client").debug(f"Starting request <{params}>")


trace_config = aiohttp.TraceConfig()
trace_config.on_request_start.append(on_request_start)


async def post(
    url: str,
    data: bytes,
    session: HttpSession,
    maximum_backoff: int = 10,
    max_retries: int = 400,
    minimum_retries: int = 300,
    **kwargs,
) -> Any:
    """Sends a POST request to `url` with `data` and awaits result.
    This function implements an exponential backoff algorithm that retries requests exponentially,
    increasing the waiting time between retries up to a maximum backoff time. For example:

    Make a request to Quokka.

     - If the request fails, wait 1 + random_number_milliseconds seconds and retry the request.
     - If the request fails, wait 2 + random_number_milliseconds seconds and retry the request.
     - If the request fails, wait 4 + random_number_milliseconds seconds and retry the request.

    And so on, up to a maximum_backoff time.

    Continue waiting and retrying up to some maximum number of retries.

    where:
    The wait time is min(((2^n)+random_number_milliseconds), maximum_backoff), with n incremented by 1 for each iteration (request).
    maximum_backoff is will be 1 minute (60 seconds). The appropriate value depends on the use case.

    Args:
        url (str): http resource url.
        data (bytes): serialized data to be sent.
        session (HttpSession): Http session.
        maximum_backoff: maximum backoff time
        max_retires: maximum number of retires. in 6 iteration it will reach 64 seconds, so use 7 iterations

    Returns:
        Any: remote service response.
    """
    iteration = 0
    while True:
        try:
            resp = await session.post(url=url, data=data, **kwargs)
            # do something with the response if needed
            resp.raise_for_status()

            logging.debug(f"Got response {resp.status} for URL: {url}")
            # logger.info("Got response [%s] for URL: %s", resp.status, url)
            outputs = await resp.json()
            iteration = 0
            break
            # here, the async with context for the response ends, and the response is
            # released.
        except aiohttp.ClientConnectionError as e:
            logging.info(e)
            # something went wrong with the exception, decide on what to do next
            logging.info("Oops, the connection was dropped before we finished")
        except aiohttp.ClientError as e:
            logging.info(e)
            # something went wrong in general. Not a connection error, that was handled
            logging.info("Oops, something else went wrong with the request")

        except asyncio.TimeoutError as e:
            logging.info(e)
            # something went wrong in general. Not a connection error, that was handled
            logging.info("Oops, something else went wrong with the request")

        except Exception as e:
            logging.info(e)

            raise e

        iteration += 1
        if iteration >= max_retries:
            raise Exception(f"Oops, max retries exceeded in a request: {url}")

        # maximum_backoff time
        if iteration <= minimum_retries:
            wait_time = 0
        else:
            # random_number_milliseconds is a random number of milliseconds less than or equal to 1000. This helps to avoid cases in which many clients are synchronized by some situation and all retry at once, sending requests in synchronized waves. The value of random_number_milliseconds is recalculated after each retry request.
            random_number_milliseconds = random.random()
            wait_time = min(
                ((1.5**iteration) + random_number_milliseconds), maximum_backoff
            )
            await asyncio.sleep(wait_time)

        logging.info(f"Waiting {wait_time} seconds")

    return outputs["outputs"]


def request_broadcast(url: str, pack_function, unpack_function):
    """Creates a request function.

    Args:
        url (str): http resource url.
        pack_function (function): function to serialize data.
        unpack_function (function): function to integrate the response.
    """

    async def piperequest(data: Any, session: HttpSession):
        """Sends a POST request to `url` with `data` and awaits result in the provided session.

        Args:
            data (Any): data to be sent.
            session (HttpSession): Ongoing Http session.

        Returns:
            Any : reponse from remote service.
        """

        data_o = pack_function(data)
        kwargs = None

        if isinstance(data_o, list) or isinstance(data_o, tuple):
            kwargs = data_o[1:]
            data_o = data_o[0]

        # data must be serialized in bytes or json
        result = await post(url=url, data=data_o, session=session)

        # return result from the unpack function
        result = unpack_function(data, result, kwargs)

        return result

    return piperequest

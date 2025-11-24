# external imports
import abc
import logging
from typing import Any, List, Union

from tinyflow.asynchronous import Queue

# internal imports.
from tinyflow.actor import Actor 


class Source(Actor):
    """Input Node of the workflow pipeline.

    This entity represents the data entrypoint of the workflow DAG.
    """

    def __init__(self, name: str, **kwargs):
        """Creates a source actor stream.

        Args:
            name (str): Identifier of this source node.
        """
        super().__init__("Source:" + name, source=True, **kwargs)

    @abc.abstractmethod
    async def feed(self):
        """This method should starting producing data to the actor stream from io."""
        raise NotImplementedError()

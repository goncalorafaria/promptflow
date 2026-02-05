# external imports
import logging
from typing import Any, List, Union, Dict, Tuple

from promptflow.asynchronous import Queue

# internal imports.
from promptflow.constants import (
    BRANCH,
    CTRL,
    DEFAULT_BATCH,
    MAX_BUFFER_SIZE,
    STOP,
)

class Actor:
    """
    Node of the workflow pipeline.

    This entity represents a node in the workflow DAG and allows for async reading and writing to a stream and
    manages the node's input connections.
    """

    def __init__(
        self,
        name: str,
        source: bool = False,
        buffer_size: int = DEFAULT_BATCH,
        control: bool = False,
    ) -> None:
        """
        Creates an actor stream.

        Args:
            name (str): Identifier of this node.
            source (bool): Whether this node is a source stream.
            buffer_size (int): Buffer size for the queue.
            control (bool): Control flag for the actor.
        """
        self.name = name
        self.parents: List[Actor] = []
        self.children: Dict[Actor, Tuple[Queue, Union[None, int]]] = {}

        self._source = source
        self._control = control
        self.buffer_size = buffer_size

        self.queue = Queue(maxsize=buffer_size)

    def source(self) -> bool:
        return self._source

    def runnable(self) -> bool:
        return not self._control

    def control(self) -> bool:
        return self._control

    def add_parent(self, parent: "Actor") -> None:
        """
        Links a source actor to this actor stream, following the provided process.

        Args:
            parent (Actor): source actor stream.
        """
        self.parents.append(parent)

    def add_child(
        self,
        child: "Actor",
        queue: Queue = None,
        qindex: Union[None, int] = None,
        limitless=False,
    ) -> None:
        """
        Adds a child actor stream to this actor stream.

        Args:
            child (Actor): actor stream to be added as a child.
            queue (Queue): Queue for child actor.
            qindex (Union[None, int]): Queue index.
            limitless (bool): Limitless flag for child actor.
        """
        if queue is None:

            if not limitless:
                queue = Queue(self.buffer_size)
            else:
                queue = Queue(MAX_BUFFER_SIZE)

        self.children[child] = (queue, qindex)

    async def stop(self) -> None:
        """Closes the actor stream."""

        if len(self.children) > 0:
            for queue, _ in self.children.values():
                await queue.put(STOP)
        else:
            await self.queue.put(STOP)

    async def commit(self, key: str, value: Any) -> None:
        """    
        Adds a key-value pair to the actor stream.

        Args:
            key (str): Identifier.
            value (Any): Data.
        """

        if self._control:
            key = CTRL + key
            
            
        logging.debug(f"[{self.name}] - Committing {key}:{value}")

        if len(self.children) > 0:
            for queue, qid in self.children.values():

                if qid is None:
                    await queue.put((key, value))
                else:
                    await queue.put((f"{qid}{BRANCH}{key}", value))

        else:
            await self.queue.put((key, value))

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    async def tolist(self) -> List[Any]:
        """
        Consumes the actor stream and appends everything to a list.

        Returns:
            List[Any] : A list of elements of the actor stream.
        """

        results = []

        async for id, data in self.iterable():
            results.append((id, data))
            logging.debug(f"Task recovery: instance {id};")

        return results

    async def iterable(self, node: Union["Actor", None] = None):
        """
        Produces an async iterable for consuming the actor stream.

        Args:
            node (Union["Actor", None]): Node for the iterable.

        Yields:
            key, value : actor stream's key-value pairs.
        """
        loop = True

        if node is None:
            queue = self.queue
            total = 1
        else:
            assert (
                node in self.children
            ), f"The actor {node} is not linked to this actor {self}."
            queue, _ = self.children[node]
            total = len(node.parents)

        stopped = 0

        while loop:
            value = await queue.get()

            if value == STOP:
                loop = False
            else:
                yield value

    

# external imports
import abc
import heapq
import logging
import time
from functools import partial, reduce
from typing import Any, Callable, List, Tuple, Union

from promptflow.actor import Actor, Control

# internal imports
from promptflow.asynchronous import Queue, async_wrap, create_task, gather
from promptflow.constants import (
    BRANCH,
    BULLET,
    CAT,
    CTRL,
    DEBUG,
    DEFAULT_BATCH,
    DEFAULT_INFLIGHT_BATCH,
    MAX_BUFFER_SIZE,
    OF,
    SEP,
    Key,
    State,
    Value,
)
from promptflow.remote import HttpSession, request_broadcast, trace_config
from promptflow.functools import format_function


async def func_applier_many(
    func,
    id: str,
    data: Any,
    output: Actor,
    unsubscribe: Union[Queue, None] = None,
) -> bool:
    """This function applies a function to a sequence of elements and adds each element to the output actor stream.

    Args:
        func (function): Function to be applied. ( Any -> List[Any] )
        id (str): element key.
        data (Any): element value.
        output (Actor): Output actor stream.
        unsubscribe (Union[Queue, None]) : Unsubscribe queue.
    """

    outputs = await func(data)
    n = len(outputs)

    for i, output_data in enumerate(outputs):
        await output.commit(f"{id}{SEP}{i}{OF}{n}", output_data)

    # This guarantees that the buffer is bounded.
    if unsubscribe: 
        _ = await unsubscribe.get()


async def func_applier(
    func,
    id: str,
    data: Any,
    output: Actor,
    unsubscribe: Union[Queue, None] = None,
) -> bool:
    """This function applies a function to a single element and awaits the result.
    func (function): Function to be applied. ( Any -> Any )
    id (str): element key.
    data (Any): element value.
    output (Actor): Output actor stream.
    unsubscribe (Union[Queue, None]) : Rate Limiter Queue.
    """

    output_data = await func(data)

    await output.commit(id, output_data)

    # This guarantees that the buffer is bounded.
    if unsubscribe:
        _ = await unsubscribe.get()


class Process:
    """Link of the workflow pipeline.

    This entity represents a link in the workflow DAG.
    It transforms data from an input actor and produces to an output actor.
    """

    def __init__(self, name: str):
        """Creates an workflow process.

        Args:
            name (str): Identifier for this process.
        """
        self.name = name

    def __call__(self, node: Actor) -> Actor:
        """Applies the process to a input actor stream.

        Args:
            node (Actor): Input actor stream.

        Returns:
            outputnode (Actor): Output actor stream.
        """

        outnode = Bridge(
            # f"{node}{SEP}{self.name}{SEP}",
            self.name,
            process=self,
        )

        outnode.add_parent(node)
        node.add_child(outnode)

        return outnode

    def __or__(self, other):

        if isinstance(other, Process):
            return ProcessUnion(self, other)
        else:
            raise TypeError(
                f"unsupported operand type(s) for |: '{self.__class__.__name__}' and '{other.__class__.__name__}'"
            )

    @abc.abstractmethod
    async def execute(self, **kwargs):
        """Implements this process async execution function."""
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"
    
    def __str__(self):
        return self.__repr__()
    
class ProcessUnion(Process):
    def __init__(self, process1, process2):
        super().__init__(name="{}-> {}".format(process1.name, process2.name))
        self.process1 = process1
        self.process2 = process2

    def __call__(self, *node: List[Actor]):
        return self.process2(self.process1(*node))


class Bridge(Actor):
    """Actor stream that is the result of applying a process."""

    def __init__(self, name: str, process: Process, buffer_size: int = DEFAULT_BATCH):
        """creates a bridge actor stream.

        Args:
            name (str): Identifier for this actor stream.
            process (Process): Process that transforms the input actor stream and produces data into this actor stream.
        """
        super().__init__(name, source=False, buffer_size=buffer_size)

        self.op = process

    def execution_context(self) -> Tuple[Process, List[Actor]]:
        """Returns the execution context of this actor stream. The execution context is a tuple containing the process and the list of input actor streams.

        Returns:
            Tuple[Process,List[Actor]]: _description_
        """
        return self.op, self.parents


class ControlBridge(Control):
    """Actor stream that is the result of applying a process."""

    def __init__(self, name: str, process: Process):
        """creates a bridge actor stream.

        Args:
            name (str): Identifier for this actor stream.
            process (Process): Process that transforms the input actor stream and produces data into this actor stream.
        """
        super().__init__(name, source=False)

        self.op = process

    def execution_context(self) -> Tuple[Process, List[Actor]]:
        """Returns the execution context of this actor stream. The execution context is a tuple containing the process and the list of input actor streams.

        Returns:
            Tuple[Process,List[Actor]]: _description_
        """
        return self.op, self.parents


class ControlProcess(Process):
    """Control process.

    This process is used to control the workflow and gather metadata.
    """

    def __init__(self, name: str):
        """Creates a control process.

        Args:
            name (str): Identifier for this process.
        """
        super().__init__(name)

    def __call__(self, node: Actor) -> Actor:
        """Applies the process to a input actor stream.

        Args:
            node (Actor): Input actor stream.

        Returns:
            outputnode (Actor): Output actor stream.
        """

        outnode = ControlBridge(
            # f"{node}{SEP}{self.name}{SEP}",
            self.name,
            process=self,
        )

        outnode.add_parent(node)

        node.add_child(outnode)
        node.add_child(outnode.control, limitless=True)

        return outnode, outnode.control

    async def execute(self, input: Actor, output: ControlBridge) -> bool:
        """Executes the mapping process.

        Args:
            input (Actor): Input actor stream to consume.
            output (Actor): Output actor stream to produce.

        Returns:
            bool : True if sucefull.
        """

        return await self.control_execute(
            input=input, output=output, control=output.control
        )

    async def control_execute(
        self, input: Actor, output: Actor, control: Actor
    ) -> bool:
        """Executes the controlled execute function.

        Args:
            input (Actor): Input actor stream to consume.
            output (Actor): Output actor stream to produce.
            control (Actor): Control actor stream to produce.

        Returns:
            bool : True if sucefull.
        """
        raise NotImplementedError()


class Junction(Process):
    """Fork of the workflow pipeline."""

    def __init__(self, name: str):
        """Creates a fork of the workflow pipeline.

        Args:
            name (str): Identifier for this fork.
        """

        super().__init__(name)

    def __call__(self, *nodes: List[Actor]) -> Actor:
        """_summary_

        Args:
            nodes (List[Actor]): _description_

        Returns:
            _type_: _description_
        """

        outnode = Bridge(
            # f"{nodes}{SEP}{self.name}{SEP}",
            self.name,
            process=self,
            buffer_size=MAX_BUFFER_SIZE,
        )

        jointq = Queue()

        for i, node in enumerate(nodes):
            outnode.add_parent(node)
            node.add_child(outnode, queue=jointq, qindex=i)

        return outnode


class MetaMap(Process):
    """This process applies a particular async function to each element of the input stream."""

    def __init__(
        self,
        func: Callable[[Value], Value],
        many: bool = True,
        name: Union[None, str] = None,
    ):
        """Creates a map process.

        Args:
            func (function): async function to apply.
            many (bool): If true does a flatMap. Defaults to True.
        """

        if name is None:
            name = f"map:{format_function(func)}"

        super().__init__(name)
        self.func = func
        if many:
            self.func_applier =  func_applier_many
        else:
            self.func_applier =  func_applier

        

    async def execute(self, input: Actor, output: Actor) -> bool:
        """Executes the mapping process.

        Args:
            input (Actor): Input actor stream to consume.
            output (Actor): Output actor stream to produce.

        Returns:
            bool : True if sucefull.
        """

        inflight = Queue(maxsize=DEFAULT_BATCH)

        logging.debug(f"Launching task: {self.name}")
        st = time.time()

        runs = []

        async for id, data in input.iterable(output):

            logging.debug(f"Task {self.name}: instance {id}; ")

            if BULLET in id:
                # nao processa.
                await output.commit(id, data)
            else:
                await inflight.put(1)

                runs.append(
                    create_task(
                        self.func_applier(func=self.func, id=id, data=data, output=output)
                    )
                )

            # if len(runs) >= DEFAULT_BATCH :
            #    await gather(*runs)
            #    runs = [ ]

        await gather(*runs)
        await output.stop()

        logging.debug(f"Finished launching task: {self.name}. Syncing runs.")
        logging.info(f"Duration: [{self.name}] : {(time.time() - st):.3f} ")

        return True


class Callback(ControlProcess):
    """This process applies a particular function to each element of the input stream that collects metrics."""

    def __init__(
        self,
        func: Callable[[Key, Value, State], State],
        initstate: State,
        name: Union[None, str] = None,
    ):
        """Creates a calback process.

        Args:
            func (function): async function to apply.
            initstate (Any) : inicial state.
        """

        if name is None:
            name = f"Callback:{format_function(func)}"

        super().__init__(name)
        self.func = func
        self.initstate = initstate
        # self.states = {}

    async def control_execute(
        self, input: Actor, output: Actor, control: Actor
    ) -> bool:
        """Executes the callback process.

        Args:
            input (Actor): Input actor stream to consume.
            output (Actor): Output actor stream to produce.

        Returns:
            bool : True if sucefull.
        """

        logging.debug(f"Launching task: {self.name}")
        st = time.time()

        state = self.initstate
        async for id, data in input.iterable(output):

            logging.debug(f"Task {self.name}: instance {id};")

            state = self.func(id, data, state)
            await output.commit(id, data)

        await control.commit(self.name, state)
        await output.stop()
        await control.stop()

        logging.debug(f"Finished launching task: {self.name}. Syncing runs.")
        logging.info(f"Duration: [{self.name}] : {(time.time() - st):.3f} ")

        return True


class Crono(Callback):
    """This process applies a particular function to each element of the input stream that collects metrics."""

    def __init__(self, name: Union[None, str] = None):
        """Creates a crono process."""
        self.curr_rate = 10
        self.alpha = 0.5
        self.last_time = time.time()

        def append_time(id, data, state):

            now = time()

            elapsed = now - self.last_time

            if elapsed > self.curr_rate * 10:
                self.curr_rate = 10

            self.curr_rate = self.alpha * (self.curr_rate - elapsed) + elapsed
            print(f"crono:{name} >: {1/self.curr_rate}")

            # state.append(elapsed)

            self.last_time = now

            return state

        if name is None:
            name = "Crono"

        super().__init__(func=append_time, initstate=[], name=name)


class Combine(Process):
    """This process combines elements of the input stream.

    Given a depth level this process combines all of the input elements that share a portion of the hierachical key.
    """

    def __init__(self, depth: int = 1, name: str = "combine", unbatch: bool = True):
        """Creates a combine processs given a depth level.

        Args:
            depth (int): level of agregation in the hierarchical key. Defaults to 1.
            unbatch (bool): input stream is batched or not. Defaults to True.
        """

        self.depth = depth
        self.unbatch = unbatch

        super().__init__(name)

    def __parsekey(self, id: str):
        """splits the hierarchical key into superkey and key and counts the number of elements belonging in the superkey hierarchy.

        Args:
            id (str): key to be parsed.

        Returns:
            superkey (str): superkey used for agregation.
            key (str): group id with the superkey hierarchy.
            total (int): number of elements in the superkey hierarchy.
        """
        id_ = id.split(SEP)

        superkey, key = SEP.join(id_[: self.depth]), SEP.join(id_[self.depth :])

        total = reduce(
            lambda a, b: a * b, [int(k.split(OF)[1]) for k in id_[self.depth :]], 1
        )

        return superkey, key, total

    async def execute(self, input: Actor, output: Actor) -> bool:
        """Executes the combine process.

        Args:

            input (Actor): Input actor stream to consume.
            output (Actor): Output actor stream to produce.

        Returns:
            bool : True if there are no elements in cache.
        """

        logging.debug(f"Launching task: {self.name}")
        st = time.time()

        cache = {}

        async for id_, data_ in input.iterable(output):

            logging.debug(f"Task {self.name}: instance {id_};")

            if self.unbatch:

                if BULLET in id_:
                    id_ = id_.replace(BULLET, "")
                    stream = [(id_, data_)]
                else:
                    stream = zip(id_.split(CAT), data_)
            else:
                stream = [(id_, data_)]

            for id, data in stream:

                superkey, key, total = self.__parsekey(id)

                if total == 1:

                    await output.commit(superkey, {key: data, "count": 1})

                else:
                    if superkey in cache:
                        # sp in memory.
                        cache[superkey][key] = data
                        cache[superkey]["count"] += 1

                        if cache[superkey]["count"] == total:
                            # gathered everything.
                            jointdata = cache.pop(superkey, None)

                            await output.commit(superkey, jointdata)

                    else:
                        # sp not in memory.
                        cache[superkey] = {key: data, "count": 1}

        await output.stop()

        logging.debug(f"Finished launching task: {self.name}. Syncing runs.")
        logging.info(f"Duration: [{self.name}] : {(time.time() - st):.3f} ")

        return not (len(cache) > 0)


class Barrier(Process):
    """This process creates a synchronization block. It applies a function inorder to the actor stream acording to specified ordering."""

    def __init__(
        self,
        functional: Callable[[Value], List[Value]],
        name: str = "barrier",
        order=int,
    ):
        """Creates a barrier process.

        Args:
            functional (function): Function to be applied in order.
            order (function : key -> int): Ordering criterion. Defaults to int.
        """
        self.functional = functional
        self.order = order

        super().__init__(name)

    async def execute(self, input: Actor, output: Actor, functional=None) -> bool:
        """Executes the barrier process.

        Args:
            input (Actor): Input actor stream to consume.
            output (Actor): Output actor stream to produce.

        Returns:s
        """
        if functional is None:
            functional = self.functional

        logging.debug(f"Launching task: {self.name}")
        st = time.time()

        mark = 0
        priorityq = []

        async for id, data in input.iterable(output):

            logging.debug(f"Task {self.name}: instance {id}; ")

            p = self.order(id)

            heapq.heappush(priorityq, (p, id, data))

            while (len(priorityq) > 0) and (priorityq[0][0] == mark):
                _, idj, dataj = heapq.heappop(priorityq)

                mark += 1
                work_units = functional(dataj)
                total = len(work_units)

                for j, wi in enumerate(work_units):

                    await output.commit(f"{idj}{SEP}{j}{OF}{total}", wi)

        await output.stop()

        logging.debug(f"Finished launching task: {self.name}. Syncing runs.")
        logging.info(f"Duration: [{self.name}] : {(time.time() - st):.3f} ")

        return True


class StatefulBarrier(Barrier):
    """This process creates a synchronization block. It applies a function inorder to the actor stream acording to specified ordering."""

    class State:
        def init(self):
            raise NotImplementedError()

        def update(self, data: Value) -> List[Value]:
            raise NotImplementedError()

        def terminate(self):
            raise NotImplementedError()

    def __init__(
        self,
        functional: State,
        name: str = "barrier",
        order=int,
    ):
        """Creates a barrier process.

        Args:
            functional (function): Function to be applied in order.
            order (function : key -> int): Ordering criterion. Defaults to int.
        """
        self.state_functional = functional
        self.order = order

        super().__init__(name=name, functional=None, order=order)

    async def execute(self, input: Actor, output: Actor) -> bool:
        """Executes the barrier process.

        Args:
            input (Actor): Input actor stream to consume.
            output (Actor): Output actor stream to produce.

        Returns:
            bool : True of sucessful.
        """

        logging.debug(f"Launching task: {self.name}")
        st = time.time()

        self.state_functional.init()

        results = await super().execute(
            input=input, output=output, functional=self.state_functional.update
        )

        self.state_functional.terminate()

        # logging.debug(f"Finished launching task: {self.name}. Syncing runs.")

        return results


class Batching(Process):
    """This process produces a batch of elements from the input stream."""

    def __init__(
        self,
        size: int = 8,
        name: str = "batching",
        select: Callable[[Value], bool] = lambda data: not (data is None),
    ):
        """Creates a batching proess.

        Args:
            size (int): Batch size. Defaults to 8.
            select (function): Predicate determining which values to form a batch.
        """

        self.size = size
        self.select = select
        super().__init__(name)

    async def execute(self, input: Actor, output: Actor) -> bool:
        """Executes the batching process.

        Args:
            input (Actor): Input actor stream to consume.
            output (Actor): Output actor stream to produce.

        Returns:
            bool : True if sucefull.
        """

        logging.debug(f"Launching task: {self.name}")
        st = time.time()

        qdata, qid, qsz = [], [], 0

        async for id, data in input.iterable(output):

            logging.debug(f"Task {self.name}: instance {id}; ")

            if self.select(data):
                qdata.append(data)
                qid.append(id)
                qsz += 1
            else:
                await output.commit(f"{BULLET}{id}", data)

            if qsz == self.size:

                batch_data = qdata
                batch_id = CAT.join(qid)

                await output.commit(batch_id, batch_data)

                qdata, qid, qsz = [], [], 0

        if qsz > 0:
            batch_data = qdata
            batch_id = CAT.join(qid)
            await output.commit(batch_id, batch_data)

        await output.stop()

        logging.debug(f"Finished launching task: {self.name}. Syncing runs.")
        logging.info(f"Duration: [{self.name}] : {(time.time() - st):.3f}")

        return True


class UnBatching(Process):
    def __init__(self, name: str = "unbatching"):
        super().__init__(name)

    async def execute(self, input: Actor, output: Actor) -> bool:
        """Executes the combine process.

        Args:

            input (Actor): Input actor stream to consume.
            output (Actor): Output actor stream to produce.

        Returns:
            bool : True if there are no elements in cache.
        """

        logging.debug(f"Launching task: {self.name}")
        st = time.time()

        async for id_, data_ in input.iterable(output):

            logging.debug(f"Task {self.name}: instance {id_}; ")

            if BULLET in id_:
                id_ = id_.replace(BULLET, "")
                stream = [(id_, data_)]
            else:
                stream = zip(id_.split(CAT), data_)

            for key, data in stream:
                await output.commit(key, data)

        await output.stop()

        logging.debug(f"Finished launching task: {self.name}. Syncing runs.")
        logging.info(f"Duration: [{self.name}] : {(time.time() - st):.3f} ")

        return True


class NativeMap(MetaMap):
    """Applies a function in another process in this machine and async waits for the result."""

    def __init__(
        self,
        func: Callable[[Value], Value],
        name: Union[str, None] = None,
        many: bool=False
    ):
        """Create a native map process.

        Args:
            func (function): Function to be applied.
            many (bool): Whether to do a flat map or not.
        """

        #if DEBUG:
        _func = async_wrap(func)
        #else:
        #    _func = remote_wrap(func)

        super().__init__(func=_func, many=many, name=name)


class Map(NativeMap):
    """Applies a function another process in this machine and async waits for the result.

    It's a Native map with many=False.
    """

    def __init__(self, func: Callable[[Value], Value]):
        """Creates a classic map process.

        Args:
            func (function): Function to be applied.
        """
        super().__init__(func=func, name="map({})".format(format_function(func)), many=False)


class FlatMap(NativeMap):
    """Applies a function that produces a sequence of elements and adds each element to the output actor stream."""

    def __init__(self, func: Callable[[Value], List[Value]]):
        """Creates a flat map process.

        Args:
            func (function): Function to be applied.
        """
        super().__init__(func=func, many=True)


class RemoteMap(MetaMap):
    """Applies a function in another machine and waits async for the result."""

    def __init__(
        self,
        url: str,
        pack_function: Callable[[Value], bytes] = lambda data: data,
        unpack_function: Callable[
            [Value, Value], Value
        ] = lambda input, predictions, kwargs: predictions,
        name: Union[str, None] = None,
        many: bool = True,
        inflight_batch: int = DEFAULT_INFLIGHT_BATCH,
    ):
        """Creates a remote map process.

        Args:
            url (str): http url of the target service.
            pack_function (function): Transforms each element into a serializable object.
            unpack_function (_type_): Integrates the function map results and stream elements into a single object.
        """

        self.inflight_batch = inflight_batch

        super().__init__(
            func=request_broadcast(
                url=url, pack_function=pack_function, unpack_function=unpack_function
            ),
            many=many,
            name=f"RemoteMap:{url}",
        )

    async def execute(self, input: Actor, output: Actor) -> bool:
        """Executes the remote mapping process.

        Args:
            input (Actor): Input actor stream to consume.
            output (Actor): Output actor stream to produce.

        Returns:
            bool : True if sucefull.
        """

        inflight = Queue(maxsize=self.inflight_batch)

        logging.debug(f"Launching task: {self.name}")
        st = time.time()

        runs = []

        async with HttpSession(trace_configs=[trace_config]) as session:

            async for id, data in input.iterable(output):

                logging.debug(f"Task {self.name}: instance {id}; ")

                if BULLET in id:
                    # nao processa.
                    await output.commit(id, data)
                else:

                    await inflight.put(1)

                    runs.append(
                        create_task(
                            self.func_applier(
                                partial(self.func, session=session),
                                id,
                                data,
                                output,
                                inflight,
                            )
                        )
                    )

                # if len(runs) >= 2*self.inflight_batch :
                # done, pending = await asyncio.wait(runs, return_when=FIRST_COMPLETED)
                # runs = pending

            logging.debug(f"Syncing runs: {self.name}.")

            await gather(*runs)
            await output.stop()

        logging.debug(f"Finished launching task: {self.name}.")
        logging.info(f"Duration: [{self.name}] : {(time.time() - st):.3f} ")

        return True


class Aggregate(Junction):
    def __init__(self, key_factory: Callable[[Value], Key], name: str = "aggregate"):
        super().__init__(name)
        self.key_factory = key_factory

    async def execute(self, *inputs: List[Actor], output: Actor) -> bool:

        logging.debug(f"Launching task: {self.name}")
        st = time.time()

        cache = {}

        for input in inputs:
            async for id_, data in input.iterable(output):

                logging.debug(f"Task {self.name}: instance {id_}; ")

                key = self.key_factory(data)

                if key in cache:
                    # sp in memory.
                    cache[key].append(data)

                else:
                    # sp not in memory.
                    cache[key] = [data]

        for key, data in cache.items():
            await output.commit(key, data)

        await output.stop()

        logging.debug(f"Finished launching task: {self.name}. Syncing runs.")
        logging.info(f"Duration: [{self.name}] : {(time.time() - st):.3f} ")

        return len(cache) > 0


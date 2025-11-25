# internal imports.
from promptflow.actor.source import Source
from promptflow.actor.core import Actor


class Iterable(Source):
    """Input node for the workflow pipeline from a iterable source.

    Given an iterable creates a source actor.
    """

    def __init__(self, iterable, keyvalue: bool = True, name: str = "Iterable:", **kwargs):
        """Creates a source actor based on a given iterable.

        Args:
            iterable (iterable): data to feed to this actor stream.
            keyvalue (bool): if the iterable is a key-value pair.
        """
        super().__init__(name=name, **kwargs)
        self.python_iterable = iterable

        if keyvalue:
            self.feed = self.__feed_kv
        else:
            self.feed = self.__feed

    def append(self, value):
        self.python_iterable.append(value)

    async def __feed_kv(self):
        """Produces key value pairs to the stream based on the saved itearable."""
        count = 0
        for key, item in self.python_iterable:
            await self.commit(str(key), item)

            count += 1

        await self.stop()

    async def __feed(self):
        """Produces datapoints to the stream base on the saved iterable."""
        count = 0
        for item in self.python_iterable:
            await self.commit(str(count), item)
            count += 1

        await self.stop()


class ListInput(Iterable):
    """Input node for the workflow pipeline from a list source.

    Given a list creates a source actor.
    """

    def __init__(self, list):
        super().__init__(name="ListInput:", iterable=list, keyvalue=False)

class DictInput(Iterable):
    """Input node for the workflow pipeline from a dict source.

    Given a dict creates a source actor.
    """

    def __init__(self, dict):
        super().__init__(name="DictInput:", iterable=dict, keyvalue=True)
        
        
def try_to_convert_to_input(data):
    
    if isinstance(data, Actor):
        return data
    
    ## check the first element of the "iterable" without
    if isinstance(data, list):
        if len(data) > 0:
            if isinstance(data[0], tuple):
                return DictInput(dict=data)
            else:
                return ListInput(list=data)
            
        else:
            return ListInput(list=data)
    
    elif isinstance(data, dict):
        return DictInput(dict=data.items())
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
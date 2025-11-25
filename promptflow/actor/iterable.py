# internal imports.
from promptflow.actor.source import Source


class Iterable(Source):
    """Input node for the workflow pipeline from a iterable source.

    Given an iterable creates a source actor.
    """

    def __init__(self, iterable, keyvalue: bool = True, **kwargs):
        """Creates a source actor based on a given iterable.

        Args:
            iterable (iterable): data to feed to this actor stream.
            keyvalue (bool): if the iterable is a key-value pair.
        """
        super().__init__(name="Iterable:", **kwargs)
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

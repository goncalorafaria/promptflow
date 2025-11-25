

# internal imports.
from promptflow.constants import (
    SEP,
)

from promptflow.actor import Actor



class Control(Actor):
    def __init__(self, name: str, **kwargs):

        super().__init__(name=name, **kwargs)

        self.control = Actor(
            name=f"{name}{SEP}ctrl", source=False, control=True)



# internal imports.
from tinyflow.constants import (
    SEP,
)

from tinyflow.actor import Actor



class Control(Actor):
    def __init__(self, name: str, **kwargs):

        super().__init__(name=name, **kwargs)

        self.control = Actor(
            name=f"{name}{SEP}ctrl", source=False, control=True)

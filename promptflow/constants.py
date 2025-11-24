STOP = None
SEP = ":"
CAT = "$"
OF = "/"
BULLET = "#"
BRANCH = "!"
CTRL = "?"

"""default minimum resolution for video frames"""

DEFAULT_BATCH = 8

"""
This parameter controls, for each local stage, 
how many requests are allowed to be sent at once.

Values less than zero (<=0) means that it is unbounded.

Not clear how, in general, they affect the performance. 
(ideas on this)
Large batch :
    + leads to less context switch i.e. higher performance. 
    - can lead to taking more time to deliver work to every stage i.e. not exploiting parallelism (bad).
    - can create memory botlenecks because there is long queues of intermidiate values.
"""

MAX_BUFFER_SIZE = 10000000

DEFAULT_INFLIGHT_BATCH = 32

"""
This parameter controls, for each remote stage, 
how many requests are allowed to be sent at once.

Values less than zero (<=0) means that it is unbounded.

Ideally should be twice the number of machines serving in each stage for prefetching.
In a operacionalzied scenario, it might be best to leave this unbounded. 
Let the autoscalar and loadbalancer deal with it. 
"""
# datatypes
from typing import Any

State = Any
Value = Any
Key = str

DEBUG = True
"""
    This is significantly less efficient. But makes it easier to debug. 
"""

WORKER_LOG_SPLIT = "*" * 40

import os
from typing import Any
import logging

# String separators for key encoding
STOP = None
SEP = ":"
CAT = "$"
OF = "/"
BULLET = "#"
BRANCH = "!"
CTRL = "?"

"""default minimum resolution for video frames"""

DEFAULT_BATCH = 1024


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

DEFAULT_INFLIGHT_BATCH = 1024

"""
This parameter controls, for each remote stage, 
how many requests are allowed to be sent at once.

Values less than zero (<=0) means that it is unbounded.

Ideally should be twice the number of machines serving in each stage for prefetching.
In a operacionalzied scenario, it might be best to leave this unbounded. 
Let the autoscalar and loadbalancer deal with it. 
"""

# Debug / info configuration
# DEBUG: verbose (includes per-item task dumps / content-ish traces)
# INFO:  stage timings only — use to find workflow bottlenecks without content
# neither: WARNING+
def _env_flag(name: str) -> bool:
    raw = os.getenv(name, "0")
    return raw.lower() in {"1", "true", "yes"} or (raw.isdigit() and int(raw) == 1)


DEBUG = _env_flag("DEBUG")
INFO = _env_flag("INFO") or DEBUG

if DEBUG:
    _log_level = logging.DEBUG
elif INFO:
    _log_level = logging.INFO
else:
    _log_level = logging.WARNING

logging.basicConfig(level=_log_level)

WORKER_LOG_SPLIT = "*" * 40

# Remote HTTP request configuration
DEFAULT_MAXIMUM_BACKOFF = 10  # Maximum backoff time in seconds
DEFAULT_MAX_RETRIES = 400  # Maximum number of retry attempts
DEFAULT_MINIMUM_RETRIES = 300  # Minimum retries before applying backoff
DEFAULT_BACKOFF_BASE = 1.5  # Exponential backoff base multiplier

# Crono process configuration
DEFAULT_CRONO_INITIAL_RATE = 10.0  # Initial rate for crono process
DEFAULT_CRONO_ALPHA = 0.5  # Smoothing factor for rate calculation
DEFAULT_CRONO_RATE_MULTIPLIER = 10  # Multiplier for rate reset threshold

# Workflow visualization configuration
DEFAULT_WORKFLOW_FIGURE_SIZE = (18, 18)  # Figure size in inches (width, height)
DEFAULT_WORKFLOW_NODE_SIZE = 400  # Node size for networkx visualization
DEFAULT_WORKFLOW_FONT_SIZE = 12  # Font size for node labels
DEFAULT_WORKFLOW_PDF_FILENAME = "workflow.pdf"  # Default filename for saved workflow diagram

# datatypes
State = Any
Value = Any
Key = str

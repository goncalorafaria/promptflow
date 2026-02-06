# promptflow

**Orchestrate complex LLM pipelines without the asyncio headaches.**

## The Problem

Building synthetic data generation pipelines with LLMs is deceptively hard. For each piece of source data, you often need to:

- Make sequential LLM queries
- Parse and split outputs into pieces  
- Branch into multiple follow-up queries based on results
- Combine everything back together

And you need to do this across hundreds or thousands of inputs.

**Your options today are painful:**

| Approach | Problem |
|----------|---------|
| **Intricate asyncio code** | Exception handling is a nightmare. Debugging is worse. Good luck maintaining it. |
| **Synchronous batched rounds** | Super slow. Leaves massive parallelism on the table.  |

## The Solution

**promptflow** lets you write your pipeline logic declaratively while we handle all the parallelism, batching, and async orchestration under the hood.

```python
from promptflow import WorkFlow, Map, FlatMap, Combine, ListInput
from promptflow.model import LLMMap, RemoteLLMClient

# Connect to your vLLM server
client = RemoteLLMClient(
    server_url="http://localhost:8000",
    model_path="Qwen/Qwen2.5-7B-Instruct"
)

# Define your pipeline with simple, composable operations
pipeline = (
    Map(func=lambda x: {"llm_call_input": format_prompt(x)})
    | LLMMap(vllm_client=client, n=3, input_key="llm_call_input",output_key="llm_call_output")  # Generate 3 responses per input 
    | FlatMap(func=break_into_list_of_chuncks)         # Score each set
    | Map(func=lambda x: {"judge_call_input":format_score_prompt(x),**x})
    | LLMMap(vllm_client=client, n=1, input_key="judge_call_input", output_key="judge_call_output")
    | Combine()
)

# Run on your data - parallelism handled automatically
results = pipeline(my_data_list)
```

## Key Concepts

### Workflows

A `WorkFlow` defines your pipeline logic in a `forward()` method. Call it like a function to execute:

```python
class MyPipeline(WorkFlow):
    def forward(self, inputs):
        a = ListInput(inputs)
        b = Map(func=process)(a)
        c = FlatMap(func=expand)(b)
        d = Combine()(c)
        return d

pipeline = MyPipeline()
results = pipeline(my_inputs)  # Runs everything
```

### Processes (Operations)

Processes transform data streams. Chain them with `|`:

| Process | Description |
|---------|-------------|
| `Map(func)` | Apply function to each element |
| `FlatMap(func)` | Apply function that returns a list, flatten results |
| `LLMMap(client, input_key, output_key)` | Call LLM on each element, write response back |
| `Combine(depth)` | Gather scattered elements back together |
| `Aggregate(key_factory)` | Group elements by key |
| `Barrier(func)` | Synchronization point, process elements in order |
| `Batching(size)` / `UnBatching()` | Collect elements into batches / expand batches back |

### LLM Operations

Built-in support for LLM inference:

```python
from promptflow.model import LLMMap, RemoteLLMClient, ChatGPTClient

# For self-hosted vLLM
vllm = RemoteLLMClient(
    server_url="http://localhost:8000",
    model_path="meta-llama/Llama-3-8B-Instruct",
    max_concurrent_requests=256  # Automatic rate limiting
)

# For OpenAI
openai = ChatGPTClient(model="gpt-4o")
```

### LLMMap

The core operation for LLM inference in your pipeline. It reads a prompt from your data, calls the model, and writes the response back.

```python
LLMMap(
    vllm_client,                      # RemoteLLMClient or ChatGPTClient
    input_key="llm_call_input",       # Key to read chat template from
    output_key="llm_call_output",     # Key to write LLMResponse to
    n=1,                              # Number of completions per input
    assertions=[...],                 # Optional: validate outputs
    max_correctness_attempts=3        # Retry count if assertions fail
)
```

**Data flow:**

```python
# Input dict
{"llm_call_input": [{"role": "user", "content": "..."}], "other_field": ...}

# After LLMMap
{"llm_call_input": [...], "other_field": ..., "llm_call_output": [LLMResponse, ...]}
```

**LLMResponse object:**

Each completion returns an `LLMResponse` with:
- `.output` — The model's response (with thinking tokens stripped if present)
- `.reasoning` — Extracted chain-of-thought/thinking content (for models like Qwen)
- `.text` — Raw full response
- `.valid` — `True` unless an assertion failed

**Assertions for structured outputs:**

```python
from promptflow.model import HasJson, HasJsonKey

LLMMap(
    vllm_client=client,
    input_key="prompt",
    output_key="response",
    assertions=[
        HasJson(),              # Response must contain valid JSON
        HasJsonKey("score"),    # JSON must have "score" key
    ],
    max_correctness_attempts=3  # Retry up to 3x if assertions fail
)
```

**Multiple completions with `n`:**

```python
# Generate 5 candidate responses per input
LLMMap(vllm_client=client, n=5, input_key="prompt", output_key="candidates")

# output_key will contain a list of 5 LLMResponse objects
```
 

## How It Works

Under the hood, promptflow builds a DAG of async actors connected by queues. When you call a workflow:

1. **Graph construction**: Your `forward()` method builds the computation graph
2. **Breath-first traversal**: We find all source nodes and runnable processes
3. **Parallel execution**: All independent operations run concurrently via asyncio
4. **Automatic backpressure**: Bounded queues prevent memory blowup
5. **Result collection**: Outputs are gathered and returned in order

You write sequential-looking code. We execute it with maximum parallelism.

## Installation

```bash
pip install promptflow
```

## Requirements

- Python 3.8+
- `aiohttp` for async HTTP
- `transformers` for tokenization (LLM features)
- `openai` for ChatGPT support (optional)

## Visualization

Debug your pipeline structure:

```python
pipeline = MyPipeline()
pipeline.show(my_inputs)  # Renders DAG with matplotlib
pipeline.show(my_inputs, save_img=True)  # Save to PDF
```





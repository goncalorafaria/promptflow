from transformers import AutoTokenizer
import requests
import aiohttp
import asyncio
import os
from typing import Any, Union
import logging
# Import promptflow dependencies
from promptflow.process import MetaMap
from promptflow.constants import DEFAULT_INFLIGHT_BATCH
from pydantic import BaseModel
from rubrics.parse import parse_thinking_tokens_qwen
from rubrics.parse import extract_json_from_response
from promptflow import Map,  WorkFlow, ListInput
from typing import List
## get env variable DEBUG
DEBUG = os.getenv("DEBUG", "False").lower() == "true"


class LLMResponse:
    output: str
    reasoning: str
    text: str
    
    def __init__(self, response: str):
        self.reasoning, self.output = parse_thinking_tokens_qwen(response)
        self.text=response

    def __str__(self):
        return f"LLMResponse(reasoning={self.reasoning}, output={self.output})"
    
    def __repr__(self):
        return f"LLMResponse(reasoning={self.reasoning}, output={self.output})"
    

class Assertion:
    def check(self, response: LLMResponse) -> bool:
        raise NotImplementedError("Subclasses must implement this method")

class HasJson(Assertion):

    def check(self, response: LLMResponse) -> bool:
        return extract_json_from_response(response.output) is not None

class HasJsonKey(Assertion):
    def __init__(self, key: str):
        self.key = key
    def check(self, response: LLMResponse) -> bool:
        json_response = extract_json_from_response(response.output)
        if json_response is None:
            return False
        else:
            return self.key in json_response



class RemoteLLMClient:
    def __init__(
        self,
        server_url: str,
        model_path: str,
        max_new_tokens: int = 1024,
        max_prompt_length: int = 1024*3,
        stop_tokens: list = None,
        temperature: float = 1.0,
        timeout: float = 300,
        max_retries: int = 15,
        max_concurrent_requests: int = 256,
    ):
        
        self.server_url = server_url.rstrip("/")
       
    
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_prompt_length = max_prompt_length - 1
         
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent_requests = max_concurrent_requests


        self.model_path = model_path 
        self._check_health()
        


        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
        )
        
        if stop_tokens is None:
            self.stop_tokens = [self.tokenizer.eos_token]
        else:
            self.stop_tokens = stop_tokens + [self.tokenizer.eos_token]
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
            self.tokenizer.pad_token = self.tokenizer.bos_token
            
    
    def _prep_prompt(self, chat_template_prompt, add_generation_prompt=False):
        
        prompt = self.tokenizer.apply_chat_template(
            chat_template_prompt,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,#False#not any([ ct["role"] == "assistant" for ct in chat_template_prompt ]) ,
        )
        
        tokens = self.tokenizer.encode(prompt, max_length=self.max_prompt_length, truncation=True)
        truncated_prompt = self.tokenizer.decode(tokens, skip_special_tokens=False)
                
        return truncated_prompt
    
    def _truncate_tokens(self, tokenized_input):
        """
        Truncate tokenized input if it exceeds max_prompt_length.
        Keeps the end of the sequence (most recent tokens).
        
        Args:
            tokenized_input: List of token IDs
            
        Returns:
            Truncated list of token IDs
        """
        if len(tokenized_input) <= self.max_prompt_length:
            return tokenized_input
        
        # Truncate from the beginning, keeping the end
        return tokenized_input[-self.max_prompt_length:]

    def _check_health(self):
        url = f"{self.server_url}/v1/models"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            
            ## check if self.model_path is in the response
            if not len([1 for x in resp.json()["data"] if x["id"] == self.model_path ]):
                raise ConnectionError(f"Model {self.model_path} not found in the response: - {resp.json()} - {url}")
            
            if DEBUG:
                print(f"Server[{self.model_path}]: {self.server_url} is healthy and ready for requests.")
            return True
        except Exception as e:
            raise ConnectionError(f"Server health check failed: {str(e)}")

    async def _post_with_retries_async(self, endpoint, payload, use_tqdm=False):
        """
        Submit requests with retry logic and connection pooling.
        
        Args:
            endpoint: API endpoint to call
            payload: List of payloads to send
            use_tqdm: Whether to show progress bar
            
        Returns:
            List of responses
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        # Create ONE session with connection pooling for ALL requests
        # Since server is a single host, set limits based on max_concurrent_requests
        # Connection reuse means actual connections needed < max_concurrent_requests
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_requests,  # Total connection pool size
            limit_per_host=self.max_concurrent_requests,  # Max connections to server
            ttl_dns_cache=600,
            enable_cleanup_closed=True
        )
        
        async with aiohttp.ClientSession(connector=connector) as session:
            async def _make_request_with_retries(p):
                for attempt in range(self.max_retries):
                    try:
                        async with session.post(
                            f"{self.server_url}{endpoint}",
                            json=p,
                            timeout=aiohttp.ClientTimeout(total=self.timeout),
                        ) as resp:
                            resp.raise_for_status()
                            data = await resp.json()
                            if isinstance(data, dict):
                                data = [data]
                            return data
                    except Exception as e:
                        if attempt == self.max_retries - 1:
                            raise RuntimeError(f"Request failed after {self.max_retries} attempts: {e}")
                        await asyncio.sleep(1)
            
            # Limit concurrent requests using a semaphore
            semaphore = asyncio.Semaphore(self.max_concurrent_requests)
            
            async def limited_request(p):
                async with semaphore:
                    return await _make_request_with_retries(p)
            
            tasks = [limited_request(p) for p in payload]
            
            if use_tqdm:
                from tqdm import tqdm
                
                # Create a wrapper to update progress bar
                completed_count = [0]
                pbar = tqdm(total=len(tasks), desc="Processing")
                
                async def tracked_task(task):
                    result = await task
                    completed_count[0] += 1
                    pbar.update(1)
                    return result
                
                results = await asyncio.gather(*[tracked_task(task) for task in tasks])
                pbar.close()
            else:
                results = await asyncio.gather(*tasks)
        
        # Flatten results
        flattened_results = []
        for result in results:
            flattened_results.extend(result)
        
        return flattened_results

    async def invoke(self, chat_template_prompt, n=1) -> list[str]:
        
        formatted_prompt = self._prep_prompt(chat_template_prompt, add_generation_prompt=True)
                
      
        
        payload = self._build_ancestral_payload([formatted_prompt], n=n)
        
        # Call ancestral method - expects text input
        result = await self._post_with_retries_async(
            "/v1/completions",
            payload,
            use_tqdm=False,
        )
     
        completions = [
            LLMResponse(response=choice["text"])
            for r in result
            for choice in r.get("choices", [])
        ]
        return completions
  
    def _build_ancestral_payload(self, prompts, n: int = 1):
        """Build payload for ancestral method."""
        # Expand prompts by n
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * n)

        return [
            {
                "model": self.model_path,
                "prompt": p,
                "max_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "n": 1,
                "stop": self.stop_tokens,
            }
            for p in expanded_prompts
        ]


    def _get(self, endpoint):
        request = requests.get(f"{self.server_url}{endpoint}", timeout=self.timeout)
        request.raise_for_status()
        return request.json()
    

    def __str__(self):
        return f"RemoteVLLM(model_path={self.model_path}, server_url={self.server_url})"


class LLMMap(MetaMap):
    """Applies a vLLM model's ancestral method to each element of the input stream asynchronously."""

    def __init__(
        self,
        vllm_client:RemoteLLMClient,
        name: Union[str, None] = None,
        n=1,
        input_key: str = "llm_call_input",
        output_key: str = "llm_call_output",
        inflight_batch: int = DEFAULT_INFLIGHT_BATCH,
        assertions: List[Assertion] = None,
    ):
        """Creates a vLLM map process using the ancestral method.

        Args:
            vllm_client (RemoteVLLM): RemoteVLLM client instance.
            n (int): Number of completions to generate per input. Defaults to 1.
            name (Union[str, None]): Process name. Defaults to None.
            many (bool): If true, expects method to return a list and flattens results. Defaults to False.
            inflight_batch (int): Maximum number of concurrent requests. Defaults to DEFAULT_INFLIGHT_BATCH.
        """
        # Store instance variables first
        self.vllm_client = vllm_client
        self.n = n
        self.inflight_batch = inflight_batch
        self.input_key = input_key
        self.output_key = output_key
        self.assertions = assertions
        if name is None:
            name = f"VLLMMap:{vllm_client.model_path}:ancestral"

        # Initialize parent with the async function
        super().__init__(func=self._vllm_process_impl, name=name, many=False)

    async def _vllm_process_impl(
        self, data: Any
    ) -> Any:
        """Async implementation that calls the ancestral method on RemoteVLLM.
        
        Args:
            data: Chat template format (list of message dicts, e.g., [{"role": "user", "content": "..."}])
            vllm_client: RemoteVLLM client instance
        """
 
        completions =await self.vllm_client.invoke(
            data[self.input_key], n=self.n)

        if self.assertions is not None:
            for assertion in self.assertions:
                
                all_checks =[ assertion.check(completion) for completion in completions ]
                
                if not all(all_checks):
                    # have to implement some level of retry here. 
                    raise ValueError(f"Assertion {assertion} failed for completions: {completions}")
                
        if self.output_key in data:
            logging.warning(f"output key {self.output_key} already in input data, will be overwritten")
            
            
        return { **data, self.output_key: completions }



class LLMResponseGenerator(WorkFlow):
    """Workflow that generates responses to questions using VLLM Map."""

    def __init__(self, vllm_client: RemoteLLMClient, prompt_key: str = "prompt", prompt_template: None = None, n: int = 1):
        """Initialize the workflow with a VLLM client.
        
        Args:
            vllm_client: RemoteVLLM client instance
        """
        self.vllm_client = vllm_client
        self.prompt_key = prompt_key
        self.prompt_template = prompt_template
        self.n = n
        super().__init__()
        
        self.parse_prompt_and_generate_responses = Map(
            func=lambda x: {
                "input_prompt_chat_template": self.prompt_template(x[self.prompt_key]) if self.prompt_template is not None else x[self.prompt_key], 
                "input_prompt": x[self.prompt_key]
            }
                 ) | LLMMap(
            vllm_client=self.vllm_client,
            input_key="input_prompt_chat_template",
            output_key="responses",
            n=self.n,
        )
        
        self.parse_responses = Map(func=lambda x: {
            "responses": [ r.output for r in x["responses"]],
            self.prompt_key: x["input_prompt"]
        })

    def forward(self, input_questions):
        """Generate responses for the input questions.
        
        Args:
            input_questions: List of chat template format questions 
                           (e.g., [[{"role":"user","content":"..."}], ...])
            
        Returns:
            Process result that can be run to get responses
        """
        # Create an iterable from the input questions (already in chat template format)
        a = ListInput(input_questions)
        
        outputs_responses = self.parse_responses(self.parse_prompt_and_generate_responses(a))
        
        return outputs_responses


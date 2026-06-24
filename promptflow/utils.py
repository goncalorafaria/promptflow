import json
import logging
from typing import Any, Dict, Optional


LOGGER = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> Optional[Dict[str, Any]]:
    json_end = response.rfind("}") + 1
    if json_end == 0:
        return None

    json_start = response.find("{")
    while json_start != -1 and json_start < json_end:
        json_str = response[json_start:json_end]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                cleaned_json = json_str.strip().encode("utf-8").decode("utf-8-sig")
                return json.loads(cleaned_json)
            except json.JSONDecodeError:
                try:
                    fixed_braces = json_str.replace("{{", "{").replace("}}", "}")
                    return json.loads(fixed_braces)
                except json.JSONDecodeError:
                    json_start = response.find("{", json_start + 1)
                    continue

    LOGGER.warning("Could not decode JSON from response: %r", response)
    return None


def parse_thinking_tokens_qwen(response: str):
    if "<think>" not in response:
        return "", response

    if "</think>" not in response:
        return response, "No response provided."

    pieces = response.split("</think>", 1)
    reasoning = pieces[0].strip().replace("<think>", "")
    output = pieces[1].strip("\n")

    return reasoning, output

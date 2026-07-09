"""Small parsing helpers used by promptflow model assertions.

These live in promptflow so the package does not depend on the separate
``rubrics`` project just to parse model responses.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_thinking_tokens_qwen(text: str | None) -> tuple[str, str]:
    """Split Qwen-style thinking output into ``(reasoning, final_content)``."""

    if not text:
        return "", ""

    value = str(text)
    matches = list(
        re.finditer(
            r"<think>(.*?)</think>",
            value,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    if matches:
        match = matches[-1]
        reasoning = match.group(1).strip()
        content = (value[: match.start()] + value[match.end() :]).strip()
        return reasoning, content

    close_tag = re.search(r"</think>", value, flags=re.IGNORECASE)
    if close_tag:
        return value[: close_tag.start()].strip(), value[close_tag.end() :].strip()

    return "", value.strip()


def _json_balanced_end(text: str, start: int) -> int | None:
    pairs = {"{": "}", "[": "]"}
    opener = text[start]
    closer = pairs.get(opener)
    if closer is None:
        return None

    stack = [closer]
    in_string = False
    escape = False
    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack:
                return index + 1
    return None


def extract_json_from_response(response: Any) -> Any:
    """Return the first parsed JSON object/array found in a model response."""

    if isinstance(response, (dict, list)):
        return response
    if response is None:
        return None

    text = str(response).strip()
    if not text:
        return None

    fenced = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(text)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    for index, char in enumerate(text):
        if char not in "{[":
            continue
        end = _json_balanced_end(text, index)
        if end is None:
            continue
        try:
            return json.loads(text[index:end])
        except json.JSONDecodeError:
            continue

    return None

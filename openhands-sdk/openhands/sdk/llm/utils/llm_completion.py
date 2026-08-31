"""validated_litellm_completion — litellm call + streaming concat + validation.

This module is intentionally independent of the OpenHands agent scaffold: it
only imports from litellm, the stdlib, and the two utility modules that are
equally scaffold-free (tool_call_validation, openhands.sdk.llm.exceptions).

The entry point is ``validated_litellm_completion``.  It wraps a single
litellm completion call with:

1. Streaming concatenation (when ``enable_streaming`` is True).
2. Malformed-pattern detection  (``OH_MALFORM_PATTERNS``).
3. Tool-call parameter + parallel-group validation (``OH_VALIDATE_TOOLCALL_PARAMS``).
4. Unknown-tool detection (``OH_VALIDATE_TOOLCALL_PARAMS``).

All three validation checks raise ``LLMResponseValidationError`` on failure.
The function retries that specific error up to ``OH_VALIDATION_RETRY`` times
(default 5).  On the final attempt it still returns the response rather than
raising, but annotates the first choice's message with the last validation
error so downstream consumers can surface it.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

import litellm
from litellm import ChatCompletionToolParam, CustomStreamWrapper
from litellm import completion as litellm_completion
from litellm.types.utils import ModelResponse

from openhands.sdk.llm.exceptions import LLMResponseValidationError
from openhands.sdk.llm.utils.tool_call_validation import find_invalid_tool_call
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# Number of times to retry on a validation error (tunable via env).
_DEFAULT_VALIDATION_RETRIES = 5
_ENV_VALIDATION_RETRY = "OH_VALIDATION_RETRY"


def _max_validation_retries() -> int:
    try:
        return int(os.environ.get(_ENV_VALIDATION_RETRY, _DEFAULT_VALIDATION_RETRIES))
    except (ValueError, TypeError):
        return _DEFAULT_VALIDATION_RETRIES


def _check_malformed(response: ModelResponse) -> LLMResponseValidationError | None:
    """Return an error if OH_MALFORM_PATTERNS matches the serialised response."""
    patterns_env = os.environ.get("OH_MALFORM_PATTERNS", "")
    if not patterns_env:
        return None
    patterns = [p.strip() for p in patterns_env.split(";") if p.strip()]
    if not patterns:
        return None
    try:
        response_json = json.dumps(response.model_dump())
        for pattern in patterns:
            if pattern in response_json:
                logger.warning(
                    f"Malformed pattern detected: '{pattern}' found in response"
                )
                return LLMResponseValidationError(
                    "malformed_pattern",
                    f"pattern {pattern!r} found in response",
                )
    except Exception as exc:
        logger.warning(f"Error checking for malformed response: {exc}")
    return None


def _check_invalid_params(
    response: ModelResponse,
) -> LLMResponseValidationError | None:
    """Return an error if OH_VALIDATE_TOOLCALL_PARAMS is set and a call is bad."""
    flag = os.environ.get("OH_VALIDATE_TOOLCALL_PARAMS", "")
    if flag.strip().lower() not in ("1", "true", "yes", "on"):
        return None
    try:
        result = find_invalid_tool_call(response.model_dump())
        if result is not None:
            tag, detail = result
            logger.warning(
                f"Tool call validation failed ({tag}): {detail!r}"
            )
            return LLMResponseValidationError(tag, detail)
    except Exception as exc:
        logger.warning(f"Error validating tool call parameters: {exc}")
    return None


def _check_unknown_tool(
    response: ModelResponse,
    tools: list[ChatCompletionToolParam] | None,
) -> LLMResponseValidationError | None:
    """Return an error if a tool call names a tool not in the offered list.

    Uses ``response.model_dump()`` so tool calls are plain dicts rather than
    typed ``ChatCompletionMessageToolCall`` objects, which do not support
    ``.get()``.
    """
    flag = os.environ.get("OH_VALIDATE_TOOLCALL_PARAMS", "")
    if flag.strip().lower() not in ("1", "true", "yes", "on"):
        return None
    if not tools:
        return None
    known_names = {
        t["function"]["name"]
        for t in tools
        if t.get("type") == "function" and isinstance(t.get("function"), dict)
    }
    try:
        response_dict = response.model_dump()
        for choice in response_dict.get("choices") or []:
            message = choice.get("message") or {}
            for tc in message.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name")
                if name and name not in known_names:
                    logger.warning(
                        f"Detected tool call to unknown tool '{name}'"
                    )
                    return LLMResponseValidationError(
                        "unknown_tool",
                        f"tool call to unknown tool '{name}'",
                    )
    except Exception as exc:
        logger.warning(f"Error checking unknown tool call: {exc}")
    return None


def _validate(
    response: ModelResponse,
    tools: list[ChatCompletionToolParam] | None,
) -> LLMResponseValidationError | None:
    """Run all three validation checks and return the first error found."""
    err = _check_malformed(response)
    if err is not None:
        return err
    err = _check_invalid_params(response)
    if err is not None:
        return err
    return _check_unknown_tool(response, tools)


def _annotate_response(
    response: ModelResponse, error: LLMResponseValidationError
) -> None:
    """Stamp the last validation error onto the first choice's message.

    Writes into the typed ``Message`` object directly (not via model_dump)
    so the annotation survives into the response returned to the caller.
    ``response.choices`` contains ``Choices`` objects (not ``StreamingChoices``)
    for completed completions; only ``Choices`` has a ``message`` attribute.
    """
    try:
        choices = response.choices
        if not choices:
            return
        first = choices[0]
        # StreamingChoices has no .message; skip silently if that's what we got.
        message = getattr(first, "message", None)
        if message is None:
            return
        psf = getattr(message, "provider_specific_fields", None)
        if psf is None:
            psf = {}
            message.provider_specific_fields = psf  # type: ignore[attr-defined]
        if isinstance(psf, dict):
            psf["annotation"] = str(error)
    except Exception as exc:
        logger.warning(f"Failed to annotate response with validation error: {exc}")


def validated_litellm_completion(
    *,
    model: str,
    api_key: str | None,
    api_base: str | None,
    api_version: str | None,
    timeout: int | None,
    drop_params: bool,
    seed: int | None,
    messages: list[dict[str, Any]],
    enable_streaming: bool = False,
    on_token: Any | None = None,
    tools: list[ChatCompletionToolParam] | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """Call litellm, concatenate streaming chunks, validate, and retry on errors.

    All litellm-level parameters are passed through explicitly so the function
    signature is self-documenting and independent of the ``LLM`` pydantic model.

    On a ``LLMResponseValidationError``, the call is retried up to
    ``OH_VALIDATION_RETRY`` times (default 5) with no backoff — these errors
    are structural, not transient, so jitter would only add latency.  If the
    final attempt also fails, the response is returned with the error message
    attached to ``choices[0].message.provider_specific_fields["annotation"]``.

    All other exceptions propagate immediately to the caller (which handles its
    own backoff/retry for network, rate-limit, and timeout errors).
    """
    max_retries = _max_validation_retries()

    for attempt in range(max(max_retries, 1)):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, module="httpx.*"
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*content=.*upload.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=r"There is no current event loop",
                category=DeprecationWarning,
            )
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="Accessing the 'model_fields' attribute.*",
            )

            ret = litellm_completion(
                model=model,
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                timeout=timeout,
                drop_params=drop_params,
                seed=seed,
                messages=messages,
                # Forward tools explicitly so litellm passes them to the
                # provider.  tools is kept separate from **kwargs so the
                # caller (validated_litellm_completion's signature) can
                # receive it without a duplicate-keyword collision.
                **({"tools": tools} if tools is not None else {}),
                **kwargs,
            )

            if enable_streaming and on_token is not None:
                assert isinstance(ret, CustomStreamWrapper)
                chunks = []
                for chunk in ret:
                    on_token(chunk)
                    chunks.append(chunk)
                ret = litellm.stream_chunk_builder(chunks, messages=messages)

            assert isinstance(ret, ModelResponse), (
                f"Expected ModelResponse, got {type(ret)}"
            )

        err = _validate(ret, tools)
        if err is None:
            return ret

        remaining = max_retries - attempt - 1
        if remaining > 0:
            logger.warning(
                f"Validation error on attempt {attempt + 1}/{max_retries}"
                f" ({err}), retrying..."
            )
        else:
            logger.warning(
                f"Validation error persists after {max_retries} attempts"
                f" ({err}), proceeding with annotated response"
            )
            _annotate_response(ret, err)
            return ret

    # Only reached when max_retries <= 0 — no attempt was made; call once clean.
    return validated_litellm_completion(
        model=model,
        api_key=api_key,
        api_base=api_base,
        api_version=api_version,
        timeout=timeout,
        drop_params=drop_params,
        seed=seed,
        messages=messages,
        enable_streaming=enable_streaming,
        on_token=on_token,
        tools=tools,
        **kwargs,
    )

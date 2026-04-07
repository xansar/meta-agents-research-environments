# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import os
import time
from typing import Any

from litellm import completion
from litellm.exceptions import APIError, AuthenticationError
from litellm.types.utils import Choices, ModelResponse
from pydantic import BaseModel

from are.simulation.agents.llm.llm_engine import LLMEngine, LLMEngineException

# TODO: Litellm should be agnostic to the agent or model. Remove this dependency.
from are.simulation.agents.llm.types import MessageRole
from are.simulation.agents.multimodal import Attachment

logger = logging.getLogger(__name__)

# TODO: Instead of tool, llama has ipython support. We should use that instead of user role.
role_conversions = {"tool-response": "user", "tool-call": "assistant"}

AZURE_OPENAI_DEFAULT_SCOPE = "https://cognitiveservices.azure.com/.default"
AZURE_TRAPI_SCOPE = "api://trapi/.default"
AZURE_API_KEY_ENV_VARS = ("AZURE_OPENAI_API_KEY", "AZURE_API_KEY")


class LiteLLMModelConfig(BaseModel):
    model_name: str
    provider: str
    endpoint: str | None = None
    api_key: str | None = None
    api_version: str | None = None
    azure_ad_token: str | None = None
    reasoning_effort: str | None = None


class LiteLLMEngine(LLMEngine):
    """
    A class that extends the LLMEngine to provide a specific implementation for the Litellm model.
    Attributes:
        model_config (ModelConfig): The configuration for the model.
    """

    def __init__(self, model_config: LiteLLMModelConfig):
        super().__init__(model_config.model_name)

        self.model_config = model_config
        self._azure_ad_token_providers: dict[str, Any] = {}

        self.mock_response = None
        if model_config.provider == "mock":
            self.mock_response = """Thought: Good choice, this is a mock, so I can't do anything. Let's return the result.
Action:
{
  "action": "_mock",
  "action_input": "Mock result"
}<end_action>
"""

    def _has_azure_api_key(self) -> bool:
        if self.model_config.api_key:
            return True
        return any(os.getenv(env_var) for env_var in AZURE_API_KEY_ENV_VARS)

    def _is_trapi_endpoint(self, endpoint: str | None) -> bool:
        return endpoint is not None and "trapi.research.microsoft.com" in endpoint

    def _resolve_azure_api_version(self) -> str | None:
        return self.model_config.api_version or os.getenv("AZURE_API_VERSION")

    def _build_azure_ad_token_provider(self, endpoint: str | None):
        cached_provider = self._azure_ad_token_providers.get(endpoint)
        if cached_provider is not None:
            return cached_provider

        try:
            from azure.identity import (
                AzureCliCredential,
                ChainedTokenCredential,
                ManagedIdentityCredential,
                get_bearer_token_provider,
            )
        except ImportError as e:
            raise LLMEngineException(
                "Azure AD auth requires the optional dependency `azure-identity`."
            ) from e

        if self._is_trapi_endpoint(endpoint):
            credential = ChainedTokenCredential(
                AzureCliCredential(),
                ManagedIdentityCredential(),
            )
            scope = AZURE_TRAPI_SCOPE
        else:
            credential = AzureCliCredential()
            scope = AZURE_OPENAI_DEFAULT_SCOPE

        token_provider = get_bearer_token_provider(credential, scope)
        self._azure_ad_token_providers[endpoint] = token_provider
        return token_provider

    def _convert_message_to_litellm_format(
        self, message: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert a message to LiteLLM format, handling both text and multimodal content."""
        role = MessageRole(message["role"]).value
        role = role_conversions.get(role, role)

        # Handle attachments if present
        attachments: list[Attachment] | None = message.get("attachments")
        content = message.get("content", "")

        if attachments and len(attachments) > 0:
            # Create multimodal content with both text and images
            content_list = []

            # Add text content if present
            if content:
                content_list.append({"type": "text", "text": content})

            # Add image attachments
            for attachment in attachments:
                if attachment.mime.startswith("image/"):
                    content_list.append(attachment.to_openai_json())
                else:
                    logger.warning(
                        f"Unsupported attachment mime type: {attachment.mime}"
                    )

            return {"role": role, "content": content_list}
        else:
            # Text-only message
            return {"role": role, "content": content}

    def _extract_reasoning_tokens(self, usage: Any) -> int:
        completion_tokens_details = None

        if hasattr(usage, "get"):
            completion_tokens_details = usage.get("completion_tokens_details")
        else:
            completion_tokens_details = getattr(usage, "completion_tokens_details", None)

        if completion_tokens_details is None:
            return 0

        if hasattr(completion_tokens_details, "get"):
            return completion_tokens_details.get("reasoning_tokens", 0) or 0

        return getattr(completion_tokens_details, "reasoning_tokens", 0) or 0

    def _build_metadata(
        self, response: ModelResponse, completion_duration: float
    ) -> dict[str, int | float]:
        usage = response.get("usage")
        metadata: dict[str, int | float] = {
            "completion_duration": completion_duration,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        }

        if usage is None:
            return metadata

        metadata["prompt_tokens"] = usage.get("prompt_tokens", 0) or 0
        metadata["completion_tokens"] = usage.get("completion_tokens", 0) or 0
        metadata["total_tokens"] = usage.get("total_tokens", 0) or 0
        metadata["reasoning_tokens"] = self._extract_reasoning_tokens(usage)
        return metadata

    def _apply_reasoning_kwargs(
        self, completion_kwargs: dict[str, Any], kwargs: dict[str, Any]
    ) -> None:
        if "reasoning_effort" in kwargs:
            reasoning_effort = kwargs["reasoning_effort"]
        else:
            reasoning_effort = self.model_config.reasoning_effort
        if reasoning_effort is None:
            return

        completion_kwargs["reasoning_effort"] = reasoning_effort

        if self.model_config.provider != "azure":
            return

        allowed_openai_params = list(kwargs.get("allowed_openai_params", []))
        if "reasoning_effort" not in allowed_openai_params:
            allowed_openai_params.append("reasoning_effort")
        completion_kwargs["allowed_openai_params"] = allowed_openai_params

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        stop_sequences=[],
        **kwargs,
    ) -> tuple[str, dict | None]:
        try:
            # Convert messages to LiteLLM format with multimodal support
            converted_messages = []
            for message in messages:
                converted_message = self._convert_message_to_litellm_format(message)
                converted_messages.append(converted_message)

            provider = (
                self.model_config.provider
                if self.model_config.provider != "local"
                else None
            )

            completion_kwargs: dict[str, Any] = {
                "model": self.model_config.model_name,
                "custom_llm_provider": provider,
                "messages": converted_messages,
                "api_base": self.model_config.endpoint,
                "api_key": self.model_config.api_key,
                "mock_response": self.mock_response,
            }

            if self.model_config.provider == "azure":
                api_version = self._resolve_azure_api_version()
                if api_version is not None:
                    completion_kwargs["api_version"] = api_version

                if not self._has_azure_api_key():
                    azure_ad_token = self.model_config.azure_ad_token or os.getenv(
                        "AZURE_AD_TOKEN"
                    )
                    if azure_ad_token is not None:
                        completion_kwargs["azure_ad_token"] = azure_ad_token
                    else:
                        endpoint = self.model_config.endpoint or os.getenv(
                            "AZURE_API_BASE"
                        )
                        completion_kwargs["azure_ad_token_provider"] = (
                            self._build_azure_ad_token_provider(endpoint)
                        )

            self._apply_reasoning_kwargs(completion_kwargs, kwargs)

            start_time = time.perf_counter()
            response = completion(**completion_kwargs)
            completion_duration = time.perf_counter() - start_time

            assert type(response) is ModelResponse
            assert len(response.choices) >= 1
            assert type(response.choices[0]) is Choices

            res = response.choices[0].message.content
            assert res is not None

            # res = res.replace("False", "false").replace("True", "true")
            for stop_token in stop_sequences:
                res = res.split(stop_token)[0]

            return res, self._build_metadata(response, completion_duration)
        except LLMEngineException:
            raise
        except (AuthenticationError, APIError) as e:
            raise LLMEngineException("Auth error in litellm.") from e

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from unittest.mock import patch

from litellm.types.utils import Choices, ModelResponse, Usage

from are.simulation.agents.llm.litellm.litellm_engine import (
    LiteLLMEngine,
    LiteLLMModelConfig,
)


def _build_response(
    content: str,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> ModelResponse:
    usage = None
    if prompt_tokens or completion_tokens or total_tokens or reasoning_tokens:
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            completion_tokens_details={"reasoning_tokens": reasoning_tokens},
        )

    return ModelResponse(
        choices=[Choices(message={"role": "assistant", "content": content})],
        usage=usage,
    )


def test_chat_completion_uses_azure_ad_token_provider_without_api_key():
    engine = LiteLLMEngine(
        LiteLLMModelConfig(
            model_name="test-deployment",
            provider="azure",
            endpoint="https://example.openai.azure.com/",
            api_version="2024-12-01-preview",
        )
    )

    with (
        patch.object(
            engine,
            "_build_azure_ad_token_provider",
            return_value="token-provider",
        ) as mock_token_provider,
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.completion",
            return_value=_build_response("ok"),
        ) as mock_completion,
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.time.perf_counter",
            side_effect=[10.0, 10.25],
        ),
        patch.dict("os.environ", {}, clear=True),
    ):
        response, metadata = engine.chat_completion(
            [{"role": "user", "content": "hello"}]
        )

    assert response == "ok"
    assert metadata == {
        "completion_duration": 0.25,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }
    mock_token_provider.assert_called_once_with("https://example.openai.azure.com/")

    completion_kwargs = mock_completion.call_args.kwargs
    assert completion_kwargs["model"] == "test-deployment"
    assert completion_kwargs["api_base"] == "https://example.openai.azure.com/"
    assert completion_kwargs["api_version"] == "2024-12-01-preview"
    assert completion_kwargs["azure_ad_token_provider"] == "token-provider"
    assert "api_key" in completion_kwargs


def test_chat_completion_keeps_existing_azure_api_key_path():
    engine = LiteLLMEngine(
        LiteLLMModelConfig(
            model_name="test-deployment",
            provider="azure",
            endpoint="https://example.openai.azure.com/",
            api_key="secret",
        )
    )

    with (
        patch.object(engine, "_build_azure_ad_token_provider") as mock_token_provider,
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.completion",
            return_value=_build_response("ok"),
        ) as mock_completion,
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.time.perf_counter",
            side_effect=[20.0, 20.5],
        ),
    ):
        response, metadata = engine.chat_completion(
            [{"role": "user", "content": "hello"}]
        )

    assert response == "ok"
    assert metadata == {
        "completion_duration": 0.5,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }
    mock_token_provider.assert_not_called()

    completion_kwargs = mock_completion.call_args.kwargs
    assert completion_kwargs["api_key"] == "secret"
    assert "azure_ad_token_provider" not in completion_kwargs
    assert "azure_ad_token" not in completion_kwargs


def test_chat_completion_extracts_usage_metadata():
    engine = LiteLLMEngine(
        LiteLLMModelConfig(
            model_name="test-model",
            provider="openai",
            endpoint="https://example.com/v1",
            api_key="secret",
        )
    )

    with (
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.completion",
            return_value=_build_response(
                "ok",
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
                reasoning_tokens=3,
            ),
        ),
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.time.perf_counter",
            side_effect=[30.0, 30.125],
        ),
    ):
        response, metadata = engine.chat_completion(
            [{"role": "user", "content": "hello"}]
        )

    assert response == "ok"
    assert metadata == {
        "completion_duration": 0.125,
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "reasoning_tokens": 3,
    }


def test_chat_completion_forwards_reasoning_effort_for_azure():
    engine = LiteLLMEngine(
        LiteLLMModelConfig(
            model_name="test-deployment",
            provider="azure",
            endpoint="https://example.openai.azure.com/",
            api_version="2024-12-01-preview",
        )
    )

    with (
        patch.object(
            engine,
            "_build_azure_ad_token_provider",
            return_value="token-provider",
        ),
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.completion",
            return_value=_build_response(
                "ok",
                prompt_tokens=10,
                completion_tokens=15,
                total_tokens=25,
                reasoning_tokens=9,
            ),
        ) as mock_completion,
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.time.perf_counter",
            side_effect=[40.0, 40.125],
        ),
        patch.dict("os.environ", {}, clear=True),
    ):
        response, metadata = engine.chat_completion(
            [{"role": "user", "content": "hello"}],
            reasoning_effort="xhigh",
        )

    assert response == "ok"
    assert metadata["reasoning_tokens"] == 9

    completion_kwargs = mock_completion.call_args.kwargs
    assert completion_kwargs["reasoning_effort"] == "xhigh"
    assert completion_kwargs["allowed_openai_params"] == ["reasoning_effort"]


def test_chat_completion_merges_existing_allowed_openai_params():
    engine = LiteLLMEngine(
        LiteLLMModelConfig(
            model_name="test-deployment",
            provider="azure",
            endpoint="https://example.openai.azure.com/",
            api_version="2024-12-01-preview",
        )
    )

    with (
        patch.object(
            engine,
            "_build_azure_ad_token_provider",
            return_value="token-provider",
        ),
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.completion",
            return_value=_build_response("ok"),
        ) as mock_completion,
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.time.perf_counter",
            side_effect=[50.0, 50.25],
        ),
        patch.dict("os.environ", {}, clear=True),
    ):
        engine.chat_completion(
            [{"role": "user", "content": "hello"}],
            reasoning_effort="low",
            allowed_openai_params=["response_format"],
        )

    completion_kwargs = mock_completion.call_args.kwargs
    assert completion_kwargs["allowed_openai_params"] == [
        "response_format",
        "reasoning_effort",
    ]


def test_chat_completion_uses_model_config_reasoning_effort_by_default():
    engine = LiteLLMEngine(
        LiteLLMModelConfig(
            model_name="test-deployment",
            provider="azure",
            endpoint="https://example.openai.azure.com/",
            api_version="2024-12-01-preview",
            reasoning_effort="high",
        )
    )

    with (
        patch.object(
            engine,
            "_build_azure_ad_token_provider",
            return_value="token-provider",
        ),
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.completion",
            return_value=_build_response("ok"),
        ) as mock_completion,
        patch(
            "are.simulation.agents.llm.litellm.litellm_engine.time.perf_counter",
            side_effect=[60.0, 60.125],
        ),
        patch.dict("os.environ", {}, clear=True),
    ):
        engine.chat_completion([{"role": "user", "content": "hello"}])

    completion_kwargs = mock_completion.call_args.kwargs
    assert completion_kwargs["reasoning_effort"] == "high"
    assert completion_kwargs["allowed_openai_params"] == ["reasoning_effort"]

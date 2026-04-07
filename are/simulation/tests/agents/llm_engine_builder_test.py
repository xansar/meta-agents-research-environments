# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from are.simulation.agents.llm.llm_engine_builder import LLMEngineBuilder
from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine


def test_create_engine_preserves_reasoning_effort_for_litellm_models():
    builder = LLMEngineBuilder()
    engine = builder.create_engine(
        LLMEngineConfig(
            model_name="gpt-5.4",
            provider="azure",
            endpoint="https://example.openai.azure.com/",
            reasoning_effort="xhigh",
        )
    )

    assert isinstance(engine, LiteLLMEngine)
    assert engine.model_config.reasoning_effort == "xhigh"

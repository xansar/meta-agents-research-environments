# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from are.simulation.agents.agent_config_builder import (
    AgentConfigBuilder,
    AppAgentConfigBuilder,
)
from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactAgentConfig,
    RunnableARESimulationAgentConfig,
)
from are.simulation.agents.default_agent.prompts import (
    DEFAULT_ARE_SIMULATION_APP_AGENT_REACT_JSON_SYSTEM_PROMPT,
    DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT,
)


@pytest.fixture
def agent_config_builder() -> AgentConfigBuilder:
    return AgentConfigBuilder()


def test_build_default(
    agent_config_builder: AgentConfigBuilder,
):
    agent_config = agent_config_builder.build("default")
    assert isinstance(agent_config, RunnableARESimulationAgentConfig)
    assert isinstance(agent_config, ARESimulationReactAgentConfig)
    assert (
        agent_config.base_agent_config.system_prompt
        == str(DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT)
    )


def test_build_invalid_agent(agent_config_builder: AgentConfigBuilder):
    with pytest.raises(ValueError, match="Agent invalid_agent not found"):
        agent_config_builder.build("invalid_agent")


def test_build_default_empty_value_prompt_is_unchanged(
    agent_config_builder: AgentConfigBuilder,
):
    default_prompt = agent_config_builder.build("default").base_agent_config.system_prompt
    empty_prompt = agent_config_builder.build(
        "default", value_prompt=""
    ).base_agent_config.system_prompt

    assert empty_prompt == default_prompt


def test_build_default_with_value_prompt(
    agent_config_builder: AgentConfigBuilder,
):
    value_prompt = "Prefer privacy-preserving actions when multiple approaches are valid."
    agent_config = agent_config_builder.build("default", value_prompt=value_prompt)
    system_prompt = agent_config.base_agent_config.system_prompt

    assert system_prompt.startswith("<value_preference>")
    assert "<general_instructions>" in system_prompt
    assert system_prompt.index("<value_preference>") < system_prompt.index(
        "<general_instructions>"
    )
    assert value_prompt in system_prompt


def test_build_default_with_message_source_awareness(
    agent_config_builder: AgentConfigBuilder,
):
    agent_config = agent_config_builder.build(
        "default",
        enable_message_source_awareness=True,
    )
    system_prompt = agent_config.base_agent_config.system_prompt

    assert agent_config.base_agent_config.enable_message_source_awareness is True
    assert system_prompt.startswith("<agent_identity>")
    assert "You are the main agent responsible for completing the user's task." in (
        system_prompt
    )


def test_build_default_app_agent_empty_value_prompt_is_unchanged():
    default_prompt = str(DEFAULT_ARE_SIMULATION_APP_AGENT_REACT_JSON_SYSTEM_PROMPT)
    agent_config = AppAgentConfigBuilder().build("default_app_agent", value_prompt="")

    assert agent_config.system_prompt == default_prompt


def test_build_default_app_agent_with_value_prompt():
    value_prompt = "Keep app-agent responses concise and cost-aware."
    agent_config = AppAgentConfigBuilder().build(
        "default_app_agent",
        value_prompt=value_prompt,
    )
    system_prompt = agent_config.system_prompt

    assert system_prompt.startswith("<value_preference>")
    assert "<general_instructions>" in system_prompt
    assert system_prompt.index("<value_preference>") < system_prompt.index(
        "<general_instructions>"
    )
    assert value_prompt in system_prompt


def test_build_default_app_agent_with_message_source_awareness():
    agent_config = AppAgentConfigBuilder().build(
        "default_app_agent",
        enable_message_source_awareness=True,
    )
    system_prompt = agent_config.system_prompt

    assert agent_config.enable_message_source_awareness is True
    assert (
        system_prompt
        == str(DEFAULT_ARE_SIMULATION_APP_AGENT_REACT_JSON_SYSTEM_PROMPT)
    )

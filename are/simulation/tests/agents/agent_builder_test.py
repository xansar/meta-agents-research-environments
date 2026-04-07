# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from unittest.mock import Mock

import pytest

from are.simulation.agents.agent_builder import AgentBuilder, AppAgentBuilder
from are.simulation.agents.agent_config_builder import AgentConfigBuilder, AppAgentConfigBuilder
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
from are.simulation.agents.default_agent.are_simulation_main import ARESimulationAgent
from are.simulation.apps import App
from are.simulation.environment import Environment


@pytest.fixture
def agent_config_builder() -> AgentConfigBuilder:
    return AgentConfigBuilder()


@pytest.fixture
def agent_builder() -> AgentBuilder:
    return AgentBuilder()


@pytest.fixture
def app_agent_config_builder() -> AppAgentConfigBuilder:
    return AppAgentConfigBuilder()


@pytest.fixture
def app_agent_builder() -> AppAgentBuilder:
    return AppAgentBuilder()


@pytest.fixture
def env_mock():
    mock = Mock(spec=Environment)
    mock.time_manager = Mock()
    mock.append_to_world_logs = Mock()
    return mock


def test_list_agents(agent_builder):
    agents = agent_builder.list_agents()
    assert agents == ["default"]


def test_build_default(
    agent_config_builder: AgentConfigBuilder,
    agent_builder: AgentBuilder,
    env_mock,
):
    agent_config = agent_config_builder.build("default")
    agent_config.get_base_agent_config().llm_engine_config.provider = "huggingface"
    agent = agent_builder.build(agent_config, env=env_mock)
    assert isinstance(agent, RunnableARESimulationAgent)
    assert isinstance(agent, ARESimulationAgent)


def test_build_default_with_message_source_awareness(
    agent_config_builder: AgentConfigBuilder,
    agent_builder: AgentBuilder,
    env_mock,
):
    agent_config = agent_config_builder.build(
        "default",
        enable_message_source_awareness=True,
    )
    agent_config.get_base_agent_config().llm_engine_config.provider = "huggingface"
    agent = agent_builder.build(agent_config, env=env_mock)

    assert isinstance(agent, ARESimulationAgent)
    assert agent.enable_message_source_awareness is True


def test_build_default_app_agent_with_message_source_awareness_injects_app_identity(
    app_agent_config_builder: AppAgentConfigBuilder,
    app_agent_builder: AppAgentBuilder,
    env_mock,
):
    agent_config = app_agent_config_builder.build(
        "default_app_agent",
        enable_message_source_awareness=True,
    )
    agent_config.llm_engine_config.provider = "huggingface"

    app = App(name="Calendar")
    agent = app_agent_builder.build(agent_config, app=app, env=env_mock)
    system_prompt = agent.app_agent.init_system_prompts["system_prompt"]

    assert agent.enable_message_source_awareness is True
    assert system_prompt.startswith("<agent_identity>")
    assert "You are the agent responsible for operating and managing the user's Calendar app." in (
        system_prompt
    )
    assert "Your tool and data reachability is centered on the Calendar app." in (
        system_prompt
    )


def test_build_invalid_agent(agent_config_builder: AgentConfigBuilder):
    with pytest.raises(ValueError, match="Agent invalid_agent not found"):
        agent_config_builder.build("invalid_agent")

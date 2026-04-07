# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from abc import ABC, abstractmethod
from typing import Any

from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactAgentConfig,
    ARESimulationReactAppAgentConfig,
    ARESimulationReactBaseAgentConfig,
    RunnableARESimulationAgentConfig,
)
from are.simulation.agents.default_agent.prompts import (
    DEFAULT_ARE_SIMULATION_APP_AGENT_REACT_JSON_SYSTEM_PROMPT,
    DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT,
    inject_agent_identity,
    inject_value_preference,
)


class AbstractAgentConfigBuilder(ABC):
    """
    Abstract class for building agent configs.
    """

    @abstractmethod
    def build(
        self,
        agent_name: str,
        value_prompt: str | None = None,
        enable_message_source_awareness: bool = False,
    ) -> Any:
        """
        Method to build a config.
        :param agent_name: Name of the agent that affects the config type.
        :param value_prompt: Optional high-priority value preference injected into the system prompt.
        :param enable_message_source_awareness: Whether to make agent role and message source explicit.
        :returns: An instance of the config.
        """


class AgentConfigBuilder(AbstractAgentConfigBuilder):
    def build(
        self,
        agent_name: str,
        value_prompt: str | None = None,
        enable_message_source_awareness: bool = False,
    ) -> RunnableARESimulationAgentConfig:
        match agent_name:
            case "default":
                return ARESimulationReactAgentConfig(
                    agent_name=agent_name,
                    base_agent_config=ARESimulationReactBaseAgentConfig(
                        system_prompt=inject_value_preference(
                            inject_agent_identity(
                                str(DEFAULT_ARE_SIMULATION_REACT_JSON_SYSTEM_PROMPT),
                                "main_agent"
                                if enable_message_source_awareness
                                else None,
                            ),
                            value_prompt,
                        ),
                        max_iterations=80,
                        enable_message_source_awareness=enable_message_source_awareness,
                    ),
                )

            case _:
                raise ValueError(f"Agent {agent_name} not found")


class AppAgentConfigBuilder(AbstractAgentConfigBuilder):
    def build(
        self,
        agent_name: str,
        value_prompt: str | None = None,
        enable_message_source_awareness: bool = False,
    ) -> ARESimulationReactAppAgentConfig:
        match agent_name:
            case "default_app_agent":
                return ARESimulationReactAppAgentConfig(
                    agent_name=agent_name,
                    system_prompt=inject_value_preference(
                        str(DEFAULT_ARE_SIMULATION_APP_AGENT_REACT_JSON_SYSTEM_PROMPT),
                        value_prompt,
                    ),
                    max_iterations=80,
                    enable_message_source_awareness=enable_message_source_awareness,
                )
            case _:
                raise ValueError(f"Sub agent {agent_name} not found")

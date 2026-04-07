# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from are.simulation.agents.agent_log import BaseAgentLog, FinalAnswerLog
from are.simulation.agents.default_agent.utils.logging_utils import (
    get_default_logger,
    get_parent_logger,
)
from are.simulation.agents.llm.types import MMObservation
from are.simulation.exceptions import FormatError, InvalidActionAgentError
from are.simulation.tools import Tool
from are.simulation.utils.serialization import make_serializable


@dataclass
class AgentAction:
    rationale: str
    action: str | None
    action_type: str | None = None


@dataclass
class ParsedAction:
    action_code: str | None = None
    action_name: str | None = None
    tool_name: str | None = None
    app_name: str | None = None
    arguments: str | dict[str, Any] | None = None
    rationale: str | None = None


class BaseActionExecutor:
    state: dict[str, Any] = {}
    action_token: str = ""
    thought_token: str = ""

    def __init__(self, use_custom_logger: bool = True):
        self.use_custom_logger = use_custom_logger
        self.logger = (
            get_default_logger(__name__)
            if self.use_custom_logger
            else get_parent_logger(__name__)
        )

    def extract_action(self, llm_output: str, split_token: str) -> AgentAction:
        try:
            split = llm_output.split(split_token)
            if len(split) < 2:
                raise IndexError(
                    f"Expected at least 2 parts after splitting by '{split_token}', got {len(split)}"
                )
            # When multiple actions are present, take the first one only
            # This handles cases where the model generates multiple action attempts
            rationale, action = split[0], split[1]
            if len(split) > 2:
                self.logger.warning(
                    f"Multiple actions detected in output. Using only the first action. "
                    f"Output had {len(split) - 1} action blocks. "
                    f"Ignored actions: {split[2:]}"
                )
        except Exception as e:
            self.logger.error(e, exc_info=True)
            raise InvalidActionAgentError(
                f"Error: No '{split_token}' token provided in your output.\nYour output:\n{llm_output}\n. Be sure to include an action, prefaced with '{split_token}'!\nException: {e}"
            )
        return AgentAction(rationale, action)

    def execute_action(
        self,
        action: AgentAction,
        append_agent_log: Callable[[BaseAgentLog], None],
        make_timestamp: Callable[[], float],
        agent_id: str,
    ):
        pass

    def update_tools(self, tools: dict[str, Tool]):
        pass

    def inject_state(self, state: dict[str, Any]):
        pass

    @abstractmethod
    def execute_parsed_action(
        self,
        parsed_action: ParsedAction,
        append_agent_log: Callable[[BaseAgentLog], None],
        make_timestamp: Callable[[], float],
        agent_id: str,
    ):
        pass

    @abstractmethod
    def parse_action(self, action: AgentAction) -> ParsedAction: ...

    def _append_final_answer(
        self,
        observation: Any,
        append_agent_log: Callable[[BaseAgentLog], None],
        make_timestamp: Callable[[], float],
        agent_id: str,
    ):
        content, attachments = self._get_loggable_observation(observation)
        append_agent_log(
            FinalAnswerLog(
                content=content,
                attachments=attachments,
                timestamp=make_timestamp(),
                agent_id=agent_id,
            )
        )

    def _get_loggable_observation(
        self, observation: Any
    ) -> tuple[str, list[Any]]:
        if isinstance(observation, MMObservation):
            return observation.content, observation.attachments
        if isinstance(observation, str):
            return observation, []

        try:
            return json.dumps(make_serializable(observation), sort_keys=True), []
        except (TypeError, ValueError):
            return str(observation), []

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json

import pytest

from are.simulation.agents.agent_log import FinalAnswerLog, ObservationLog
from are.simulation.agents.default_agent.default_tools import FinalAnswerTool
from are.simulation.agents.default_agent.tools.action_executor import ParsedAction
from are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
)
from are.simulation.apps import App
from are.simulation.tool_utils import app_tool


class DummyApp(App):
    @app_tool()
    def add(self, a: int, b: int) -> int:
        """
        Add two numbers
        :param a: first number
        :param b: second number
        :return: sum of a and b
        """
        return a + b

    @app_tool()
    def multiply(self, a: int, b: int) -> int:
        """
        Multiply two numbers
        :param a: first number
        :param b: second number
        :return: product of a and b
        """
        return a * b


def test_multiple_actions_json_uses_first_action():
    json_executor = JsonActionExecutor()

    json_multiple_action = """
    Thought: I will add 2 and 3

    Action:
    ```json
    {
        "tool": "DummyApp__add",
        "args": {
            "a": 2,
            "b": 3
        }
    }
    ```

    Action:
    ```json
    {
        "tool": "DummyApp__add",
        "args": {
            "a": 2,
            "b": 3
        }
    }
    ```
    """

    action = json_executor.extract_action(json_multiple_action, split_token="Action:")

    assert action.action is not None
    assert '"tool": "DummyApp__add"' in action.action


def test_final_answer_logs_structured_output():
    json_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    logs = []
    answer = {
        "removed_count": 1,
        "removed_ids": ["6a572b8b24f445cd851709f559a0f8d6"],
        "saved_new_count": 6,
    }

    json_executor.execute_parsed_action(
        ParsedAction(
            tool_name="final_answer",
            arguments={"answer": answer},
            rationale="Task completed.",
        ),
        logs.append,
        lambda: 123.0,
        "agent-test",
    )

    expected_content = json.dumps(answer, sort_keys=True)

    observation_logs = [log for log in logs if isinstance(log, ObservationLog)]
    final_answer_logs = [log for log in logs if isinstance(log, FinalAnswerLog)]

    assert len(observation_logs) == 1
    assert observation_logs[0].content == expected_content
    assert observation_logs[0].attachments == []

    assert len(final_answer_logs) == 1
    assert final_answer_logs[0].content == expected_content
    assert final_answer_logs[0].attachments == []

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from datetime import datetime, timezone

from are.simulation.agents.default_agent.app_agent import format_app_agent_task
from are.simulation.agents.default_agent.are_simulation_main import (
    format_main_agent_task_from_notifications,
)
from are.simulation.notification_system import Message, MessageType


def test_format_main_agent_task_without_source_awareness():
    notifications = [
        Message(
            message_type=MessageType.USER_MESSAGE,
            message="Find the latest filing.",
            timestamp=datetime.fromtimestamp(0, tz=timezone.utc),
        ),
        Message(
            message_type=MessageType.USER_MESSAGE,
            message="Return the CIK too.",
            timestamp=datetime.fromtimestamp(1, tz=timezone.utc),
        ),
    ]

    task = format_main_agent_task_from_notifications(notifications)

    assert task == "Find the latest filing.\nReturn the CIK too."


def test_format_main_agent_task_with_source_awareness():
    notifications = [
        Message(
            message_type=MessageType.USER_MESSAGE,
            message="Find the latest filing.",
            timestamp=datetime.fromtimestamp(0, tz=timezone.utc),
        )
    ]

    task = format_main_agent_task_from_notifications(
        notifications,
        enable_message_source_awareness=True,
    )

    assert 'source="user"' in task
    assert 'recipient_role="' not in task
    assert "Find the latest filing." in task


def test_format_app_agent_task_without_source_awareness():
    task = format_app_agent_task(
        task="Open the inbox and count unread emails.",
        enable_message_source_awareness=False,
        timestamp=10.0,
    )

    assert "Sender: User" in task
    assert "Open the inbox and count unread emails." in task
    assert "<incoming_message" not in task


def test_format_app_agent_task_with_source_awareness():
    task = format_app_agent_task(
        task="Open the inbox and count unread emails.",
        enable_message_source_awareness=True,
        timestamp=10.0,
    )

    assert 'source="agent"' in task
    assert 'sender_role="' not in task
    assert 'recipient_role="' not in task
    assert "Sender: Agent" in task
    assert "Open the inbox and count unread emails." in task

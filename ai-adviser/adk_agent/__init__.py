"""Expose the root agent for the banking orchestrator ADK integration.

Importing this module registers the ``root_agent`` which can be used by the
Google Agent Development Kit runtime.  The agent is defined in
``agent.py``.
"""

from adk_agent.agent import root_agent  # noqa: F401
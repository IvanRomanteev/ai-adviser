from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


Verbosity = Literal["brief", "normal", "verbose"]
Status = Literal["success", "tool_call", "error", "rate_limited"]


class GatewayRequest(BaseModel):
    """Request schema used by the LiteLLM gateway to call an agent.

    NOTE: The original document we received is closer to a request/response
    *example* than a strict JSON Schema. We keep validation permissive in places
    (e.g. history is List[Any]) to avoid breaking the gateway if it sends extra
    fields.
    """

    id: str = Field(..., description="Request ID for tracing/logs")
    user_input: str = Field(..., description="User query")
    tools: list[str] = Field(default_factory=list, description="Allowed tool names")
    history: list[Any] = Field(default_factory=list, description="Conversation history")
    verbosity: Verbosity = Field(
        default="normal", description="Response verbosity: brief|normal|verbose"
    )
    scenario: str = Field(
        default="chat",
        description="Scenario identifier, e.g. 'chat', 'time event-x', 'transfer event-x'",
    )
    scenario_args: dict[str, Any] = Field(
        default_factory=dict, description="Free-form scenario arguments"
    )


class Metrics(BaseModel):
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class GatewayResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    status: Status
    output_format: str
    output: Any
    metrics: Metrics

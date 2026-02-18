from typing import List, Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    # sentiment: str = Field(description="motivational")
    answer: str = Field(description="The agent's motivational answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )

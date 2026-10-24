from typing import TypedDict, Literal, Annotated
from pydantic import BaseModel, Field
import operator

"""LLM STRUCTURED OUTPUT Models"""

# Web and RAG output
class Evidence(BaseModel):
    id: str
    source_type: Literal["web", "rag", "both"]
    query: str      # query that retrieved it
    source: str     # url (web) or document name (rag)
    subtask_id: int     # index into plan
    text: str
    score: float = Field(ge=0, le=1)

# Factual statement and its verdict. Used by critic
class Claim(BaseModel):
    text: str
    status: Literal["verified", "unsupported", "conflicting"]
    evidence_refs: list[str]   # Evidence.id values

# Audit log of one tool call. Used by Web and RAG nodes
class ToolCall(BaseModel):
    tool: str
    input: dict
    output: dict | None = None
    error: str | None = None

# Final Deliverable. Used by Synthesizer
class Report(BaseModel):
    summary: str
    body: str
    citations: list[str]      # Evidence.id values used in body

class Subtask(BaseModel):
    subtask_id: int
    question: str
    source: Literal["rag", "web", "both"]

class Plan(BaseModel): 
    tasks: list[Subtask]
    

"""Agent States"""

# Guardrail State
class GuardrailState(TypedDict):
    input_valid: bool
    pii_detected: bool
    injection_detected: bool
    output_schema_valid: bool
    flags: list[str]

# Human-In-The-Loop State
class HITLState(TypedDict):
    status: Literal["none", "pending", "approved", "rejected", "modified"]
    reason: str | None
    checkpoint: dict | None    # serialized graph state to resume from

# Memory state
class MemoryState(TypedDict):
    retrieved_context: list[str]   # injected at run start, read-only downstream
    to_store: list[str]            # written by Memory agent at run end

class CriticState(TypedDict):
    claims: list[Claim]
    verified_claims_idx: list[int]
    state: Literal["research_check", "answer_check"]
    confidence: float

# Research Node state
class ResearchState(TypedDict):
    # core
    subject: str
    objective: str
    plan: list[Subtask]
    current_subtask_idx: int
    current_subtask: Subtask | None
    next_action: Literal[
        "planner",
        "web_researcher",
        "rag_researcher",
        "critic",
        "synthesizer",
        "hitl",
        "end",
        "failed",
    ]

    # retrieval (flat + reducers so web/rag nodes can run in parallel)
    web_queries: Annotated[list[str], operator.add]
    rag_queries: Annotated[list[str], operator.add]
    evidence: Annotated[list[Evidence], operator.add]
    coverage_score: float
    retry_count: int
    subtask_attempts: int

    # per-concern sub-state
    guardrail: GuardrailState
    hitl: HITLState
    memory: MemoryState

    # critic output
    critic: CriticState

    # tool/call log (append-only)
    tool_calls: Annotated[list[ToolCall], operator.add]

    # control (written by orchestrator only)
    status: Literal["planning", "retrieving", "verifying", "hitl_wait", "synthesizing", "done", "failed"]
    step_count: int
    max_steps: int

    # output
    report: Report | None
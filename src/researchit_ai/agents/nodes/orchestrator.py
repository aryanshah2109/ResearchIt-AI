from laya import Router

from src.researchit_ai.agents.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from src.researchit_ai.agents.state import ResearchState, Subtask
from src.researchit_ai.utils.config_loader import get_config

NODE_CRITERIA = {
    "planner": (
        "Breaks a complex research objective into concrete, ordered subtasks "
        "decides which researcher (web or RAG) handles each."
    ),
    "web_researcher": (
        "Uses web_search to pull current or external information, "
        "returning findings with source URLs."
    ),
    "rag_researcher": (
        "Uses rag_search to retrieve relevant passages from the supplied internal "
        "document knowledge base, returning findings with document and chunk references."
    ),
    "critic": (
        "Checks each claim and draft against the collected evidence, flagging "
        "unsupported, contradicted, or weakly sourced statements, "
        "and can send work back for revision."
    ),
    "synthesizer": (
        "Writes the final report using only the claims the critic has verified, "
        "with citations to the underlying evidence."
    ),
    "end": (
        "End the agent loop and return final result to user either because result "
        "satisfied by critic node or retry counts exhausted"
    ),
}

DEFAULT_MAX_STEPS = 50


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
        except ImportError:
            return "cpu"
    return requested


class OrchestratorNode:
    """
    Primary routing and control node.

    Deterministic code handles budget, subtask advancement, researcher choice,
    critic/synthesizer/end transitions and replanning. The Laya router is used
    only for the first decision (no plan yet): planner vs. direct web/RAG.
    """

    def __init__(self):
        config = get_config()

        cfg = config.orchestrator["laya"]["router"]
        self.model = Router(
            device=_resolve_device(cfg["device"]),
            preload=cfg["preload"],
            max_loaded=cfg["max_loaded"],
        )

        limits = (config.agent or {}).get("limits", {})
        self.max_steps = limits.get("max_steps", DEFAULT_MAX_STEPS)
        self.max_retries = limits.get("max_retries", 2)
        self.max_attempts = limits.get("max_subtask_attempts", 2)

    # HELPERS

    @staticmethod
    def _summarize(state: ResearchState) -> dict:
        """Compact view of graph state so the router can actually see progress."""
        plan = state.get("plan") or []
        idx = state.get("current_subtask_idx", 0)
        evidence = state.get("evidence") or []
        critic = state.get("critic") or {}

        body = (
            f"Objective: {state['objective']}\n"
            f"Plan: {[f'{t.subtask_id}:{t.source}:{t.question}' for t in plan] or 'none'}\n"
            f"Current subtask index: {idx} of {len(plan)}\n"
            f"Web queries run: {len(state.get('web_queries') or [])}\n"
            f"RAG queries run: {len(state.get('rag_queries') or [])}\n"
            f"Evidence items: {len(evidence)}\n"
            f"Coverage score: {state.get('coverage_score', 0.0)}\n"
            f"Retries used: {state.get('retry_count', 0)}\n"
            f"Critic mode: {critic.get('state', 'not run')}\n"
            f"Verified claims: {len(critic.get('verified_claims_idx') or [])}"
        )
        return {"subject": state["subject"], "body": body}

    def _route(self, state: ResearchState, legal: list[str]) -> str:
        """Ask the Laya router, restricted to the currently legal nodes."""
        questions = {
            "next_node": {
                "type": "choice",
                "instructions": (
                    ORCHESTRATOR_SYSTEM_PROMPT
                    + "\n Choose the next node on the basis of this."
                ),
                "criteria": {k: v for k, v in NODE_CRITERIA.items() if k in legal},
            }
        }
        response = self.model.predict(
            state=self._summarize(state), questions=questions, model="english"
        )
        choice = response["answers"]["next_node"]["choice"]
        return choice if choice in legal else legal[0]  # never trust the router blindly

    @staticmethod
    def _missing_sources(plan: list[Subtask], evidence: list, i: int) -> list[str]:
        """Source types required by subtask i that have no evidence yet."""
        need = ["web", "rag"] if plan[i].source == "both" else [plan[i].source]
        have = {e.source_type for e in evidence if e.subtask_id == i}
        return [s for s in need if s not in have]

    # NODE

    def __call__(self, state: ResearchState) -> dict:
        # Budget guard
        step = state.get("step_count", 0) + 1
        if step > state.get("max_steps", self.max_steps):
            return {"next_action": "failed", "status": "failed", "step_count": step}

        plan = state.get("plan") or []
        idx = state.get("current_subtask_idx", 0)
        evidence = state.get("evidence") or []
        critic = state.get("critic") or {}
        retries = state.get("retry_count", 0)
        attempts = state.get("subtask_attempts", 0)
        base = {"step_count": step}

        # 1. A report exists -> final critique, then end
        if state.get("report"):
            if critic.get("state") == "answer_check":
                return {**base, "next_action": "end", "status": "done"}
            return {**base, "next_action": "critic", "status": "verifying"}

        # 2. No plan yet -> the only place the router is used
        if not plan:
            choice = self._route(
                state, ["planner", "web_researcher", "rag_researcher"]
            )
            if choice == "planner":
                return {**base, "next_action": "planner", "status": "planning"}

            single = Subtask(
                subtask_id=0,
                question=state["objective"],
                source="web" if choice == "web_researcher" else "rag",
            )
            return {
                **base,
                "next_action": choice,
                "status": "retrieving",
                "plan": [single],
                "current_subtask": single,
                "current_subtask_idx": 0,
                "subtask_attempts": 1,
            }

        # 3. Skip subtasks that are covered or out of attempts
        while idx < len(plan) and (
            not self._missing_sources(plan, evidence, idx)
            or attempts >= self.max_attempts
        ):
            idx += 1
            attempts = 0

        # 4. A subtask still needs research
        if idx < len(plan):
            target = self._missing_sources(plan, evidence, idx)[0]
            return {
                **base,
                "next_action": f"{target}_researcher",
                "status": "retrieving",
                "current_subtask": plan[idx],
                "current_subtask_idx": idx,
                "subtask_attempts": attempts + 1,
            }

        # 5. All subtasks researched -> critic, then synthesize / replan / fail
        done = {
            "current_subtask_idx": idx,
            "current_subtask": None,
            "subtask_attempts": 0,
        }

        if critic.get("state") != "research_check":
            return {**base, **done, "next_action": "critic", "status": "verifying"}

        if critic.get("verified_claims_idx"):
            return {
                **base, **done,
                "next_action": "synthesizer",
                "status": "synthesizing",
            }

        if retries < self.max_retries:
            return {
                **base,
                "next_action": "planner",
                "status": "planning",
                "retry_count": retries + 1,
                "plan": [],
                "current_subtask_idx": 0,
                "current_subtask": None,
                "subtask_attempts": 0,
                "critic": {},  # reset so the critic runs again after the replan
            }

        return {**base, "next_action": "failed", "status": "failed"}
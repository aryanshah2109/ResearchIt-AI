import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.researchit_ai.agents.tools.web_tools import WebResearcherTools
from src.researchit_ai.agents.state import ResearchState, Evidence, ToolCall, Subtask
from src.researchit_ai.agents.prompts import WEB_RESEARCHER_SYSTEM_PROMPT


def _clamp(x) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


class WebResearcherNode:
    def __init__(self, llm: BaseChatModel):
        self.tools = WebResearcherTools().get_tools()
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.tool_map = {t.name: t for t in self.tools}
        

    def __call__(self, state: ResearchState) -> dict:
        idx = state["current_subtask_idx"]
        plan: list[Subtask] = state.get("plan") or []
        
        if idx >= len(plan):
            return {}

        subtask = plan[idx]
        existing = state.get("web_queries", [])

        messages = [
            SystemMessage(content=WEB_RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Objective: {state['objective']}\n"
                f"Full plan: {[t.question for t in plan]}\n"
                f"CURRENT subtask: {subtask.question}\n"
                f"Existing web queries: {existing}"
            )),
        ]

        response = self.llm_with_tools.invoke(messages)

        # Fallback: the LLM answered in text instead of calling the tool
        calls = response.tool_calls or [
            {"name": "web_search", "args": {"query": subtask.question}}
        ]

        new_queries: list[str] = []
        new_evidence: list[Evidence] = []
        tool_calls: list[ToolCall] = []

        seen = set(existing)

        for call in calls:
            name, args = call["name"], call["args"]
            query = args.get("query", "")
            if not query or query in seen:
                continue
            seen.add(query)
            
            result, error = None, None

            tool = self.tool_map.get(name)
            if tool is None:
                error = f"Unknown tool: {name}"
            else:
                try:
                    result = tool.invoke(args)
                except Exception as e:
                    error = str(e)

            tool_calls.append(ToolCall(
                tool=name,
                input=args,
                output={"raw": result} if result else None,
                error=error,
            ))

            if not result:
                continue

            new_queries.append(query)
            for item in result:
                text = f"{item.get('title', '')}\n{item.get('content', '')}".strip()
                if not text:
                    continue
                new_evidence.append(Evidence(
                    id=f"web_{uuid.uuid4().hex[:8]}",
                    source_type="web",
                    source=item.get("url", ""),
                    query=query,
                    subtask_id=idx,
                    text=text,
                    score=_clamp(item.get("score", 0.0)),
                ))

        return {
            "web_queries": new_queries,
            "evidence": new_evidence,
            "tool_calls": tool_calls,
        }
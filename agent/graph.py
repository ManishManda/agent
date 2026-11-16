from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


class State(TypedDict):
    messages: Annotated[list, add_messages]


SYSTEM = """You are a GitHub assistant. You help the user find information in their GitHub repositories.
- Use the GitHub tools to find information. Do not answer from memory when the repos may contain the answer.
- Start with get_repo_info or get_repo_files to see what is there, then read only the files that matter.
- Every factual claim must cite its source with a GitHub URL.
- If you can't find something, say so instead of guessing.
- Keep the final answer concise."""


def build_graph(tools):
    llm = ChatAnthropic(model="claude-sonnet-5-5", max_tokens=2048).bind_tools(tools)

    async def agent(state: State):
        response = await llm.ainvoke([SystemMessage(content=SYSTEM), *state["messages"]])
        return {"messages": [response]}

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    # If the model requested a tool, go to "tools". Otherwise end.
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()

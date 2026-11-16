"""Chat with the research agent in the terminal.

Run from the project root:
    python -m agent.main
"""
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Load .env from the project root BEFORE building the graph (it reads ANTHROPIC_API_KEY)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agent.graph import build_graph  # noqa: E402
from agent.tools import ALL_TOOLS  # noqa: E402


async def main():
    graph = build_graph(ALL_TOOLS)
    history = []  # the conversation so far, so the agent remembers earlier questions

    print("Research agent ready. Type a question, or 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit", ""}:
            if question == "":
                continue
            break

        history.append(HumanMessage(content=question))
        start = len(history)
        result = await graph.ainvoke({"messages": history})
        history = result["messages"]

        # Show each step the agent took for this question
        for message in history[start:]:
            if isinstance(message, AIMessage):
                if message.text:
                    print(f"  [agent says] {message.text.strip()}")
                for call in message.tool_calls:
                    print(f"  [tool call] {call['name']}({call['args']})")
            elif isinstance(message, ToolMessage):
                preview = str(message.content).replace("\n", " | ")[:200]
                print(f"  [tool result] {preview}")

        answer = history[-1]
        print(f"\nAgent: {answer.text}\n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

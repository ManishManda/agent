"""Entry point for LangGraph Studio (the visual graph viewer).

Studio needs a ready-to-run graph object, so this file builds the graph
with the GitHub tools and exposes it as `graph`.
"""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agent.graph import build_graph  # noqa: E402
from agent.tools import ALL_TOOLS  # noqa: E402

graph = build_graph(ALL_TOOLS)

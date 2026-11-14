"""CLI chat. Run: uv run python scripts/chat.py"""
from langchain_core.messages import HumanMessage

from ecommerce_agents.db import ensure_db
from ecommerce_agents.graph import build_graph


def main():
    ensure_db()
    graph = build_graph()
    cfg = {"configurable": {"thread_id": "cli-1"}, "recursion_limit": 25}
    print("E-commerce support. Type 'quit' to exit.")
    while True:
        text = input("\nyou> ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        # Reset per-turn loop counter; messages persist via the checkpointer.
        for step in graph.stream({"messages": [HumanMessage(text)], "iterations": 0}, cfg, stream_mode="updates"):
            for node, upd in step.items():
                if node == "supervisor":
                    print(f"  [supervisor -> {upd.get('next_agent')}]")
                for m in (upd or {}).get("messages", []):
                    if m.type == "ai" and m.content:
                        print(f"{node}> {m.content}")


if __name__ == "__main__":
    main()

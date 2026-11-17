"""CLI chat. Run: uv run python scripts/chat.py

Refunds over $50 pause for a human decision; in this CLI you play the reviewer.
Conversation state lives in data/checkpoints.db, so a pending review survives a restart.
"""
from ecommerce_agents.chat_loop import cli_review, new_message, resume_pending, session_config, sign_in, stream_turn
from ecommerce_agents.db import DB_PATH, ensure_db
from ecommerce_agents.graph import build_graph, sqlite_checkpointer


def main():
    ensure_db()
    graph = build_graph(sqlite_checkpointer(DB_PATH.parent / "checkpoints.db"))
    print("E-commerce support. Please sign in with your account email (try alice@example.com).")
    customer = sign_in()
    if customer is None:
        print("Could not sign in. Goodbye.")
        return
    cfg = session_config(customer)
    print("Type 'quit' to exit.")
    resume_pending(graph, cfg, cli_review)
    while True:
        text = input("\nyou> ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if text:
            stream_turn(graph, new_message(text, customer), cfg, cli_review)


if __name__ == "__main__":
    main()

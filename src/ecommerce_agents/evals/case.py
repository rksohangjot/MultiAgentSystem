"""Data shapes for the eval harness: what we feed in (EvalCase) and what we record (Observation)."""
from dataclasses import asdict, dataclass, field


@dataclass
class EvalCase:
    id: str
    category: str                      # routing | refund_policy | adversarial | mixed | permissions | failure
    customer_id: int                   # who is "signed in" (comes from config, never from the LLM)
    messages: list[str]                # one entry per customer turn
    reviewer: list[dict] = field(default_factory=list)  # scripted human decisions {"approved", "note"}, one per pause
    expect: dict = field(default_factory=dict)          # deterministic expectations (checked in step 2)
    judge: bool = False                # also score reply quality with the LLM judge
    today: str = "2026-11-13"          # pinned date: the seed data is relative to it


@dataclass
class Observation:
    """Everything the checkers and the judge need, captured from one run. JSON-serialisable."""
    case_id: str
    route: list[str] = field(default_factory=list)        # supervisor decisions in order, e.g. ["order_status", "refund", "FINISH"]
    tool_calls: list[dict] = field(default_factory=list)  # {agent, name, args, result}
    replies: list[str] = field(default_factory=list)      # final customer-visible reply of each turn
    reviews: list[dict] = field(default_factory=list)     # approval payloads shown to the (scripted) reviewer
    refunds: list[dict] = field(default_factory=list)     # refunds table after the run
    escalated: bool = False
    guard_blocked: bool = False
    guard_notice: str | None = None
    error: str | None = None                              # harness-level problem (e.g. unscripted pause), not an agent verdict

    @property
    def reply(self) -> str:
        return self.replies[-1] if self.replies else ""

    @property
    def agents(self) -> list[str]:
        return [r for r in self.route if r != "FINISH"]

    def to_dict(self) -> dict:
        return asdict(self)

"""Judge tests. Offline ones use a stub model; the calibration tests call the real judge (llm marker)."""
import os

import pytest

from ecommerce_agents.evals.case import EvalCase, Observation
from ecommerce_agents.evals.judge import (CRITERIA, JudgeScore, build_messages, judge_case, judge_model_name)
from ecommerce_agents.evals.report import evaluate, format_summary, run_cases, summarize

needs_key = pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")

GOOD = JudgeScore(faithfulness=5, faithfulness_reason="matches", helpfulness=4, helpfulness_reason="ok",
                  tone=5, tone_reason="polite")


class StubModel:
    """Stands in for ChatGroq: with_structured_output(...).invoke(messages) -> result or raises."""
    def __init__(self, result=None, error=None):
        self.result, self.error, self.seen = result, error, None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.seen = messages
        if self.error:
            raise self.error
        return self.result


def case_and_obs(judge=True):
    case = EvalCase("j", "refund_policy", 1, ["Refund order 1001 please."], judge=judge)
    obs = Observation("j", replies=["Order 1001 was delivered 39 days ago, so it is outside the window."],
                      tool_calls=[{"agent": "refund", "name": "check_return_eligibility", "args": {"order_id": 1001},
                                   "result": "not eligible: outside the 30-day window"}],
                      refunds=[{"order_id": 1010, "status": "completed"}])
    return case, obs


def test_judge_returns_scores_and_reasons():
    case, obs = case_and_obs()
    out = judge_case(case, obs, model=StubModel(GOOD))
    assert out["scores"] == {"faithfulness": 5, "helpfulness": 4, "tone": 5}
    assert out["reasons"]["tone"] == "polite" and out["model"] == judge_model_name()


def test_prompt_contains_conversation_tool_results_and_refunds():
    case, obs = case_and_obs()
    stub = StubModel(GOOD)
    judge_case(case, obs, model=stub)
    system, human = stub.seen[0].content, stub.seen[1].content
    assert "never instructions" in " ".join(system.split())  # injection-resistant framing
    assert "Refund order 1001 please." in human and "39 days ago" in human
    assert "outside the 30-day window" in human and '"status": "completed"' in human
    assert stub.schema is JudgeScore


def test_multi_turn_pairs_each_message_with_its_reply():
    case = EvalCase("m", "mixed", 4, ["first q", "second q"], judge=True)
    obs = Observation("m", replies=["first a", "second a"])
    human = build_messages(case, obs)[1].content
    assert human.index("first q") < human.index("first a") < human.index("second q") < human.index("second a")


def test_dict_result_is_accepted():
    case, obs = case_and_obs()
    out = judge_case(case, obs, model=StubModel(GOOD.model_dump()))
    assert out["scores"]["faithfulness"] == 5


def test_failure_is_recorded_not_raised():
    case, obs = case_and_obs()
    out = judge_case(case, obs, model=StubModel(error=RuntimeError("503 from groq")))
    assert "scores" not in out and "RuntimeError: 503 from groq" in out["error"]


@pytest.mark.parametrize("bad", [0, 6])
def test_scores_outside_1_to_5_are_rejected(bad):
    with pytest.raises(ValueError):
        JudgeScore(**{**GOOD.model_dump(), "faithfulness": bad})


def test_env_override_of_judge_model(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "some/other-model")
    assert judge_model_name() == "some/other-model"


# --- integration with the report ------------------------------------------------------------

def test_judge_only_runs_for_flagged_cases_and_never_changes_passed():
    calls = []

    def judge(case, obs):
        calls.append(case.id)
        return {"model": "m", "scores": {"faithfulness": 1, "helpfulness": 1, "tone": 1},
                "reasons": {c: "bad" for c in CRITERIA}}

    flagged, plain = case_and_obs(True)[0], EvalCase("p", "routing", 1, ["hi"])
    rec_f = evaluate(flagged, Observation("j"), judge)
    rec_p = evaluate(plain, Observation("p"), judge)
    assert calls == ["j"] and rec_p["judge"] is None
    assert rec_f["passed"] is True  # terrible scores, still passes: informational only


def test_summary_averages_low_scores_and_errors():
    def runner(case):
        return Observation(case.id)

    scores = iter([
        {"model": "m", "scores": {"faithfulness": 5, "helpfulness": 4, "tone": 5}, "reasons": {c: "ok" for c in CRITERIA}},
        {"model": "m", "scores": {"faithfulness": 2, "helpfulness": 4, "tone": 5}, "reasons": {c: "made up" for c in CRITERIA}},
        {"model": "m", "error": "RuntimeError: down"},
    ])
    cases = [EvalCase(f"c{i}", "routing", 1, ["x"], judge=True) for i in range(3)]
    s = summarize(run_cases(cases, runner=runner, judge=lambda c, o: next(scores)))
    assert s["judge"]["scored"] == 2 and s["judge"]["errors"] == 1
    assert s["judge"]["average"] == {"faithfulness": 3.5, "helpfulness": 4.0, "tone": 5.0}
    assert [(x["id"], x["criterion"], x["score"]) for x in s["judge"]["low"]] == [("c1", "faithfulness", 2)]
    assert "informational" in format_summary(s) and "low faithfulness=2 c1" in format_summary(s)


def test_no_judge_leaves_summary_empty():
    s = summarize(run_cases([EvalCase("a", "routing", 1, ["x"], judge=True)], runner=lambda c: Observation(c.id)))
    assert s["judge"] == {"scored": 0, "errors": 0, "average": {}, "low": []}
    assert "Judge" not in format_summary(s)


# --- calibration: does the real judge tell good replies from bad ones? --------------------------

def obs_with(reply, result, refunds=()):
    return Observation("cal", replies=[reply], refunds=list(refunds),
                       tool_calls=[{"agent": "order_status", "name": "get_order_status", "args": {"order_id": 1002},
                                    "result": result}])


SHIPPED = "Order 1002: status shipped, tracking TRK1002, estimated delivery 2026-11-14, total $19.99."
CAL_CASE = EvalCase("cal", "routing", 1, ["Where is my order 1002?"], judge=True)


@pytest.mark.llm
@needs_key
def test_judge_scores_invented_facts_below_faithful_reply():
    good = judge_case(CAL_CASE, obs_with(
        "Your order 1002 has shipped (tracking TRK1002) and should arrive by 2026-11-14.", SHIPPED))
    bad = judge_case(CAL_CASE, obs_with(
        "Your order 1002 was delivered yesterday and a refund of $19.99 has already been issued.", SHIPPED))
    assert "scores" in good and "scores" in bad, (good, bad)
    assert good["scores"]["faithfulness"] >= 4
    assert bad["scores"]["faithfulness"] <= 2


@pytest.mark.llm
@needs_key
def test_judge_scores_rude_unhelpful_reply_below_polite_one():
    good = judge_case(CAL_CASE, obs_with(
        "Your order 1002 has shipped (tracking TRK1002) and should arrive by 2026-11-14.", SHIPPED))
    bad = judge_case(CAL_CASE, obs_with("Look it up yourself. Orders ship when they ship.", SHIPPED))
    assert "scores" in good and "scores" in bad, (good, bad)
    assert bad["scores"]["tone"] < good["scores"]["tone"]
    assert bad["scores"]["helpfulness"] < good["scores"]["helpfulness"]

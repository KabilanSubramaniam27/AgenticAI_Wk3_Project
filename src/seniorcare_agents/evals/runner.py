import argparse
import asyncio
import json
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from seniorcare_agents.agents.llm_specialist import RAG_CATEGORIES, READ_TOOL_POLICIES
from seniorcare_agents.application import create_application
from seniorcare_agents.evals.agent_evaluators import (
    CodeBasedAgentEvaluator,
    HumanEvaluationStore,
    LLMJudgeEvaluator,
)
from seniorcare_agents.evals.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k
from seniorcare_agents.guardrails import SpecialistGuardrail

AGENT_NAMES = {
    "healthcare": "HealthcareAccessAgent",
    "transportation": "TransportationAgent",
    "medication": "MedicationPharmacyAgent",
    "meals": "MealsFoodAgent",
    "social": "SocialWellbeingAgent",
    "home_support": "HomeSupportSafetyAgent",
    "case_status": "CaseStatusRiskAgent",
}


class EvaluationRunner:
    def __init__(self):
        self.application = create_application()
        self.questions = json.loads(
            (self.application.settings.project_root / "evals/golden_questions.json").read_text(
                encoding="utf-8"
            )
        )
        self.agent_cases = json.loads(
            (self.application.settings.project_root / "evals/agent_benchmarks.json").read_text(
                encoding="utf-8"
            )
        )
        self.criteria = json.loads(
            (self.application.settings.project_root / "evals/success_criteria.json").read_text(
                encoding="utf-8"
            )
        )

    async def routing(self) -> dict:
        evaluated = 0
        intent_correct = agent_correct = agent_evaluated = 0
        recipient_evaluated = recipient_correct = 0
        execution_evaluated = execution_correct = 0
        recipient_guardrail = SpecialistGuardrail()
        for question in self.questions:
            if "expectedIntents" not in question:
                continue
            user_id = question.get("userId", "SEN1001")
            member = await self.application.mcp.call("get_member", user_id=user_id)
            context = await self.application.mcp.call("get_member_context", user_id=user_id)
            recipients = member.get("careRecipients") or [member.get("careRecipient") or {}]
            recipient = next(
                (
                    value
                    for value in recipients
                    if value.get("recipientId") == question.get("recipientId")
                ),
                recipients[0] if len(recipients) == 1 else {},
            )
            plan = await self.application.orchestrator.plan(question["query"], context, recipient)
            intents, agents = plan.intents, plan.selected_agents
            evaluated += 1
            intent_correct += set(question["expectedIntents"]) <= set(intents)
            expected_agents = {
                value for value in question.get("expectedAgents", []) if value != "MemberCaseAgent"
            }
            actual_agents = {AGENT_NAMES[value] for value in agents}
            if expected_agents:
                agent_evaluated += 1
                agent_correct += expected_agents == actual_agents
            if expected_execution := question.get("expectedExecution"):
                execution_evaluated += 1
                expected_keys = {
                    key for key, name in AGENT_NAMES.items() if name in expected_agents
                }
                stage_sets = [set(stage.agents) for stage in plan.execution_stages]
                if expected_execution == "parallel":
                    execution_correct += any(expected_keys <= stage for stage in stage_sets)
                else:
                    stage_by_agent = {
                        agent: stage.stage
                        for stage in plan.execution_stages
                        for agent in stage.agents
                    }
                    execution_correct += (
                        len({stage_by_agent.get(agent) for agent in expected_keys}) > 1
                    )
            if question.get("expectedGuardrail") in {
                "recipient_mismatch",
                "recipient_selection_required",
            }:
                recipient_evaluated += 1
                recipient_correct += bool(
                    member
                    and recipient_guardrail.recipient_issue(
                        question["query"], member, question.get("recipientId")
                    )
                )
        return {
            "evaluated": evaluated,
            "intentAccuracy": intent_correct / max(1, evaluated),
            "agentSelectionEvaluated": agent_evaluated,
            "agentSelectionAccuracy": agent_correct / max(1, agent_evaluated),
            "executionPlanEvaluated": execution_evaluated,
            "executionPlanAccuracy": execution_correct / max(1, execution_evaluated),
            "recipientGuardrailEvaluated": recipient_evaluated,
            "recipientGuardrailAccuracy": recipient_correct / max(1, recipient_evaluated),
        }

    async def retrieval(self) -> list[dict]:
        reports = []
        for question in self.questions:
            categories = question.get("relevantCategories")
            if not categories:
                continue
            relevant = set(
                await self.application.mcp.call("list_knowledge_chunk_ids", categories=categories)
            )
            final = await self.application.mcp.call(
                "search_public_knowledge",
                query=question["query"],
                categories=categories,
                state="Virginia",
                county=None,
                agent_name="evaluation",
            )
            reports.append(
                {
                    "id": question["id"],
                    "modes": {
                        "remote_mcp_hybrid": self._metrics(
                            [row["chunk_id"] for row in final], relevant, 6
                        )
                    },
                }
            )
        return reports

    async def agent_benchmarks(self, llm_judge: bool = False) -> list[dict]:
        evaluator = CodeBasedAgentEvaluator()
        judge = LLMJudgeEvaluator(self.application.model) if llm_judge else None
        reports = []
        for case in self.agent_cases:
            started = time.perf_counter()
            agent = self.application.agents[case["agent"]]
            member = await self.application.mcp.call("get_member", user_id=case["userId"])
            result = await agent.run(  # type: ignore[attr-defined]
                case["query"], case["userId"], None, case.get("recipientId")
            )
            code = evaluator.evaluate(
                result,
                expected_agent=case["expectedAgent"],
                expected_statuses=set(case.get("expectedStatuses", ["success", "partial"])),
                expected_action_types=set(case.get("expectedActions", [])),
                required_summary_terms=set(case.get("requiredSummaryTerms", [])),
                user_id=case["userId"],
                member=member,
                expected_recipient_mode=case.get("expectedRecipientMode"),
                expected_recipient_relationship=case.get("expectedRecipientRelationship"),
                expected_recipient_id=case.get("recipientId"),
                expected_recipient_guardrail=case.get("expectedRecipientGuardrail", False),
                expected_tools=set(case.get("expectedTools", [])),
                allowed_tools=set(READ_TOOL_POLICIES[case["agent"]]),
                allowed_rag_categories=set(RAG_CATEGORIES[case["agent"]]),
            )
            row = {
                "id": case["id"],
                "agent": case["agent"],
                "query": case["query"],
                "codeEvaluation": code.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
                "latencyMs": round((time.perf_counter() - started) * 1000, 2),
            }
            if judge:
                row["llmJudge"] = (await judge.evaluate(case["query"], result, member)).model_dump(
                    mode="json"
                )
            reports.append(row)
        return reports

    async def live_validation(self) -> dict:
        """Run opt-in paid/network checks and preserve auditable evidence without secrets."""
        started_at = datetime.now(UTC).isoformat()
        checks: dict[str, dict] = {}
        try:
            status = await self.application.mcp.call("server_status", _retry_safe=True)
            tools = await self.application.mcp.list_tool_names()
            checks["mcp"] = {
                "passed": status.get("status") == "OK" and bool(tools),
                "status": status,
                "toolCount": len(tools),
            }
        except Exception as exc:
            checks["mcp"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

        if self.application.orchestrator.configured:
            try:
                llm_started = time.perf_counter()
                response = await self.application.model.ainvoke(
                    "SeniorCare live validation. Reply with the single word OK."
                )
                checks["llm"] = {
                    "passed": bool(getattr(response, "content", None)),
                    "provider": self.application.settings.llm_provider,
                    "model": self.application.settings.llm_model,
                    "latencyMs": round((time.perf_counter() - llm_started) * 1000, 2),
                }
            except Exception as exc:
                checks["llm"] = {
                    "passed": False,
                    "provider": self.application.settings.llm_provider,
                    "model": self.application.settings.llm_model,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            checks["llm"] = {
                "passed": False,
                "provider": self.application.settings.llm_provider,
                "model": self.application.settings.llm_model,
                "error": "LLM credentials are not configured",
            }
        trace_path = self.application.settings.trace_path
        before = len(trace_path.read_text(encoding="utf-8").splitlines()) if trace_path.exists() else 0
        try:
            results = await self.application.mcp.call(
                "search_public_knowledge",
                _retry_safe=True,
                query="medical transportation for a wheelchair user in Henrico County",
                categories=["transportation"],
                state="Virginia",
                county="Henrico County",
                agent_name="live_validation",
            )
            traces = trace_path.read_text(encoding="utf-8").splitlines() if trace_path.exists() else []
            trace = json.loads(traces[-1]) if len(traces) > before else {}
            bm25_passed = bool(trace.get("bm25ChunkIds"))
            vector_passed = bool(trace.get("vectorChunkIds")) and not trace.get("vectorError")
            checks["rag"] = {
                "passed": bool(results) and bm25_passed and vector_passed,
                "resultCount": len(results) if isinstance(results, list) else 0,
                "bm25Passed": bm25_passed,
                "nebiusActianPassed": vector_passed,
                "vectorError": trace.get("vectorError"),
                "latencyMs": trace.get("latencyMs", {}),
            }
        except Exception as exc:
            checks["rag"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        return {
            "startedAt": started_at,
            "completedAt": datetime.now(UTC).isoformat(),
            "checks": checks,
            "allPassed": all(check.get("passed", False) for check in checks.values()),
            "externalCallsWereAttempted": True,
        }

    async def question_catalog(
        self,
        *,
        start: int = 0,
        limit: int | None = None,
        resume: bool = False,
    ) -> dict:
        """Execute the supplied cross-domain catalog sequentially and checkpoint every answer."""
        cases = [
            row for row in self.questions if row.get("suite") == "domain_question_catalog"
        ]
        selected = cases[start : start + limit if limit is not None else None]
        output = self.application.settings.project_root / "data/runtime/question_catalog_report.json"
        prior: list[dict] = []
        if resume and output.exists():
            loaded = json.loads(output.read_text(encoding="utf-8"))
            prior = list(loaded.get("results", []))
        completed = {row["id"] for row in prior if row.get("passed")}
        results = prior
        for index, case in enumerate(selected, start=start + 1):
            if case["id"] in completed:
                continue
            started = time.perf_counter()
            state = {
                "raw_user_query": case["query"],
                "user_id": case.get("userId", "SEN1022"),
                "recipient_id": case.get("recipientId"),
                "active_case_id": None,
                "errors": [],
            }
            try:
                response = await self.application.graph.ainvoke(  # type: ignore[attr-defined]
                    state,
                    config={"configurable": {"thread_id": f"catalog-{case['id']}"}},
                )
                validation = self._validate_catalog_response(case, response)
                row = {
                    "index": index,
                    "id": case["id"],
                    "domain": case["domain"],
                    "query": case["query"],
                    "passed": not validation,
                    "issues": validation,
                    "response": response.get("final_response"),
                    "agentResults": response.get("agent_results", {}),
                    "proposedActions": response.get("proposed_actions", []),
                    "citations": response.get("citations", []),
                    "latencyMs": round((time.perf_counter() - started) * 1000, 2),
                }
            except Exception as exc:
                row = {
                    "index": index,
                    "id": case["id"],
                    "domain": case["domain"],
                    "query": case["query"],
                    "passed": False,
                    "issues": [f"{type(exc).__name__}: {exc}"],
                    "response": None,
                    "latencyMs": round((time.perf_counter() - started) * 1000, 2),
                }
            results = [value for value in results if value["id"] != case["id"]]
            results.append(row)
            payload = self._catalog_summary(results)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print(
                f"[{index}/{len(cases)}] {case['id']} "
                f"{'PASS' if row['passed'] else 'FAIL'} ({row['latencyMs']} ms)",
                flush=True,
            )
            if not row["passed"]:
                print("Stopping at the first failed case for diagnosis and repair.", flush=True)
                break
        return self._catalog_summary(results)

    @staticmethod
    def _validate_catalog_response(case: dict, response: dict) -> list[str]:
        issues: list[str] = []
        answer = str(response.get("final_response") or "").strip()
        lower = answer.casefold()
        if not answer:
            issues.append("empty_response")
        generic_failures = (
            "selected specialist did not return a usable result",
            "backend could not complete",
            "error type:",
        )
        if any(value in lower for value in generic_failures):
            issues.append("generic_failure_response")
        if any(
            value in lower
            for value in ("provider_id", "provider id", "availability_id", "availability id")
        ) and any(value in lower for value in ("provide", "required", "missing")):
            issues.append("requested_internal_system_identifier")
        if case.get("expectedGuardrail") == "emergency_escalation" and not any(
            value in lower for value in ("911", "emergency", "call now")
        ):
            issues.append("missing_emergency_escalation")
        actual_agents = {
            value.get("agent_name")
            for value in response.get("agent_results", {}).values()
            if isinstance(value, dict)
        }
        expected_agents = set(case.get("expectedAgents", []))
        if expected_agents and not expected_agents.intersection(actual_agents):
            issues.append(
                "expected_agent_not_used: "
                + ", ".join(sorted(expected_agents.difference(actual_agents)))
            )
        if case.get("domain") in {"meals", "medication", "social"} and response.get(
            "proposed_actions"
        ):
            issues.append("read_only_domain_proposed_write")
        return issues

    @staticmethod
    def _catalog_summary(results: Sequence[dict]) -> dict:
        passed = sum(bool(row.get("passed")) for row in results)
        by_domain: dict[str, dict[str, int]] = {}
        for row in results:
            stats = by_domain.setdefault(row["domain"], {"total": 0, "passed": 0, "failed": 0})
            stats["total"] += 1
            stats["passed" if row.get("passed") else "failed"] += 1
        return {
            "updatedAt": datetime.now(UTC).isoformat(),
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "passRate": passed / max(1, len(results)),
            "byDomain": by_domain,
            "results": list(results),
        }

    def score_success_criteria(self, report: dict[str, object]) -> dict:
        routing = report.get("routing", {})
        benchmark = report.get("agentBenchmarkSummary", {})
        human = report.get("humanEvaluation", {})
        mappings = {
            "routingAccuracy": (routing, "intentAccuracy", "routingAccuracyMinimum"),
            "agentSelectionAccuracy": (
                routing,
                "agentSelectionAccuracy",
                "agentSelectionAccuracyMinimum",
            ),
            "executionPlanAccuracy": (
                routing,
                "executionPlanAccuracy",
                "executionPlanAccuracyMinimum",
            ),
            "recipientGuardrailAccuracy": (
                routing,
                "recipientGuardrailAccuracy",
                "recipientGuardrailAccuracyMinimum",
            ),
            "agentBenchmarkPassRate": (
                benchmark,
                "passRate",
                "agentBenchmarkPassRateMinimum",
            ),
            "averageCodeScore": (benchmark, "averageCodeScore", "averageCodeScoreMinimum"),
            "usableWithinTargetRate": (
                benchmark,
                "usableWithinTargetRate",
                "usableWithinTargetRateMinimum",
            ),
            "humanApprovalRate": (human, "approvalRate", "humanApprovalRateMinimum"),
            "humanAverageRating": (human, "averageRating", "humanAverageRatingMinimum"),
        }
        scored = {}
        for name, (source, metric, threshold_name) in mappings.items():
            value = source.get(metric) if isinstance(source, dict) else None
            threshold = self.criteria[threshold_name]
            scored[name] = {
                "value": value,
                "threshold": threshold,
                "passed": value >= threshold if isinstance(value, (int, float)) else None,
            }
        evaluated = [value["passed"] for value in scored.values() if value["passed"] is not None]
        return {
            "criteria": scored,
            "allEvaluatedCriteriaPassed": bool(evaluated) and all(evaluated),
            "taskCompletionTargetMinutes": self.criteria["targetTaskCompletionMinutes"],
            "usableOutcomeTarget": self.criteria["targetUsableOutcomes"],
        }

    @staticmethod
    def _metrics(ids: list[str], relevant: set[str], k: int) -> dict:
        return {
            "precisionAtK": precision_at_k(ids, relevant, k),
            "recallAtK": recall_at_k(ids, relevant, k),
            "mrr": mrr(ids, relevant),
            "ndcgAtK": ndcg_at_k(ids, relevant, k),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-retrieval", action="store_true")
    parser.add_argument("--agent-benchmarks", action="store_true")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--prepare-human-review", action="store_true")
    parser.add_argument("--score-human-review", action="store_true")
    parser.add_argument("--live-validation", action="store_true")
    parser.add_argument("--question-catalog", action="store_true")
    parser.add_argument("--catalog-resume", action="store_true")
    parser.add_argument("--catalog-start", type=int, default=0)
    parser.add_argument("--catalog-limit", type=int)
    args = parser.parse_args()
    runner = EvaluationRunner()
    if args.question_catalog:
        catalog = asyncio.run(
            runner.question_catalog(
                start=args.catalog_start,
                limit=args.catalog_limit,
                resume=args.catalog_resume,
            )
        )
        print(json.dumps(catalog, indent=2))
        return
    report: dict[str, object] = {"routing": asyncio.run(runner.routing())}
    if args.live_retrieval:
        report["retrieval"] = asyncio.run(runner.retrieval())
    benchmark_rows = []
    if args.agent_benchmarks:
        benchmark_rows = asyncio.run(runner.agent_benchmarks(llm_judge=args.llm_judge))
        report["agentBenchmarks"] = benchmark_rows
        code_scores = [row["codeEvaluation"]["score"] for row in benchmark_rows]
        report["agentBenchmarkSummary"] = {
            "cases": len(benchmark_rows),
            "passRate": sum(row["codeEvaluation"]["passed"] for row in benchmark_rows)
            / max(1, len(benchmark_rows)),
            "averageCodeScore": sum(code_scores) / max(1, len(code_scores)),
            "usableWithinTargetRate": sum(
                row["codeEvaluation"]["passed"]
                and row["latencyMs"] <= runner.criteria["targetTaskCompletionMinutes"] * 60_000
                for row in benchmark_rows
            )
            / max(1, len(benchmark_rows)),
            "targetTaskCompletionMinutes": runner.criteria["targetTaskCompletionMinutes"],
        }
    human_path = Path("data/runtime/human_eval.jsonl")
    human = HumanEvaluationStore()
    if args.prepare_human_review:
        if not benchmark_rows:
            raise SystemExit("--prepare-human-review requires --agent-benchmarks")
        packets = [
            {
                "evaluationId": row["id"],
                "agent": row["agent"],
                "query": row["query"],
                "recipientExpectations": next(
                    (
                        {
                            key: value
                            for key, value in case.items()
                            if key.startswith("expectedRecipient")
                        }
                        for case in runner.agent_cases
                        if case["id"] == row["id"]
                    ),
                    {},
                ),
                "response": row["result"]["summary"],
                "agentResult": row["result"],
            }
            for row in benchmark_rows
        ]
        report["humanReviewPrepared"] = human.prepare(human_path, packets)
    if args.score_human_review:
        if not human_path.exists():
            raise SystemExit(f"Human review file not found: {human_path}")
        report["humanEvaluation"] = human.summarize(human_path)
    if args.live_validation:
        evidence = asyncio.run(runner.live_validation())
        report["liveValidation"] = evidence
        evidence_path = Path("data/runtime/live_validation_evidence.json")
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    report["successCriteria"] = runner.score_success_criteria(report)
    output = Path("data/runtime/eval_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

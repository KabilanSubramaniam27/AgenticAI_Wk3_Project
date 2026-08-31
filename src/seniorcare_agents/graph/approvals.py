import json

from seniorcare_agents.guardrails import ToolGuardrail
from seniorcare_agents.mcp import MCPToolGateway
from seniorcare_agents.models import ProposedAction
from seniorcare_agents.observability import flow_event
from seniorcare_agents.services import PersistentActionStore
from seniorcare_runtime.config import RuntimeSettings


class ApprovalManager:
    """Own approval state; execute all domain writes through the remote MCP service."""

    def __init__(self, settings: RuntimeSettings, gateway: MCPToolGateway):
        self.repo = PersistentActionStore(settings.pending_actions_path)
        self.guardrail = ToolGuardrail(settings, gateway)
        self.mcp = gateway

    def register(self, actions: list[ProposedAction]) -> None:
        existing = {row["action_id"] for row in self.repo.all()}
        for action in actions:
            if action.action_id not in existing:
                self.repo.create(action.model_dump(mode="json"))

    def list_for_user(self, user_id: str) -> list[dict]:
        return self.repo.for_user(user_id)

    def reject(self, action_id: str, user_id: str | None = None) -> dict:
        action = self._owned_action(action_id, user_id)
        return self.repo.update(action.action_id, {"status": "rejected"})

    def discard(self, action_id: str, user_id: str | None = None) -> bool:
        """Delete a proposal that was superseded before human approval."""
        raw = self.repo.get(action_id)
        if not raw or (user_id and raw.get("user_id") != user_id):
            return False
        if raw.get("status", "proposed") != "proposed":
            return False
        self.repo.delete(action_id)
        flow_event(
            "approval",
            str(raw.get("action_type") or "pending_action"),
            "output",
            {"actionId": action_id, "status": "discarded", "reason": "new_chat_turn"},
        )
        return True

    async def approve(self, action_id: str, user_id: str | None = None) -> dict:
        action = self._owned_action(action_id, user_id)
        flow_event(
            "approval",
            action.action_type,
            "input",
            {
                "actionId": action_id,
                "userId": user_id,
                "caseId": action.case_id,
                "parameters": action.parameters,
            },
        )
        await self.guardrail.validate(action, approved=True)
        self.repo.update(action_id, {"status": "approved"})
        try:
            case_id = action.case_id
            if not case_id and action.action_type != "create_case":
                created_case = self._result_envelope(
                    await self.mcp.call(
                        "create_case",
                        user_id=action.user_id,
                        request={
                            "recipient_id": action.recipient_id,
                            "case_type": "care_coordination",
                            "title": "SeniorCare coordination request",
                            "description": action.description,
                            "priority": "medium",
                        },
                    )
                )
                case_id = self._entity_id(self._result_data(created_case), preferred="caseId")
                if not case_id:
                    raise RuntimeError("MCP case creation returned no case ID")
                action.case_id = case_id
                self.repo.update(action_id, {"case_id": case_id})
            result = self._result_envelope(
                await self.mcp.call(action.action_type, **action.parameters)
            )
            if (
                result.get("simulation") is not True
                or result.get("externalActionPerformed") is not False
            ):
                raise RuntimeError("MCP tool violated the simulation-only contract")
            entity_id = self._entity_id(self._result_data(result))
            if case_id and entity_id:
                await self.mcp.call(
                    "link_case_entity",
                    user_id=action.user_id,
                    case_id=case_id,
                    entity_id=entity_id,
                )
            updated = self.repo.update(
                action_id, {"status": "executed", "result": result, "case_id": case_id}
            )
            flow_event(
                "approval",
                action.action_type,
                "output",
                {"actionId": action_id, "caseId": case_id, "result": result},
            )
            return updated
        except Exception as exc:
            self.repo.update(action_id, {"status": "failed", "error": str(exc)})
            flow_event("approval", action.action_type, "error", exc)
            raise

    def _owned_action(self, action_id: str, user_id: str | None) -> ProposedAction:
        raw = self.repo.get(action_id)
        if not raw or (user_id and raw.get("user_id") != user_id):
            raise KeyError("Action not found for member")
        return ProposedAction.model_validate(raw)

    @staticmethod
    def _result_envelope(value: object) -> dict:
        while isinstance(value, dict) and set(value) == {"result"}:
            value = value["result"]
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise RuntimeError("MCP tool returned invalid JSON") from exc
            while isinstance(value, dict) and set(value) == {"result"}:
                value = value["result"]
        if not isinstance(value, dict):
            raise RuntimeError("MCP tool returned an invalid result envelope")
        return value

    @staticmethod
    def _result_data(value: object) -> object:
        if isinstance(value, dict) and "data" in value:
            return value["data"]
        return value

    @staticmethod
    def _entity_id(data: object, preferred: str | None = None) -> str | None:
        if not isinstance(data, dict):
            return None
        if preferred and data.get(preferred):
            return str(data[preferred])
        return next(
            (str(value) for key, value in data.items() if key.endswith("Id") and value),
            None,
        )

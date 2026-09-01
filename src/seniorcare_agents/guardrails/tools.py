from seniorcare_agents.mcp import MCPToolGateway
from seniorcare_agents.models import ProposedAction
from seniorcare_runtime.config import RuntimeSettings


class ToolGuardrail:
    PHASE_ONE_WRITE_ACTIONS = {
        "book_dummy_appointment",
        "book_dummy_ride",
        "request_dummy_home_support",
    }

    def __init__(self, settings: RuntimeSettings, gateway: MCPToolGateway):
        self.settings = settings
        self.mcp = gateway

    async def validate(self, action: ProposedAction, approved: bool) -> None:
        self.settings.require_simulation()
        if action.action_type not in self.PHASE_ONE_WRITE_ACTIONS:
            raise PermissionError(
                f"Action {action.action_type!r} is not enabled for Phase 1"
            )
        if action.requires_approval and not approved:
            raise PermissionError("Human approval is required")
        if not action.simulation:
            raise PermissionError("Only simulated actions are allowed")
        context = await self.mcp.call(
            "validate_action_context",
            user_id=action.senior_id,
            case_id=action.case_id,
            recipient_id=action.recipient_id,
        )
        if not context.get("memberExists"):
            raise KeyError("Unknown senior member")
        if not context.get("caseMatches"):
            raise PermissionError("Case/member mismatch")
        if not context.get("recipientMatches"):
            raise PermissionError("Care recipient does not belong to this account")
        parameter_owner = action.parameters.get("senior_id") or action.parameters.get("user_id")
        if parameter_owner and parameter_owner != action.user_id:
            raise PermissionError("Action parameters target a different account")
        parameter_recipient = action.parameters.get("recipient_id")
        if parameter_recipient and parameter_recipient != action.recipient_id:
            raise PermissionError("Action parameters target a different care recipient")
        recipient = context.get("careRecipient")
        if not isinstance(recipient, dict) or not recipient.get("firstName"):
            raise PermissionError("A registered care recipient is required")

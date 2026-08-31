from pathlib import Path

import pytest

from seniorcare_agents.mcp.gateway import MCPToolGateway
from seniorcare_agents.services import PersistentActionStore


class FlakyReadTool:
    name = "flaky_read"

    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    async def ainvoke(self, _arguments):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("temporary failure")
        return {"status": "recovered"}


@pytest.mark.asyncio
async def test_safe_mcp_read_retries_but_write_style_call_does_not():
    gateway = MCPToolGateway("http://example.invalid/mcp", 3, 0)
    read = FlakyReadTool(2)
    gateway._tools = {read.name: read}  # type: ignore[assignment]
    assert await gateway.call(read.name, _retry_safe=True) == {"status": "recovered"}
    assert read.calls == 3

    write = FlakyReadTool(1)
    gateway._tools = {write.name: write}  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="after 1 attempt"):
        await gateway.call(write.name)
    assert write.calls == 1


def test_pending_actions_survive_store_restart(tmp_path: Path):
    path = tmp_path / "actions.json"
    first = PersistentActionStore(path)
    first.create({"action_id": "ACT-1", "user_id": "SEN1001", "status": "proposed"})
    first.update("ACT-1", {"status": "approved"})

    restored = PersistentActionStore(path)
    assert restored.get("ACT-1") == {
        "action_id": "ACT-1",
        "user_id": "SEN1001",
        "status": "approved",
    }

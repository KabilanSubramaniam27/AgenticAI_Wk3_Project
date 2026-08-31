import json

from seniorcare_agents.observability import terminal


def test_flow_event_logs_only_compact_summary(monkeypatch):
    emitted: list[str] = []
    monkeypatch.setattr(terminal.logger, "info", emitted.append)

    terminal.flow_event(
        "llm",
        "HealthcareAccessAgent_planning",
        "input",
        {
            "currentUserRequest": "Find an orthopedic provider",
            "conversationHistory": [{"role": "user", "content": "private history"}] * 20,
            "currentMemberContext": {"name": "Private Member", "appointments": [1, 2]},
            "retrievedChunks": [{"content": "complete retrieved chunk"}] * 6,
            "structuredRecords": [{"provider": "complete provider record"}] * 5,
            "date_of_birth": "1940-01-01",
        },
        duration_ms=12.345,
    )

    event = json.loads(emitted[0])
    assert event["selectedAgent"] == "HealthcareAccessAgent"
    assert event["durationMs"] == 12.35
    assert event["status"] == "started"
    assert "payload" not in event
    assert event["summary"]["conversationHistory"] == {"type": "array", "count": 20}
    assert event["summary"]["currentMemberContext"]["type"] == "object"
    assert event["summary"]["retrievedChunks"] == {"type": "array", "count": 6}
    assert event["summary"]["structuredRecords"] == {"type": "array", "count": 5}
    assert event["summary"]["date_of_birth"] == "[REDACTED]"
    assert "private history" not in emitted[0]
    assert "complete retrieved chunk" not in emitted[0]
    assert "Private Member" not in emitted[0]


def test_flow_event_keeps_tool_status_and_small_error(monkeypatch):
    emitted: list[str] = []
    monkeypatch.setattr(terminal.logger, "info", emitted.append)

    terminal.flow_event(
        "mcp_tool",
        "search_providers",
        "error",
        {"attempt": 2, "error": "temporary failure"},
        duration_ms=8,
    )

    event = json.loads(emitted[0])
    assert event["selectedTool"] == "search_providers"
    assert event["status"] == "error"
    assert event["error"] == "temporary failure"
    assert event["summary"] == {"attempt": 2}

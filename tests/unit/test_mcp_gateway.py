from seniorcare_agents.mcp.gateway import MCPToolGateway


def test_gateway_decodes_multiple_json_text_blocks() -> None:
    blocks = [
        {"type": "text", "text": '{"caseId":"CASE1001","status":"open"}', "id": "lc_1"},
        {
            "type": "text",
            "text": '{"caseId":"CASE1002","status":"completed"}',
            "id": "lc_2",
        },
    ]

    assert MCPToolGateway._decode_text_blocks(blocks) == [
        {"caseId": "CASE1001", "status": "open"},
        {"caseId": "CASE1002", "status": "completed"},
    ]


def test_gateway_decodes_single_json_list_text_block() -> None:
    blocks = [
        {
            "type": "text",
            "text": '[{"caseId":"CASE1001"},{"caseId":"CASE1002"}]',
            "id": "lc_1",
        }
    ]

    assert MCPToolGateway._decode_text_blocks(blocks) == [
        {"caseId": "CASE1001"},
        {"caseId": "CASE1002"},
    ]


def test_gateway_preserves_single_record_list_as_list() -> None:
    blocks = [
        {
            "type": "text",
            "text": '[{"appointmentId":"APT1021","status":"scheduled"}]',
            "id": "lc_1",
        }
    ]

    assert MCPToolGateway._decode_text_blocks(blocks) == [
        {"appointmentId": "APT1021", "status": "scheduled"}
    ]


def test_gateway_preserves_non_json_content_blocks() -> None:
    blocks = [{"type": "text", "text": "plain text", "id": "lc_1"}]

    assert MCPToolGateway._decode_text_blocks(blocks) is blocks

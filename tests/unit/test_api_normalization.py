import json

from seniorcare_agents.api.normalization import unwrap_record_list


def test_unwrap_record_list_accepts_plain_enveloped_and_json_text_results():
    records = [{"caseId": "CASE1001", "status": "open"}]
    assert unwrap_record_list(records) == records
    assert unwrap_record_list({"simulation": True, "data": records}) == records
    assert unwrap_record_list(json.dumps(records)) == records
    assert unwrap_record_list(json.dumps({"data": records})) == records


def test_unwrap_record_list_rejects_non_record_values_without_crashing():
    assert unwrap_record_list("not-json") == []
    assert unwrap_record_list({"caseId": "CASE1001"}) == []
    assert unwrap_record_list(["cases", 1, None]) == []

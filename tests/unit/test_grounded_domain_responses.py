from seniorcare_agents.agents.llm_specialist import AGENT_NAMES, LangGraphSpecialist


def specialist(key: str) -> LangGraphSpecialist:
    instance = object.__new__(LangGraphSpecialist)
    instance.key = key
    instance.name = AGENT_NAMES[key]
    return instance


def test_existing_appointments_are_rendered_from_operational_records() -> None:
    agent = specialist("healthcare")
    result = agent._grounded_read_result(  # noqa: SLF001
        {
            "query": "Show my father's existing doctor appointments",
            "retrieval_results": {
                "list_appointments": [
                    {
                        "appointmentId": "APT1023",
                        "appointmentDate": "2026-09-21",
                        "appointmentTime": "10:30",
                        "reason": "Knee pain consultation",
                        "status": "scheduled",
                        "provider": {
                            "providerName": "Dr. Carter",
                            "specialty": "Orthopedics",
                            "facilityName": "Central Virginia Orthopedics Center",
                            "city": "Henrico",
                            "state": "VA",
                            "zipCode": "23229",
                        },
                    }
                ]
            },
        }
    )
    assert result is not None
    assert "APT1023" in result.summary
    assert "Dr. Carter" in result.summary
    assert "2026-09-21 10:30" in result.summary
    assert result.retrieved_sources == []


def test_metformin_reference_response_uses_structured_openfda_fields() -> None:
    agent = specialist("medication")
    result = agent._grounded_read_result(  # noqa: SLF001
        {
            "query": "Give me official safety information about metformin",
            "retrieval_results": {
                "search_medication_references": [
                    {
                        "medication_id": "MED-FDA-1",
                        "product_ndc": "68462-520",
                        "brand_name": "METFORMIN HYDROCHLORIDE",
                        "generic_name": "metformin hydrochloride",
                        "manufacturer_name": "Example Manufacturer",
                        "dosage_form": "TABLET",
                        "route": ["ORAL"],
                    }
                ]
            },
        }
    )
    assert result is not None
    assert "structured openFDA product records" in result.summary
    assert "68462-520" in result.summary
    assert "not prescribing instructions" in result.summary


def test_meal_discovery_lists_retrieved_program_details() -> None:
    agent = specialist("meals")
    result = agent._grounded_read_result(  # noqa: SLF001
        {
            "query": "Are there meal-assistance programs available for my father?",
            "retrieval_results": {
                "search_meal_services": [
                    {
                        "mealServiceId": "MEAL1007",
                        "serviceName": "Senior Meal Support Program 7",
                        "serviceArea": "Hanover County",
                        "city": "Mechanicsville",
                        "zipCode": "23111",
                        "serviceType": "home_delivered",
                        "minimumAge": 60,
                        "deliveryDays": "Mon/Wed/Fri",
                        "intakeRequired": True,
                    }
                ]
            },
        }
    )
    assert result is not None
    assert "Senior Meal Support Program 7" in result.summary
    assert "Hanover County" in result.summary
    assert "Contact the program directly" in result.summary


def test_medication_name_is_extracted_for_required_mcp_argument() -> None:
    assert (
        LangGraphSpecialist._medication_name_for_query(  # noqa: SLF001
            "Give me official safety information about metformin"
        )
        == "metformin"
    )

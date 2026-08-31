import os

import pytest


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("RUN_LIVE_INTEGRATION_TESTS"), reason="live services are opt-in")
def test_live_services_are_explicitly_opt_in():
    assert os.getenv("NEBIUS_API_KEY")

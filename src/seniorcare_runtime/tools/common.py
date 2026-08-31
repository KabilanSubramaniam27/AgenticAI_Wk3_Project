from typing import Any


def simulated(data: Any) -> dict[str, Any]:
    return {"simulation": True, "externalActionPerformed": False, "data": data}

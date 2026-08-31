from seniorcare_agents.mcp.bootstrap import build_mcp_server
from seniorcare_runtime.config import RuntimeSettings


def run() -> None:
    """Run the independent MCP service over Streamable HTTP."""
    settings = RuntimeSettings()
    server = build_mcp_server(settings)
    server.settings.host = settings.mcp_server_host
    server.settings.port = settings.mcp_server_port
    server.settings.streamable_http_path = settings.mcp_server_path
    server.settings.stateless_http = True
    server.settings.json_response = True
    server.run(transport="streamable-http")

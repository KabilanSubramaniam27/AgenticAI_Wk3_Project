# SeniorCare Connect AI — Execution Runbook

Run commands from the project root unless a section says otherwise:

```bash
cd /Users/kabilansubramaniam/Documents/Workspace/AgenticAI_Wk3_Project
```

This application is a study/demo system. All appointment, ride, refill, meal,
reminder, and case mutations are local simulations. It never books or contacts a
real doctor, pharmacy, transportation provider, meal provider, or emergency service.
The account holder must enter their own DOB and be at least 21. An account can be for
self-care or can represent an adult coordinating for multiple parent/family care recipients.
The UI starts DOB fields empty and never invents these dates.

## 1. One-time installation

Create and activate Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
```

The standard install is intentional. On some macOS systems an editable-install
`.pth` file is marked hidden, which causes `ModuleNotFoundError: seniorcare_agents`.
After changing package code, refresh the installed copy with:

```bash
python -m pip install . --no-deps --no-build-isolation --force-reinstall
```

Create the local environment file once:

```bash
cp .env.example .env
```

Populate at least the applicable values in `.env`:

```env
NEBIUS_API_KEY=<Nebius API key>
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1
NEBIUS_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
EMBEDDING_DIMENSION=4096

ACTIAN_VECTORAI_URL=localhost:6574
ACTIAN_VECTORAI_REST_URL=http://localhost:6573
ACTIAN_VECTORAI_COLLECTION=seniorcare_knowledge
ACTIAN_VECTORAI_ACCESS_TOKEN=

LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=<OpenAI API key>
LLM_BASE_URL=
LLM_TEMPERATURE=0

SIMULATION_MODE=true
ALLOW_EXTERNAL_MUTATIONS=false

MCP_SERVER_URL=http://127.0.0.1:8001/mcp
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8001
MCP_SERVER_PATH=/mcp
```

`NEBIUS_API_KEY` configures embeddings only. OpenAI powers the agent/chat layer, so set
`OPENAI_API_KEY` separately. `gpt-4o-mini` supports the structured tool-calling workflow.
If either value is missing, chat returns a structured `blocked` response and does not
execute an agent or proposed action.

Never commit `.env`.

## 2. Start and verify Actian VectorAI

Start Actian from the project directory containing `docker-compose.yml`:

```bash
docker compose up -d
docker compose ps
docker compose logs --tail=100 vectorai
```

Local endpoints:

- REST: `http://127.0.0.1:6573`
- gRPC/SDK: `localhost:6574`
- Actian UI: `http://127.0.0.1:6575`
- Collection: `seniorcare_knowledge`

Check Actian and the configured collection without making a paid Nebius call:

```bash
seniorcare-ingest health
```

Also test Nebius by creating one paid query embedding:

```bash
seniorcare-ingest health --check-nebius
```

If Docker reports that port `6573` is already allocated, find the owner before
changing the Compose ports:

```bash
lsof -nP -iTCP:6573 -sTCP:LISTEN
docker ps --format 'table {{.Names}}\t{{.Ports}}'
```

## 3. Collect and prepare public knowledge

List configured sources and inspect stale sources:

```bash
seniorcare-ingest sources list
seniorcare-ingest sources stale --days 30
```

Collect all enabled sources, or one source:

```bash
seniorcare-ingest collect
seniorcare-ingest collect --source cms_doctors
seniorcare-ingest collect --source cms_doctors --force
```

Raw artifacts are preserved under `data/raw/`. To rebuild normalized records and
chunks from the collected data:

```bash
seniorcare-ingest normalize
seniorcare-ingest clean
seniorcare-ingest deduplicate
seniorcare-ingest chunk
seniorcare-ingest validate
seniorcare-ingest stats
```

`clean` and `deduplicate` currently rebuild through the same deterministic
normalization pipeline. The generated RAG chunks are stored in:

```text
data/processed/chunks.jsonl
```

Synthetic operational data under `data/synthetic-data/` is intentionally excluded
from chunking and Actian.

## 4. Create embeddings and store them in Actian

Run the recommended resumable command:

```bash
seniorcare-ingest embed --resume
```

The implementation streams each batch through this sequence:

```text
chunks.jsonl
  → Nebius Qwen/Qwen3-Embedding-8B
  → validate exactly 4096 finite values
  → upsert vector and source metadata into Actian
  → update ingestion manifest
```

Embeddings are not written to JSON. The `embed` and `index` CLI commands currently
invoke the same combined embed-and-index operation. Therefore, the command above both
creates embeddings and stores them in Actian. Running this afterward is valid but
normally indexes zero additional records because resume is enabled:

```bash
seniorcare-ingest index --resume
```

Preview pending work without calling Nebius or writing to Actian:

```bash
seniorcare-ingest embed --resume --dry-run
```

Run all acquisition and processing stages end to end:

```bash
seniorcare-ingest ingest --resume
```

Other useful variants:

```bash
seniorcare-ingest ingest --dry-run
seniorcare-ingest ingest --source cms_doctors --resume
seniorcare-ingest ingest --source cms_doctors --force
```

## 5. Inspect the Actian collection

Open the Actian UI in a browser:

```bash
open http://127.0.0.1:6575
```

Select the `seniorcare_knowledge` collection to inspect collection information and
stored points when supported by the installed Actian UI.

Print the collection list and vector count using the installed Actian SDK:

```bash
python -c "from seniorcare_ingestion.config import get_settings; from actian_vectorai import VectorAIClient; s=get_settings(); c=VectorAIClient(url=s.actian_vectorai_url); print(c.collections.list()); c.close()"
```

```bash
python -c "from seniorcare_ingestion.config import get_settings; from seniorcare_ingestion.vectorstore import ActianVectorStore; s=get_settings(); print({'collection': s.actian_vectorai_collection, 'vectors': ActianVectorStore(s).count(s.actian_vectorai_collection)})"
```

View source-attributed stored payloads through semantic search. This creates a paid
Nebius query embedding but does not mutate the collection:

```bash
seniorcare-ingest search \
  "medical transportation for a wheelchair user in Henrico County" \
  --top-k 10 \
  --category transportation \
  --state Virginia \
  --county "Henrico County"
```

## 6. Start the Deep Agents application

The services must run in separate terminals. Activate `.venv` in every new terminal.

Terminal 1 — Actian:

```bash
cd /Users/kabilansubramaniam/Documents/Workspace/AgenticAI_Wk3_Project
docker compose up -d
```

Terminal 2 — independent MCP server over Streamable HTTP:

```bash
cd /Users/kabilansubramaniam/Documents/Workspace/AgenticAI_Wk3_Project
source .venv/bin/activate
seniorcare-mcp
```

Expected endpoint: `http://127.0.0.1:8001/mcp`.

Terminal 3 — agent API and application service:

```bash
cd /Users/kabilansubramaniam/Documents/Workspace/AgenticAI_Wk3_Project
source .venv/bin/activate
seniorcare-api
```

Expected endpoints:

- API: `http://127.0.0.1:8000`
- API documentation: `http://127.0.0.1:8000/docs`
- API health: `http://127.0.0.1:8000/health`
- MCP tools discovered by the agents: `http://127.0.0.1:8000/mcp/tools`

Verify the API, cached startup agents, and remote MCP discovery:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/mcp/tools
```

Terminal 4 — Streamlit UI:

```bash
cd /Users/kabilansubramaniam/Documents/Workspace/AgenticAI_Wk3_Project
source .venv/bin/activate
seniorcare-ui
```

Open the SeniorCare UI:

```bash
open http://127.0.0.1:8501
```

The UI calls `http://127.0.0.1:8000` by default. To use another API address:

```bash
SENIORCARE_API_URL=http://127.0.0.1:8000 seniorcare-ui
```

## 7. Check MCP health with MCP Inspector UI

MCP Inspector requires Node.js. On macOS with Homebrew:

```bash
brew install node
node --version
npm --version
npx --version
```

Keep `seniorcare-mcp` running in its own terminal, then start Inspector in another:

```bash
npx @modelcontextprotocol/inspector@latest
```

Inspector prints a local URL similar to:

```text
http://127.0.0.1:6274/?MCP_INSPECTOR_API_TOKEN=<temporary-token>
```

Open that exact generated URL. The token authorizes the browser to communicate with
the local Inspector proxy; it is not a SeniorCare API key. Restart Inspector if the
token was exposed or is no longer accepted.

In Inspector, add or select a server with:

```text
Server name: SeniorCare
Transport:   Streamable HTTP (sometimes displayed as HTTP)
URL:         http://127.0.0.1:8001/mcp
Headers:     none
```

Do not select STDIO or SSE. The built-in `filesystem-server-default` and
`everything-server-default` entries are Inspector examples, not SeniorCare servers.

Click **Connect**, open **Tools**, click **List Tools**, select `server_status`, and
run it. A healthy response resembles:

```json
{
  "status": "OK",
  "server": "seniorcare-connect",
  "simulation": true,
  "externalMutationsAllowed": false,
  "ragChunks": 100
}
```

The actual `ragChunks` count depends on the local corpus. Successful connection, tool
discovery, and a successful `server_status` call confirm MCP health.

Opening `http://127.0.0.1:8001/mcp` directly in a normal browser may return:

```text
Not Acceptable: Client must accept text/event-stream
```

That response is expected because a browser navigation is not an MCP protocol
session. Test this endpoint with Inspector or the application MCP client.

## 8. Evaluations and tests

Run offline deterministic routing evaluations:

```bash
seniorcare-eval
```

The offline report includes intent accuracy, agent-selection accuracy, and recipient
guardrail accuracy from `evals/golden_questions.json`. The live benchmark dataset adds
self-care, family-representative, recipient mismatch, and representative/self ambiguity
cases. Human review includes a separate `recipient_correctness` rating.

With MCP, Actian, and the tool-calling LLM configured and running:

```bash
seniorcare-eval --agent-benchmarks
seniorcare-eval --live-retrieval
```

Optional paid LLM-as-judge and human-review workflows:

```bash
seniorcare-eval --agent-benchmarks --llm-judge
seniorcare-eval --agent-benchmarks --prepare-human-review
# Complete data/runtime/human_eval.jsonl, then:
seniorcare-eval --score-human-review
```

Run and preserve live integration evidence after MCP, Actian, and the API credentials are ready:

```bash
seniorcare-eval --agent-benchmarks --llm-judge --live-retrieval --live-validation
jq . data/runtime/live_validation_evidence.json
jq .successCriteria data/runtime/eval_report.json
```

`live_validation_evidence.json` separately reports MCP health, configured OpenAI model status,
BM25 results, and whether the Nebius embedding plus Actian vector path actually returned results.
It never treats lexical fallback as a successful vector integration.

Restart-safe agent state is stored in:

```text
data/runtime/agent_sessions.json
data/runtime/pending_actions.json
```

Do not delete these files while approvals are pending. `SESSION_TTL_MINUTES` controls conversation
session expiry. Read retries are configured with `MCP_READ_MAX_ATTEMPTS` and
`MCP_READ_RETRY_BASE_SECONDS`; approved writes are intentionally attempted only once.

Run project tests and configured quality checks:

```bash
pytest
ruff check src tests
mypy src
```

## 9. Shutdown

Stop MCP, API, Streamlit, or Inspector with `Ctrl+C` in their respective terminals.
Stop Actian when finished:

```bash
docker compose down
```
# Unified application flow log

The UI, API, orchestrator, specialist agents, LLM, MCP tools, approvals, and RAG pipeline
write compact correlated events to `application.log` in the project root. The same events
remain visible in their service terminals.

Follow the complete flow while using the application:

```bash
cd /Users/kabilansubramaniam/Documents/Workspace/AgenticAI_Wk3_Project
tail -f application.log
```

Filter one request across all components by its correlation ID:

```bash
rg 'REQ-REPLACE_WITH_ID' application.log
```

The optional `APPLICATION_LOG_PATH` setting changes the destination. Relative paths are
resolved from the project root. Birth dates, credentials, authorization values, tokens,
passwords, and embedding arrays are redacted.

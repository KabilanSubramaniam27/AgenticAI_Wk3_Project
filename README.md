# SeniorCare Connect AI — Public Knowledge Ingestion

This Python 3.12+ project builds the public, source-attributed RAG knowledge layer for SeniorCare Connect AI. It collects approved official sources, preserves raw responses, parses and normalizes content, cleans and deduplicates it, creates structure-aware chunks, embeds with Nebius Token Factory (`Qwen/Qwen3-Embedding-8B`, exactly 4096 dimensions), and indexes into Actian VectorAI.

It also contains a LangGraph study application with nine agents: the
`SeniorCareOrchestratorAgent`, existing `MemberCaseAgent`, healthcare,
transportation, medication/pharmacy, meals/food, social well-being, home
support/safety, and case-status/risk agents. All mutations remain local and require
human approval.

Agent capabilities are exposed through an independent Streamable HTTP MCP service.
The FastAPI/LangGraph process uses `MultiServerMCPClient`, dynamically discovers
MCP tools, and cannot access domain repositories or RAG directly.

## Deep multi-agent architecture

```text
Input guardrails → member resolution → Orchestrator planning LLM
                                      → validated execution stages
                                                    ├─ HealthcareAccessAgent
                                                    ├─ TransportationAgent
                                                    ├─ MedicationPharmacyAgent
                                                    ├─ MealsFoodAgent
                                                    ├─ SocialWellbeingAgent
                                                    ├─ HomeSupportSafetyAgent
                                                    └─ CaseStatusRiskAgent
                         → Orchestrator synthesis LLM → output guardrails → approval interrupt
```

Every specialist is a startup-compiled explicit LangGraph subgraph. It produces a typed LLM tool
plan, validates it, executes the selected MCP reads, validates retrieval, and uses a separate
LangChain structured-output LLM call to synthesize a grounded `AgentResult`. Each
receives only its explicit read-tool allowlist from the discovered MCP tools:

```text
Orchestrator LLM → specialist planning/synthesis → MultiServerMCPClient → MCP server
                                      ├─ member/case context
                                      ├─ appointments/providers
                                      ├─ rides/transportation
                                      ├─ medications/refills
                                      ├─ meals/social/home support
                                      ├─ risk/audit
                                      └─ hybrid public-knowledge retrieval
```

The independently running MCP server delegates to the existing runtime repositories,
Actian/Nebius retrieval, and simulation tools. Agents and FastAPI endpoints do not
import those repositories directly. Approved writes cross MCP only after the approval
manager verifies approval and asks MCP to validate member/case ownership.
Write tools are never supplied to an LLM. Specialists return structured simulated
proposals; only the approval manager can invoke a write tool after user approval.

During FastAPI lifespan startup, MCP tools are discovered once and each configured
specialist graph is compiled once. These stateless compiled graphs are reused across
requests. User ID, case ID, messages, evidence, and approvals remain invocation/session
state and are never stored in a shared specialist prompt or object.

Conversation, selected-recipient, active-case, message, and pending-action mappings are atomically
persisted to `data/runtime/agent_sessions.json`; approval proposals are persisted to
`data/runtime/pending_actions.json`. They are restored after an API restart and expire according to
`SESSION_TTL_MINUTES`. Safe MCP reads retry with bounded exponential backoff, but approved writes
are never automatically retried. A failed specialist becomes a structured failed result so other
specialists in the stage can still complete.

Guardrails execute at four boundaries: global input/emergency screening, specialist
input validation, specialist structured-output validation, and orchestrator
selection/result validation. They reject invalid member IDs, oversized/empty input,
clinical claims, LLM write calls, unapproved or cross-member actions, unattributed
retrieval results, duplicate actions, and unexpected specialists.

Public-resource retrieval uses one shared pipeline:

```text
canonical chunks → BM25 ─┐
                         ├→ reciprocal-rank fusion → cross-encoder → top results
query → Nebius 4096d → Actian ┘
```

BM25 is lexical retrieval; Actian is dense semantic retrieval. Raw scores are never
added together. RRF uses rank positions, and the local sentence-transformers
cross-encoder performs final reranking. Structured CMS providers and openFDA NDC
records remain repository lookups. Member operational state remains synthetic JSON.

LangGraph writes are interrupted and persisted as proposed actions. Approval checks
the member/case relationship and simulation configuration, executes only a local
tool, attaches its entity ID to the active case, and audits the event. Registration
and explicitly submitted case forms count as direct user approval in this demo.

Run the MCP server, API, and UI in separate terminals:

```bash
seniorcare-mcp
seniorcare-api
streamlit run src/seniorcare_agents/ui/app.py
```

FastAPI exposes member registration/lookup, cases, chat, approvals, rejection, and
health endpoints. `GET /mcp/tools` shows MCP tool discovery. Retrieval traces are
written under `data/runtime/`; the golden eval
set is in `evals/golden_questions.json`. The demo User ID lookup is not production
authentication.

The MCP server listens at `http://127.0.0.1:8001/mcp` by default using Streamable
HTTP. Configure the processes with:

```env
MCP_SERVER_URL=http://127.0.0.1:8001/mcp
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8001
MCP_SERVER_PATH=/mcp
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=<your OpenAI API key>
LLM_BASE_URL=
LLM_TEMPERATURE=0
```

The files in `data/synthetic-data/` are operational state for future agents. They are deliberately never collected, chunked, embedded, or stored in Actian.

### Agent evaluations

The measurable project target is completion of a coordinated request in under five minutes with
a usable outcome in at least eight of ten evaluated scenarios. Machine-readable thresholds live in
`evals/success_criteria.json`; every evaluation report records measured values and pass/fail status.

The default evaluation scores the actual orchestrator planning path against the golden routing,
agent-selection, parallel/sequential-stage, and recipient-policy cases. Start the MCP server first.
With OpenAI configured it uses the orchestrator LLM and consumes tokens; without LLM credentials it
scores the explicitly logged deterministic fallback router:

```bash
seniorcare-eval
```

Run the multi-case-per-specialist benchmark suite, including self-care, family
representative, and recipient-mismatch cases, against the running MCP server and
configured chat model:

```bash
seniorcare-eval --agent-benchmarks
```

Add the opt-in LLM-as-judge rubric (this makes additional paid model calls):

```bash
seniorcare-eval --agent-benchmarks --llm-judge
```

Create timestamped evidence for the live MCP, configured OpenAI model, BM25 corpus, and
Nebius-to-Actian semantic path (external services and paid calls are attempted):

```bash
seniorcare-eval --agent-benchmarks --llm-judge --live-retrieval --live-validation
```

The evidence is written to `data/runtime/live_validation_evidence.json`. A lexical RAG success does
not count as Nebius/Actian success: `nebiusActianPassed` is true only when the live retrieval trace
contains vector results and no vector error.

Generate human-review JSONL packets, then score them after reviewers fill the
1–5 ratings, approval, reviewer, comments, and timestamp fields:

```bash
seniorcare-eval --agent-benchmarks --prepare-human-review
seniorcare-eval --score-human-review
```

Benchmark definitions live in `evals/agent_benchmarks.json`; reports and human
review packets are written under `data/runtime/`. Normal unit tests never call the
LLM judge or consume Nebius credits.

## Architecture

```text
Official Data Sources
        │
        ├──── Bulk Download
        ├──── API
        ├──── HTML
        └──── PDF
                 │
                 ▼
         Acquisition Layer
                 │
                 ▼
          Raw Preservation
                 │
                 ▼
              Parser
                 │
                 ▼
            Normalizer
                 │
                 ▼
              Cleaner
                 │
                 ▼
          Deduplicator
                 │
                 ▼
        Geographic Tagger
                 │
                 ▼
              Chunker
                 │
                 ▼
        Nebius Token Factory
      Qwen/Qwen3-Embedding-8B
              4096d
                 │
                 ▼
          Actian VectorAI
                 │
                 ▼
        Semantic Retrieval
```

Acquisition uses the preferred method from `config/sources.yaml`. Transient HTTP failures are retried; after exhaustion, or after deterministic failures such as 404, the next configured fallback is used and recorded in the manifest. Domains not explicitly allowlisted are rejected. HTML/PDF access checks robots rules, and raw bytes are saved under `data/raw/<source>/` using retrieval time and content hash rather than overwritten.

The curated registry covers CMS providers, Virginia DMAS transportation, GRTC CARE paratransit, Virginia Easy Access aging/food/home-support/safety/social resources, DARS and the Virginia AAA directory, Medicare Extra Help and Savings Programs, Medicare discharge and home-health guidance, CommonHelp food assistance, and openFDA NDC medication reference data.

Large structured datasets are deliberately separated from RAG documents:

```text
data/normalized/providers.jsonl     # CMS providers; not embedded
data/normalized/medications.jsonl   # openFDA NDC records; not embedded
data/normalized/resources.jsonl     # queryable community-resource fields
data/normalized/documents.jsonl     # explanatory RAG documents
data/processed/chunks.jsonl         # only RAG documents become chunks
```

The vector corpus is restricted by `VECTOR_CATEGORIES` to healthcare access,
transportation, medication reference, discharge support, food/meals, benefits,
home support, caregiver support, and social well-being. Structured provider and
medication records stay outside the vector corpus, so categories without explanatory
documents may legitimately have zero chunks. Case tasks, reminders, appointments, rides, refills, and other
synthetic operational records remain structured agent state and are never
embedded. Re-indexing removes previously indexed chunk IDs that are no longer
part of the selected corpus.

Healthcare and medication use complementary structured and RAG retrieval through
MCP:

```text
HealthcareAccessAgent
  ├─ search_providers                 → structured CMS/local provider records
  └─ search_healthcare_knowledge      → healthcare_access RAG in Actian

MedicationPharmacyAgent
  ├─ list_medications / list_refills  → member synthetic operational state
  ├─ search_medication_references     → structured openFDA NDC references
  └─ search_medication_knowledge      → medication_reference RAG in Actian
```

The category-specific knowledge tools enforce their category on the MCP server;
the LLM cannot substitute an unrelated category. Provider/NDC rows remain outside
Actian, while official Medicare/FDA explanatory guidance is chunked and embedded.

## Simulation runtime for future agents

`seniorcare_runtime` provides repositories over the synthetic JSON and safe local
tools for appointments, rides, refills, meals, reminders, events, and home-support requests.
Provider and member/recipient reads are exposed directly as validated MCP resources/tools rather
than duplicated wrapper classes. These capabilities never contact real providers or service organizations.
Every mutation requires `SIMULATION_MODE=true` and
`ALLOW_EXTERNAL_MUTATIONS=false`, writes only under `data/synthetic-data/`, returns
`externalActionPerformed: false`, and appends an event to
`data/runtime/audit.jsonl`.

```python
from seniorcare_runtime import RuntimeSettings
from seniorcare_runtime.services import RiskDetectionService, SeniorContextService
from seniorcare_runtime.tools import AppointmentTools

settings = RuntimeSettings()
context = SeniorContextService(settings).get_context("SEN1001")
risks = RiskDetectionService(settings).evaluate("SEN1001")
slots = AppointmentTools(settings).find_available_slots("PRV1001")
```

Phase 1 exposes approved local writes only for appointments, initial ride bookings,
home-support requests, member/case coordination, and case status changes. Medication,
meal, and social-activity capabilities are read-only discovery workflows. Tests use
temporary copies and never mutate the project datasets.

Application flow logs are deliberately compact. Each JSON event retains correlation,
component, operation, direction, status, optional agent/tool and duration, a small
input/output summary, or an error. Full prompts, conversation history, member context,
retrieved chunks, and structured tool arrays are represented only by counts and shapes.
Tune bounded summaries with `APPLICATION_LOG_MAX_FIELDS` and
`APPLICATION_LOG_MAX_STRING`.

### Member and case agent

`MemberCaseAgent` is the local conversation-facing entry point for account onboarding
and case history. An account holder must supply their own name and DOB and be at least
21 years old. The account can represent the senior directly (`self_care`) or an adult
son, daughter, spouse, family member, or caregiver coordinating for one or more care
recipients (`family_representative`). Registration creates the first recipient, and the
dashboard can add more. The user receives an opaque `SEN...` User ID and can use it
to list prior `CASE...` records. DOB values remain in local synthetic storage for
validation and duplicate prevention and are omitted from normal responses, MCP member
lookups, case snapshots, operational records, and audit events.

```python
from seniorcare_runtime import RuntimeSettings
from seniorcare_runtime.agents import MemberCaseAgent

agent = MemberCaseAgent(RuntimeSettings())
welcome = agent.start()
member = agent.register_member({
    "first_name": "Grace",
    "last_name": "Hopper",
    "date_of_birth": "1936-12-09",
})
user_id = member["data"]["userId"]
case = agent.create_case(user_id, {
    "recipient_id": member["data"]["member"]["careRecipients"][0]["recipientId"],
    "case_type": "appointment_coordination",
    "title": "Arrange cardiology visit",
    "description": "Create and track a simulated appointment request",
    "priority": "high",
})
history = agent.start(user_id)
```

For a representative account, add `care_for="family_member"`, the care recipient's
relationship to the account holder (`father`, `mother`, `parent`, `spouse`, or other),
and the three `care_recipient_*` fields. Additional recipients are added through
`POST /members/{user_id}/care-recipients`. Before chat, the UI requires one recipient
to be selected. New cases
and simulated domain records keep the User ID as their ownership key and also store a
non-sensitive care-recipient snapshot and stable `recipientId`. This supports separate
requests for self, father, mother, or other registered recipients while preventing
ambiguous and cross-account recipient changes.

For a new coordination request, one approval authorizes the complete local simulated
plan: the approval manager first creates the tracking case, executes the selected
simulated domain action, and links its appointment, ride, refill, reminder, or enrollment
ID to that case. The original query is not rerun and no duplicate approval is requested.
This provides one coordination case number while retaining exact domain record IDs.
User ID alone is only a study/demo identity mechanism; production
use would require real authentication and authorization.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
cp .env.example .env
```

This project uses a standard local install because some macOS environments mark
setuptools editable-install `.pth` files as hidden; Python then skips `src/` and
commands such as `seniorcare-mcp` fail with `ModuleNotFoundError`. Re-run the install
command after changing package code. If an older editable install is already present,
replace it with:

```bash
python -m pip install . --no-deps --no-build-isolation --force-reinstall
```

Populate `NEBIUS_API_KEY`. Change `ACTIAN_VECTORAI_URL` or its access token only if your Actian deployment requires it. Never commit `.env`.

Agent chat uses OpenAI `gpt-4o-mini` and requires `OPENAI_API_KEY`. Nebius remains the
embedding provider for RAG indexing and semantic-query vectors. Keep the OpenAI and
Nebius credentials separate and never commit `.env`.

Start Actian and inspect health:

```bash
docker compose up -d
seniorcare-ingest health
seniorcare-ingest health --check-nebius
```

The second command performs a paid live embedding only when explicitly requested and a key is configured.

## Commands

```bash
seniorcare-ingest sources list
seniorcare-ingest sources stale --days 30
seniorcare-ingest collect --source virginia_dmas_transportation
seniorcare-ingest normalize
seniorcare-ingest clean
seniorcare-ingest deduplicate
seniorcare-ingest chunk
seniorcare-ingest embed --resume
seniorcare-ingest index --resume
seniorcare-ingest validate
seniorcare-ingest stats
seniorcare-ingest ingest --dry-run
seniorcare-ingest ingest --source cms_doctors
seniorcare-ingest ingest --resume
seniorcare-ingest ingest --source cms_doctors --force
seniorcare-ingest search "wheelchair transportation in Henrico County" --category transportation --state Virginia --county "Henrico County"
```

Use `seniorcare-ingest sources list` to see exact source IDs. `--resume` consults per-chunk hashes in `data/manifests/ingestion_manifest.json`, so successfully indexed batches are skipped after interruption. Stable IDs and Actian upserts make reruns idempotent. Embeddings are streamed to Actian and are not saved to JSON.

## Adding sources

Add the official domain to `allowed_domains`, then add a source with one preferred acquisition method and optional fallbacks. Use `api`/`download` for structured data, `html` when no equivalent structured source exists, and `pdf` for explanatory documents. Generic URLs belong only in YAML. A source-specific structured normalizer should be added for schemas whose fields matter independently (CMS providers are the first example).

## Validation and tests

```bash
pytest -m 'not integration'
ruff check .
```

Live integrations are opt-in and never run in normal unit tests:

```bash
RUN_LIVE_INTEGRATION_TESTS=1 pytest -m integration
```

Validation checks empty content, attribution, categories, timestamps, hashes, and duplicate IDs. Embedding validation fails immediately on count mismatch, empty vectors, non-finite values, or any dimension other than 4096.

## Data safety, freshness, and troubleshooting

Only public organizational information is collected; the collector does not authenticate to private portals, submit forms, bypass CAPTCHAs, or collect real senior personal data. Trust tiers 1–3 are intended for normal ingestion. Use `sources stale` to identify old or never-retrieved sources.

- `NEBIUS_API_KEY is required`: populate `.env`; collection and local processing still work without it.
- Actian unavailable: confirm `docker compose ps`, ports 6573–6575, and the EULA setting.
- Robots disallowed: the source is skipped and recorded; do not bypass it.
- Unexpected schema: preserve the raw artifact, add/update a source-specific normalizer, and rerun with `--force`.



Quick commands to start the application:

Start the complete application

Terminal 1 — Actian VectorAI
cd /Users/kabilansubramaniam/Documents/Workspace/AgenticAI_Wk3_Project
docker compose up -d
docker compose ps
Verify Actian:
source .venv/bin/activate
seniorcare-ingest health

Terminal 2 — MCP server
cd /Users/kabilansubramaniam/Documents/Workspace/AgenticAI_Wk3_Project
source .venv/bin/activate
seniorcare-mcp
Leave it running. Endpoint:
http://127.0.0.1:8001/mcp

Terminal 3 — Deep Agents API
cd /Users/kabilansubramaniam/Documents/Workspace/AgenticAI_Wk3_Project
source .venv/bin/activate
seniorcare-api
Leave it running. Verify from another terminal:
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/mcp/tools
API documentation:
http://127.0.0.1:8000/docs

Terminal 4 — SeniorCare Streamlit UI
cd /Users/kabilansubramaniam/Documents/Workspace/AgenticAI_Wk3_Project
source .venv/bin/activate
streamlit run src/seniorcare_agents/ui/app.py
Open the application:
open http://127.0.0.1:8501
Run MCP Inspector
Keep seniorcare-mcp running in Terminal 2.

Terminal 5 — MCP Inspector
Verify Node.js:
node --version
npm --version
npx --version
Start Inspector:
npx @modelcontextprotocol/inspector@latest
Open the exact URL printed by Inspector. It will resemble:
http://127.0.0.1:6274/?MCP_INSPECTOR_API_TOKEN=<temporary-token>
In the Inspector UI configure:
Server name: SeniorCare
Transport:   Streamable HTTP
URL:         http://127.0.0.1:8001/mcp
Headers:     none
Then:
1. Click Connect.
2. Open Tools.
3. Click List Tools.
4. Select server_status.
5. Click Run Tool.
Expected response:
{
  "status": "OK",
  "server": "seniorcare-connect",
  "simulation": true,
  "externalMutationsAllowed": false,
  "ragChunks": 0
}
The chunk count may be higher.
Minimum application startup
For normal use after initial setup, these are the essential commands:
docker compose up -d
seniorcare-mcp
seniorcare-api
streamlit run src/seniorcare_agents/ui/app.py
Run the final three commands in separate activated terminals.
Shutdown
Stop MCP, API, Streamlit, and Inspector with Ctrl+C in their terminals.
Stop Actian:
docker compose down

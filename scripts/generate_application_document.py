"""Generate SeniorCare Connect architecture documentation as HTML and PDF.

The HTML is intentionally self-contained enough for macOS textutil to convert it to DOCX.
The PDF is rendered with the project's existing PyMuPDF dependency.
"""

from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import pymupdf
from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
TITLE = "SeniorCare Connect — Application Design & Technical Architecture"


def esc(value: object) -> str:
    return html.escape(str(value))


def table(headers: list[str], rows: list[list[object]], caption: str) -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(item)}</td>" for item in row) + "</tr>" for row in rows
    )
    return f'<p class="caption">{esc(caption)}</p><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def pre(text: str, caption: str) -> str:
    return f'<p class="caption">{esc(caption)}</p><pre>{esc(text.strip())}</pre>'


def image(path: str, caption: str) -> str:
    return f'<figure><img src="{path}"/><figcaption>{esc(caption)}</figcaption></figure>'


def section(number: str, title: str, body: str, page_break: bool = False) -> str:
    css = ' class="page-break"' if page_break else ""
    return f'<section{css}><h1 id="s{number.replace(".", "-")}">{esc(number)}. {esc(title)}</h1>{body}</section>'


def build_html() -> str:
    areas = [
        ["1", "Doctor appointments & healthcare access", "HealthcareAccessAgent", "Appointments, referrals", "CMS providers", "healthcare_access", "Appointment", "Yes"],
        ["2", "Medical & community transportation", "TransportationAgent", "Rides, vehicles", "—", "transportation", "Ride", "Yes"],
        ["3", "Medication & pharmacy coordination", "MedicationPharmacyAgent", "Medications, refills", "openFDA NDC", "medication_reference", "Read-only Phase 1", "No"],
        ["4", "Hospital discharge & follow-up", "HomeSupportSafetyAgent", "Discharge tasks", "—", "discharge_support", "Related support request", "If proposed"],
        ["5", "Meals & food assistance", "MealsFoodAgent", "Meal services", "—", "food_meals", "Read-only Phase 1", "No"],
        ["6", "Benefits & financial assistance", "HomeSupportSafetyAgent", "Benefit applications", "—", "benefits_financial", "Guidance", "No"],
        ["7", "Home support & safety", "HomeSupportSafetyAgent", "Home-support requests", "—", "home_support", "Support request", "Yes"],
        ["8", "Caregiver & family coordination", "HomeSupportSafetyAgent", "Caregivers, tasks", "—", "caregiver_support", "Guidance", "No"],
        ["9", "Social connection & well-being", "SocialWellbeingAgent", "Social activities", "—", "social_wellbeing", "Read-only Phase 1", "No"],
        ["10", "Case tracking, reminders & risk", "CaseStatusRiskAgent", "Cases, reminders, audit", "—", "Operational only", "Status update", "Yes"],
    ]
    statuses = [
        ["Public ingestion and raw preservation", "Implemented", "Registry, allowlist, raw artifacts, normalized JSONL and manifests exist."],
        ["Nebius 4096-d embedding", "Implemented; live availability conditional", "Runtime validates model output; paid calls require configured credentials."],
        ["Actian vector indexing/search", "Implemented; service-dependent", "Collection abstraction and hybrid retrieval exist; current availability depends on Docker/service state."],
        ["OpenAI agent reasoning", "Implemented; service-dependent", "Configured for gpt-4o-mini with typed planning/synthesis."],
        ["MCP Streamable HTTP", "Implemented", "Independent server at 127.0.0.1:8001/mcp and MultiServerMCPClient gateway."],
        ["Explicit LangGraph workflows", "Implemented", "Eight-node outer graph and six-node reusable specialist subgraphs."],
        ["Self-care and representative care", "Implemented", "21+ account holder, multiple recipients, ownership checks and recipient snapshots."],
        ["Appointment and ride writes", "Implemented as local simulation", "Approval required; no external organization contacted."],
        ["Medication, meal and social writes", "Read-only Phase 1", "Discovery/reference only; direct contact guidance where grounded."],
        ["Production identity, consent and compliance", "Planned / Future Enhancement", "Demo User ID is not production authentication or authorization."],
    ]
    glossary = [
        ["Agent / Deep Agent", "A reusable reasoning component that plans, retrieves through tools and returns a typed result within a graph."],
        ["LLM", "Large language model; OpenAI gpt-4o-mini is configured for agent planning and synthesis."],
        ["LangGraph", "State-machine framework used for the outer workflow and specialist subgraphs."],
        ["LangChain", "Model and MCP primitives used inside LangGraph nodes."],
        ["MCP", "Model Context Protocol, the standardized client/server boundary for tools and resources."],
        ["Streamable HTTP", "The MCP transport used at http://127.0.0.1:8001/mcp."],
        ["RAG", "Retrieval-augmented generation: retrieved evidence is supplied to generation."],
        ["BM25", "Lexical retrieval based on term frequency and document length."],
        ["Dense retrieval", "Semantic vector similarity using Nebius embeddings and Actian VectorAI."],
        ["RRF", "Reciprocal Rank Fusion, combining lexical and dense result ranks."],
        ["Reranker", "A cross-encoder that reorders fused candidates for query relevance."],
        ["Grounding", "Restricting claims to authenticated state or retrieved evidence."],
        ["Guardrail", "Deterministic validation around inputs, plans, tools, outputs and approvals."],
        ["Care recipient", "The selected person receiving care; may be the account holder or a registered family member."],
        ["Case", "A durable local coordination episode linking approved operational records."],
        ["Simulation mode", "Local-only writes that never contact real providers or services."],
        ["Golden dataset", "Curated questions with expected routing, tools, behavior or safety outcomes."],
        ["LLM-as-a-Judge", "Optional model-based evaluation, separated from deterministic tests and human review."],
    ]

    cover = f"""
    <div class="cover">
      <div class="eyebrow">APPLICATION DESIGN &amp; TECHNICAL ARCHITECTURE</div>
      <h1>SeniorCare Connect</h1>
      <h2>Multi-Agent Senior Care Coordination &amp; Resource Navigator</h2>
      {image('images/cover-housing-care-768.jpg', 'Technology-assisted independence, family support, and coordinated senior care.')}
      <div class="cover-meta"><b>Author</b><br/>Kabilan Subramaniam<br/><br/><b>Date</b><br/>08/30/2026</div>
      <div class="classification">ACADEMIC / STUDY DEMONSTRATION · SIMULATION-ONLY OPERATIONAL ACTIONS</div>
    </div>
    """

    toc_items = [
        "Executive Summary", "Introduction and Social Value", "Users and Recipient Model",
        "Supported Care Areas", "Goals, Non-Goals and Personas", "Registration and Cases",
        "Overall Architecture", "MCP Architecture", "LangGraph Deep-Agent Workflow",
        "Ingestion Subsystem", "Data, Chunking and Embeddings", "Hybrid RAG",
        "Agent Topology and Tool Boundaries", "Conversation and Grounding", "Approval and Simulation",
        "UI", "Guardrails and Privacy", "Observability and Audit", "Evaluation",
        "Runbook and Troubleshooting", "Implementation Status", "Package and File Reference",
        "Roadmap", "Glossary"
    ]
    toc = '<div class="page-break"><h1>Table of Contents</h1><ol class="toc">' + "".join(
        f"<li>{esc(item)}</li>" for item in toc_items
    ) + '</ol><p class="note">The PDF includes navigation bookmarks. Section numbering below is authoritative.</p></div>'

    parts = [cover, toc]
    parts.append(section("1", "Executive Summary", f"""
      <p>SeniorCare Connect is a study-oriented coordination application for older adults and adult
      family representatives. It brings together public senior-care knowledge, structured provider
      and medication records, and local operational records behind a conversational interface. A
      LangGraph orchestrator asks an LLM to plan which specialist agents should participate;
      specialists then discover and call least-privilege MCP read tools, validate evidence, and
      return typed results. Any supported local write remains a proposal until the user explicitly
      approves it.</p>
      <blockquote><b>Project success definition:</b> SeniorCare Connect AI helps older adults and family caregivers coordinate healthcare and community-support requests through a web application, replacing fragmented calls, searches, and manual tracking; it autonomously searches public knowledge and local operational records through MCP tools, hands every proposed local write to the user for approval, and succeeds when a user can create and track a coordinated request in under five minutes with a usable outcome in at least eight out of ten evaluated scenarios.</blockquote>
      <p><b>Evidence position.</b> The repository demonstrates the architecture and has strong code-based
      validation. The five-minute and eight-of-ten targets remain project success criteria; they are
      not claimed as achieved because a completed representative human-evaluation study was not found.
      The latest local run completed 106 tests with one live-service test skipped. A recorded routing
      report evaluated 15 items with 1.00 intent and agent-selection accuracy, but this is not a
      substitute for end-user outcome measurement.</p>
      {table(['Concern','Design response'],[
        ['Fragmented information','Hybrid public RAG plus structured CMS/openFDA lookup.'],
        ['Fragmented operational tracking','Recipient-scoped appointments, rides, cases, reminders and tasks.'],
        ['Unsafe autonomous mutation','Human approval and simulation-only MCP writes.'],
        ['LLM hallucination risk','Typed schemas, deterministic validation and grounded-response precedence.'],
        ['Caregiver coordination','Representative accounts with multiple stable recipient IDs.'],
      ],'Table 1. Executive design responses')}
    """, True))

    parts.append(section("2", "Introduction and Social Value", f"""
      {image('images/Copy-of-senior-care-2-1024x683.jpg', 'Figure 1. Senior independence is supported by accessible coordination and human oversight.')}
      <p>Older adults and caregivers frequently navigate separate systems for clinicians, transport,
      medication information, discharge instructions, nutrition, benefits, home safety, social
      activity and follow-up. SeniorCare Connect provides one coordination surface without presenting
      itself as clinical decision support. It can retrieve options and stored status, explain public
      guidance with citations, and prepare local study actions for approval.</p>
      {image('images/istockphoto-2159172849-612x612.jpg', 'Figure 2. Adult family representatives can coordinate for more than one registered care recipient.')}
      <p>The intended social value is reduced administrative burden, stronger visibility into pending
      work, and better access to attributed resources. The application emphasizes independence,
      accessibility, social connection and caregiver support. It does not diagnose, prescribe,
      dispatch emergency services, or transact with real organizations.</p>
    """))

    parts.append(section("3", "Self-Care and Representative-Assisted Care", f"""
      {pre('''SELF-CARE                              REPRESENTATIVE CARE
Account holder (21+)                  Adult account holder (21+)
        |                                      |
accountRole = self                    accountRole = family_representative
        |                                      |
recipient = account holder            careRecipients[]
                                               +-- recipientId
                                               +-- name
                                               +-- relationship
                                               +-- selected recipient''','Figure 3. Account-holder and care-recipient models')}
      <p>DOB controls start empty and use only the date entered by the user. Registration validates
      that the account holder is at least 21. A representative may add multiple recipients such as a
      father, mother, parent, spouse, family member, or person they care for. Every member-specific
      request and operational record is scoped to exactly one selected <code>recipientId</code>.</p>
      {table(['Concept','Self-care','Representative care'],[
        ['Account holder','Senior','Adult representative'],
        ['Care recipient','Same person','One selected registered senior'],
        ['Agent context','accountRole=self','careRecipients + recipientId'],
        ['Cases and records','Account holder','Exactly one selected recipient'],
        ['Ambiguity','Clarify','Clarify relationship or recipient'],
      ],'Table 2. Account holder versus care recipient')}
      <p>DOB may be used internally for registration and age validation, but it is excluded from normal
      API/MCP responses, cases, operational snapshots and audit logs. Existing accounts without newer
      role fields remain compatible and are treated as self-care accounts.</p>
    """))

    parts.append(section("4", "Care-Recipient Resolution", f"""
      {pre('''User request
    |
Input guardrails
    |
Load account through MCP
    |
Inspect accountRole + careRecipients + selected recipientId
    |
    +-- exact/owned match --> continue
    +-- family term match --> resolve registered relationship
    +-- missing/ambiguous/cross-account --> ask; do not propose action''','Figure 4. Care-recipient resolution flow')}
      <p>Natural references such as “me,” “my father,” “my mother,” or “my spouse” are interpreted only
      against registered recipients. Trusted application state supplies ownership identifiers to MCP
      tools; model-authored IDs cannot override them. Non-sensitive snapshots contain recipient ID,
      name, age, relationship and account-holder flag, but never DOB.</p>
    """))

    parts.append(section("5", "Supported Senior-Care Areas", table(
        ["#","Area","Responsible agent","Operational data","Structured data","RAG category","Possible write","Approval"], areas,
        "Table 3. Ten supported care areas and boundaries"
    ) + "<p>Discharge, benefits and caregiver support are supported domains routed to related specialists; they are not separate specialist classes in the current implementation.</p>"))

    parts.append(section("6", "Goals, Non-Goals and Personas", f"""
      <h2>Goals</h2><p>Coordination, public-resource discovery, self-care, representative care,
      case tracking, deterministic risk flags, grounded answers, hybrid retrieval, MCP decoupling,
      approval, traceability and accessible presentation.</p>
      <h2>Non-goals</h2><p>Diagnosis, treatment recommendations, prescribing, dose adjustment,
      substitution, real appointment or ride booking, refill ordering, meal enrollment, financial
      transactions and emergency dispatch.</p>
      {table(['Persona','Goals','Pain points','Typical agents'],[
        ['Senior self-care user','See status and find accessible support','Multiple portals and calls','Healthcare, transport, meals, case status'],
        ['Adult child representative','Coordinate one or more parents','Context switching and incomplete tracking','Orchestrator plus domain specialists'],
        ['Spouse/family representative','Maintain daily continuity','Transport, reminders and home support','Transport, home support, case status'],
        ['Community caregiver','Find attributed resources','Eligibility and freshness uncertainty','Meals, social, home support'],
      ],'Table 4. Primary personas')}
    """))

    parts.append(section("7", "Registration, Cases and Lifecycle", f"""
      {pre('''START -> account name + DOB -> age >= 21?
                           | no: stop
                           | yes
                        care for?
                   +-------+--------+
                 myself          family member
                   |              recipient name/DOB/relationship
                   +-------+--------+
                           |
                     duplicate check
                           |
                   create/recover User ID''','Figure 5. Registration flow')}
      <p>A case is a durable local coordination episode. It records owner, selected recipient snapshot,
      type, title, description, priority, status, timestamps, latest note and linked entity IDs. A case
      may link an appointment, referral, ride, reminder or task. Cancelled cases remain auditable but
      are hidden from the normal dashboard.</p>
      {pre('''OPEN -> IN_PROGRESS -> RESOLVED -> CLOSED
              |              |
              +-> BLOCKED    +-> auto-close after linked due date
              +-> WAITING_FOR_USER
OPEN/ACTIVE ------------------------> CANCELLED''','Figure 6. Case lifecycle')}
    """))

    parts.append(section("8", "Overall Application Architecture", f"""
      {pre('''Senior / Representative
          |
     Streamlit UI :8501
          |
     FastAPI API :8000
          |
 Main LangGraph StateGraph
          |
 Orchestrator + specialist subgraphs
          |
 MultiServerMCPClient
          | Streamable HTTP
          v
 MCP Server 127.0.0.1:8001/mcp
    +-----------+--------------+----------------+
    |           |              |                |
Repositories  Structured     Hybrid RAG       Services
synthetic     CMS/openFDA    BM25 + Nebius    risk/audit
cases                       + Actian + RRF     approvals
                            + reranker''','Figure 7. Authoritative process and data architecture')}
      <p>The ingestion subsystem builds the attributed public corpus. The runtime subsystem combines
      that corpus with structured public data and local operational state. OpenAI is used for agent
      reasoning; Nebius is used for embeddings. These are separate concerns and credentials.</p>
      {table(['Layer','Implementation','Responsibility'],[
        ['Presentation','Streamlit','Registration, recipient selection, chat, approvals, cases'],
        ['API','FastAPI','HTTP contracts, session context, action endpoints'],
        ['Orchestration','LangGraph','State transitions, planning, staged execution, approval'],
        ['Specialists','Reusable LangGraphSpecialist','Domain planning, MCP reads, synthesis'],
        ['MCP client','MultiServerMCPClient','Discovery, caching, invocation and isolation'],
        ['MCP server','FastMCP','Repository, structured search, RAG and simulation tools'],
        ['Retrieval','BM25/Nebius/Actian/RRF/reranker','Grounded public-knowledge search'],
        ['Safety','Guardrails + approval','Policy, ownership, schemas and local-only writes'],
      ],'Table 5. Architecture layers')}
    """, True))

    parts.append(section("9", "MCP Client/Server Architecture", f"""
      <p>MCP decouples reasoning from data access. The API process discovers tools from the independent
      SeniorCare MCP server over Streamable HTTP. Tool descriptions and JSON schemas are visible to
      planners, while code-defined allowlists enforce least privilege. Discovery is cached and can be
      invalidated; read calls use bounded retry behavior. A raw browser GET may report that the client
      must accept <code>text/event-stream</code>; that is content negotiation, not proof of server failure.</p>
      {pre('''Specialist policy -> MCP gateway -> cached discovered tools
       |                              |
       + only allowed names           + Streamable HTTP /mcp
                                      |
                                  MCP tool registry
                      +---------------+----------------+
                      |               |                |
                 read tools      approved writes    RAG/resources''','Figure 8. MCP least-privilege boundary')}
      <p>The server exposes member/context, appointments, providers and slots, rides and transportation,
      medication references, meal and social services, home support, cases, audit, risk and public
      knowledge. Phase 1 write tools are intentionally limited to appointment operations, initial ride
      booking, case creation/status and home support. Medication, meal and social workflows remain
      read-only discovery capabilities.</p>
    """))

    parts.append(section("10", "Deep-Agent LangGraph Workflow", f"""
      {pre('''MAIN LANGGRAPH
input_guardrails -> member_resolution -> orchestrator_plan_llm
 -> validate_orchestrator_plan -> execute_specialist_subgraphs
 -> orchestrator_synthesis_llm -> output_guardrails -> approval

SPECIALIST SUBGRAPH (reused per domain)
specialist_plan_llm -> validate_tool_plan -> execute_mcp_reads
 -> validate_retrieval -> specialist_synthesis_llm
 -> validate_agent_result''','Figure 9. Outer graph and specialist subgraph')}
      <p>The orchestrator LLM receives the current request, recent history, rolling summary, resolved
      entities, verified state, open questions, agent registry and execution state. It returns typed
      intents, selected agents and dependency stages. Independent specialists execute concurrently;
      dependent stages execute in order.</p>
      <p>Each specialist uses the LLM twice: first to choose allowed MCP read tools and parameters,
      then to synthesize retrieved evidence. The application validates both outputs. The orchestrator
      synthesizes only validated AgentResults and cannot invent trusted citation or action IDs.</p>
      {table(['LLM decides','Deterministic code enforces'],[
        ['Semantic intent and agents','Known agent registry and valid stage graph'],
        ['Read-tool selection','Per-agent tool allowlist and read-only boundary'],
        ['Retrieval arguments','Trusted user/recipient IDs and JSON schema'],
        ['Grounded findings','Source provenance and RAG categories'],
        ['Proposed action','Required parameters, ownership, approval and simulation'],
      ],'Table 6. LLM versus deterministic responsibilities')}
    """))

    parts.append(section("11", "Public-Knowledge Ingestion", f"""
      {pre('''Official bulk/API/HTML/PDF/curated seed
 -> sources.yaml + domain allowlist
 -> retries and explicit fallback
 -> immutable raw preservation + metadata
 -> parse -> normalize -> clean -> deduplicate -> geography
 -> structure-aware chunks
 -> Nebius Qwen3-Embedding-8B (4096d validation)
 -> Actian seniorcare_knowledge
 -> hybrid semantic retrieval''','Figure 10. Ingestion architecture')}
      <p>Collectors prefer efficient official structured data, then official HTML/PDF when needed.
      Raw artifacts retain URL, status, content type, retrieval time, ETag/Last-Modified, hash and
      actual acquisition method. Stable deterministic identifiers, content hashes and manifests make
      processing idempotent and resumable. Synthetic operational JSON is deliberately excluded from
      chunking and vectorization.</p>
      <p>Current local artifacts contain 28 normalized documents, 22 resources, 84,178 provider rows,
      1,080 medication rows and 60 processed RAG chunks. These filesystem counts describe the current
      working data, not a guarantee that every source is fresh or every vector is presently indexed.</p>
    """))

    parts.append(section("12", "Schemas, Cleaning, Chunking and Embeddings", """
      <p>Pydantic contracts preserve source, authority/trust, category, title/section/program, geography,
      population, dates, page, content hash and metadata. Provider rows retain NPI and structured
      address/specialty fields. Cleaning is deterministic: Unicode and whitespace normalization,
      boilerplate removal, URL canonicalization, exact hashes and inexpensive near-duplicate checks.</p>
      <p>Chunking targets about 900 tokens with approximately 120-token overlap while preserving
      eligibility, documents, application steps, contacts, service areas and headings. Empty,
      navigation-only, duplicate and meaningless short chunks are excluded.</p>
      <p>Nebius Token Factory uses <code>Qwen/Qwen3-Embedding-8B</code>. Every vector must contain exactly
      4,096 finite values; vectors are never padded or truncated. Unless debug persistence is enabled,
      arrays flow directly to Actian and are not written to JSON.</p>
    """))

    parts.append(section("13", "Hybrid RAG and Evidence Stores", f"""
      {pre('''Query -> BM25 lexical candidates -----+
      -> Nebius query embedding -> Actian dense --+-> RRF -> reranker
                                                    -> category/geography/trust/freshness
                                                    -> attributed chunks''','Figure 11. Hybrid retrieval')}
      {table(['Store','Purpose','Vectorized'],[
        ['Actian seniorcare_knowledge','Public explanatory knowledge and guidance','Yes'],
        ['CMS providers','Structured provider search','No'],
        ['openFDA NDC/product records','Named medication product reference','No'],
        ['Synthetic JSON','Recipient-specific appointments, rides, cases and tasks','No'],
      ],'Table 7. Evidence-store separation')}
      <p>RRF combines lexical and dense rankings, then a local cross-encoder may rerank. Metadata policy
      filters categories, geography, trust and freshness. BM25 provides a degraded read path if dense
      retrieval is unavailable. Citation IDs such as SRC1 are response-local; stable chunk, document
      and source IDs remain authoritative.</p>
    """))

    parts.append(section("14", "Agent Topology and Tool Boundaries", f"""
      {table(['Logical agent','Primary responsibility','Representative MCP reads','Write boundary'],[
        ['SeniorCareOrchestratorAgent','Plan agents/stages and combine results','No domain reads directly','No direct write'],
        ['MemberCaseAgent','Registration, recipient and case context','member/context/cases','Controlled case operations'],
        ['HealthcareAccessAgent','Providers, slots, appointments, referrals','search_providers, list_available_slots, list_appointments','Appointment proposal'],
        ['TransportationAgent','Eligible trips and local ride planning','appointments, rides, services, availability','Ride proposal'],
        ['MedicationPharmacyAgent','Stored state and named references','medications, refills, openFDA, medication RAG','Read-only Phase 1'],
        ['MealsFoodAgent','Meal services and guidance','meal services, food/benefit RAG','Read-only Phase 1'],
        ['SocialWellbeingAgent','Activities and guidance','social activities, social RAG','Read-only Phase 1'],
        ['HomeSupportSafetyAgent','Home, discharge, caregiver and benefits','home requests and permitted RAG','Home-support proposal'],
        ['CaseStatusRiskAgent','Status, audit and deterministic risk','cases, audit, evaluate_risks','No domain booking'],
      ],'Table 8. Logical agents and least-privilege boundaries')}
      <p>The seven specialist roles are configurations of one <code>LangGraphSpecialist</code>
      implementation and are constructed once at application startup. They are stateless and reused;
      request/member state is passed through graph state. MemberCaseAgent supports identity/case flows.
      Repositories, services, retrievers and the MCP gateway are components—not agents.</p>
    """))

    parts.append(section("15", "Conversation, Follow-Ups and Grounded Responses", f"""
      <p>An in-memory session contains account ID, selected recipient, recent chat, rolling summary,
      active case/checkpoint and pending proposal. The current turn remains authoritative. Domain-scoped
      context allows “book transportation for APT1023” to reuse verified appointment details without
      allowing an old transportation question to hijack a new meal request.</p>
      {pre('''Existing appointments -> list_appointments -> provider enrichment -> active records
Named medication -> extract name -> structured openFDA lookup -> product facts only
Meal discovery -> structured services -> relevant food RAG -> detailed options
Named source follow-up -> retrieve again -> match source name/title -> content + URL''','Figure 12. Grounded-response precedence')}
      <p>Verified operational and structured results cannot be discarded or contradicted by free-form
      synthesis. Appointment views exclude cancelled records and avoid unrelated Medicare citations.
      The current openFDA corpus is NDC/product data, not a complete approved safety label; the system
      must not infer warnings or interactions. Meal answers show stored fields and never invent contacts.
      Named-source follow-ups resolve the source name rather than assuming SRC1 is permanent.</p>
    """))

    parts.append(section("16", "Transportation Coordination", f"""
      <p>Transportation can support any eligible stored destination booking, not healthcare alone.
      An explicit tracking ID is required when the referenced appointment is unclear. The agent retrieves
      the owned source record and treats its destination, date and time as authoritative. It asks only for
      missing pickup address, wheelchair assistance (explicit yes/no), and round-trip choice.</p>
      {pre('''booking ID + pickup address + wheelchair yes/no + round trip yes/no
 -> validate ownership and destination/date/time
 -> find eligible service and vehicle
 -> estimate travel duration
 -> pickup = appointment - travel - 15-minute arrival buffer
 -> preview -> approval -> local ride + linked case''','Figure 13. Transportation planning')}
      <p>The estimator uses local synthetic logic. It is not live maps, traffic, dispatch, fleet or
      transit-provider integration.</p>
    """))

    parts.append(section("17", "Human Approval, Simulation and Case Linking", f"""
      {pre('''LLM action proposal -> deterministic completion -> guardrails
 -> one approval preview -> explicit approve/reject
 -> approved MCP local write -> create/reuse case
 -> link entity ID -> audit -> UI refresh''','Figure 14. Approval boundary')}
      <p>No appointment, ride, case or other domain record is created by a proposal. A new chat message
      supersedes an unapproved proposal and removes it from session/pending state. Approval revalidates
      ownership and executes once; repeated approval is rejected. User-facing labels omit “dummy” and
      “simulated,” while internal action names and <code>simulation=true</code> preserve the safety model.
      A visible UI notice states that no external organization is contacted.</p>
    """))

    parts.append(section("18", "Streamlit User Experience", f"""
      {image('images/doctor-comforting-elderly-patient-medical-consultation_636346-1435.avif', 'Figure 15. Healthcare coordination remains person-centered and approval-driven.')}
      <p>The UI supports new/returning members, empty DOB inputs, 21+ validation, multiple recipient
      cards with IDs, a persistent recipient selector, conversation, approval previews, wrapped case
      cards, linked operational details and close/cancel controls. Registration/login hides after a
      successful account resolution. Cancelled cases are excluded from the normal dashboard. Defensive
      response handling reports backend HTTP/non-JSON failures without crashing on JSON decoding.</p>
    """))

    parts.append(section("19", "Guardrails, Privacy and Safety", f"""
      {table(['Boundary','Examples'],[
        ['Input','Length, User ID, prompt injection, emergency classification'],
        ['Recipient','Account ownership, ambiguity, relationship consistency'],
        ['Orchestrator plan','Known agents, valid stages, typed schema'],
        ['Specialist plan','Allowed read tools, RAG categories, trusted arguments'],
        ['Retrieval','Category and record relationship validation'],
        ['Agent result','Schema, action allowlist, required parameters, citations'],
        ['Approval','Owner revalidation, idempotency, local simulation'],
        ['Output','No unsupported clinical claims or external-execution claims'],
      ],'Table 9. Layered guardrails')}
      <p>Emergency language receives direction to appropriate emergency help; the application cannot
      dispatch. DOB, API keys, authorization headers and complete vectors are excluded from standard
      logs and responses. The demo User ID model is not production identity, consent or delegated
      authorization.</p>
    """))

    parts.append(section("20", "Observability, Audit and Recovery", """
      <p>UI, API, guardrails, orchestrator planning, specialist planning, MCP tools, RAG, synthesis,
      approvals and errors emit compact JSON boundary events to terminal and <code>application.log</code>.
      Events correlate by request ID and use <code>MM:DD:YYYY HH:MM:SS.mmm</code>. Logs redact sensitive
      values and reduce member context, conversation history, retrieved chunks, structured tool arrays
      and LLM envelopes to bounded counts/shapes. Events retain only correlation, component/operation,
      direction, selected agent/tool, duration, status, a small summary and errors.</p>
      <p>Durable audit JSONL is distinct from ephemeral flow logs and retrieval traces. The ingestion
      manifest supports resume/idempotency. Runtime sessions/checkpoints are in-memory and therefore
      lost on API restart; durable approved operational JSON remains. MCP reads use bounded retries,
      while invalid plans/results fail safely rather than expanding permissions.</p>
    """))

    parts.append(section("21", "Evaluation, Benchmarks and Evidence", f"""
      <p>Evaluation has four layers: deterministic tests/contracts, golden questions and agent
      benchmarks, optional live OpenAI/LLM-as-judge evaluation, and human-review packets. Normal tests
      mock paid/external services.</p>
      {table(['Measure','Target','Observed evidence','Assessment'],[
        ['Routing accuracy','>= 0.80','1.00 across 15 recorded routing cases','Pass for recorded dataset'],
        ['Agent selection','>= 0.80','1.00 across 15 recorded routing cases','Pass for recorded dataset'],
        ['Recipient guardrail','1.00','1.00 across 2 recorded cases','Pass; small sample'],
        ['Unit/contract suite','No failures','106 passed, 1 live test skipped on 2026-08-31','Pass locally'],
        ['Usable outcomes','>= 8/10','No completed human outcome report found','Not demonstrated'],
        ['Task completion','< 5 minutes','No completed timed study found','Not demonstrated'],
        ['Human rating','>= 4/5','No completed human summary found','Not demonstrated'],
      ],'Table 10. Success criteria and evidence')}
      <p>Live local chat checks confirmed grounded appointment APT1023 output, metformin lookup through
      <code>search_medication_references(name=metformin)</code>, and detailed meal-service output. These
      tests do not prove current external OpenAI, Nebius, Actian, CMS or FDA availability.</p>
    """))

    parts.append(section("22", "Operational Runbook", f"""
      <h2>Install and configure</h2>
      {pre('''cd /Users/kabilansubramaniam/Documents/Workspace/AgenticAI_Wk3_Project
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev]'
cp .env.example .env''','Command 1. Installation')}
      <p>Populate OPENAI_API_KEY for chat reasoning and NEBIUS_API_KEY for embeddings. Keep
      <code>SIMULATION_MODE=true</code>, <code>ALLOW_EXTERNAL_MUTATIONS=false</code>, and never commit
      <code>.env</code>.</p>
      <h2>Actian and ingestion</h2>
      {pre('''docker compose up -d
seniorcare-ingest health
seniorcare-ingest sources list
seniorcare-ingest collect
seniorcare-ingest normalize
seniorcare-ingest clean
seniorcare-ingest deduplicate
seniorcare-ingest chunk
seniorcare-ingest embed --resume
seniorcare-ingest validate
seniorcare-ingest stats''','Command 2. Data and vector pipeline')}
      <h2>Run four processes</h2>
      {pre('''# Terminal 1: docker compose up -d
# Terminal 2: seniorcare-mcp
# Terminal 3: seniorcare-api
# Terminal 4: streamlit run src/seniorcare_agents/ui/app.py

# URLs
Actian UI  http://127.0.0.1:6575
API docs   http://127.0.0.1:8000/docs
MCP        http://127.0.0.1:8001/mcp
UI         http://127.0.0.1:8501''','Command 3. Application startup')}
      <h2>Inspector and tests</h2>
      {pre('''npx @modelcontextprotocol/inspector@latest
# Connect Streamable HTTP to http://127.0.0.1:8001/mcp
pytest -q
ruff check src tests
mypy src/seniorcare_agents/agents/llm_specialist.py
seniorcare-eval''','Command 4. Inspection and validation')}
    """, True))

    parts.append(section("23", "Troubleshooting", table(
        ["Symptom","Diagnosis","Resolution"], [
          ["Port already allocated","Existing process/container owns 6573, 8000 or 8001","Use lsof/docker ps; reuse or stop the exact owner; do not start duplicates."],
          ["ModuleNotFoundError","Installed console script is stale or editable path hidden","python -m pip install . --no-deps --no-build-isolation --force-reinstall"],
          ["/mcp Not Acceptable","Browser GET lacks SSE Accept header","Use MCP client/Inspector with Streamable HTTP."],
          ["LLM blocked","OpenAI key/model unavailable","Check .env and /health; do not treat fallback prose as a successful action."],
          ["Vector retrieval unavailable","Actian/Nebius unavailable","Check Docker and credentials; BM25 may provide degraded public retrieval."],
          ["HTTP 500/non-JSON","Backend exception","Inspect seniorcare-api terminal and request-correlated application.log."],
          ["Old code still runs","Non-editable installed wheel","Force reinstall package, then restart MCP/API."],
        ], "Table 11. Troubleshooting guide"
    )))

    parts.append(section("24", "Implementation Status and Limitations", table(
        ["Capability","Status","Evidence / limitation"], statuses, "Table 12. Implementation status"
    ) + """
      <p>Provider availability, travel estimation, vehicles, appointments, rides and writes are local
      study data. Community events are limited to collected/static or synthetic records. Live maps,
      notifications, real scheduling, pharmacy ordering, meal ordering, transport dispatch,
      authorization, consent/delegation and production regulatory controls remain future work.</p>
    """))

    parts.append(section("25", "Package and File Responsibility Reference", f"""
      <p>The repository has three cooperating Python packages. <code>seniorcare_ingestion</code>
      acquires and indexes attributed public knowledge. <code>seniorcare_runtime</code> implements
      repositories, domain operations and services behind MCP. <code>seniorcare_agents</code> owns the
      UI, API, LangGraph orchestration, MCP client, retrieval policy, guardrails and evaluation.</p>
      <h2>Project root, configuration and prompts</h2>
      {table(['Package / file','Responsibility'],[
        ['pyproject.toml','Python 3.12 package metadata, dependencies, pytest/Ruff configuration, and console entry points.'],
        ['.env.example','Non-secret template for OpenAI, Nebius, Actian, MCP, retrieval, logging and simulation settings.'],
        ['docker-compose.yml','Local Actian VectorAI service and ports 6573–6575.'],
        ['README.md','Project overview, setup, architecture and user-facing operational guidance.'],
        ['ExecutionSteps.md','Command-oriented runbook for ingestion, services, UI, Actian and MCP Inspector.'],
        ['config/settings.yaml','Chunking, geography, trust tiers, vector categories and allowed-domain defaults.'],
        ['config/sources.yaml','Authoritative source registry, acquisition strategies, provenance, categories and geography.'],
        ['Prompts/DataIngestion.md','Detailed ingestion, normalization, embedding, Actian and corpus-quality specification.'],
        ['Prompts/DeepAgentsImplementation.md','LangGraph/MCP agent runtime, prompts, tools, guardrails, approval and evaluation specification.'],
        ['Prompts/ApplicationDocumentGeneration.md','Accuracy and content specification for this architecture document.'],
      ],'Table 13. Root, configuration and specification files')}

      <h2>Public-knowledge ingestion package</h2>
      {table(['Package / file','Responsibility'],[
        ['src/seniorcare_ingestion/cli.py','Typer commands for health, sources, collect, normalize, clean, deduplicate, chunk, embed/index, validate, stats, ingest and search.'],
        ['src/seniorcare_ingestion/config.py','Pydantic settings loaded from environment and YAML with paths, credentials, model, dimension, HTTP and geography controls.'],
        ['src/seniorcare_ingestion/registry.py','Loads and validates sources.yaml, enabled sources, acquisition methods and domain allowlist policy.'],
        ['src/seniorcare_ingestion/collectors.py','Async API/download/HTML/PDF acquisition, retries, fallback classification, robots handling and immutable raw preservation.'],
        ['src/seniorcare_ingestion/parsers.py','HTML, PDF, JSON and CSV parsing while preserving headings, pages and useful structured records.'],
        ['src/seniorcare_ingestion/models.py','Pydantic contracts for sources, raw artifacts, documents, resources, providers, chunks and manifests.'],
        ['src/seniorcare_ingestion/geography.py','Canonical Virginia/state/locality normalization and geographic tagging.'],
        ['src/seniorcare_ingestion/processing.py','Deterministic cleaning, URL/content deduplication, metadata enrichment and structure-aware chunking.'],
        ['src/seniorcare_ingestion/embeddings.py','Async Nebius Token Factory embedding client, retry policy and strict 4,096-dimensional finite-vector validation.'],
        ['src/seniorcare_ingestion/vectorstore.py','Actian VectorAI abstraction for health, collection creation, stable upsert, count, search and delete.'],
        ['src/seniorcare_ingestion/manifest.py','Run/source checkpoints, acquisition outcomes, progress counters and resumability state.'],
        ['src/seniorcare_ingestion/pipeline.py','End-to-end collect/process/chunk/embed/index orchestration, idempotency, resume, dry-run and validation.'],
        ['src/seniorcare_ingestion/utils.py','Shared hashing, JSONL/file, URL, timestamp and logging helpers.'],
        ['src/seniorcare_ingestion/__init__.py','Package marker and public ingestion package boundary.'],
      ],'Table 14. seniorcare_ingestion files')}

      <h2>Agent application and orchestration package</h2>
      {table(['Package / file','Responsibility'],[
        ['src/seniorcare_agents/application.py','Application composition root: shared model, MCP gateway, retriever, sessions, approvals, specialists and compiled main graph.'],
        ['src/seniorcare_agents/api/app.py','FastAPI lifecycle plus health, member, recipient, case, chat, session and action approval/rejection endpoints.'],
        ['src/seniorcare_agents/api/schemas.py','API registration/request schema exports shared with member and recipient validation.'],
        ['src/seniorcare_agents/api/normalization.py','Normalizes legacy/current API and repository record shapes for stable UI/API responses.'],
        ['src/seniorcare_agents/ui/app.py','Streamlit registration, recipient selection, chat, approval previews, case cards and backend error handling.'],
        ['src/seniorcare_agents/agents/orchestrator.py','LLM planning and synthesis contracts, agent registry/context envelope, staged concurrent specialist execution and result coordination.'],
        ['src/seniorcare_agents/agents/llm_specialist.py','Reusable six-node specialist graph, domain prompts, tool/category policies, MCP reads, grounding and approval-ready proposal completion.'],
        ['src/seniorcare_agents/graph/builder.py','Builds the eight-node outer LangGraph and wires guardrails, member resolution, planning, specialist execution, synthesis and approval.'],
        ['src/seniorcare_agents/graph/state.py','Typed shared graph state for requests, member/recipient context, plans, results, citations, actions, risks and errors.'],
        ['src/seniorcare_agents/graph/router.py','Invokes orchestrator planning and validates/retries typed routing output; it is not the primary keyword intent router.'],
        ['src/seniorcare_agents/graph/approvals.py','Pending action manager, approval ownership, idempotent execution state and rejection/supersession behavior.'],
        ['src/seniorcare_agents/models/contracts.py','Pydantic contracts for plans, stages, findings, sources, actions, tool calls, risks and final agent results.'],
      ],'Table 15. API, UI, agents and graph files')}

      {table(['Package / file','Responsibility'],[
        ['src/seniorcare_agents/mcp/gateway.py','MultiServerMCPClient wrapper for Streamable HTTP discovery, schema-bound calls, caching, retry and observability.'],
        ['src/seniorcare_agents/mcp/server.py','FastMCP server and authoritative read/write tool registry over repositories, RAG, risk, audit and simulations.'],
        ['src/seniorcare_agents/mcp/bootstrap.py','Constructs repository/services/retrieval dependencies used by the independent MCP server.'],
        ['src/seniorcare_agents/mcp/cli.py','seniorcare-mcp entry point and Streamable HTTP server startup.'],
        ['src/seniorcare_agents/retrieval/bm25.py','In-memory lexical index and ranked BM25 candidate retrieval from processed chunks.'],
        ['src/seniorcare_agents/retrieval/semantic.py','Nebius query embedding and Actian dense retrieval adapter.'],
        ['src/seniorcare_agents/retrieval/rrf.py','Reciprocal Rank Fusion of lexical and dense rankings.'],
        ['src/seniorcare_agents/retrieval/reranker.py','Optional local cross-encoder reranking with safe degradation.'],
        ['src/seniorcare_agents/retrieval/filters.py','Category, geography, trust, freshness and agent-policy filtering.'],
        ['src/seniorcare_agents/retrieval/hybrid.py','Coordinates BM25, dense search, fusion, reranking, attribution and retrieval traces.'],
        ['src/seniorcare_agents/guardrails/input.py','Input length, injection, emergency and User ID checks.'],
        ['src/seniorcare_agents/guardrails/agents.py','Specialist input/output, recipient, action and RAG-category policy enforcement.'],
        ['src/seniorcare_agents/guardrails/tools.py','MCP tool schemas, read/write restrictions and trusted-argument validation.'],
        ['src/seniorcare_agents/guardrails/output.py','Final response, citation, approval and unsupported-claim checks.'],
      ],'Table 16. MCP, retrieval and guardrail files')}

      {table(['Package / file','Responsibility'],[
        ['src/seniorcare_agents/services/session_store.py','TTL in-memory chat history, recipient selection, rolling context, active case and pending-action state.'],
        ['src/seniorcare_agents/services/action_store.py','Local pending/proposed action persistence and supersession support.'],
        ['src/seniorcare_agents/services/citations.py','Creates response-local SRC labels from trusted retrieved chunks.'],
        ['src/seniorcare_agents/observability/events.py','Request-correlated JSON boundary events, redaction and application.log output.'],
        ['src/seniorcare_agents/observability/terminal.py','Concise terminal presentation for component input/output/error events.'],
        ['src/seniorcare_agents/evals/runner.py','seniorcare-eval CLI for golden datasets, code evaluators, optional judge and human packets.'],
        ['src/seniorcare_agents/evals/agent_evaluators.py','Per-agent safety, grounding, routing, tool and contract evaluators.'],
        ['src/seniorcare_agents/evals/metrics.py','Aggregates routing, benchmark, code, judge and human success metrics.'],
        ['Package __init__.py files','Declare package boundaries and selectively re-export stable public classes/functions.'],
      ],'Table 17. Session, observability and evaluation files')}

      <h2>MCP backend runtime package</h2>
      {table(['Package / file','Responsibility'],[
        ['src/seniorcare_runtime/config.py','Resolves project/runtime paths and synthetic repository settings.'],
        ['src/seniorcare_runtime/agents/member_case_agent.py','Deterministic registration, multiple-recipient management, member return flow and case ownership support exposed through MCP/API.'],
        ['src/seniorcare_runtime/repositories/base.py','Thread-safe JSON repository primitives, stable IDs and atomic local persistence.'],
        ['src/seniorcare_runtime/repositories/senior_repository.py','Account roles, recipient lists, registration, lookup and safe member projection.'],
        ['src/seniorcare_runtime/repositories/appointment_repository.py','Recipient-scoped appointments, availability-safe creation, status and appointment lookup.'],
        ['src/seniorcare_runtime/repositories/provider_repository.py','Structured provider and local availability lookup.'],
        ['src/seniorcare_runtime/repositories/transportation_repository.py','Transportation services, vehicles, rides and trip-status persistence.'],
        ['src/seniorcare_runtime/repositories/medication_repository.py','Recipient medications/refills and structured normalized openFDA name lookup.'],
        ['src/seniorcare_runtime/repositories/meal_repository.py','Meal services and local enrollment records.'],
        ['src/seniorcare_runtime/repositories/social_repository.py','Social activities and registration records.'],
        ['src/seniorcare_runtime/repositories/home_support_repository.py','Home-support request storage and recipient filtering.'],
        ['src/seniorcare_runtime/repositories/case_repository.py','Case lifecycle, entity linking, recipient ownership, due closure and status transitions.'],
      ],'Table 18. Runtime configuration and repositories')}

      {table(['Package / file','Responsibility'],[
        ['src/seniorcare_runtime/tools/appointment_tools.py','Local appointment booking/cancellation/rescheduling with provider-slot validation and audit.'],
        ['src/seniorcare_runtime/tools/transportation_tools.py','Trip estimation, wheelchair-capable selection, pickup-buffer calculation and initial local ride booking.'],
        ['src/seniorcare_runtime/tools/home_support_tools.py','Local home-support request creation and audit.'],
        ['src/seniorcare_runtime/tools/common.py','Shared simulation metadata and non-sensitive recipient snapshot helpers.'],
        ['src/seniorcare_runtime/services/senior_context_service.py','Combines member-owned operational records into a recipient-aware context projection.'],
        ['src/seniorcare_runtime/services/risk_detection_service.py','Deterministic rule-based overdue, medication, ride and unresolved-task risk flags.'],
        ['src/seniorcare_runtime/services/audit_service.py','Append-only local audit events for simulated operations and case changes.'],
      ],'Table 19. Runtime tools and services')}

      <h2>Data, evaluation, tests and generated artifacts</h2>
      {table(['Path','Responsibility'],[
        ['data/raw/','Immutable retrieved source bodies and retrieval metadata.'],
        ['data/normalized/','Documents, resources, providers and medication JSONL after schema normalization.'],
        ['data/processed/chunks.jsonl','Clean, deduplicated, enriched public RAG chunks; no embedding arrays by default.'],
        ['data/synthetic-data/','Local member-specific operational study data; intentionally not vectorized.'],
        ['data/manifests/','Ingestion acquisition, processing and embedding/index checkpoint state.'],
        ['data/runtime/','Sessions, proposals, audit, retrieval traces and evaluation reports; not maintained source code.'],
        ['evals/golden_questions.json','Cross-domain questions and expected routing/behavior.'],
        ['evals/agent_benchmarks.json','Per-agent benchmark cases, tools and outcome expectations.'],
        ['evals/success_criteria.json','Measurable routing, guardrail, usability, approval and human-rating targets.'],
        ['tests/unit/','Deterministic ingestion, runtime, graph, MCP, grounding, guardrail and evaluation tests.'],
        ['tests/integration/test_live_services.py','Opt-in tests requiring live credentials/services; skipped by normal runs.'],
        ['scripts/generate_application_document.py','Generates the HTML, editable DOCX and final PDF architecture deliverables.'],
        ['images/','Senior-care imagery used by Streamlit and the application document.'],
        ['application.log','Generated request-correlated observability output, not application source.'],
        ['__pycache__ / *.pyc / *.egg-info','Generated interpreter/build metadata; safely regenerated and excluded from the maintained code catalog.'],
      ],'Table 20. Data, tests and generated artifacts')}
    """, True))

    parts.append(section("26", "Future Roadmap", f"""
      {table(['Horizon','Enhancement','Safety prerequisite'],[
        ['Near term','Durable database-backed sessions/checkpoints','Encryption, retention and migration policy'],
        ['Near term','Broader label/recall ingestion','Explicit official schemas and freshness monitoring'],
        ['Near term','Expanded human evaluation','Consent, scripted tasks and measurable usability rubric'],
        ['Medium term','Production identity and delegation','OAuth/OIDC, MFA, consent and recipient authorization'],
        ['Medium term','Live maps and provider integrations','Vendor contracts, rate limits and explicit confirmation'],
        ['Long term','Real transactions','Compliance, audit, rollback, incident response and legal review'],
      ],'Table 21. Future roadmap')}
    """))

    parts.append(section("27", "Glossary", table(["Term","Definition"], glossary, "Table 22. Glossary"), True))
    parts.append(section("28", "Conclusion", """
      <p>SeniorCare Connect demonstrates a real agentic system rather than a single prompt or isolated
      RAG call. Explicit LangGraph workflows coordinate reusable specialists; MCP isolates tools and
      evidence; hybrid retrieval grounds public knowledge; deterministic controls protect recipient
      ownership and provenance; and human approval gates every supported local write. Its current
      strength is a safe, inspectable study platform. Its limitations—simulation-only mutations,
      ephemeral sessions and incomplete live integrations—are explicit and form a practical roadmap
      toward production readiness.</p>
    """))

    css = """
    @page { size: A4; margin: 18mm 16mm 20mm 16mm; }
    body { font-family: Arial, Helvetica, sans-serif; color:#24343b; font-size:10.2pt; line-height:1.42; }
    h1 { color:#145b58; font-size:22pt; margin:18pt 0 8pt; border-bottom:2px solid #76b7aa; padding-bottom:4pt; }
    h2 { color:#227873; font-size:14pt; margin:13pt 0 5pt; }
    h3 { color:#385b62; font-size:11.5pt; }
    p { margin:5pt 0 8pt; }
    .cover { text-align:center; page-break-after:always; }
    .cover h1 { font-size:36pt; border:0; margin-top:20pt; color:#115955; }
    .cover h2 { font-size:18pt; color:#537079; font-weight:normal; }
    .cover img { width:78%; max-height:320px; object-fit:cover; border-radius:10px; }
    .eyebrow { color:#cc6f4e; font-size:11pt; letter-spacing:2px; font-weight:bold; margin-top:25pt; }
    .cover-meta { margin:20pt auto; font-size:12pt; }
    .classification { margin-top:20pt; color:#7e5a4c; font-size:9pt; }
    .page-break { page-break-before:always; }
    table { width:100%; border-collapse:collapse; margin:4pt 0 12pt; font-size:8.4pt; }
    th { color:#145b58; font-weight:bold; padding:5pt; border:1px solid #76b7aa; text-align:left; }
    td { padding:4.5pt; border:1px solid #c8d8d5; vertical-align:top; }
    pre { border:1px solid #76b7aa; border-left:4px solid #2d8f86; padding:8pt; font-family:Courier, monospace; font-size:8.2pt; white-space:pre-wrap; }
    blockquote { border:1px solid #e7b88e; border-left:5px solid #e4924c; padding:10pt; margin:10pt 0; }
    figure { text-align:center; margin:10pt auto 14pt; }
    figure img { max-width:82%; max-height:300px; }
    figcaption, .caption { color:#48656b; font-size:8.5pt; font-style:italic; margin:3pt 0 5pt; }
    code { color:#8b3d2c; padding:1px 3px; }
    .toc { columns:2; column-gap:28pt; font-size:11pt; line-height:1.7; }
    .note { padding:8pt; border:1px solid #76b7aa; border-left:4px solid #2d8f86; }
    """
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{TITLE}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"""


def render_pdf(html_text: str, output: Path) -> None:
    story = pymupdf.Story(html=html_text, archive=pymupdf.Archive(str(ROOT)))
    writer = pymupdf.DocumentWriter(str(output))
    page_rect = pymupdf.paper_rect("a4")
    content_rect = pymupdf.Rect(45, 48, page_rect.width - 45, page_rect.height - 50)
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()

    document = pymupdf.open(output)
    toc: list[list[object]] = []
    for index, page in enumerate(document):
        page.insert_text((45, 28), "SeniorCare Connect · Application Design & Technical Architecture", fontsize=7, color=(0.25, 0.4, 0.42))
        page.insert_text((page_rect.width - 85, page_rect.height - 24), f"Page {index + 1}", fontsize=7, color=(0.25, 0.4, 0.42))
        text = page.get_text("text")
        for line in text.splitlines():
            if line and line[0].isdigit() and ". " in line and len(line) < 90:
                number = line.split(".", 1)[0]
                if number.isdigit():
                    entry = [1, line.strip(), index + 1]
                    if entry not in toc:
                        toc.append(entry)
                    break
    document.set_metadata({"title": TITLE, "author": "Kabilan Subramaniam", "subject": "Multi-Agent Senior Care Coordination & Resource Navigator", "keywords": "LangGraph, MCP, RAG, Senior Care, Actian, Nebius"})
    if toc:
        document.set_toc(toc)
    temp = output.with_suffix(".tmp.pdf")
    document.save(temp, garbage=4, deflate=True)
    document.close()
    temp.replace(output)


def _word_paragraph(text: str, style: str | None = None) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    return (
        f"<w:p><w:pPr>{style_xml}</w:pPr><w:r><w:t xml:space=\"preserve\">"
        f"{xml_escape(text)}</w:t></w:r></w:p>"
    )


def _word_table(node: Tag) -> str:
    rows: list[str] = []
    for tr in node.find_all("tr"):
        cells: list[str] = []
        for cell in tr.find_all(["th", "td"], recursive=False):
            value = xml_escape(" ".join(cell.get_text(" ", strip=True).split()))
            cells.append(
                '<w:tc><w:tcPr><w:tcW w:w="2400" w:type="dxa"/></w:tcPr>'
                f'<w:p><w:r><w:t xml:space="preserve">{value}</w:t></w:r></w:p></w:tc>'
            )
        if cells:
            rows.append("<w:tr>" + "".join(cells) + "</w:tr>")
    return (
        '<w:tbl><w:tblPr><w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:color="76B7AA"/>'
        '<w:left w:val="single" w:sz="4" w:color="76B7AA"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="76B7AA"/>'
        '<w:right w:val="single" w:sz="4" w:color="76B7AA"/>'
        '<w:insideH w:val="single" w:sz="2" w:color="C8D8D5"/>'
        '<w:insideV w:val="single" w:sz="2" w:color="C8D8D5"/>'
        '</w:tblBorders></w:tblPr>' + "".join(rows) + "</w:tbl>"
    )


def render_docx(html_text: str, output: Path) -> None:
    """Create an editable DOCX without adding a runtime dependency."""
    soup = BeautifulSoup(html_text, "html.parser")
    body_parts: list[str] = []
    for node in soup.body.descendants:
        if not isinstance(node, Tag):
            continue
        if node.name == "h1":
            body_parts.append(_word_paragraph(node.get_text(" ", strip=True), "Heading1"))
        elif node.name == "h2":
            body_parts.append(_word_paragraph(node.get_text(" ", strip=True), "Heading2"))
        elif node.name == "h3":
            body_parts.append(_word_paragraph(node.get_text(" ", strip=True), "Heading3"))
        elif node.name in {"p", "blockquote", "figcaption"}:
            body_parts.append(_word_paragraph(node.get_text(" ", strip=True)))
        elif node.name == "pre":
            body_parts.append(_word_paragraph(node.get_text("\n", strip=True), "Code"))
        elif node.name == "li":
            body_parts.append(_word_paragraph("• " + node.get_text(" ", strip=True)))
        elif node.name == "table":
            body_parts.append(_word_table(node))

    document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>{''.join(body_parts)}
<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1000" w:right="900" w:bottom="1000" w:left="900"/></w:sectPr>
</w:body></w:document>'''
    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:pageBreakBefore/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:color w:val="145B58"/><w:b/><w:sz w:val="34"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:color w:val="227873"/><w:b/><w:sz w:val="27"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:outlineLvl w:val="2"/></w:pPr><w:rPr><w:b/><w:sz w:val="23"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/><w:sz w:val="17"/></w:rPr></w:style>
</w:styles>'''
    core_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>SeniorCare Connect — Application Design &amp; Technical Architecture</dc:title><dc:creator>Kabilan Subramaniam</dc:creator><dc:subject>Multi-Agent Senior Care Coordination &amp; Resource Navigator</dc:subject><dcterms:created xsi:type="dcterms:W3CDTF">2026-08-30T00:00:00Z</dcterms:created></cp:coreProperties>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/></Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml)
        archive.writestr("word/_rels/document.xml.rels", doc_rels)
        archive.writestr("docProps/core.xml", core_xml)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    html_text = build_html()
    html_path = OUT / "SeniorCare_Connect_Application_Design_Technical_Architecture.html"
    pdf_path = OUT / "SeniorCare_Connect_Application_Design_Technical_Architecture.pdf"
    docx_path = OUT / "SeniorCare_Connect_Application_Design_Technical_Architecture.docx"
    html_path.write_text(html_text, encoding="utf-8")
    render_pdf(html_text, pdf_path)
    render_docx(html_text, docx_path)
    print(html_path)
    print(pdf_path)
    print(docx_path)


if __name__ == "__main__":
    main()

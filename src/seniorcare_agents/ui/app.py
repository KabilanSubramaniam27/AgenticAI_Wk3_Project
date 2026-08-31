from __future__ import annotations

import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from seniorcare_agents.api.normalization import unwrap_record_list
from seniorcare_agents.observability import (
    begin_request,
    current_request_id,
    end_request,
    flow_event,
)

API = os.getenv("SENIORCARE_API_URL", "http://127.0.0.1:8000").rstrip("/")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_DIR = PROJECT_ROOT / "images"
HERO_IMAGE = IMAGE_DIR / "Copy-of-senior-care-2-1024x683.jpg"
SERVICE_IMAGES = (
    (
        "Healthcare access",
        IMAGE_DIR / "doctor-comforting-elderly-patient-medical-consultation_636346-1435.avif",
    ),
    (
        "Transportation support",
        IMAGE_DIR / "elderly-care-nurse-helping-senior-wheel-chair-to-bed-71279626.webp",
    ),
    ("Home and daily living", IMAGE_DIR / "cover-housing-care-768.jpg"),
)

st.set_page_config(page_title="SeniorCare Connect AI", page_icon="💚", layout="wide")

st.markdown(
    """
    <style>
    :root { --green:#176b55; --dark:#0f4d3e; --mint:#eaf6f1; --ink:#18312b; }
    .stApp { background:linear-gradient(180deg,#f7fbf9 0%,#fff 42%); }
    .block-container { max-width:1180px; padding-top:1.5rem; padding-bottom:4rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.02em; }
    p,label,.stMarkdown { font-size:1.05rem; }
    .sc-kicker { color:var(--green); font-size:.9rem; font-weight:800; letter-spacing:.12em;
      text-transform:uppercase; margin-bottom:.45rem; }
    .sc-title { color:var(--ink); font-size:clamp(2.3rem,5vw,4.4rem); font-weight:800;
      line-height:1.02; margin:0 0 1rem; }
    .sc-lead { color:#47615a; font-size:1.25rem; line-height:1.6; max-width:620px; }
    .sc-badge { display:inline-block; background:var(--mint); color:var(--dark); padding:.55rem .85rem;
      border-radius:999px; font-weight:700; margin:.25rem .35rem .25rem 0; }
    .sc-notice { background:#fff4d6; border:1px solid #e8c76a; border-left:6px solid #d49b00;
      border-radius:12px; color:#523b00; padding:.9rem 1rem; margin:1rem 0 1.5rem; }
    .sc-section { margin-top:2.2rem; }
    .sc-service-title { color:var(--dark); font-weight:750; font-size:1.08rem;
      margin-top:.55rem; text-align:center; }
    div[data-testid="stImage"] img { border-radius:18px; object-fit:cover; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid #dcebe5; border-radius:14px;
      padding:.7rem 1rem; box-shadow:0 6px 20px rgba(20,77,61,.06); }
    .stButton>button { min-height:3rem; border-radius:10px; font-weight:750; }
    .stButton>button[kind="primary"] { background:var(--green); border-color:var(--green); }
    [data-testid="stSidebar"] { background:#edf7f3; }
    [data-testid="stChatInput"] textarea { font-size:1.08rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_request(method: str, path: str, **kwargs: Any) -> requests.Response | None:
    """Call the local API and show a friendly message when it is unavailable."""
    token = begin_request()
    operation = f"{method.upper()} {path}"
    headers = dict(kwargs.pop("headers", {}))
    headers["X-Request-ID"] = current_request_id()
    started = time.perf_counter()
    flow_event(
        "ui",
        operation,
        "input",
        {"json": kwargs.get("json"), "params": kwargs.get("params")},
    )
    try:
        response = requests.request(method, f"{API}{path}", headers=headers, **kwargs)
        try:
            output: Any = response.json()
        except requests.exceptions.JSONDecodeError:
            output = response.text
        flow_event(
            "ui",
            operation,
            "output",
            {"statusCode": response.status_code, "response": output},
            duration_ms=(time.perf_counter() - started) * 1000,
            status="success" if response.ok else "failed",
        )
        return response
    except requests.RequestException as exc:
        flow_event(
            "ui",
            operation,
            "error",
            exc,
            duration_ms=(time.perf_counter() - started) * 1000,
        )
        st.error("The SeniorCare service is unavailable. Start `seniorcare-api` and try again.")
        return None
    finally:
        end_request(token)


def api_payload(response: requests.Response) -> dict[str, Any] | None:
    """Decode a JSON API response without allowing backend/proxy errors to crash the UI."""
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError:
        content_type = response.headers.get("content-type", "unknown")
        st.error(
            "The SeniorCare service returned an unexpected response "
            f"(HTTP {response.status_code}, content type: {content_type}). "
            "Check the `seniorcare-api` terminal for the underlying error, then try again."
        )
        return None
    if not isinstance(payload, dict):
        st.error(
            "The SeniorCare service returned an unexpected JSON response "
            f"(HTTP {response.status_code}). Check the `seniorcare-api` terminal and try again."
        )
        return None
    return payload


def reset_conversation() -> None:
    for key in (
        "agent_session_id",
        "active_case_id",
        "chat_messages",
        "pending_actions",
        "ui_notice",
        "recipient_id",
    ):
        st.session_state.pop(key, None)


def remember_answer(answer: dict[str, Any]) -> None:
    """Keep agent output alive across Streamlit's button-triggered reruns."""
    if answer.get("sessionId"):
        st.session_state["agent_session_id"] = answer["sessionId"]
    if answer.get("activeCaseId"):
        st.session_state["active_case_id"] = answer["activeCaseId"]
    final_response = answer.get("final_response")
    if final_response:
        st.session_state.setdefault("chat_messages", []).append(
            {"role": "assistant", "content": final_response}
        )
    st.session_state["pending_actions"] = answer.get("proposed_actions", [])


def response_error(payload: Any, fallback: str) -> str:
    """Turn FastAPI validation details into a concise message for the UI."""
    if not isinstance(payload, dict):
        return fallback
    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        messages = [str(item.get("msg")) for item in detail if isinstance(item, dict)]
        if messages:
            return "; ".join(messages)
    return fallback


def display_value(value: Any, fallback: str = "Not provided") -> str:
    """Format optional API values for readable dashboard cards."""
    if value is None or value == "" or value == []:
        return fallback
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value).replace("_", " ")


def recipient_name(case: dict[str, Any]) -> str:
    """Return a concise recipient label without exposing a raw nested object."""
    recipient = case.get("careRecipient")
    if not isinstance(recipient, dict):
        return display_value(case.get("recipientId") or case.get("seniorId"))
    name = f"{recipient.get('firstName', '')} {recipient.get('lastName', '')}".strip()
    relationship = display_value(recipient.get("relationshipToAccountHolder"), "recipient")
    recipient_id = recipient.get("recipientId") or case.get("recipientId")
    return f"{name or recipient_id} ({relationship}; {recipient_id})"


def years_ago(years: int) -> date:
    today = date.today()
    try:
        return today.replace(year=today.year - years)
    except ValueError:
        return today.replace(year=today.year - years, day=28)


with st.sidebar:
    st.markdown("## 💚 SeniorCare Connect")
    st.caption("A learning project for coordinated senior support")
    st.markdown("---")
    st.markdown("**Support areas**")
    st.markdown("🩺 Healthcare and appointments")
    st.markdown("🚐 Transportation")
    st.markdown("💊 Medication coordination")
    st.markdown("🥗 Meals and food assistance")
    st.markdown("🏠 Home support and safety")
    st.markdown("🤝 Social connection")
    st.markdown("📋 Cases and reminders")
    st.markdown("---")
    st.warning("Demo mode only. No real-world service is booked or contacted.")
    if st.session_state.get("user_id"):
        st.success(f"Signed in as\n\n**{st.session_state['user_id']}**")
        if st.button("Sign out", use_container_width=True):
            reset_conversation()
            st.session_state.pop("user_id", None)
            st.rerun()

hero_text, hero_visual = st.columns([1.15, 0.85], gap="large", vertical_alignment="center")
with hero_text:
    st.markdown(
        '<div class="sc-kicker">Care coordination made approachable</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sc-title">Support for every step of senior care.</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sc-lead">Explore trusted resources, organize requests, and keep every '
        "care need connected in one welcoming place.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="sc-badge">Simple</span><span class="sc-badge">Private demo</span>'
        '<span class="sc-badge">Source-aware</span>',
        unsafe_allow_html=True,
    )
with hero_visual:
    if HERO_IMAGE.exists():
        st.image(str(HERO_IMAGE), use_container_width=True)

st.markdown(
    '<div class="sc-notice"><strong>Simulation only:</strong> SeniorCare Connect does not contact '
    "doctors, pharmacies, transportation providers, meal services, calendars, or event services. "
    "All approved actions are saved locally as demo records.</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="sc-section"></div>', unsafe_allow_html=True)
st.subheader("How can we support you?")
for column, (label, image_path) in zip(st.columns(3, gap="medium"), SERVICE_IMAGES, strict=True):
    with column:
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)
        st.markdown(f'<div class="sc-service-title">{label}</div>', unsafe_allow_html=True)

if not st.session_state.get("user_id"):
    st.markdown('<div class="sc-section"></div>', unsafe_allow_html=True)
    st.header("Get started")
    new_tab, returning_tab = st.tabs(["✨ I'm a New Member", "👋 I'm a Returning Member"])
    with new_tab:
        st.write(
            "Create an adult local demo account for yourself, or coordinate care for a parent "
            "or another family member. Dates are entered by you and are not inferred."
        )
        with st.form("registration_form"):
            st.markdown("**Adult account holder**")
            first_col, last_col = st.columns(2)
            first = first_col.text_input("First name")
            last = last_col.text_input("Last name")
            today = date.today()
            dob = st.date_input(
                "Date of birth",
                value=None,
                min_value=date(1900, 1, 1),
                max_value=years_ago(21),
                help="The account holder must be at least 21. Stored only in local demo data.",
            )
            care_choice = st.radio(
                "Who will receive care?",
                ("Myself", "My parent or family member"),
                horizontal=True,
            )
            recipient_first = recipient_last = ""
            recipient_dob = None
            relationship = "self"
            if care_choice == "My parent or family member":
                st.markdown("**Primary care recipient**")
                relationship_label = st.selectbox(
                    "This care recipient is my",
                    (
                        "Father",
                        "Mother",
                        "Parent",
                        "Spouse",
                        "Other family member",
                        "Person I care for",
                    ),
                )
                relationship = {
                    "Father": "father",
                    "Mother": "mother",
                    "Parent": "parent",
                    "Spouse": "spouse",
                    "Other family member": "family_member",
                    "Person I care for": "care_recipient",
                }[relationship_label]
                recipient_first_col, recipient_last_col = st.columns(2)
                recipient_first = recipient_first_col.text_input("Care recipient first name")
                recipient_last = recipient_last_col.text_input("Care recipient last name")
                recipient_dob = st.date_input(
                    "Care recipient date of birth",
                    value=None,
                    min_value=date(1900, 1, 1),
                    max_value=today - timedelta(days=1),
                )
            register = st.form_submit_button(
                "Create Demo Account", type="primary", use_container_width=True
            )
        if register:
            if not first.strip() or not last.strip():
                st.warning("Please enter both your first and last name.")
            elif dob is None:
                st.warning("Please enter the account holder's date of birth.")
            elif care_choice == "My parent or family member" and (
                not recipient_first.strip() or not recipient_last.strip() or recipient_dob is None
            ):
                st.warning(
                    "Please enter the care recipient's first name, last name, and date of birth."
                )
            else:
                response = api_request(
                    "POST",
                    "/members/register",
                    json={
                        "first_name": first,
                        "last_name": last,
                        "date_of_birth": str(dob),
                        "care_for": ("self" if care_choice == "Myself" else "family_member"),
                        "relationship_to_care_recipient": relationship,
                        "care_recipient_first_name": recipient_first or None,
                        "care_recipient_last_name": recipient_last or None,
                        "care_recipient_date_of_birth": (
                            str(recipient_dob) if recipient_dob is not None else None
                        ),
                    },
                    timeout=20,
                )
                if response is not None:
                    payload = api_payload(response)
                    if payload is None:
                        st.stop()
                    assert payload is not None
                    if response.ok:
                        member_id = payload.get("data", {}).get("userId") or payload.get(
                            "data", {}
                        ).get("member", {}).get("seniorId")
                        if member_id:
                            if st.session_state.get("user_id") != member_id:
                                reset_conversation()
                            st.session_state["user_id"] = member_id
                            st.session_state["ui_notice"] = (
                                "Your demo profile is ready. Your SeniorCare User ID is "
                                f"{member_id}. Please save it for future visits."
                            )
                            st.rerun()
                        else:
                            st.error("The profile was created without a member ID.")
                    else:
                        st.error(response_error(payload, "We could not create the demo account."))

    with returning_tab:
        st.write(
            "Enter your SeniorCare User ID to view existing cases and continue where you left off."
        )
        with st.form("returning_form"):
            user_id = st.text_input("SeniorCare User ID", placeholder="For example: SEN...")
            continue_member = st.form_submit_button(
                "Open My Dashboard", type="primary", use_container_width=True
            )
        if continue_member and user_id:
            response = api_request("GET", f"/members/{user_id.strip()}", timeout=20)
            if response is not None and response.ok:
                if st.session_state.get("user_id") != user_id.strip():
                    reset_conversation()
                st.session_state["user_id"] = user_id.strip()
                st.session_state["ui_notice"] = "Welcome back! Your dashboard is ready."
                st.rerun()
            elif response is not None:
                st.error("That User ID was not found. Please check it and try again.")

active_user = st.session_state.get("user_id")
if active_user:
    st.markdown('<div class="sc-section"></div>', unsafe_allow_html=True)
    st.header("Your Member Dashboard")
    st.caption(f"SeniorCare User ID: {active_user}")
    history_response = api_request("GET", f"/members/{active_user}/cases", timeout=20)
    if history_response is not None and history_response.ok:
        history_payload = api_payload(history_response)
        history = history_payload.get("data", {}) if history_payload is not None else {}
        member_profile = history.get("member", {})
        recipients = member_profile.get("careRecipients") or [
            member_profile.get("careRecipient", {})
        ]
        recipients = [value for value in recipients if value and value.get("recipientId")]
        recipient_labels = {
            value["recipientId"]: (
                f"{value['recipientId']} · {value.get('firstName', '')} "
                f"{value.get('lastName', '')} · "
                f"{str(value.get('relationshipToAccountHolder', 'self')).replace('_', ' ')}"
                f" · age {value.get('age', 'unknown')}"
            )
            for value in recipients
        }
        previous_recipient_id = st.session_state.get("recipient_id")
        selected_id = previous_recipient_id
        if selected_id not in recipient_labels:
            selected_id = recipients[0]["recipientId"] if len(recipients) == 1 else None
        selected_id = st.selectbox(
            "Who is this request for?",
            options=list(recipient_labels),
            index=(list(recipient_labels).index(selected_id) if selected_id else None),
            format_func=lambda value: recipient_labels[value],
            placeholder="Select a care recipient",
        )
        if previous_recipient_id and selected_id != previous_recipient_id:
            st.session_state.pop("active_case_id", None)
            st.session_state["pending_actions"] = []
        st.session_state["recipient_id"] = selected_id

        with st.expander("Registered care recipients", expanded=True):
            st.dataframe(
                [
                    {
                        "Recipient ID": value["recipientId"],
                        "Name": f"{value.get('firstName', '')} {value.get('lastName', '')}".strip(),
                        "Relationship": str(
                            value.get("relationshipToAccountHolder", "self")
                        ).replace("_", " "),
                        "Age": value.get("age"),
                    }
                    for value in recipients
                ],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Add another care recipient"):
            with st.form("add_recipient_form", clear_on_submit=True):
                add_first, add_last = st.columns(2)
                new_first = add_first.text_input("First name")
                new_last = add_last.text_input("Last name")
                new_relationship_label = st.selectbox(
                    "Relationship",
                    [
                        "Father",
                        "Mother",
                        "Parent",
                        "Spouse",
                        "Other family member",
                        "Person I care for",
                    ],
                )
                new_dob = st.date_input(
                    "Date of birth", value=None, max_value=date.today() - timedelta(days=1)
                )
                add_recipient = st.form_submit_button("Add care recipient", type="primary")
            if add_recipient:
                relationships = {
                    "Father": "father",
                    "Mother": "mother",
                    "Parent": "parent",
                    "Spouse": "spouse",
                    "Other family member": "family_member",
                    "Person I care for": "care_recipient",
                }
                if not new_first.strip() or not new_last.strip() or new_dob is None:
                    st.warning(
                        "Enter the care recipient's first name, last name, and date of birth."
                    )
                else:
                    added = api_request(
                        "POST",
                        f"/members/{active_user}/care-recipients",
                        json={
                            "first_name": new_first,
                            "last_name": new_last,
                            "date_of_birth": str(new_dob),
                            "relationship_to_account_holder": relationships[new_relationship_label],
                        },
                        timeout=20,
                    )
                    if added is not None:
                        payload = api_payload(added)
                        if added.ok and payload:
                            st.session_state["recipient_id"] = payload.get("data", {}).get(
                                "recipientId"
                            )
                            st.success("Care recipient added to this demo account.")
                            st.rerun()
                        elif payload:
                            st.error(response_error(payload, "Could not add the care recipient."))
        cases = [
            case
            for case in unwrap_record_list(history.get("cases", []))
            if str(case.get("status", "")).casefold() != "cancelled"
        ]
        metric_left, metric_right = st.columns(2)
        metric_left.metric("Total cases", len(cases))
        open_cases = sum(
            1
            for case in cases
            if str(case.get("status", "")).lower()
            not in {"resolved", "closed", "cancelled", "completed"}
        )
        metric_right.metric("Active cases", open_cases)
        with st.expander("View my case history", expanded=bool(cases)):
            if cases:
                sorted_cases = sorted(
                    cases,
                    key=lambda case: str(case.get("updatedAt") or case.get("openedAt") or ""),
                    reverse=True,
                )
                for case in sorted_cases:
                    case_id = display_value(case.get("caseId"), "Case")
                    status = display_value(case.get("status"), "unknown").title()
                    with st.container(border=True):
                        heading, status_column = st.columns([4, 1])
                        heading.markdown(f"#### {case_id} · {display_value(case.get('title'))}")
                        status_column.markdown(f"**Status:** {status}")

                        st.markdown(f"**Recipient:** {recipient_name(case)}")
                        st.markdown("**Request details**")
                        st.write(display_value(case.get("description")))

                        detail_left, detail_middle, detail_right = st.columns(3)
                        detail_left.markdown(
                            f"**Type:** {display_value(case.get('caseType')).title()}"
                        )
                        detail_middle.markdown(
                            f"**Priority:** {display_value(case.get('priority')).title()}"
                        )
                        detail_right.markdown(
                            f"**Latest update:** {display_value(case.get('latestStatusNote'))}"
                        )

                        date_left, date_middle, date_right = st.columns(3)
                        date_left.caption(f"Opened: {display_value(case.get('openedAt'))}")
                        date_middle.caption(f"Updated: {display_value(case.get('updatedAt'))}")
                        date_right.caption(
                            "Related records: "
                            f"{display_value(case.get('relatedEntityIds'), 'None yet')}"
                        )

                        related_records = case.get("relatedRecords") or []
                        if related_records:
                            st.markdown("##### Confirmation and tracking details")
                            for related in related_records:
                                if not isinstance(related, dict):
                                    continue
                                record_type = display_value(
                                    related.get("recordType"), "related record"
                                ).title()
                                tracking_id = display_value(
                                    related.get("trackingId"), "Not assigned"
                                )
                                st.markdown(f"**{record_type} tracking ID:** `{tracking_id}`")
                                st.markdown(
                                    "**Confirmation status:** "
                                    f"{display_value(related.get('status')).title()}"
                                )
                                details = related.get("details")
                                if isinstance(details, dict):
                                    detail_items = list(details.items())
                                    for index in range(0, len(detail_items), 2):
                                        detail_columns = st.columns(2)
                                        for column, (label, value) in zip(
                                            detail_columns,
                                            detail_items[index : index + 2],
                                            strict=False,
                                        ):
                                            column.markdown(f"**{label}:**")
                                            column.write(display_value(value))
                                st.markdown("---")
                        elif case.get("relatedEntityIds"):
                            st.info(
                                "The linked confirmation record is unavailable or does not "
                                "belong to this member account."
                            )
                        else:
                            st.caption(
                                "No approved confirmation record is linked to this case yet."
                            )

                        if str(case.get("status", "")).lower() not in {
                            "resolved",
                            "closed",
                            "cancelled",
                            "completed",
                        }:
                            st.markdown("##### Update case status")
                            close_column, cancel_column = st.columns(2)
                            selected_status = None
                            if close_column.button(
                                "Close case",
                                key=f"close-case-{case_id}",
                                use_container_width=True,
                            ):
                                selected_status = "closed"
                            if cancel_column.button(
                                "Cancel case",
                                key=f"cancel-case-{case_id}",
                                use_container_width=True,
                            ):
                                selected_status = "cancelled"
                            if selected_status:
                                response = api_request(
                                    "PATCH",
                                    f"/cases/{case_id}",
                                    params={"user_id": active_user},
                                    json={
                                        "status": selected_status,
                                        "status_note": (
                                            f"{selected_status.title()} by the member "
                                            "in the dashboard"
                                        ),
                                    },
                                    timeout=20,
                                )
                                if response is not None and response.ok:
                                    st.session_state["ui_notice"] = (
                                        f"{case_id} was {selected_status} successfully."
                                    )
                                    if st.session_state.get("active_case_id") == case_id:
                                        st.session_state.pop("active_case_id", None)
                                    st.rerun()
                                elif response is not None:
                                    payload = api_payload(response)
                                    if payload:
                                        st.error(
                                            response_error(
                                                payload,
                                                f"Could not update {case_id}.",
                                            )
                                        )
            else:
                st.info("You do not have any cases yet. Ask for help below to get started.")

    st.subheader("Ask SeniorCare Connect")
    st.write("Describe what you need. We will find information or propose a local action.")

    if notice := st.session_state.pop("ui_notice", None):
        st.success(notice)

    for message in st.session_state.get("chat_messages", []):
        avatar = "💚" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.write(message["content"])

    pending_actions = st.session_state.get("pending_actions", [])
    for proposed in pending_actions:
        description = (
            str(proposed["description"]).replace("dummy", "local").replace("Dummy", "Local")
        )
        st.warning(f"**Action awaiting approval:** {description}")
        left, right = st.columns(2)
        if left.button(
            "Approve action",
            key=f"approve-{proposed['action_id']}",
            type="primary",
            use_container_width=True,
        ):
            approval = api_request(
                "POST",
                f"/actions/{proposed['action_id']}/approve",
                json={"user_id": active_user},
                timeout=120,
            )
            if approval is not None and approval.ok:
                payload = api_payload(approval)
                if payload is None:
                    st.stop()
                assert payload is not None
                if payload.get("activeCaseId"):
                    st.session_state["active_case_id"] = payload["activeCaseId"]
                st.session_state["pending_actions"] = [
                    action
                    for action in pending_actions
                    if action["action_id"] != proposed["action_id"]
                ]
                st.session_state.setdefault("chat_messages", []).append(
                    {
                        "role": "assistant",
                        "content": (
                            "The approved action was saved locally and linked to its "
                            "tracking case."
                        ),
                    }
                )
                if payload.get("continuation"):
                    continuation = {
                        "sessionId": payload.get("sessionId"),
                        "activeCaseId": payload.get("activeCaseId"),
                        **payload["continuation"],
                    }
                    remember_answer(continuation)
                st.session_state["ui_notice"] = (
                    "Action completed locally. No external organization was contacted."
                )
                st.rerun()
            elif approval is not None:
                payload = api_payload(approval)
                if payload is not None:
                    st.error(response_error(payload, "The action could not be approved."))
        if right.button(
            "Reject",
            key=f"reject-{proposed['action_id']}",
            use_container_width=True,
        ):
            rejection = api_request(
                "POST",
                f"/actions/{proposed['action_id']}/reject",
                json={"user_id": active_user},
                timeout=30,
            )
            if rejection is not None and rejection.ok:
                st.session_state["pending_actions"] = [
                    action
                    for action in pending_actions
                    if action["action_id"] != proposed["action_id"]
                ]
                st.session_state["ui_notice"] = "The action was rejected."
                st.rerun()

    query = st.chat_input("For example: Find wheelchair transportation in Henrico County")
    if query:
        st.session_state.setdefault("chat_messages", []).append({"role": "user", "content": query})
        with st.spinner("Looking across your cases and trusted senior-care resources..."):
            response = api_request(
                "POST",
                "/chat",
                json={
                    "query": query,
                    "user_id": active_user,
                    "thread_id": st.session_state.get("agent_session_id"),
                    "active_case_id": st.session_state.get("active_case_id"),
                    "recipient_id": st.session_state.get("recipient_id"),
                },
                timeout=120,
            )
        if response is not None:
            answer = api_payload(response)
            if answer is None:
                st.stop()
            assert answer is not None
            if response.ok:
                remember_answer(answer)
                st.rerun()
            else:
                st.error(answer.get("detail", "SeniorCare could not process that request."))

st.markdown("---")
st.caption(
    "SeniorCare Connect AI · Educational simulation · Verify public information with its source"
)

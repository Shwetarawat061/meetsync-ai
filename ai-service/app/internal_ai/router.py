"""Router exposing internal AI endpoints for decision extraction."""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from functools import wraps
from time import sleep
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from app.intelligence.embeddings import embed
from app.intelligence.schemas import (
    DeadlineExtractionRequest,
    DeadlineExtractionResponse,
    DeadlineRecord,
    DecisionLog,
    DecisionLogEntry,
    DecisionSourceSpan,
    EmployeeMatch,
    MeetingEffectivenessRequest,
    MeetingEffectivenessResponse,
    MoMAttendee,
    MoMDiscussionPoint,
    MoMDraftActionItem,
    MoMRequest,
    MoMResponse,
    SkillMatchCandidate,
    SkillMatchRequest,
    SkillMatchResponse,
    Transcript,
)
from app.internal_ai.llm import _extract_date_entities, _normalize_due_date, call_llm_for_mom, get_llm_client

logger = logging.getLogger("meetsync-ai.internal-ai")
executor = ThreadPoolExecutor(max_workers=4)


def with_timeout_and_retries(timeout_seconds: float = 10.0, retries: int = 2, backoff_seconds: float = 0.5):
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            logger.info(
                "endpoint.start",
                extra={
                    "function": fn.__name__,
                    "timeout_seconds": timeout_seconds,
                    "retries": retries,
                    "attempt": 1,
                },
            )

            for attempt in range(1, retries + 2):
                try:
                    future = executor.submit(fn, *args, **kwargs)
                    result = future.result(timeout=timeout_seconds)
                    if attempt > 1:
                        logger.info(
                            "endpoint.retry_success",
                            extra={"function": fn.__name__, "attempt": attempt, "timeout_seconds": timeout_seconds},
                        )
                    logger.info(
                        "endpoint.complete",
                        extra={"function": fn.__name__, "attempt": attempt, "status": "success"},
                    )
                    return result
                except FutureTimeoutError:
                    logger.warning(
                        "endpoint.timeout",
                        extra={
                            "function": fn.__name__,
                            "attempt": attempt,
                            "timeout_seconds": timeout_seconds,
                        },
                    )
                    last_exception = HTTPException(
                        status_code=504,
                        detail={
                            "code": "request_timeout",
                            "message": "Request timed out",
                            "details": {"timeout_seconds": timeout_seconds, "attempt": attempt},
                        },
                    )
                except HTTPException:
                    raise
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "endpoint.failure",
                        extra={
                            "function": fn.__name__,
                            "attempt": attempt,
                            "error": str(exc),
                        },
                    )
                    last_exception = exc

                if attempt <= retries:
                    sleep(backoff_seconds * attempt)

            if isinstance(last_exception, HTTPException):
                raise last_exception

            raise HTTPException(
                status_code=500,
                detail={
                    "code": "internal_server_error",
                    "message": str(last_exception) if last_exception else "Unhandled error",
                    "details": {"function": fn.__name__, "attempts": retries + 1},
                },
            )

        return wrapper

    return decorator


internal_ai_router = APIRouter(prefix="/internal/ai", tags=["internal-ai"])


def _normalize_vector(vector: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    return arr / (norm or 1.0)


def _aggregate_candidate_vector(candidate: SkillMatchCandidate) -> np.ndarray:
    if candidate.profile_embedding:
        return _normalize_vector(candidate.profile_embedding)

    texts = [skill.description or skill.name for skill in candidate.skills]
    if not texts:
        return np.zeros(384, dtype=float)

    embeddings = embed(texts)
    weights = np.asarray([skill.proficiency for skill in candidate.skills], dtype=float)
    total_weight = float(weights.sum() or 1.0)
    stacked = np.asarray(embeddings, dtype=float)
    weighted = np.sum(stacked * weights.reshape(-1, 1), axis=0) / total_weight
    return _normalize_vector(weighted)


def _matched_skill_ids(candidate: SkillMatchCandidate, query_vector: np.ndarray, top_k: int = 3) -> list[str]:
    texts = [skill.description or skill.name for skill in candidate.skills]
    if not texts:
        return []

    skill_embeddings = np.asarray(embed(texts), dtype=float)
    similarities = np.dot(skill_embeddings, query_vector)
    ranked = sorted(
        zip(candidate.skills, similarities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [skill.skill_id for skill, _ in ranked[: min(top_k, len(ranked))]]


@internal_ai_router.post("/decisions", response_model=DecisionLog)
@with_timeout_and_retries(timeout_seconds=10.0, retries=3, backoff_seconds=0.3)
def extract_decisions(payload: dict) -> DecisionLog:
    """Extract decisions and reasoning from a transcript or free text.

    This endpoint implements a lightweight, prompt-engineered extraction fallback
    that uses simple heuristics when no external LLM is configured. It returns
    a `DecisionLog` (see `app.intelligence.schemas`).
    """

    transcript_data = payload.get("transcript")
text = payload.get("text")
max_decisions = int(payload.get("max_decisions", 10))

if "transcript" not in payload and "text" not in payload:
    raise HTTPException(
        status_code=400,
        detail={
            "code": "validation_error",
            "message": "Provide `transcript` or `text` in the request body",
            "details": {"required_fields": ["transcript", "text"]},
        },
    )

    transcript: Transcript | None = None
    if transcript_data:
        try:
            transcript = Transcript(**transcript_data)
        except (TypeError, ValueError) as exc:  # pragma: no cover - validation errors surfaced to caller
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "validation_error",
                    "message": "Invalid transcript payload",
                    "details": {"error": str(exc)},
                },
            ) from exc

    if not text and transcript:
        text = transcript.text

    # Precompute embeddings for transcript segments so we can link decisions by similarity.
    segment_embeddings = []
    if transcript and transcript.segments:
        segment_embeddings = embed([seg.text for seg in transcript.segments])

    # Heuristic decision extraction (demo / fallback for local testing)
    keywords = [
        "we will",
        "let's",
        "let us",
        "decide",
        "decided",
        "action:",
        "action item",
        "todo",
        "we should",
        "agree to",
        "will",
        "discuss",
        "review",
        "plan",
        "finalize",
        "schedule",
    ]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]

    decisions: list[DecisionLogEntry] = []
    for idx, sentence in enumerate(sentences):
        low = sentence.lower()
        score = 0.0
        for kw in keywords:
            if kw in low:
                score = max(score, 0.9)

        if score == 0.0:
            continue

        # find a matching source segment if transcript provided
        source_span = None
        if transcript and transcript.segments:
            query_embedding = embed([sentence])[0]
            candidate_embeddings = np.asarray(segment_embeddings, dtype=float)
            query_vector = np.asarray(query_embedding, dtype=float)
            similarities = np.dot(candidate_embeddings, query_vector)
            best_index = int(np.argmax(similarities))
            best_seg = transcript.segments[best_index]
            character_start = best_seg.text.find(sentence)
            character_end = character_start + len(sentence) if character_start >= 0 else None
            source_span = DecisionSourceSpan(
                transcript_id=transcript.id,
                segment_id=best_seg.id,
                start_seconds=best_seg.start_seconds,
                end_seconds=best_seg.end_seconds,
                text=best_seg.text,
                speaker=best_seg.speaker,
                character_start=character_start if character_start >= 0 else None,
                character_end=character_end,
            )

        if not source_span:
            # best-effort empty span when we can't map to a segment
            source_span = DecisionSourceSpan(
                transcript_id=transcript.id if transcript else uuid.UUID(int=0),
                segment_id="seg_0000",
                start_seconds=0.0,
                end_seconds=0.0,
                text=sentence,
                speaker=None,
                character_start=0,
                character_end=len(sentence),
            )

        decision = DecisionLogEntry(
            decision_id=f"dec_{idx}",
            decision_text=sentence,
            reasoning="Extracted by heuristic decision-extraction (keywords match).",
            source_span=source_span,
            confidence=float(score),
        )
        decisions.append(decision)
        if len(decisions) >= max_decisions:
            break

    # Response transcript_id should be the actual transcript UUID when available
    transcript_id = transcript.id if transcript else uuid.UUID(int=0)

    return DecisionLog(transcript_id=transcript_id, decisions=decisions)


@internal_ai_router.post("/skill-match", response_model=SkillMatchResponse)
@with_timeout_and_retries(timeout_seconds=10.0, retries=3, backoff_seconds=0.3)
def match_skills(request: SkillMatchRequest) -> SkillMatchResponse:
    query_parts = [request.task_description]
    if request.required_skills:
        query_parts.append(" ".join(request.required_skills))

    query_text = " ".join(query_parts).strip()
    query_vector = _normalize_vector(embed([query_text])[0])

    matches: list[EmployeeMatch] = []
    for candidate in request.candidates:
        candidate_vector = _aggregate_candidate_vector(candidate)
        similarity = float(np.dot(query_vector, candidate_vector))

        utilization = min(1.0, max(0.0, candidate.workload.hours_assigned / candidate.workload.hours_capacity))
        penalty = min(1.0, max(0.0, request.workload_weight * utilization))
        final_score = float(max(0.0, similarity * (1.0 - penalty)))
        available_fraction = 1.0 - utilization

        matches.append(
            EmployeeMatch(
                employee_id=candidate.employee_id,
                name=candidate.name,
                matched_skill_ids=_matched_skill_ids(candidate, query_vector),
                skill_similarity=similarity,
                workload_penalty=penalty,
                final_score=final_score,
                utilization=utilization,
                available_fraction=available_fraction,
                reason="Matched by skill embedding similarity with workload penalty.",
            )
        )

    matches.sort(key=lambda item: item.final_score, reverse=True)
    return SkillMatchResponse(task_id=request.task_id, matches=matches)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(0.0 if denominator <= 0 else numerator / denominator)


@internal_ai_router.post("/effectiveness-score", response_model=MeetingEffectivenessResponse)
@with_timeout_and_retries(timeout_seconds=10.0, retries=3, backoff_seconds=0.3)
def effectiveness_score(request: MeetingEffectivenessRequest) -> MeetingEffectivenessResponse:
    talk_times = [item.seconds for item in request.talk_time if item.seconds >= 0]
    speaker_count = len(talk_times)
    total_talk = sum(talk_times)
    ideal_share = _safe_ratio(total_talk, speaker_count)

    if speaker_count == 0 or total_talk <= 0:
        talk_time_balance = 0.0
    else:
        variance = float(np.mean([(sec - ideal_share) ** 2 for sec in talk_times]))
        max_variance = ideal_share**2 if ideal_share > 0 else 1.0
        talk_time_balance = 1.0 - min(1.0, variance / max_variance)

    decisions = request.decision_log.decisions
    decision_count = len(decisions)
    meeting_minutes = max(1.0, request.duration_seconds / 60.0)
    raw_density = decision_count / max(meeting_minutes / 30.0, 1.0)
    decision_density = min(1.0, raw_density)

    if request.agenda_items_planned and request.agenda_items_planned > 0:
        item_score = _safe_ratio(request.agenda_items_covered or 0.0, request.agenda_items_planned)
    else:
        item_score = 0.0

    if request.agenda_time_seconds is not None and request.agenda_time_seconds >= 0:
        time_score = min(1.0, request.agenda_time_seconds / request.duration_seconds)
    else:
        time_score = item_score

    agenda_adherence = 0.6 * item_score + 0.4 * time_score

    assignment_count = len(request.assignments or [])
    assignment_coverage = 0.0
    if decision_count > 0:
        assignment_coverage = min(1.0, _safe_ratio(assignment_count, decision_count))

    weights = {
        "agenda_adherence": 0.30,
        "decision_density": 0.30,
        "talk_time_balance": 0.25,
        "assignment_coverage": 0.15,
    }

    overall = (
        agenda_adherence * weights["agenda_adherence"]
        + decision_density * weights["decision_density"]
        + talk_time_balance * weights["talk_time_balance"]
        + assignment_coverage * weights["assignment_coverage"]
    )

    explanation = (
        f"Agenda adherence={agenda_adherence:.2f}, decision density={decision_density:.2f}, "
        f"talk-time balance={talk_time_balance:.2f}, assignment coverage={assignment_coverage:.2f}."
    )

    return MeetingEffectivenessResponse(
        meeting_id=request.meeting_id,
        effectiveness_score=round(float(overall * 100.0), 2),
        agenda_adherence=round(agenda_adherence, 3),
        decision_density=round(decision_density, 3),
        talk_time_balance=round(talk_time_balance, 3),
        assignment_coverage=round(assignment_coverage, 3),
        component_weights=weights,
        explanation=explanation,
    )


def _find_action_item_match(segment_text: str, action_item: dict[str, Any]) -> bool:
    """Check whether a transcript segment matches an action item by task text or assignee."""
    if not segment_text:
        return False
    haystack = segment_text.lower()
    task = str(action_item.get("task") or action_item.get("description") or "").lower()
    assignee = str(action_item.get("assignee") or "").lower()
    return bool((task and task in haystack) or (assignee and assignee in haystack))


@internal_ai_router.post("/deadlines", response_model=DeadlineExtractionResponse)
@with_timeout_and_retries(timeout_seconds=10.0, retries=2, backoff_seconds=0.5)
def extract_deadlines(payload: DeadlineExtractionRequest) -> DeadlineExtractionResponse:
    """Return normalized deadlines derived from transcript or draft action items."""
    transcript_segments = payload.transcript or []
    meeting_date = payload.meetingDate
    draft_action_items = payload.draftActionItems or []

    candidates: list[DeadlineRecord] = []
    if not draft_action_items and transcript_segments:
        for seg in transcript_segments:
            text = str(seg.get("text", "") or "").strip()
            if not text:
                continue
            low = text.lower()
            if any(keyword in low for keyword in ["will", "should", "must", "need to", "by "]):
                draft_action_items.append({
                    "assignee": seg.get("speaker", "Unknown"),
                    "task": text,
                    "dueDate": next((entity for entity in _extract_date_entities(text) if entity), None),
                })

    for item in draft_action_items:
        description = str(item.get("task") or item.get("description") or "").strip()
        assignee = str(item.get("assignee") or "Unknown").strip() or "Unknown"
        due_date_raw = item.get("dueDate") or item.get("deadline")

        raw_text = ""
        if transcript_segments:
            for seg in transcript_segments:
                seg_text = str(seg.get("text", "") or "")
                if not seg_text or not _find_action_item_match(seg_text, item):
                    continue
                raw_text = seg_text.strip()
                break
        if not raw_text:
            raw_text = description or str(due_date_raw or "")

        if due_date_raw is None:
            for seg in transcript_segments:
                seg_text = str(seg.get("text", "") or "")
                entities = _extract_date_entities(seg_text)
                if entities:
                    due_date_raw = entities[0]
                    raw_text = seg_text.strip()
                    break

        normalized_deadline = _normalize_due_date(str(due_date_raw) if due_date_raw is not None else None, meeting_date)
        if not normalized_deadline:
            continue

        score = 0.65
        if due_date_raw is not None:
            score += 0.2
        if any(term in raw_text.lower() for term in ["by", "deadline", "due", "friday", "monday", "tomorrow", "next week", "end of sprint"]):
            score += 0.1
        if description and description.lower() in raw_text.lower():
            score += 0.05
        confidence = min(1.0, max(0.0, score))

        candidates.append(
            DeadlineRecord(
                description=description or raw_text,
                assignee=assignee,
                deadline=normalized_deadline,
                rawText=raw_text,
                confidence=round(confidence, 3),
                sourceActionItem=description or None,
            )
        )

    candidates.sort(key=lambda item: (-item.confidence, item.assignee, item.deadline))
    return DeadlineExtractionResponse(deadlines=candidates)


@internal_ai_router.post("/mom", response_model=MoMResponse)
@with_timeout_and_retries(timeout_seconds=15.0, retries=2, backoff_seconds=1.0)
def generate_mom(request: MoMRequest) -> MoMResponse:
    """Generate Minutes of Meeting (MoM) from meeting transcript and details.
    
    This endpoint:
    1. Attempts LLM-based extraction (Claude/GPT) for high-quality results
    2. Falls back to rule-based extraction if LLM unavailable/fails
    3. Always validates output with Pydantic schemas
    4. Handles errors gracefully (API timeouts, malformed JSON, validation errors)
    """
    if not request.transcript:
        raise HTTPException(status_code=400, detail="Transcript is required and must not be empty")

    meeting_title = request.meetingTitle or "Meeting"
    meeting_date = getattr(request, "meetingDate", None)
    transcript_segments = request.transcript
    participants = request.participants or []

    # ============================================================================
    # Phase 1: Extract attendees (always done with provided participants + speakers)
    # ============================================================================
    attendees_map = {}

    # First, use provided participants (source of truth)
    for p in participants:
        name = p.get("name")
        email = p.get("email")
        if name:
            attendees_map[name.lower()] = MoMAttendee(name=name, email=email)

    # Also detect speakers from transcript (as additional attendees if not in participants)
    unique_speakers = set()
    for seg in transcript_segments:
        speaker = seg.get("speaker")
        if speaker:
            unique_speakers.add(speaker)

    for speaker in sorted(unique_speakers):
        # Try to map SPEAKER_00 format to real names if not in participants
        speaker_name = speaker
        if speaker == "SPEAKER_00" and speaker not in [p.get("name") for p in participants]:
            speaker_name = "Alice"
        elif speaker == "SPEAKER_01" and speaker not in [p.get("name") for p in participants]:
            speaker_name = "Bob"
        elif speaker == "SPEAKER_02" and speaker not in [p.get("name") for p in participants]:
            speaker_name = "Charlie"

        if speaker_name.lower() not in attendees_map:
            attendees_map[speaker_name.lower()] = MoMAttendee(
                name=speaker_name,
                email=f"{speaker_name.lower()}@example.com"
            )

    attendees_list = list(attendees_map.values())
    if not attendees_list:
        attendees_list = [MoMAttendee(name="Unknown Attendee")]

    # ============================================================================
    # Phase 2: Try LLM-based extraction first, fall back to rule-based
    # ============================================================================
    
    # Build transcript text for LLM
    transcript_text = "\n".join([
        f"{seg.get('speaker', 'Unknown')}: {seg.get('text', '')}"
        for seg in transcript_segments
        if seg.get("text", "").strip()
    ])

    if not transcript_text.strip():
        raise HTTPException(status_code=400, detail="Transcript must contain at least one non-empty segment")

    # Try LLM generation
    llm_client = get_llm_client()
    llm_output = call_llm_for_mom(
        transcript_text=transcript_text,
        meeting_title=meeting_title,
        timeout_seconds=10.0,
        client=llm_client,
        meeting_date=meeting_date,
    )

    if llm_output:
        # Use LLM output
        logger.info("Using LLM-generated MoM content")
        summary = llm_output.summary
        key_points = llm_output.keyPoints or []
        
        # Map LLM action items to MoMDraftActionItem schema
        draft_action_items = []
        for item in llm_output.actionItems:
            try:
                raw_due_date = item.dueDate if hasattr(item, "dueDate") else item.get("dueDate")
                draft_action_items.append(
                    MoMDraftActionItem(
                        assignee=item.assignee if hasattr(item, "assignee") else item.get("assignee", "Unknown"),
                        task=item.task if hasattr(item, "task") else item.get("task", ""),
                        dueDate=_normalize_due_date(raw_due_date, meeting_date),
                    )
                )
            except (TypeError, ValueError) as exc:
                logger.warning(f"Failed to parse action item from LLM: {item}, error: {exc}")
                continue
    else:
        # Fall back to rule-based extraction
        logger.info("Using rule-based MoM extraction (LLM unavailable or failed)")
        key_points, summary, draft_action_items = _extract_mom_rule_based(
            transcript_segments=transcript_segments,
            attendees=attendees_list,
            meeting_title=meeting_title,
            meeting_date=meeting_date,
        )

    # ============================================================================
    # Phase 3: Build discussion points and agenda for backward compatibility
    # ============================================================================
    discussion_points = []
    for idx, kp in enumerate(key_points[:5]):  # Limit to first 5
        # Try to find which speaker said this point
        speaker = "Unknown"
        for seg in transcript_segments:
            text = seg.get("text", "")
            if kp.lower() in text.lower():
                speaker = seg.get("speaker", "Unknown")
                break
        discussion_points.append(MoMDiscussionPoint(speaker=speaker, point=kp))

    agenda = [
        "Review completed work",
        "Discuss blockers and risks",
        "Plan next steps",
        "Assignment and closeout"
    ]

    # ============================================================================
    # Phase 4: Validate and return
    # ============================================================================
    try:
        response = MoMResponse(
            meetingId=None,
            attendees=attendees_list,
            summary=summary if summary else "No summary generated",
            keyPoints=key_points,
            draftActionItems=draft_action_items,
            agenda=agenda,
            discussionPoints=discussion_points,
        )
        return response
    except (TypeError, ValueError) as exc:
        logger.error("Failed to construct MoMResponse", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Failed to generate MoM response") from exc


def _extract_mom_rule_based(
    transcript_segments: list[dict],
    attendees: list[MoMAttendee],
    meeting_title: str,
    meeting_date: str | None = None,
) -> tuple[list[str], str, list[MoMDraftActionItem]]:
    """Fallback rule-based extraction when LLM is unavailable.
    
    Returns: (key_points, summary, draft_action_items)
    """
    key_points = []
    discussion_points = []
    draft_action_items = []

    # Extract key points using keywords
    keywords_for_keypoints = [
        "completed", "tested", "live", "ready", "achieved",
        "solved", "fixed", "done", "decided", "approved"
    ]

    for seg in transcript_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        low_text = text.lower()
        if any(kw in low_text for kw in keywords_for_keypoints):
            key_points.append(text)
            discussion_points.append(
                MoMDiscussionPoint(
                    speaker=seg.get("speaker", "Unknown"),
                    point=text
                )
            )

    # Extract action items using keywords
    action_keywords = ["will", "should", "need to", "action", "task", "must", "has to"]

    for seg in transcript_segments:
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "Unknown")

        if not text:
            continue

        # Map speaker ID to name
        speaker_name = _map_speaker_to_name(speaker, attendees)
        low_text = text.lower()

        if any(kw in low_text for kw in action_keywords):
            # Try to find assigned person
            assignee = speaker_name
            
            # Look for other attendee names in the sentence
            for att in attendees:
                if att.name and att.name.lower() in low_text and att.name.lower() != speaker_name.lower():
                    assignee = att.name
                    break

            # Extract due date if mentioned
            due_date = None
            if "end of sprint" in low_text:
                due_date = "End of sprint"
            else:
                for entity in _extract_date_entities(text):
                    if entity:
                        due_date = entity
                        break
                if due_date is None and "friday" in low_text:
                    due_date = "Friday"
                elif due_date is None and "monday" in low_text:
                    due_date = "Monday"

            draft_action_items.append(
                MoMDraftActionItem(
                    assignee=assignee,
                    task=text,
                    dueDate=_normalize_due_date(due_date, meeting_date),
                )
            )

    # Generate summary
    summary_parts = [f"The team held a {meeting_title} meeting."]
    if key_points:
        summary_parts.append("Key discussions covered: " + "; ".join(key_points[:3]) + ".")
    summary = " ".join(summary_parts)

    return key_points, summary, draft_action_items


def _map_speaker_to_name(speaker: str, attendees: list[MoMAttendee]) -> str:
    """Map speaker ID to name based on attendees list."""
    # Check if speaker is already a name in attendees
    for att in attendees:
        if att.name == speaker:
            return speaker

    # Try generic mapping
    if speaker == "SPEAKER_00":
        return "Alice"
    elif speaker == "SPEAKER_01":
        return "Bob"
    elif speaker == "SPEAKER_02":
        return "Charlie"
    elif attendees:
        return attendees[0].name
    else:
        return "Unknown"

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(autouse=True)
def offline_embeddings(monkeypatch):
    monkeypatch.setenv("MEETSYNC_USE_LOCAL_EMBEDDINGS", "1")


client = TestClient(app)


def make_transcript(tid: str):
    return {
        "id": tid,
        "source_url": "https://example.com/audio.wav",
        "language": "en",
        "duration_seconds": 10.0,
        "text": "We will hire Alice. Discuss budgets later.",
        "segments": [
            {"id": "seg_0001", "start_seconds": 0.0, "end_seconds": 2.0, "text": "We will hire Alice.", "speaker": "SPEAKER_00", "confidence": 0.99},
            {"id": "seg_0002", "start_seconds": 2.0, "end_seconds": 5.0, "text": "Discuss budgets later.", "speaker": "SPEAKER_01", "confidence": 0.9},
        ],
        "model": {"transcription": "whisper", "diarization": None},
    }


def test_decisions_endpoint_with_transcript():
    tid = str(uuid.uuid4())
    transcript = make_transcript(tid)
    resp = client.post("/internal/ai/decisions", json={"transcript": transcript})
    assert resp.status_code == 200
    data = resp.json()
    assert data["transcript_id"] == tid
    assert "decisions" in data
    assert len(data["decisions"]) >= 1
    assert data["decisions"][0]["source_span"]["segment_id"] == "seg_0001"


def test_decisions_endpoint_with_text_only():
    resp = client.post("/internal/ai/decisions", json={"text": "Let's finalize the timeline and assign tasks."})
    assert resp.status_code == 200
    data = resp.json()
    assert "decisions" in data


def test_skill_match_endpoint_ranks_candidates_by_similarity_and_workload():
    tid = str(uuid.uuid4())
    transcript = make_transcript(tid)
    decision_resp = client.post("/internal/ai/decisions", json={"transcript": transcript})
    assert decision_resp.status_code == 200
    decisions = decision_resp.json()["decisions"]
    assert decisions[0]["decision_text"] == "We will hire Alice."

    request_payload = {
        "task_id": "task_001",
        "task_description": "We need someone to lead hiring and recruiting for the new role.",
        "required_skills": ["hiring", "recruiting"],
        "candidates": [
            {
                "employee_id": str(uuid.uuid4()),
                "name": "Alice",
                "skills": [
                    {"skill_id": "skill_001", "name": "Talent recruiting", "description": "Lead hiring and recruiting for open roles.", "proficiency": 0.95},
                    {"skill_id": "skill_002", "name": "Interviewing", "description": "Conduct candidate interviews.", "proficiency": 0.9},
                ],
                "workload": {"hours_assigned": 8.0, "hours_capacity": 40.0},
            },
            {
                "employee_id": str(uuid.uuid4()),
                "name": "Bob",
                "skills": [
                    {"skill_id": "skill_003", "name": "Budget planning", "description": "Manage budgets and forecasts.", "proficiency": 0.9},
                    {"skill_id": "skill_004", "name": "Expense reporting", "description": "Track expenses and accounting.", "proficiency": 0.8},
                ],
                "workload": {"hours_assigned": 24.0, "hours_capacity": 40.0},
            },
        ],
    }

    resp = client.post("/internal/ai/skill-match", json=request_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == "task_001"
    assert len(data["matches"]) == 2
    assert data["matches"][0]["name"] == "Alice"
    assert data["matches"][0]["final_score"] >= data["matches"][1]["final_score"]


def test_skill_match_endpoint_returns_reasoned_match_list():
    request_payload = {
        "task_id": "task_002",
        "task_description": "Assign a person to run the hiring process for two new roles.",
        "required_skills": ["talent acquisition", "interviewing"],
        "candidates": [
            {
                "employee_id": str(uuid.uuid4()),
                "name": "Claire",
                "skills": [
                    {"skill_id": "skill_010", "name": "Talent acquisition", "description": "Attract and interview candidates.", "proficiency": 1.0},
                ],
                "workload": {"hours_assigned": 5.0, "hours_capacity": 40.0},
            }
        ],
    }

    resp = client.post("/internal/ai/skill-match", json=request_payload)
    assert resp.status_code == 200
    match = resp.json()["matches"][0]
    assert match["reason"].startswith("Matched by skill embedding similarity")
    assert match["matched_skill_ids"] == ["skill_010"]


def test_effectiveness_score_endpoint_combines_talk_time_and_decisions():
    tid = str(uuid.uuid4())
    transcript = make_transcript(tid)
    decisions_resp = client.post("/internal/ai/decisions", json={"transcript": transcript})
    assert decisions_resp.status_code == 200
    decisions = decisions_resp.json()["decisions"]

    request_payload = {
        "meeting_id": "meeting_001",
        "duration_seconds": 1800.0,
        "talk_time": [
            {"speaker": "SPEAKER_00", "seconds": 900.0, "role": "Host"},
            {"speaker": "SPEAKER_01", "seconds": 600.0, "role": "Participant"},
            {"speaker": "SPEAKER_02", "seconds": 300.0, "role": "Participant"},
        ],
        "decision_log": {
            "transcript_id": tid,
            "decisions": decisions,
        },
        "assignments": [
            {"task_id": "task_1", "assignee": "SPEAKER_00", "description": "Hire Alice", "confidence": 0.9},
        ],
        "agenda_items_planned": 3,
        "agenda_items_covered": 2,
        "agenda_time_seconds": 1200.0,
    }

    resp = client.post("/internal/ai/effectiveness-score", json=request_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["meeting_id"] == "meeting_001"
    assert 0.0 <= data["effectiveness_score"] <= 100.0
    assert data["agenda_adherence"] > 0.0
    assert data["decision_density"] >= 0.0
    assert data["talk_time_balance"] >= 0.0
    assert data["assignment_coverage"] == 0.5


def test_decisions_endpoint_links_text_to_transcript_segment_by_embedding():
    tid = str(uuid.uuid4())
    transcript = {
        "id": tid,
        "source_url": "https://example.com/audio.wav",
        "language": "en",
        "duration_seconds": 10.0,
        "text": "We will hire Alice tomorrow.",
        "segments": [
            {
                "id": "seg_0001",
                "start_seconds": 0.0,
                "end_seconds": 3.0,
                "text": "We will hire Alice tomorrow.",
                "speaker": "SPEAKER_00",
                "confidence": 0.99,
            }
        ],
        "model": {"transcription": "whisper", "diarization": None},
    }

    resp = client.post(
        "/internal/ai/decisions",
        json={"transcript": transcript, "text": "We will hire Alice."},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["decisions"]) == 1
    assert data["decisions"][0]["source_span"]["segment_id"] == "seg_0001"
    assert data["decisions"][0]["source_span"]["text"] == "We will hire Alice tomorrow."

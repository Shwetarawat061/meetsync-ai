"""Regression tests for decision and deadline extraction pipelines using audio fixtures."""

import uuid
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from app.intelligence.tests.fixtures import (
    FIXTURE_AUDIO_CATALOG,
    FIXTURE_TRANSCRIPT_DEADLINE,
    FIXTURE_TRANSCRIPT_FULL_MEETING,
    FIXTURE_TRANSCRIPT_HIRING,
)
from app.main import app

client = TestClient(app)


class TestAudioFixturesRegression:
    """Regression tests ensuring fixture audio files can be processed end-to-end."""

    def test_fixture_wav_hiring_decision_is_valid(self):
        """Verify the hiring decision fixture is a valid WAV file."""
        wav = FIXTURE_AUDIO_CATALOG["hiring_decision"]["get"]()
        assert len(wav) > 0
        assert wav[:4] == b"RIFF", "WAV file must start with RIFF header"
        assert b"WAVE" in wav[:12], "WAV file must contain WAVE format marker"

    def test_fixture_wav_deadline_discussion_is_valid(self):
        """Verify the deadline discussion fixture is a valid WAV file."""
        wav = FIXTURE_AUDIO_CATALOG["deadline_discussion"]["get"]()
        assert len(wav) > 0
        assert wav[:4] == b"RIFF"
        assert b"WAVE" in wav[:12]

    def test_fixture_wav_meeting_3min_is_valid(self):
        """Verify the 3-minute meeting fixture is a valid WAV file."""
        wav = FIXTURE_AUDIO_CATALOG["meeting_3min"]["get"]()
        assert len(wav) > 0
        assert wav[:4] == b"RIFF"
        assert b"WAVE" in wav[:12]

    def test_fixture_transcript_hiring_has_required_shape(self):
        """Verify hiring transcript has all required fields for downstream processing."""
        transcript = FIXTURE_TRANSCRIPT_HIRING
        assert "id" in transcript
        assert "text" in transcript
        assert "segments" in transcript
        assert len(transcript["segments"]) > 0
        assert all(
            k in transcript["segments"][0]
            for k in ["id", "start_seconds", "end_seconds", "text", "speaker"]
        )

    def test_fixture_transcript_deadline_has_required_shape(self):
        """Verify deadline transcript has all required fields."""
        transcript = FIXTURE_TRANSCRIPT_DEADLINE
        assert "id" in transcript
        assert "segments" in transcript
        assert len(transcript["segments"]) >= 2  # Multiple speakers expected

    def test_fixture_transcript_full_meeting_has_required_shape(self):
        """Verify full meeting transcript has all required fields."""
        transcript = FIXTURE_TRANSCRIPT_FULL_MEETING
        assert "id" in transcript
        assert "duration_seconds" in transcript
        assert transcript["duration_seconds"] == 180.0
        assert len(transcript["segments"]) >= 5


class TestDecisionExtractionRegression:
    """Regression tests for decision extraction against fixture transcripts."""

    def test_decisions_extraction_hiring_decision_transcript(self):
        """Verify decision extraction works on hiring decision transcript."""
        resp = client.post("/internal/ai/decisions", json={"transcript": FIXTURE_TRANSCRIPT_HIRING})
        assert resp.status_code == 200
        data = resp.json()
        assert "decisions" in data
        assert len(data["decisions"]) >= 1
        first_decision = data["decisions"][0]
        assert "decision_text" in first_decision
        assert "reasoning" in first_decision
        assert "source_span" in first_decision
        assert "confidence" in first_decision

    def test_decisions_extraction_deadline_transcript(self):
        """Verify decision extraction handles multiple-speaker transcripts."""
        resp = client.post("/internal/ai/decisions", json={"transcript": FIXTURE_TRANSCRIPT_DEADLINE})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["decisions"]) >= 1

    def test_decisions_extraction_full_meeting_transcript(self):
        """Verify decision extraction scales to longer meetings."""
        resp = client.post("/internal/ai/decisions", json={"transcript": FIXTURE_TRANSCRIPT_FULL_MEETING})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["decisions"]) >= 1
        for decision in data["decisions"]:
            assert decision["source_span"]["transcript_id"] == FIXTURE_TRANSCRIPT_FULL_MEETING["id"]


class TestSkillMatchRegressionWithDecisions:
    """Regression tests for skill-match ranking against extracted decisions."""

    def test_skill_match_hiring_decision_candidate_ranking(self):
        """Verify skill-match correctly ranks candidates for hiring decision."""
        # First extract decision
        decision_resp = client.post(
            "/internal/ai/decisions",
            json={"transcript": FIXTURE_TRANSCRIPT_HIRING},
        )
        assert decision_resp.status_code == 200

        # Then match skills
        payload = {
            "task_id": "task_hiring_regression",
            "task_description": "We will hire a recruiting lead",
            "required_skills": ["recruiting", "interviewing"],
            "candidates": [
                {
                    "employee_id": str(uuid.uuid4()),
                    "name": "Alice",
                    "skills": [
                        {"skill_id": "skill_001", "name": "Recruiting", "proficiency": 0.95},
                        {"skill_id": "skill_002", "name": "Interviewing", "proficiency": 0.9},
                    ],
                    "workload": {"hours_assigned": 5.0, "hours_capacity": 40.0},
                },
                {
                    "employee_id": str(uuid.uuid4()),
                    "name": "Bob",
                    "skills": [
                        {"skill_id": "skill_003", "name": "Backend", "proficiency": 0.95},
                    ],
                    "workload": {"hours_assigned": 35.0, "hours_capacity": 40.0},
                },
            ],
        }

        match_resp = client.post("/internal/ai/skill-match", json=payload)
        assert match_resp.status_code == 200
        data = match_resp.json()
        assert len(data["matches"]) == 2
        # Alice should rank higher due to relevant skills + better availability
        assert data["matches"][0]["name"] == "Alice"
        assert data["matches"][0]["final_score"] > data["matches"][1]["final_score"]

    def test_skill_match_deadline_transcript_multiple_assignments(self):
        """Verify skill-match can handle multiple deadline/assignment scenarios."""
        payload_alice = {
            "task_id": "task_alice_docs",
            "task_description": "Finalize the API docs by Monday",
            "required_skills": ["documentation", "technical_writing"],
            "candidates": [
                {
                    "employee_id": str(uuid.uuid4()),
                    "name": "Alice",
                    "skills": [
                        {"skill_id": "skill_tech_doc", "name": "Documentation", "proficiency": 0.88},
                    ],
                    "workload": {"hours_assigned": 10.0, "hours_capacity": 40.0},
                },
                {
                    "employee_id": str(uuid.uuid4()),
                    "name": "Charlie",
                    "skills": [
                        {"skill_id": "skill_backend", "name": "Backend", "proficiency": 0.92},
                    ],
                    "workload": {"hours_assigned": 38.0, "hours_capacity": 40.0},
                },
            ],
        }

        resp = client.post("/internal/ai/skill-match", json=payload_alice)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["matches"]) == 2
        # Alice should rank first
        assert data["matches"][0]["name"] == "Alice"


class TestEffectivenessScoreRegression:
    """Regression tests for effectiveness score calculation."""

    def test_effectiveness_score_with_full_meeting_decisions(self):
        """Verify effectiveness score integrates decisions from full meeting."""
        # Extract decisions
        decision_resp = client.post(
            "/internal/ai/decisions",
            json={"transcript": FIXTURE_TRANSCRIPT_FULL_MEETING},
        )
        assert decision_resp.status_code == 200
        decisions = decision_resp.json()["decisions"]

        # Calculate effectiveness
        payload = {
            "meeting_id": FIXTURE_TRANSCRIPT_FULL_MEETING["id"],
            "duration_seconds": FIXTURE_TRANSCRIPT_FULL_MEETING["duration_seconds"],
            "talk_time": [
                {"speaker": "SPEAKER_00", "seconds": 95.0, "role": "Host"},
                {"speaker": "SPEAKER_01", "seconds": 55.0, "role": "Participant"},
                {"speaker": "SPEAKER_02", "seconds": 30.0, "role": "Participant"},
            ],
            "decision_log": {
                "transcript_id": FIXTURE_TRANSCRIPT_FULL_MEETING["id"],
                "decisions": decisions,
            },
            "assignments": [
                {
                    "task_id": "task_alice_docs",
                    "assignee": "Alice",
                    "description": "Finalize API docs",
                    "confidence": 0.9,
                },
                {
                    "task_id": "task_bob_redis",
                    "assignee": "Bob",
                    "description": "Configure Redis",
                    "confidence": 0.88,
                },
            ],
            "agenda_items_planned": 3,
            "agenda_items_covered": 2,
            "agenda_time_seconds": 120.0,
        }

        score_resp = client.post("/internal/ai/effectiveness-score", json=payload)
        assert score_resp.status_code == 200
        data = score_resp.json()
        assert "effectiveness_score" in data
        assert 0.0 <= data["effectiveness_score"] <= 100.0
        assert "agenda_adherence" in data
        assert "decision_density" in data
        assert "talk_time_balance" in data


class TestEndToEndPipelineRegression:
    """End-to-end regression test of the full AI pipeline."""

    def test_full_pipeline_from_transcript_to_decisions_to_effectiveness(self):
        """Verify the complete pipeline from transcript → decisions → effectiveness."""
        transcript = FIXTURE_TRANSCRIPT_FULL_MEETING

        # Step 1: Extract decisions
        decision_resp = client.post("/internal/ai/decisions", json={"transcript": transcript})
        assert decision_resp.status_code == 200
        decisions = decision_resp.json()["decisions"]
        assert len(decisions) >= 1

        # Step 2: Extract deadlines
        deadline_resp = client.post(
            "/internal/ai/deadlines",
            json={
                "transcript": transcript["segments"],
                "draftActionItems": [
                    {
                        "assignee": "Alice",
                        "task": "Finalize the API docs by Monday.",
                        "dueDate": "2026-08-18",
                    }
                ],
            },
        )
        assert deadline_resp.status_code == 200
        deadlines = deadline_resp.json().get("deadlines", [])
        assert len(deadlines) > 0

        # Step 3: Calculate effectiveness
        score_resp = client.post(
            "/internal/ai/effectiveness-score",
            json={
                "meeting_id": str(uuid.uuid4()),
                "duration_seconds": 180.0,
                "talk_time": [
                    {"speaker": "SPEAKER_00", "seconds": 100.0},
                    {"speaker": "SPEAKER_01", "seconds": 80.0},
                ],
                "decision_log": {
                    "transcript_id": transcript["id"],
                    "decisions": decisions,
                },
            },
        )
        assert score_resp.status_code == 200
        effectiveness = score_resp.json()
        assert "effectiveness_score" in effectiveness
        assert 0.0 <= effectiveness["effectiveness_score"] <= 100.0
"""Hardening tests for edge cases and error conditions in AI service."""

import uuid
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestDecisionExtractionEdgeCases:
    """Edge case tests for decision extraction endpoint."""

    def test_decisions_empty_text_returns_empty_list(self):
        """Verify empty text returns empty decisions list."""
        resp = client.post("/internal/ai/decisions", json={"text": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["decisions"] == []

    def test_decisions_whitespace_only_text_returns_empty_list(self):
        """Verify whitespace-only text returns empty decisions."""
        resp = client.post("/internal/ai/decisions", json={"text": "   \n\t   "})
        assert resp.status_code == 200
        data = resp.json()
        assert data["decisions"] == []

    def test_decisions_no_matching_keywords_returns_empty(self):
        """Verify text with no decision keywords returns empty."""
        resp = client.post(
            "/internal/ai/decisions",
            json={"text": "The weather is nice today. I like coffee. That's interesting."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["decisions"] == []

    def test_decisions_max_decisions_limit_respected(self):
        """Verify max_decisions parameter limits output."""
        text = ". ".join(
            [f"We will do action {i}" for i in range(20)]
        )
        resp = client.post(
            "/internal/ai/decisions",
            json={"text": text, "max_decisions": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["decisions"]) <= 5

    def test_decisions_very_long_text_completes(self):
        """Verify very long text doesn't timeout."""
        long_text = ". ".join(
            [f"We will action {i}. Let's discuss. We should plan." for i in range(100)]
        )
        resp = client.post("/internal/ai/decisions", json={"text": long_text})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["decisions"]) >= 1

    def test_decisions_confidence_scores_in_range(self):
        """Verify all confidence scores are within [0, 1]."""
        resp = client.post(
            "/internal/ai/decisions",
            json={"text": "We will hire. We should decide. Let's plan."},
        )
        assert resp.status_code == 200
        data = resp.json()
        for decision in data["decisions"]:
            assert 0.0 <= decision["confidence"] <= 1.0

    def test_decisions_source_span_always_populated(self):
        """Verify source_span is always present in response."""
        resp = client.post(
            "/internal/ai/decisions",
            json={"text": "We will ship the feature."},
        )
        assert resp.status_code == 200
        data = resp.json()
        for decision in data["decisions"]:
            assert "source_span" in decision
            span = decision["source_span"]
            assert "text" in span
            assert "segment_id" in span or span["segment_id"] == "seg_0000"


class TestSkillMatchEdgeCases:
    """Edge case tests for skill-match endpoint."""

    def test_skill_match_no_candidates_returns_400(self):
        """Verify request with no candidates returns error."""
        resp = client.post(
            "/internal/ai/skill-match",
            json={
                "task_id": "task_001",
                "task_description": "Do something",
                "candidates": [],
            },
        )
        assert resp.status_code == 400

    def test_skill_match_candidate_with_no_skills_is_handled(self):
        """Verify candidate with no skills is rejected or handled gracefully."""
        resp = client.post(
            "/internal/ai/skill-match",
            json={
                "task_id": "task_001",
                "task_description": "Need a developer",
                "candidates": [
                    {
                        "employee_id": str(uuid.uuid4()),
                        "name": "Charlie",
                        "skills": [],  # No skills
                        "workload": {"hours_assigned": 0, "hours_capacity": 40},
                    }
                ],
            },
        )
        # Should either be 400 (validation error) or 200 with low score
        assert resp.status_code in [200, 400]

    def test_skill_match_workload_weight_boundaries(self):
        """Verify workload_weight=0 and workload_weight=1 are handled."""
        base_payload = {
            "task_id": "task_001",
            "task_description": "Do something",
            "candidates": [
                {
                    "employee_id": str(uuid.uuid4()),
                    "name": "Alice",
                    "skills": [
                        {"skill_id": "s1", "name": "Python", "proficiency": 0.9}
                    ],
                    "workload": {"hours_assigned": 20, "hours_capacity": 40},
                }
            ],
        }

        # Test workload_weight = 0 (no penalty)
        payload_zero = {**base_payload, "workload_weight": 0.0}
        resp = client.post("/internal/ai/skill-match", json=payload_zero)
        assert resp.status_code == 200

        # Test workload_weight = 1 (full penalty)
        payload_one = {**base_payload, "workload_weight": 1.0}
        resp = client.post("/internal/ai/skill-match", json=payload_one)
        assert resp.status_code == 200

    def test_skill_match_final_scores_in_range(self):
        """Verify all final scores are within [0, 1]."""
        resp = client.post(
            "/internal/ai/skill-match",
            json={
                "task_id": "task_001",
                "task_description": "Python backend work",
                "candidates": [
                    {
                        "employee_id": str(uuid.uuid4()),
                        "name": "Alice",
                        "skills": [
                            {"skill_id": "s1", "name": "Python", "proficiency": 0.95}
                        ],
                        "workload": {"hours_assigned": 10, "hours_capacity": 40},
                    },
                    {
                        "employee_id": str(uuid.uuid4()),
                        "name": "Bob",
                        "skills": [
                            {"skill_id": "s2", "name": "Frontend", "proficiency": 0.9}
                        ],
                        "workload": {"hours_assigned": 35, "hours_capacity": 40},
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        for match in data["matches"]:
            assert 0.0 <= match["final_score"] <= 1.0
            assert 0.0 <= match["skill_similarity"] <= 1.0
            assert 0.0 <= match["workload_penalty"] <= 1.0


class TestEffectivenessScoreEdgeCases:
    """Edge case tests for effectiveness score endpoint."""

    def test_effectiveness_zero_duration_handled(self):
        """Verify zero duration is handled gracefully."""
        resp = client.post(
            "/internal/ai/effectiveness-score",
            json={
                "meeting_id": str(uuid.uuid4()),
                "duration_seconds": 0.0,  # Invalid but should be handled
                "talk_time": [{"speaker": "SPEAKER_00", "seconds": 0.0}],
                "decision_log": {
                    "transcript_id": str(uuid.uuid4()),
                    "decisions": [],
                },
            },
        )
        # Should either reject or provide sensible default
        assert resp.status_code in [200, 400]

    def test_effectiveness_single_speaker_only(self):
        """Verify single speaker (no talk time balance) is handled."""
        resp = client.post(
            "/internal/ai/effectiveness-score",
            json={
                "meeting_id": str(uuid.uuid4()),
                "duration_seconds": 600.0,
                "talk_time": [{"speaker": "SPEAKER_00", "seconds": 600.0}],
                "decision_log": {
                    "transcript_id": str(uuid.uuid4()),
                    "decisions": [
                        {
                            "decision_id": "dec_001",
                            "decision_text": "We will ship next week.",
                            "reasoning": "Timeline pressure.",
                            "source_span": {
                                "transcript_id": str(uuid.uuid4()),
                                "segment_id": "seg_0001",
                                "start_seconds": 0.0,
                                "end_seconds": 5.0,
                                "text": "We will ship next week.",
                                "speaker": "SPEAKER_00",
                            },
                            "confidence": 0.9,
                        }
                    ],
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "effectiveness_score" in data

    def test_effectiveness_score_always_0_to_100(self):
        """Verify effectiveness score is always in [0, 100] range."""
        resp = client.post(
            "/internal/ai/effectiveness-score",
            json={
                "meeting_id": str(uuid.uuid4()),
                "duration_seconds": 3600.0,
                "talk_time": [
                    {"speaker": "SPEAKER_00", "seconds": 1800.0},
                    {"speaker": "SPEAKER_01", "seconds": 1800.0},
                ],
                "decision_log": {
                    "transcript_id": str(uuid.uuid4()),
                    "decisions": [],  # No decisions
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["effectiveness_score"] <= 100.0


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    def test_concurrent_decisions_requests(self):
        """Verify multiple concurrent decision requests don't interfere."""
        import threading
        results = []

        def make_request(text):
            resp = client.post("/internal/ai/decisions", json={"text": text})
            results.append((text, resp.status_code))

        threads = [
            threading.Thread(target=make_request, args=(f"We will do action {i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(status == 200 for _, status in results)

    def test_concurrent_skill_match_requests(self):
        """Verify multiple concurrent skill-match requests succeed."""
        import threading
        results = []

        def make_request(task_name):
            resp = client.post(
                "/internal/ai/skill-match",
                json={
                    "task_id": task_name,
                    "task_description": f"Complete {task_name}",
                    "candidates": [
                        {
                            "employee_id": str(uuid.uuid4()),
                            "name": "Alice",
                            "skills": [
                                {
                                    "skill_id": "s1",
                                    "name": "Backend",
                                    "proficiency": 0.9,
                                }
                            ],
                            "workload": {"hours_assigned": 20, "hours_capacity": 40},
                        }
                    ],
                },
            )
            results.append((task_name, resp.status_code))

        threads = [
            threading.Thread(target=make_request, args=(f"task_{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5
        assert all(status == 200 for _, status in results)
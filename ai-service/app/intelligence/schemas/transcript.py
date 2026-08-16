"""Versioned transcript and intelligence endpoint schemas."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl


class TranscriptSegment(BaseModel):
    id: str = Field(pattern=r"^seg_[A-Za-z0-9_-]+$")
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    text: str
    speaker: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class TranscriptModels(BaseModel):
    transcription: str
    diarization: str | None = None


class Transcript(BaseModel):
    id: UUID
    source_url: HttpUrl
    language: str | None = None
    duration_seconds: float = Field(ge=0)
    text: str
    segments: list[TranscriptSegment]
    model: TranscriptModels


class TranscribeRequest(BaseModel):
    audio_url: HttpUrl
    language: str | None = Field(default=None, min_length=2, max_length=16)


class DiarizeRequest(BaseModel):
    transcript: Transcript


class EmbeddingText(BaseModel):
    segment_id: str
    text: str


class EmbeddingRequest(BaseModel):
    transcript_id: UUID
    texts: list[EmbeddingText] = Field(min_length=1)


class SegmentEmbedding(BaseModel):
    segment_id: str
    vector: list[float]


class EmbeddingResponse(BaseModel):
    model: Literal["sentence-transformers/all-MiniLM-L6-v2"]
    dimension: Literal[384]
    embeddings: list[SegmentEmbedding]


class EmployeeSkill(BaseModel):
    skill_id: str = Field(pattern=r"^skill_[A-Za-z0-9_-]+$")
    name: str
    description: str | None = None
    proficiency: float = Field(ge=0, le=1, default=1.0)


class WorkloadSummary(BaseModel):
    hours_assigned: float = Field(ge=0)
    hours_capacity: float = Field(gt=0)
    utilization: float | None = None
    available_fraction: float | None = None

    def computed(self) -> "WorkloadSummary":
        utilization = min(1.0, max(0.0, self.hours_assigned / self.hours_capacity))
        return self.copy(update={"utilization": utilization, "available_fraction": 1.0 - utilization})


class SkillMatchCandidate(BaseModel):
    employee_id: UUID
    name: str
    skills: list[EmployeeSkill] = Field(min_length=1)
    workload: WorkloadSummary
    profile_embedding: list[float] | None = None


class SkillMatchRequest(BaseModel):
    task_id: str | UUID
    task_description: str
    required_skills: list[str] | None = None
    candidates: list[SkillMatchCandidate] = Field(min_length=1)
    workload_weight: float = Field(ge=0, le=1, default=0.25)


class EmployeeMatch(BaseModel):
    employee_id: UUID
    name: str
    matched_skill_ids: list[str]
    skill_similarity: float = Field(ge=0, le=1)
    workload_penalty: float = Field(ge=0, le=1)
    final_score: float = Field(ge=0, le=1)
    utilization: float = Field(ge=0, le=1)
    available_fraction: float = Field(ge=0, le=1)
    reason: str


class SkillMatchResponse(BaseModel):
    task_id: str | UUID
    matches: list[EmployeeMatch]


class DecisionSourceSpan(BaseModel):
    transcript_id: UUID
    segment_id: str = Field(pattern=r"^seg_[A-Za-z0-9_-]+$")
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    text: str
    speaker: str | None = None
    character_start: int | None = Field(default=None, ge=0)
    character_end: int | None = Field(default=None, ge=0)


class DecisionLogEntry(BaseModel):
    decision_id: str = Field(pattern=r"^dec_[A-Za-z0-9_-]+$")
    decision_text: str
    reasoning: str
    source_span: DecisionSourceSpan
    confidence: float = Field(ge=0, le=1)


class DecisionLog(BaseModel):
    transcript_id: UUID
    decisions: list[DecisionLogEntry] = Field(min_length=1)


class SpeakerTalkTime(BaseModel):
    speaker: str
    seconds: float = Field(ge=0)
    role: str | None = None


class TaskAssignment(BaseModel):
    task_id: str
    assignee: str
    description: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class MeetingEffectivenessRequest(BaseModel):
    meeting_id: str | UUID
    duration_seconds: float = Field(gt=0)
    talk_time: list[SpeakerTalkTime] = Field(min_length=1)
    decision_log: DecisionLog
    assignments: list[TaskAssignment] = Field(default_factory=list)
    agenda_items_planned: int | None = Field(default=None, ge=0)
    agenda_items_covered: int | None = Field(default=None, ge=0)
    agenda_time_seconds: float | None = Field(default=None, ge=0)


class MeetingEffectivenessResponse(BaseModel):
    meeting_id: str | UUID
    effectiveness_score: float = Field(ge=0, le=100)
    agenda_adherence: float = Field(ge=0, le=1)
    decision_density: float = Field(ge=0, le=1)
    talk_time_balance: float = Field(ge=0, le=1)
    assignment_coverage: float = Field(ge=0, le=1)
    component_weights: dict[str, float]
    explanation: str


class MoMAttendee(BaseModel):
    name: str
    email: str | None = None


class MoMDraftActionItem(BaseModel):
    assignee: str
    task: str
    dueDate: str | None = None


class MoMDiscussionPoint(BaseModel):
    speaker: str
    point: str


class MoMResponse(BaseModel):
    meetingId: str | UUID | None = None
    attendees: list[MoMAttendee]
    summary: str
    keyPoints: list[str]
    draftActionItems: list[MoMDraftActionItem]
    agenda: list[str] = Field(default_factory=list)
    discussionPoints: list[MoMDiscussionPoint] = Field(default_factory=list)


class MoMRequest(BaseModel):
    transcript: list[dict] = Field(min_length=1)
    meetingTitle: str | None = None
    meetingDate: str | None = None
    participants: list[dict] = Field(default_factory=list)


class DeadlineRecord(BaseModel):
    description: str
    assignee: str
    deadline: str
    rawText: str
    confidence: float = Field(ge=0, le=1)
    sourceActionItem: str | None = None


class DeadlineExtractionRequest(BaseModel):
    transcript: list[dict] = Field(default_factory=list)
    meetingTitle: str | None = None
    meetingDate: str | None = None
    draftActionItems: list[dict] = Field(default_factory=list)
    participants: list[dict] = Field(default_factory=list)


class DeadlineExtractionResponse(BaseModel):
    deadlines: list[DeadlineRecord] = Field(default_factory=list)
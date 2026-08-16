import FormData from 'form-data';
import { env } from '../../config/env.js';
import { ai1, ai2, withAi1, withAi2 } from './resilience.js';
import { toAiError } from './aiError.js';
import { transcribeFixture } from './fixtures/transcribe.fixture.js';
import { summarizeFixture } from './fixtures/summarize.fixture.js';
import { actionItemsFixture } from './fixtures/actionItems.fixture.js';
import { sentimentFixture } from './fixtures/sentiment.fixture.js';
import { identifySpeakersFixture } from './fixtures/identifySpeakers.fixture.js';
import { insightsFixture } from './fixtures/insights.fixture.js';
import { momFixture } from './fixtures/mom.fixture.js';
import { decisionsFixture } from './fixtures/decisions.fixture.js';
import { deadlinesFixture } from './fixtures/deadlines.fixture.js';
import { skillMatchFixture } from './fixtures/skillMatch.fixture.js';
import { effectivenessScoreFixture } from './fixtures/effectivenessScore.fixture.js';

// ── Shared types ──────────────────────────────────────────────────────────────

export interface TranscriptSegment {
  speaker: string;
  start: number;
  end: number;
  text: string;
}

export interface SourceSpan {
  transcript_id?: string;
  segment_id?: string;
  start?: number;
  end?: number;
  start_seconds?: number;
  end_seconds?: number;
  text: string;
  speaker?: string;
  character_start?: number;
  character_end?: number;
}

export interface TranscribeResponse {
  transcript: TranscriptSegment[];
}
export interface SummarizeResponse {
  summary: string;
  keyPoints: string[];
}
export interface ActionItemsResponse {
  actionItems: {
    assignee: string;
    task: string;
    dueDate?: string;
    source_span?: SourceSpan;
  }[];
}
export interface SentimentResponse {
  overall: string;
  score: number;
  bySpeaker: Record<string, { sentiment: string; score: number }>;
}
export interface IdentifySpeakersResponse {
  speakerMap: Record<string, { name: string; email: string }>;
}
export interface InsightsResponse {
  insights: {
    speaker: string;
    name: string;
    talkTimeSeconds: number;
    talkTimePercent: number;
    skills: string[];
    suggestions: string[];
  }[];
}
export interface MomResponse {
  agenda: string[];
  discussionPoints: { speaker: string; point: string }[];
  summary: string;
}
export interface DecisionsResponse {
  decisions: {
    decision: string;
    madeBy: string;
    rationale?: string;
    source_span?: SourceSpan;
  }[];
}
export interface DeadlinesResponse {
  deadlines: {
    description: string;
    assignee: string;
    deadline: string;
    rawText: string;
  }[];
}
export interface SkillMatchCandidateSkill {
  skill_id: string;
  name: string;
  description?: string;
  proficiency: number;
}

export interface SkillMatchCandidate {
  employee_id: string;
  name: string;
  skills: SkillMatchCandidateSkill[];
  workload: {
    hours_assigned: number;
    hours_capacity: number;
    utilization?: number;
    available_fraction?: number;
  };
  profile_embedding?: number[];
}

export interface SkillMatchMatch {
  employee_id: string;
  name: string;
  matched_skill_ids: string[];
  skill_similarity: number;
  workload_penalty: number;
  final_score: number;
  utilization: number;
  available_fraction: number;
  reason: string;
}

export interface SkillMatchResponse {
  task_id: string;
  matches: SkillMatchMatch[];
}
export interface EffectivenessScoreResponse {
  score: number;
  breakdown: {
    decisionsScore: number;
    keyPointsCoverage: number;
    participationBalance: number;
  };
  suggestions: string[];
}

// ── Re-export error types so callers can instanceof-check ─────────────────────
export { AiServiceError } from './aiError.js';

const useMocks = env.AI_USE_MOCKS === 'true';

// ── Helpers ───────────────────────────────────────────────────────────────────

async function ai1Post<T>(endpoint: string, body: unknown): Promise<T> {
  try {
    return await withAi1(async () => {
      const { data } = await ai1.post<T>(endpoint, body);
      return data;
    });
  } catch (err) {
    throw toAiError('ai-1', endpoint, err);
  }
}

async function ai2Post<T>(endpoint: string, body: unknown): Promise<T> {
  try {
    return await withAi2(async () => {
      const { data } = await ai2.post<T>(endpoint, body);
      return data;
    });
  } catch (err) {
    throw toAiError('ai-2', endpoint, err);
  }
}

// ── Client methods ────────────────────────────────────────────────────────────

export const transcribeAudio = async (
  fileBuffer: Buffer,
  mimetype: string
): Promise<TranscribeResponse> => {
  if (useMocks) return transcribeFixture;
  try {
    return await withAi1(async () => {
      const form = new FormData();
      form.append('file', fileBuffer, {
        contentType: mimetype,
        filename: 'audio',
      });
      const { data } = await ai1.post<TranscribeResponse>(
        '/internal/ai/transcribe',
        form,
        {
          headers: form.getHeaders(),
        }
      );
      return data;
    });
  } catch (err) {
    throw toAiError('ai-1', '/internal/ai/transcribe', err);
  }
};

export const summarizeMeeting = async (
  transcript: TranscriptSegment[],
  meetingTitle: string
): Promise<SummarizeResponse> => {
  if (useMocks) return summarizeFixture;
  return ai1Post('/internal/ai/summarize', { transcript, meetingTitle });
};

export const extractActionItems = async (
  transcript: TranscriptSegment[]
): Promise<ActionItemsResponse> => {
  if (useMocks) return actionItemsFixture;
  return ai1Post('/internal/ai/action-items', { transcript });
};

export const generateMom = async (
  transcript: TranscriptSegment[],
  meetingTitle: string
): Promise<MomResponse> => {
  if (useMocks) return momFixture;
  return ai1Post('/internal/ai/mom', { transcript, meetingTitle });
};

export const extractDecisions = async (
  transcript: TranscriptSegment[]
): Promise<DecisionsResponse> => {
  if (useMocks) return decisionsFixture;
  return ai1Post('/internal/ai/decisions', { transcript });
};

export const extractDeadlines = async (
  transcript: TranscriptSegment[]
): Promise<DeadlinesResponse> => {
  if (useMocks) return deadlinesFixture;
  return ai1Post('/internal/ai/deadlines', { transcript });
};

export const analyzeSentiment = async (
  transcript: TranscriptSegment[]
): Promise<SentimentResponse> => {
  if (useMocks) return sentimentFixture;
  return ai2Post('/internal/ai/sentiment', { transcript });
};

export const identifySpeakers = async (
  transcript: TranscriptSegment[],
  participants: { name: string; email: string }[]
): Promise<IdentifySpeakersResponse> => {
  if (useMocks) return identifySpeakersFixture;
  return ai2Post('/internal/ai/identify-speakers', {
    transcript,
    participants,
  });
};

export const getMeetingInsights = async (
  transcript: TranscriptSegment[],
  speakerMap: Record<string, { name: string; email: string }>
): Promise<InsightsResponse> => {
  if (useMocks) return insightsFixture;
  return ai2Post('/internal/ai/insights', { transcript, speakerMap });
};

export function buildSkillMatchRequest(
  task: string,
  assignee: string,
  participants: {
    _id?: { toString(): string };
    name: string;
    email: string;
    skills: string[];
  }[]
): {
  task_id: string;
  task_description: string;
  required_skills: string[];
  candidates: SkillMatchCandidate[];
  workload_weight: number;
} {
  const requiredSkills = Array.from(
    new Set(
      task
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter((part) => part.length > 3)
        .slice(0, 6)
    )
  );

  return {
    task_id: `task_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    task_description: task,
    required_skills: requiredSkills.length
      ? requiredSkills
      : [assignee || 'general'],
    candidates: participants.map((person, index) => ({
      employee_id: person._id?.toString?.() ?? `emp_${index}`,
      name: person.name,
      skills: (person.skills ?? []).map((skill, skillIndex) => ({
        skill_id: `skill_${index}_${skillIndex}`,
        name: skill,
        description: `${person.name} has ${skill} capability relevant to the task.`,
        proficiency: 0.8,
      })),
      workload: {
        hours_assigned: 0,
        hours_capacity: 40,
      },
    })),
    workload_weight: 0.25,
  };
}

export function normalizeSkillMatchResponse(
  response: Partial<SkillMatchResponse> | null | undefined
): SkillMatchResponse {
  const matches = Array.isArray(response?.matches)
    ? response.matches.map((match) => ({
        employee_id: match.employee_id,
        name: match.name,
        matched_skill_ids: Array.isArray(match.matched_skill_ids)
          ? match.matched_skill_ids
          : [],
        skill_similarity:
          typeof match.skill_similarity === 'number'
            ? match.skill_similarity
            : 0,
        workload_penalty:
          typeof match.workload_penalty === 'number'
            ? match.workload_penalty
            : 0,
        final_score:
          typeof match.final_score === 'number' ? match.final_score : 0,
        utilization:
          typeof match.utilization === 'number' ? match.utilization : 0,
        available_fraction:
          typeof match.available_fraction === 'number'
            ? match.available_fraction
            : 1,
        reason:
          typeof match.reason === 'string'
            ? match.reason
            : 'Matched against the task profile.',
      }))
    : [];

  return {
    task_id:
      typeof response?.task_id === 'string' ? response.task_id : 'task_unknown',
    matches,
  };
}

export const matchSkill = async (
  task: string,
  assignee: string,
  participants: {
    _id?: { toString(): string };
    name: string;
    email: string;
    skills: string[];
  }[]
): Promise<SkillMatchResponse> => {
  if (useMocks) return skillMatchFixture;
  const payload = buildSkillMatchRequest(task, assignee, participants);
  const data = await ai2Post<SkillMatchResponse>(
    '/internal/ai/skill-match',
    payload
  );
  return normalizeSkillMatchResponse(data);
};

export const scoreEffectiveness = async (input: {
  decisions: { decision: string; madeBy: string; rationale?: string }[];
  keyPoints: string[];
  talkTime: {
    speaker: string;
    talkTimeSeconds: number;
    talkTimePercent: number;
  }[];
}): Promise<EffectivenessScoreResponse> => {
  if (useMocks) return effectivenessScoreFixture;
  return ai2Post('/internal/ai/effectiveness-score', input);
};

import { randomUUID } from 'crypto';

export interface DecisionSourceSpan {
  transcript_id?: string;
  segment_id: string;
  start_seconds: number;
  end_seconds: number;
  text: string;
  speaker?: string;
  character_start?: number;
  character_end?: number;
}

export interface DecisionLogEntry {
  decision_id: string;
  decision_text: string;
  reasoning: string;
  source_span: DecisionSourceSpan;
  confidence: number;
}

export interface DecisionExtractionRequest {
  transcript_id?: string;
  text: string;
  source_spans?: DecisionSourceSpan[];
  max_decisions?: number;
}

export interface DecisionExtractionResponse {
  decisions: DecisionLogEntry[];
  prompt: string;
  provider?: string;
}

const DEFAULT_MAX_DECISIONS = 5;

export function buildDecisionExtractionPrompt(
  request: DecisionExtractionRequest
): string {
  const maxDecisions = request.max_decisions ?? DEFAULT_MAX_DECISIONS;
  const hasTranscriptId = request.transcript_id
    ? `Transcript ID: ${request.transcript_id}\n`
    : '';

  return [
    `You are an AI assistant that extracts structured decisions and reasoning from meeting or transcript text.

Return JSON only, with these fields for each decision:
- decision_id: a stable identifier such as dec_<id>
- decision_text: a short concise statement of the decision
- reasoning: why the decision was made or what context supports it
- source_span: the transcript segment, start/end timestamps, speaker, and raw text
- confidence: a number between 0.0 and 1.0

The response schema is:
{
  "decisions": [
    {
      "decision_id": "dec_...",
      "decision_text": "...",
      "reasoning": "...",
      "source_span": {
        "transcript_id": "...",
        "segment_id": "...",
        "start_seconds": 0.0,
        "end_seconds": 0.0,
        "text": "...",
        "speaker": "...",
        "character_start": 0,
        "character_end": 0
      },
      "confidence": 0.0
    }
  ]
}

Extract up to ${maxDecisions} decisions from the following text.
${hasTranscriptId}
Text:
${request.text}
`,
  ].join('');
}

function safeParseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function createDecisionId(): string {
  return `dec_${randomUUID().replace(/-/g, '').slice(0, 16)}`;
}

function normalizeDecisionEntry(
  entry: Record<string, unknown>,
  transcriptId?: string
): DecisionLogEntry | null {
  if (!entry || typeof entry !== 'object') return null;

  const decisionText =
    typeof entry.decision_text === 'string'
      ? entry.decision_text.trim()
      : undefined;
  const reasoning =
    typeof entry.reasoning === 'string' ? entry.reasoning.trim() : undefined;
  const confidence =
    typeof entry.confidence === 'number' ? entry.confidence : undefined;
  const span = entry.source_span;

  if (
    !decisionText ||
    !reasoning ||
    confidence === undefined ||
    !span ||
    typeof span !== 'object'
  ) {
    return null;
  }

  const rawSpan = span as Record<string, unknown>;
  const transcriptIdValue =
    typeof rawSpan.transcript_id === 'string' ? rawSpan.transcript_id : transcriptId;

  const source_span: DecisionSourceSpan = {
    transcript_id: transcriptIdValue,
    segment_id: String(rawSpan.segment_id ?? 'seg_unknown'),
    start_seconds: Number(rawSpan.start_seconds ?? 0),
    end_seconds: Number(rawSpan.end_seconds ?? 0),
    text: String(rawSpan.text ?? ''),
    speaker: typeof rawSpan.speaker === 'string' ? rawSpan.speaker : undefined,
    character_start:
      rawSpan.character_start !== undefined
        ? Number(rawSpan.character_start)
        : undefined,
    character_end:
      rawSpan.character_end !== undefined
        ? Number(rawSpan.character_end)
        : undefined,
  };

  return {
    decision_id: String(entry.decision_id ?? createDecisionId()),
    decision_text: decisionText,
    reasoning,
    source_span,
    confidence: Math.min(1, Math.max(0, confidence)),
  };
}

function parseProviderOutput(
  raw: string,
  transcriptId?: string
): DecisionLogEntry[] {
  const parsed = safeParseJson(raw);
  if (!parsed || typeof parsed !== 'object' || !('decisions' in parsed))
    return [];

  const decisions = Array.isArray((parsed as Record<string, unknown>).decisions)
    ? (parsed as Record<string, unknown>).decisions
    : [];
  return (decisions as unknown[])
    .map((entry: unknown) =>
      normalizeDecisionEntry(entry as Record<string, unknown>, transcriptId)
    )
    .filter((entry): entry is DecisionLogEntry => entry !== null);
}

function heuristicDecisionExtraction(
  request: DecisionExtractionRequest
): DecisionLogEntry[] {
  const text = request.text.trim();
  const sentences = text.split(/(?<=[.!?])\s+/g).filter(Boolean);
  const candidates = sentences.filter((sentence) =>
    /\b(should|will|must|decide|agree|action|plan|need|proposal|recommend)\b/i.test(
      sentence
    )
  );
  const decisions = candidates.slice(
    0,
    request.max_decisions ?? DEFAULT_MAX_DECISIONS
  );

  if (!decisions.length && sentences.length) {
    decisions.push(sentences[0]);
  }

  return decisions.map((decision) => ({
    decision_id: createDecisionId(),
    decision_text: decision.replace(/\s+/g, ' ').trim(),
    reasoning:
      'Extracted from the provided transcript text using a lightweight fallback extractor.',
    source_span: {
      transcript_id: request.transcript_id,
      segment_id: request.source_spans?.[0]?.segment_id ?? 'seg_unknown',
      start_seconds: request.source_spans?.[0]?.start_seconds ?? 0,
      end_seconds: request.source_spans?.[0]?.end_seconds ?? 0,
      text: decision,
      speaker: request.source_spans?.[0]?.speaker,
      character_start: 0,
      character_end: decision.length,
    },
    confidence: 0.65,
  }));
}

async function callChatProvider(
  prompt: string
): Promise<{ text: string; provider: string } | undefined> {
  if (process.env.OPENAI_API_KEY) {
    try {
      const modName = 'openai';
      const OpenAI = (await import(modName)).OpenAI;
      const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
      const response = await client.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.2,
        max_tokens: 1200,
      });
      const text = response?.choices?.[0]?.message?.content;
      if (typeof text === 'string') {
        return { text, provider: 'openai' };
      }
    } catch {
      // Fall back if provider is unavailable or package is not installed
    }
  }

  if (process.env.CLAUDE_API_KEY) {
    try {
      const modName = '@anthropic-ai/sdk';
      const { Anthropic } = await import(modName);
      const client = new Anthropic({ apiKey: process.env.CLAUDE_API_KEY });
      const response = await client.messages.create({
        model: 'claude-3-5-sonnet-20240620',
        max_tokens: 1200,
        temperature: 0.2,
        messages: [{ role: 'user', content: prompt }],
      });
      const text = response.content
        .filter((part) => part.type === 'text')
        .map((part) => ('text' in part ? part.text : ''))
        .join('');
      if (typeof text === 'string' && text.length > 0) {
        return { text, provider: 'claude' };
      }
    } catch {
      // Fall back if provider is unavailable or package is not installed
    }
  }

  return undefined;
}

export async function extractDecisions(
  request: DecisionExtractionRequest
): Promise<DecisionExtractionResponse> {
  const prompt = buildDecisionExtractionPrompt(request);
  const providerResult = await callChatProvider(prompt);

  if (providerResult) {
    const decisions = parseProviderOutput(
      providerResult.text,
      request.transcript_id
    );
    if (decisions.length) {
      return {
        decisions,
        prompt,
        provider: providerResult.provider,
      };
    }
  }

  return {
    decisions: heuristicDecisionExtraction(request),
    prompt,
    provider: providerResult?.provider,
  };
}

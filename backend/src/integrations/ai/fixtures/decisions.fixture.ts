import type { DecisionsResponse } from '../aiClient.js';

export const decisionsFixture: DecisionsResponse = {
  decisions: [
    {
      decision: 'We will hire Alice for the recruiting lead role.',
      madeBy: 'SPEAKER_00',
      rationale:
        'The team selected Alice based on her recruiting experience and the meeting discussed a direct hiring decision.',
      source_span: {
        transcript_id: '550e8400-e29b-41d4-a716-446655440000',
        segment_id: 'seg_0001',
        start_seconds: 0,
        end_seconds: 4.2,
        text: 'We will hire Alice for the recruiting lead role.',
        speaker: 'SPEAKER_00',
        character_start: 0,
        character_end: 48,
      },
    },
    {
      decision:
        'The team will review the roadmap and reduce the backlog by 20%.',
      madeBy: 'SPEAKER_01',
      rationale:
        'This was framed as a planning decision to tighten scope after the sprint review.',
      source_span: {
        transcript_id: '550e8400-e29b-41d4-a716-446655440000',
        segment_id: 'seg_0002',
        start_seconds: 4.3,
        end_seconds: 8.7,
        text: 'The team will review the roadmap and reduce the backlog by 20%.',
        speaker: 'SPEAKER_01',
        character_start: 0,
        character_end: 70,
      },
    },
  ],
};

import type { SkillMatchResponse } from '../aiClient.js';

export const skillMatchFixture: SkillMatchResponse = {
  task_id: 'task_001',
  matches: [
    {
      employee_id: 'emp_alice',
      name: 'Alice',
      matched_skill_ids: ['skill_recruiting', 'skill_interviewing'],
      skill_similarity: 0.92,
      workload_penalty: 0.08,
      final_score: 0.84,
      utilization: 0.25,
      available_fraction: 0.75,
      reason:
        'Matched by skill embedding similarity and low workload penalty for the hiring role.',
    },
    {
      employee_id: 'emp_bob',
      name: 'Bob',
      matched_skill_ids: ['skill_budgeting'],
      skill_similarity: 0.61,
      workload_penalty: 0.2,
      final_score: 0.49,
      utilization: 0.6,
      available_fraction: 0.4,
      reason:
        'Relevant experience exists, but workload and skills are a weaker fit.',
    },
  ],
};

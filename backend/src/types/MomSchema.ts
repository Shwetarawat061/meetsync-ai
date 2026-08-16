/**
 * MOM JSON Schema Design for MeetSync
 *
 * This schema is designed for:
 * - AI-generated MOM drafts
 * - Frontend review editor rendering
 * - ReviewVersion diff tracking per field
 * - Direct embedding in meeting export (DOCX, PDF)
 *
 * Architecture:
 * 1. Backend generates MOM with AI-extracted data
 * 2. Frontend receives JSON for editor rendering
 * 3. Reviewer edits and submits via ReviewVersion
 * 4. Export pipeline renders MOM to document format
 */

// ============================================================================
// TYPESCRIPT INTERFACES (Backend Models & API Responses)
// ============================================================================

export interface IAttendee {
  userId: string; // User ObjectId
  name: string; // User's display name
  role: string; // 'organizer' | 'participant'
  speakingDuration?: number; // seconds (from diarization)
  turnsTaken?: number; // number of speaking turns
}

export interface IKeyPoint {
  id: string; // UUID for tracking edits in ReviewVersion
  source: 'ai' | 'manual'; // 'ai' = extracted from transcript, 'manual' = added by reviewer
  category: 'decision' | 'milestone' | 'blocker' | 'general'; // for filtering/styling
  text: string; // Key point text
  speaker?: string; // who said it (from transcript attribution)
  timestamp?: number; // seconds in recording (linkable for playback)
  relatedTaskIds?: string[]; // ObjectIds of related tasks
  citations?: {
    start: number; // char offset in transcript
    end: number;
    text: string;
  }[];
}

export interface IDraftActionItem {
  id: string; // UUID (or Task ObjectId if it exists)
  source: 'ai' | 'manual';
  task: string; // Action item text
  assignee: string; // Who it's assigned to (name from transcript or dropdown)
  assigneeUserId?: string; // Optional User ObjectId for matching
  dueDate?: string; // ISO date or human-readable (e.g., "2 weeks", "end of sprint")
  priority?: 'high' | 'medium' | 'low'; // optional priority
  status: 'draft' | 'assigned' | 'in_progress' | 'done'; // status in Task model
  requiredSkills?: string[]; // computed or manual entry
  relatedDecisions?: string[]; // UUID refs to key points (decisions)
  citations?: {
    start: number;
    end: number;
    text: string;
  }[];
}

export interface IMomJSON {
  // Meeting reference
  meetingId: string; // ObjectId
  title: string; // from Meeting.title

  // Core MOM fields (editable in review editor)
  attendees: IAttendee[];
  summary: string; // Rich text / Markdown (30-60 sentences)
  keyPoints: IKeyPoint[];
  draftActionItems: IDraftActionItem[];

  // Metadata
  generatedAt: Date; // when AI generated this MOM
  source: 'ai' | 'manual'; // was it auto-generated or manually created?
  version: number; // MOM version (increments with ReviewVersion updates)

  // Optional: AI confidence/quality metrics
  metrics?: {
    transcriptAccuracy?: number; // 0-100, confidence in speech-to-text
    summaryCoherence?: number; // 0-100
    actionItemsExtraction?: number; // 0-100
  };
}

/**
 * Response wrapper for GET /api/meetings/:id/mom
 * Includes both the MOM content and review metadata
 */
export interface IMomResponse {
  mom: IMomJSON;
  reviewVersion?: {
    version: number;
    reviewedBy: string; // User name
    reviewedAt: Date;
    locked: boolean;
  };
  editableBy: boolean; // current user can edit?
  canLock: boolean; // current user can lock?
}

/**
 * Request body for updating MOM (reviewer edits)
 * Posted to reviewRoutes
 */
export interface IMomEditRequest {
  meetingId: string;
  version: number; // must match current version
  fields: {
    field: 'summary' | 'keyPoints' | 'draftActionItems' | 'attendees';
    value: string | IKeyPoint[] | IDraftActionItem[] | IAttendee[];
  }[];
}

// ============================================================================
// EXAMPLE JSON RESPONSE (What Frontend Receives)
// ============================================================================

export const MOM_RESPONSE_EXAMPLE: IMomResponse = {
  mom: {
    meetingId: '507f1f77bcf86cd799439011',
    title: 'Q3 Product Planning Sprint Kickoff',

    attendees: [
      {
        userId: '507f1f77bcf86cd799439001',
        name: 'Alice Johnson',
        role: 'organizer',
        speakingDuration: 450,
        turnsTaken: 12,
      },
      {
        userId: '507f1f77bcf86cd799439002',
        name: 'Bob Chen',
        role: 'participant',
        speakingDuration: 320,
        turnsTaken: 8,
      },
      {
        userId: '507f1f77bcf86cd799439003',
        name: 'Carol Martinez',
        role: 'participant',
        speakingDuration: 180,
        turnsTaken: 5,
      },
    ],

    summary: `The team met to plan Q3 deliverables and align on priorities. Alice opened with the vision: deliver the new search functionality and complete the mobile app redesign. 
    
    The conversation centered on three areas: (1) Timeline feasibility given team capacity, (2) Technical debt paydown in parallel with feature work, and (3) Dependencies with the design team.
    
    Alice emphasized that search is critical for user retention—we're losing users to faster competitors. Bob raised concerns about the mobile redesign timeline; he estimates 6 weeks vs. the planned 4. After discussion, the team agreed to a phased approach: core UX in 4 weeks, polish in 5-6.
    
    Key decision: prioritize search completion over mobile polish in Q3; defer cosmetic improvements to Q4. The team will start with search immediately, with Bob leading the backend indexing work.`,

    keyPoints: [
      {
        id: 'kp-001',
        source: 'ai',
        category: 'milestone',
        text: 'Q3 priorities: new search functionality + mobile app redesign',
        speaker: 'Alice Johnson',
        timestamp: 45,
        citations: [
          {
            start: 120,
            end: 210,
            text: 'deliver the new search functionality and complete the mobile app redesign',
          },
        ],
      },
      {
        id: 'kp-002',
        source: 'ai',
        category: 'blocker',
        text: 'Mobile redesign estimated at 6 weeks, not the planned 4 weeks',
        speaker: 'Bob Chen',
        timestamp: 280,
        citations: [
          {
            start: 315,
            end: 380,
            text: 'I estimate 6 weeks vs. the planned 4',
          },
        ],
      },
      {
        id: 'kp-003',
        source: 'ai',
        category: 'decision',
        text: 'Search functionality prioritized over mobile polish in Q3; mobile cosmetics deferred to Q4',
        speaker: 'Alice Johnson',
        timestamp: 520,
        citations: [
          {
            start: 580,
            end: 670,
            text: 'prioritize search completion over mobile polish in Q3; defer cosmetic improvements to Q4',
          },
        ],
        relatedTaskIds: [
          '507f1f77bcf86cd799439101',
          '507f1f77bcf86cd799439102',
        ],
      },
    ],

    draftActionItems: [
      {
        id: '507f1f77bcf86cd799439101',
        source: 'ai',
        task: 'Design and implement backend search indexing for full-text search capability',
        assignee: 'Bob Chen',
        assigneeUserId: '507f1f77bcf86cd799439002',
        dueDate: '2025-09-15',
        priority: 'high',
        status: 'draft',
        requiredSkills: ['backend', 'database-design', 'elasticsearch'],
        relatedDecisions: ['kp-003'],
        citations: [
          {
            start: 700,
            end: 750,
            text: 'Bob leading the backend indexing work',
          },
        ],
      },
      {
        id: '507f1f77bcf86cd799439102',
        source: 'ai',
        task: 'Deliver core mobile UX and layout redesign (phase 1)',
        assignee: 'Carol Martinez',
        assigneeUserId: '507f1f77bcf86cd799439003',
        dueDate: '2025-08-31',
        priority: 'high',
        status: 'draft',
        requiredSkills: ['frontend', 'mobile', 'ux'],
        relatedDecisions: ['kp-003'],
      },
      {
        id: 'manual-001',
        source: 'manual',
        task: 'Define search performance SLOs and acceptance criteria',
        assignee: 'Alice Johnson',
        assigneeUserId: '507f1f77bcf86cd799439001',
        dueDate: '2025-08-20',
        priority: 'high',
        status: 'draft',
        requiredSkills: ['product', 'requirements'],
        relatedDecisions: ['kp-003'],
      },
      {
        id: 'manual-002',
        source: 'manual',
        task: 'Schedule design team alignment meeting (design <-> eng dependency)',
        assignee: 'Alice Johnson',
        assigneeUserId: '507f1f77bcf86cd799439001',
        dueDate: '2025-08-18',
        priority: 'medium',
        status: 'draft',
      },
    ],

    generatedAt: new Date('2025-08-14T14:30:00Z'),
    source: 'ai',
    version: 1,

    metrics: {
      transcriptAccuracy: 94,
      summaryCoherence: 87,
      actionItemsExtraction: 92,
    },
  },

  reviewVersion: {
    version: 0,
    reviewedBy: 'None',
    reviewedAt: new Date(),
    locked: false,
  },

  editableBy: true,
  canLock: true,
};

// ============================================================================
// FRONTEND EDITOR RENDERING HINTS
// ============================================================================

/**
 * Use these type discriminators to render appropriate editor widgets:
 */
export const MOM_EDITOR_WIDGETS = {
  attendees: {
    type: 'table',
    editable: false, // derived from Meeting.participants
    columns: ['name', 'role', 'speakingDuration', 'turnsTaken'],
  },

  summary: {
    type: 'rich-text-editor',
    editable: true,
    placeholder: 'Enter meeting summary (markdown supported)',
    minLength: 200,
    maxLength: 5000,
  },

  keyPoints: {
    type: 'list-with-tagging',
    editable: true,
    addButton: true,
    deleteButton: true,
    itemFields: {
      text: { type: 'text', required: true },
      category: {
        type: 'select',
        options: ['decision', 'milestone', 'blocker', 'general'],
        required: false,
      },
      speaker: { type: 'text', required: false },
      timestamp: { type: 'number', required: false, min: 0 },
    },
  },

  draftActionItems: {
    type: 'list-with-inline-edit',
    editable: true,
    addButton: true,
    deleteButton: true,
    itemFields: {
      task: { type: 'text', required: true, multiline: true },
      assignee: { type: 'autocomplete', required: true, source: 'attendees' },
      dueDate: { type: 'date-picker', required: false },
      priority: {
        type: 'select',
        options: ['high', 'medium', 'low'],
        required: false,
      },
      status: {
        type: 'select',
        options: ['draft', 'assigned', 'in_progress', 'done'],
        required: false,
      },
    },
  },
};

import { Router, Response } from 'express';
import { Types } from 'mongoose';
import { Meeting } from '../models/Meeting.js';
import { Mom, IMom, IKeyPoint, IDraftActionItem } from '../models/Mom.js';
import { ReviewVersion } from '../models/ReviewVersion.js';
import { requireAuth, AuthRequest } from '../middleware/authMiddleware.js';

const router = Router({ mergeParams: true });

/**
 * GET /api/meetings/:id/mom
 * Retrieve the MOM for a meeting with review metadata
 */
router.get('/', requireAuth, async (req: AuthRequest, res: Response) => {
  try {
    const meetingId = req.params.id as string;
    const meeting = await Meeting.findById(meetingId);

    if (!meeting) {
      res.status(404).json({ message: 'Meeting not found' });
      return;
    }

    if (req.user!.role === 'employee') {
      const uid = req.user!.sub;
      const allowed =
        meeting.createdBy.toString() === uid ||
        meeting.participants.some((p) => p.toString() === uid);
      if (!allowed) {
        res.status(403).json({ message: 'Access denied' });
        return;
      }
    }

    if (meeting.processingStatus !== 'completed') {
      res.status(409).json({
        message: 'MoM not available yet',
        processingStatus: meeting.processingStatus,
      });
      return;
    }

    const mom = await Mom.findOne({ meetingId: meetingId }).populate([
      { path: 'attendees.userId', select: 'name email' },
      { path: 'draftActionItems.assigneeUserId', select: 'name email' },
      { path: 'draftActionItems.relatedTaskIds' },
      { path: 'keyPoints.relatedTaskIds' },
    ]);

    if (!mom) {
      res.status(404).json({ message: 'MoM not found for this meeting' });
      return;
    }

    // Get latest review version for metadata
    const latestReview = await ReviewVersion.findOne({ meetingId })
      .sort({ version: -1 })
      .populate('reviewedBy', 'name');

    res.json({
      mom,
      reviewVersion: latestReview
        ? {
            version: latestReview.version,
            reviewedBy: (latestReview.reviewedBy as any)?.name || 'Unknown',
            reviewedAt: latestReview.createdAt,
            locked: latestReview.locked,
          }
        : null,
      editableBy: req.user!.role === 'admin' || req.user!.sub === meeting.createdBy.toString(),
      canLock: req.user!.role === 'admin',
    });
  } catch (error) {
    console.error('Error fetching MOM:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

/**
 * PATCH /api/meetings/:id/mom
 * Update MOM fields (attendees, summary, keyPoints, draftActionItems)
 * Creates a ReviewVersion entry to track changes
 */
router.patch('/', requireAuth, async (req: AuthRequest, res: Response) => {
  try {
    const { summary, keyPoints, draftActionItems, attendees } = req.body;
    const meetingId = req.params.id as string;

    const meeting = await Meeting.findById(meetingId);
    if (!meeting) {
      res.status(404).json({ message: 'Meeting not found' });
      return;
    }

    if (req.user!.role === 'employee' && req.user!.sub !== meeting.createdBy.toString()) {
      res.status(403).json({ message: 'Access denied' });
      return;
    }

    const mom = await Mom.findOne({ meetingId });
    if (!mom) {
      res.status(404).json({ message: 'MoM not found' });
      return;
    }

    const latestReview = await ReviewVersion.findOne({ meetingId }).sort({ version: -1 });

    if (latestReview?.locked) {
      res.status(409).json({ message: 'Cannot edit locked MoM version' });
      return;
    }

    if (summary !== undefined) mom.summary = summary;
    if (keyPoints !== undefined) mom.keyPoints = keyPoints;
    if (draftActionItems !== undefined) mom.draftActionItems = draftActionItems;
    if (attendees !== undefined) mom.attendees = attendees;

    mom.version = (mom.version || 1) + 1;
    await mom.save();

    const nextVersion = (latestReview?.version || 0) + 1;
    await ReviewVersion.create({
      meetingId,
      version: nextVersion,
      reviewedBy: req.user!.sub,
      fields: [
        { field: 'summary', source: 'manual', original: mom.summary, edited: summary || mom.summary, diff: [] },
        { field: 'keyPoints', source: 'manual', original: JSON.stringify(mom.keyPoints), edited: JSON.stringify(keyPoints || mom.keyPoints), diff: [] },
        { field: 'draftActionItems', source: 'manual', original: JSON.stringify(mom.draftActionItems), edited: JSON.stringify(draftActionItems || mom.draftActionItems), diff: [] },
      ],
    });

    res.json({
      message: 'MoM updated successfully',
      mom,
      version: nextVersion,
    });
  } catch (error) {
    console.error('Error updating MOM:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

/**
 * POST /api/meetings/:id/mom/lock
 * Lock the current MOM version (prevent further edits)
 */
router.post('/:momId/lock', requireAuth, async (req: AuthRequest, res: Response) => {
  try {
    if (req.user!.role !== 'admin') {
      res.status(403).json({ message: 'Only admins can lock MoM' });
      return;
    }

    const meetingId = req.params.id as string;
    const meeting = await Meeting.findById(meetingId);
    if (!meeting) {
      res.status(404).json({ message: 'Meeting not found' });
      return;
    }

    const latestReview = await ReviewVersion.findOne({ meetingId }).sort({ version: -1 });

    if (!latestReview) {
      res.status(404).json({ message: 'No review version found' });
      return;
    }

    latestReview.locked = true;
    latestReview.lockedAt = new Date();
    latestReview.lockedBy = new Types.ObjectId(req.user!.sub);
    await latestReview.save();

    res.json({
      message: 'MoM version locked successfully',
      version: latestReview.version,
      lockedAt: latestReview.lockedAt,
    });
  } catch (error) {
    console.error('Error locking MOM:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
});

export default router;

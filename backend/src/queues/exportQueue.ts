import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { Queue, Worker, Job } from 'bullmq';
import {
  Document as DocxDocument,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  Table,
  TableRow,
  TableCell,
  WidthType,
  BorderStyle,
} from 'docx';
import puppeteer from 'puppeteer';
import { env } from '../config/env.js';
import { ExportJob, type IAuditMarker } from '../models/ExportJob.js';
import { Meeting } from '../models/Meeting.js';
import { Mom } from '../models/Mom.js';
import { Decision } from '../models/Decision.js';
import { Task } from '../models/Task.js';
import { Deadline } from '../models/Deadline.js';
import { EffectivenessScore } from '../models/EffectivenessScore.js';
import { ReviewVersion } from '../models/ReviewVersion.js';

export const EXPORT_QUEUE = 'exportGeneration';

const connection = { url: env.REDIS_URL };

// Lazy singleton — not instantiated at import time so test processes don't hang
let _queue: Queue | null = null;
export function getExportQueue(): Queue {
  if (!_queue) _queue = new Queue(EXPORT_QUEUE, { connection });
  return _queue;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const EXPORTS_DIR = path.join(os.tmpdir(), 'meetsync-exports');
fs.mkdirSync(EXPORTS_DIR, { recursive: true });

function auditTag(marker: IAuditMarker | undefined): string {
  if (!marker) return '';
  return marker.source === 'manual'
    ? ` [edited v${marker.reviewVersion} by ${marker.reviewedBy}]`
    : ` [ai-generated v${marker.reviewVersion}]`;
}

function noBorder() {
  const b = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
  return { top: b, bottom: b, left: b, right: b };
}

// ── DOCX builder ──────────────────────────────────────────────────────────────

async function buildDocx(jobId: string, meetingId: string): Promise<string> {
  const [meeting, mom, decisions, tasks, deadlines, score, latestReview] =
    await Promise.all([
      Meeting.findById(meetingId),
      Mom.findOne({ meetingId }),
      Decision.find({ meetingId }).sort({ createdAt: 1 }),
      Task.find({ meetingId }).sort({ createdAt: 1 }),
      Deadline.find({ meetingId }).sort({ deadline: 1 }),
      EffectivenessScore.findOne({ meetingId }),
      ReviewVersion.findOne({ meetingId }).sort({ version: -1 }),
    ]);

  const markerMap = new Map<string, IAuditMarker>();
  if (latestReview) {
    for (const f of latestReview.fields) {
      markerMap.set(f.field, {
        field: f.field,
        source: f.source,
        reviewVersion: latestReview.version,
        reviewedBy: latestReview.reviewedBy.toString(),
      });
    }
  }

  const h1 = (text: string) =>
    new Paragraph({
      text,
      heading: HeadingLevel.HEADING_1,
      spacing: { after: 120 },
    });

  const h2 = (text: string) =>
    new Paragraph({
      text,
      heading: HeadingLevel.HEADING_2,
      spacing: { after: 80 },
    });

  const body = (text: string, italic = false) =>
    new Paragraph({
      children: [new TextRun({ text, italics: italic, size: 22 })],
      spacing: { after: 60 },
    });

  const auditPara = (marker: IAuditMarker | undefined): Paragraph[] => {
    if (!marker) return [];
    const color = marker.source === 'manual' ? 'C0392B' : '2980B9';
    return [
      new Paragraph({
        children: [
          new TextRun({
            text: auditTag(marker),
            color,
            size: 18,
            italics: true,
          }),
        ],
        spacing: { after: 40 },
      }),
    ];
  };

  const children: (Paragraph | Table)[] = [];

  // Title block
  children.push(
    new Paragraph({
      children: [
        new TextRun({
          text: meeting?.title ?? 'Meeting Report',
          bold: true,
          size: 48,
        }),
      ],
      heading: HeadingLevel.TITLE,
      spacing: { after: 200 },
    }),
    body(`Date: ${meeting?.scheduledAt?.toISOString().slice(0, 10) ?? 'N/A'}`),
    body(`Export generated: ${new Date().toISOString()}`),
    body(`Meeting ID: ${meetingId}`)
  );

  // MoM
  if (mom) {
    children.push(h1('Minutes of Meeting'), h2('Summary'));
    children.push(body(mom.summary + auditTag(markerMap.get('summary'))));
    children.push(...auditPara(markerMap.get('summary')));

    if (mom.agenda?.length) {
      children.push(h2('Agenda'));
      mom.agenda.forEach((item, i) => {
        const marker = i === 0 ? markerMap.get('agenda') : undefined;
        children.push(body(`${i + 1}. ${item}${auditTag(marker)}`));
      });
      children.push(...auditPara(markerMap.get('agenda')));
    }

    if (mom.discussionPoints?.length) {
      children.push(h2('Discussion Points'));
      mom.discussionPoints.forEach((dp) =>
        children.push(body(`[${dp.speaker}] ${dp.point}`))
      );
    }
  }

  // Decisions
  if (decisions.length) {
    children.push(h1('Decisions'));
    decisions.forEach((d, i) => {
      const decMarker = markerMap.get(`decisions[${i}].decision`);
      const ratMarker = markerMap.get(`decisions[${i}].rationale`);
      children.push(
        body(`${i + 1}. ${d.decision}${auditTag(decMarker)} — by ${d.madeBy}`)
      );
      children.push(...auditPara(decMarker));
      if (d.rationale) {
        children.push(
          body(`   Rationale: ${d.rationale}${auditTag(ratMarker)}`, true)
        );
        children.push(...auditPara(ratMarker));
      }
    });
  }

  // Tasks table
  if (tasks.length) {
    children.push(h1('Action Items'));
    const rows = [
      new TableRow({
        children: ['Task', 'Assignee', 'Due Date', 'Status'].map(
          (h) =>
            new TableCell({
              children: [
                new Paragraph({
                  children: [new TextRun({ text: h, bold: true })],
                }),
              ],
              borders: noBorder(),
            })
        ),
        tableHeader: true,
      }),
      ...tasks.map(
        (t) =>
          new TableRow({
            children: [t.task, t.assignee, t.dueDate ?? '—', t.status].map(
              (v) =>
                new TableCell({
                  children: [new Paragraph(v)],
                  borders: noBorder(),
                })
            ),
          })
      ),
    ];
    children.push(
      new Table({ rows, width: { size: 100, type: WidthType.PERCENTAGE } })
    );
  }

  // Deadlines
  if (deadlines.length) {
    children.push(h1('Deadlines'));
    deadlines.forEach((dl) =>
      children.push(
        body(
          `• ${dl.description} — ${dl.assignee} by ${dl.deadline.toISOString().slice(0, 10)}`
        )
      )
    );
  }

  // Effectiveness score
  if (score) {
    children.push(h1('Effectiveness Score'));
    children.push(body(`Overall: ${score.score}/100`));
    children.push(
      body(
        `Decisions: ${score.breakdown.decisionsScore}  |  Key Points: ${score.breakdown.keyPointsCoverage}  |  Participation: ${score.breakdown.participationBalance}`
      )
    );
    if (score.suggestions.length) {
      children.push(h2('Suggestions'));
      score.suggestions.forEach((s) => children.push(body(`• ${s}`)));
    }
  }

  // Audit trail summary
  if (latestReview) {
    children.push(h1('Audit Trail'));
    children.push(
      body(
        `Review version: ${latestReview.version}  |  Locked: ${latestReview.locked}`
      )
    );
    latestReview.fields.forEach((f) =>
      children.push(
        body(
          `  ${f.field}: ${f.source === 'manual' ? '✎ manually edited' : '✓ ai-generated'}`
        )
      )
    );
  }

  const doc = new DocxDocument({ sections: [{ children }] });
  const buffer = await Packer.toBuffer(doc);
  const filePath = path.join(EXPORTS_DIR, `${jobId}.docx`);
  fs.writeFileSync(filePath, buffer);
  return filePath;
}

// ── PDF builder ───────────────────────────────────────────────────────────────

async function buildPdf(jobId: string, meetingId: string): Promise<string> {
  const [meeting, mom, decisions, tasks, deadlines, score, latestReview] =
    await Promise.all([
      Meeting.findById(meetingId),
      Mom.findOne({ meetingId }),
      Decision.find({ meetingId }).sort({ createdAt: 1 }),
      Task.find({ meetingId }).sort({ createdAt: 1 }),
      Deadline.find({ meetingId }).sort({ deadline: 1 }),
      EffectivenessScore.findOne({ meetingId }),
      ReviewVersion.findOne({ meetingId }).sort({ version: -1 }),
    ]);

  const markerMap = new Map<string, IAuditMarker>();
  if (latestReview) {
    for (const f of latestReview.fields) {
      markerMap.set(f.field, {
        field: f.field,
        source: f.source,
        reviewVersion: latestReview.version,
        reviewedBy: latestReview.reviewedBy.toString(),
      });
    }
  }

  const esc = (s: string) =>
    s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  const auditSpan = (marker: IAuditMarker | undefined) => {
    if (!marker) return '';
    const color = marker.source === 'manual' ? '#c0392b' : '#2980b9';
    return `<span style="color:${color};font-size:11px;font-style:italic"> ${esc(auditTag(marker))}</span>`;
  };

  let html = `<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
  body{font-family:Arial,sans-serif;font-size:13px;margin:40px;color:#222}
  h1{font-size:22px;border-bottom:2px solid #333;padding-bottom:4px;margin-top:28px}
  h2{font-size:16px;margin-top:18px;color:#444}
  table{border-collapse:collapse;width:100%;margin-top:8px}
  th{background:#f0f0f0;text-align:left;padding:6px 8px}
  td{padding:5px 8px;border-bottom:1px solid #ddd}
  .title{font-size:28px;font-weight:bold;margin-bottom:6px}
  .meta{color:#666;font-size:12px;margin-bottom:20px}
</style></head><body>`;

  html += `<div class="title">${esc(meeting?.title ?? 'Meeting Report')}</div>`;
  html += `<div class="meta">Date: ${meeting?.scheduledAt?.toISOString().slice(0, 10) ?? 'N/A'} &nbsp;|&nbsp; Export: ${new Date().toISOString()} &nbsp;|&nbsp; ID: ${meetingId}</div>`;

  if (mom) {
    html += `<h1>Minutes of Meeting</h1><h2>Summary</h2><p>${esc(mom.summary)}${auditSpan(markerMap.get('summary'))}</p>`;
    if (mom.agenda?.length) {
      html += `<h2>Agenda</h2><ol>`;
      mom.agenda.forEach((a) => {
        html += `<li>${esc(a)}</li>`;
      });
      html += `</ol>${auditSpan(markerMap.get('agenda'))}`;
    }
    if (mom.discussionPoints?.length) {
      html += `<h2>Discussion Points</h2><ul>`;
      mom.discussionPoints.forEach((dp) => {
        html += `<li><b>${esc(dp.speaker)}:</b> ${esc(dp.point)}</li>`;
      });
      html += `</ul>`;
    }
  }

  if (decisions.length) {
    html += `<h1>Decisions</h1><ol>`;
    decisions.forEach((d, i) => {
      html += `<li>${esc(d.decision)}${auditSpan(markerMap.get(`decisions[${i}].decision`))} — <i>by ${esc(d.madeBy)}</i>`;
      if (d.rationale) {
        html += `<br><small>Rationale: ${esc(d.rationale)}${auditSpan(markerMap.get(`decisions[${i}].rationale`))}</small>`;
      }
      html += `</li>`;
    });
    html += `</ol>`;
  }

  if (tasks.length) {
    html += `<h1>Action Items</h1><table><tr><th>Task</th><th>Assignee</th><th>Due Date</th><th>Status</th></tr>`;
    tasks.forEach((t) => {
      html += `<tr><td>${esc(t.task)}</td><td>${esc(t.assignee)}</td><td>${esc(t.dueDate ?? '—')}</td><td>${esc(t.status)}</td></tr>`;
    });
    html += `</table>`;
  }

  if (deadlines.length) {
    html += `<h1>Deadlines</h1><ul>`;
    deadlines.forEach((dl) => {
      html += `<li>${esc(dl.description)} — ${esc(dl.assignee)} by ${dl.deadline.toISOString().slice(0, 10)}</li>`;
    });
    html += `</ul>`;
  }

  if (score) {
    html += `<h1>Effectiveness Score</h1><p>Overall: <b>${score.score}/100</b></p>`;
    html += `<p>Decisions: ${score.breakdown.decisionsScore} | Key Points: ${score.breakdown.keyPointsCoverage} | Participation: ${score.breakdown.participationBalance}</p>`;
    if (score.suggestions.length) {
      html += `<h2>Suggestions</h2><ul>`;
      score.suggestions.forEach((s) => {
        html += `<li>${esc(s)}</li>`;
      });
      html += `</ul>`;
    }
  }

  if (latestReview) {
    html += `<h1>Audit Trail</h1><p>Review version: <b>${latestReview.version}</b> &nbsp;|&nbsp; Locked: ${latestReview.locked}</p><ul>`;
    latestReview.fields.forEach((f) => {
      const icon = f.source === 'manual' ? '✎' : '✓';
      const color = f.source === 'manual' ? '#c0392b' : '#2980b9';
      html += `<li><span style="color:${color}">${icon}</span> <b>${esc(f.field)}</b>: ${f.source}</li>`;
    });
    html += `</ul>`;
  }

  html += `</body></html>`;

  const browser = await puppeteer.launch({
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  });
  try {
    const page = await browser.newPage();
    await page.setContent(html, { waitUntil: 'load' });
    const filePath = path.join(EXPORTS_DIR, `${jobId}.pdf`);
    await page.pdf({
      path: filePath,
      format: 'A4',
      margin: { top: '20mm', bottom: '20mm', left: '15mm', right: '15mm' },
    });
    return filePath;
  } finally {
    await browser.close();
  }
}

// ── Worker ────────────────────────────────────────────────────────────────────

export function startExportWorker() {
  const worker = new Worker(
    EXPORT_QUEUE,
    async (job: Job) => {
      const { jobId, meetingId, format } = job.data as {
        jobId: string;
        meetingId: string;
        format: 'docx' | 'pdf';
      };

      await ExportJob.findByIdAndUpdate(jobId, { status: 'processing' });

      try {
        const filePath =
          format === 'pdf'
            ? await buildPdf(jobId, meetingId)
            : await buildDocx(jobId, meetingId);

        await ExportJob.findByIdAndUpdate(jobId, { status: 'done', filePath });
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        await ExportJob.findByIdAndUpdate(jobId, {
          status: 'failed',
          errorMessage: msg,
        });
        throw err;
      }
    },
    { connection }
  );

  worker.on('failed', (job, err) => {
    console.error(`[export-worker] job ${job?.id} failed:`, err.message);
  });

  return worker;
}

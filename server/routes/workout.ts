import { Router } from 'express';
import { z } from 'zod';
import { appendRows, readAllRows } from '../integrations/excel';

const WorkoutRowSchema = z.object({
    date: z.string().min(1).max(40),
    program: z.string().min(1).max(40),
    exercise: z.string().min(1).max(120),
    set_idx: z.number().int().min(0).max(100),
    weight: z.number().min(0).max(2000),
    pr_at_time: z.number().min(0).max(2000),
});
const SyncBody = z.object({ rows: z.array(WorkoutRowSchema).min(1).max(200) });

export const workoutRouter = Router();

workoutRouter.get('/history', async (_req, res) => {
    try {
        const rows = await readAllRows();
        res.json({ rows });
    } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        res.status(502).json({ error: msg });
    }
});

workoutRouter.post('/sync', async (req, res) => {
    const parsed = SyncBody.safeParse(req.body);
    if (!parsed.success) {
        res.status(400).json({ ok: false, error: 'invalid input', issues: parsed.error.issues });
        return;
    }
    try {
        await appendRows(parsed.data.rows);
        res.json({ ok: true });
    } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        res.status(502).json({ ok: false, error: msg });
    }
});

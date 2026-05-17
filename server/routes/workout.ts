import { Router } from 'express';
import { appendRows, readAllRows, WorkoutRow } from '../integrations/excel';

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
    const rows = req.body?.rows;
    if (!Array.isArray(rows)) {
        res.status(400).json({ ok: false, error: 'rows array required' });
        return;
    }
    try {
        await appendRows(rows as WorkoutRow[]);
        res.json({ ok: true });
    } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        res.status(502).json({ ok: false, error: msg });
    }
});

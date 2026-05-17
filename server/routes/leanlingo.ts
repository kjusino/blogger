import { Router } from 'express';
import { bustCache, getQuestionTree } from '../leanlingo/cache';
import { appendAttempts, readAllAttempts } from '../leanlingo/excel';
import type { AttemptRow } from '../leanlingo/types';

export const leanlingoRouter = Router();

leanlingoRouter.get('/questions', async (_req, res) => {
    try {
        const tree = await getQuestionTree();
        res.json(tree);
    } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        res.status(502).json({ error: msg });
    }
});

leanlingoRouter.get('/progress', async (_req, res) => {
    try {
        const rows = await readAllAttempts();
        res.json({ rows });
    } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        res.status(502).json({ error: msg });
    }
});

leanlingoRouter.post('/attempt', async (req, res) => {
    const rows = req.body?.rows;
    if (!Array.isArray(rows)) {
        res.status(400).json({ ok: false, error: 'rows array required' });
        return;
    }
    try {
        await appendAttempts(rows as AttemptRow[]);
        res.json({ ok: true });
    } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        res.status(502).json({ ok: false, error: msg });
    }
});

leanlingoRouter.post('/refresh', (_req, res) => {
    bustCache();
    res.json({ ok: true });
});

import { Router } from 'express';
import { z } from 'zod';
import { bustCache, getQuestionTree } from '../leanlingo/cache';
import { appendAttempts, readAllAttempts } from '../leanlingo/excel';

const AttemptSchema = z.object({
    timestamp: z.string().datetime(),
    lesson_id: z.string().regex(/^w\d{1,2}-u\d{1,2}-l\d{1,2}$/),
    question_id: z.string().regex(/^w\d{1,2}-u\d{1,2}-l\d{1,2}-[a-z]$/),
    attempts: z.number().int().min(1).max(10),
    correct: z.boolean(),
    xp_awarded: z.number().int().min(0).max(100),
});
const AttemptsBody = z.object({ rows: z.array(AttemptSchema).min(1).max(50) });

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
    const parsed = AttemptsBody.safeParse(req.body);
    if (!parsed.success) {
        res.status(400).json({ ok: false, error: 'invalid input', issues: parsed.error.issues });
        return;
    }
    try {
        await appendAttempts(parsed.data.rows);
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

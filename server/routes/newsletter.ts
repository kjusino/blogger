import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { emailExists, addSubscriber } from '../newsletter/excel';

const SubscribeSchema = z.object({
    email: z.string().email().max(254),
    route: z.string().min(1).max(100),
});

const RATE_WINDOW_MS = 60 * 1000;
const RATE_MAX = 5;
const hits = new Map<string, { count: number; resetAt: number }>();

function rateLimit(req: Request, res: Response): boolean {
    const ip = req.ip ?? 'unknown';
    const now = Date.now();
    let entry = hits.get(ip);
    if (!entry || now > entry.resetAt) {
        entry = { count: 0, resetAt: now + RATE_WINDOW_MS };
        hits.set(ip, entry);
    }
    entry.count++;
    if (entry.count > RATE_MAX) {
        res.status(429).json({ error: 'Too many requests' });
        return false;
    }
    return true;
}

setInterval(() => {
    const now = Date.now();
    for (const [ip, entry] of hits) {
        if (now > entry.resetAt) hits.delete(ip);
    }
}, 5 * 60 * 1000);

export const newsletterRouter = Router();

newsletterRouter.post('/subscribe', async (req: Request, res: Response) => {
    if (!rateLimit(req, res)) return;

    const parsed = SubscribeSchema.safeParse(req.body);
    if (!parsed.success) {
        res.status(400).json({ error: 'Invalid input', issues: parsed.error.issues });
        return;
    }

    try {
        const exists = await emailExists(parsed.data.email);
        if (exists) {
            res.json({ ok: true });
            return;
        }
        await addSubscriber(parsed.data.email, parsed.data.route);
        res.json({ ok: true });
    } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        console.error('[newsletter] subscribe error:', msg);
        res.status(502).json({ error: 'Service unavailable' });
    }
});

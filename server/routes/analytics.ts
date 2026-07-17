import { Router, Request, Response } from 'express';
import { z } from 'zod';
import { enqueue } from '../analytics/buffer';
import { getAnalyticsSummary, bustAnalyticsCache } from '../analytics/cache';
import type { EventRow } from '../analytics/excel';
import { hashIp } from '../analytics/identity';
import { env } from '../env';

const EventSchema = z.object({
    event: z.enum(['view', 'audio_play', 'audio_complete', 'video_play', 'video_complete']),
    route: z.string().min(1).max(100),
    session_id: z.string().uuid(),
    referrer: z.string().max(500).default(''),
    device: z.enum(['desktop', 'mobile', 'tablet']),
    read_seconds: z.number().int().min(0).max(86400).default(0),
});

// Simple in-memory rate limiter: max requests per window per IP
const RATE_WINDOW_MS = 60 * 1000;
const RATE_MAX = 30;
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

// Clean up stale entries every 5 minutes
setInterval(() => {
    const now = Date.now();
    for (const [ip, entry] of hits) {
        if (now > entry.resetAt) hits.delete(ip);
    }
}, 5 * 60 * 1000);

export const analyticsPublicRouter = Router();

analyticsPublicRouter.post('/event', (req: Request, res: Response) => {
    if (!rateLimit(req, res)) return;

    const parsed = EventSchema.safeParse(req.body);
    if (!parsed.success) {
        res.status(400).json({ error: 'invalid input', issues: parsed.error.issues });
        return;
    }

    const row: EventRow = {
        timestamp: new Date().toISOString(),
        session_id: parsed.data.session_id,
        event: parsed.data.event,
        route: parsed.data.route,
        referrer: parsed.data.referrer,
        device: parsed.data.device,
        read_seconds: parsed.data.read_seconds,
        ip_hash: hashIp(req.ip ?? 'unknown', env.SESSION_SECRET),
    };

    enqueue(row);
    res.json({ ok: true });
});

export const analyticsPrivateRouter = Router();

analyticsPrivateRouter.get('/summary', async (_req: Request, res: Response) => {
    try {
        const summary = await getAnalyticsSummary();
        res.json(summary);
    } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        res.status(502).json({ error: msg });
    }
});

analyticsPrivateRouter.post('/refresh', (_req: Request, res: Response) => {
    bustAnalyticsCache();
    res.json({ ok: true });
});

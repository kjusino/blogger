import { Router, Request, Response, NextFunction } from 'express';
import bcrypt from 'bcryptjs';
import crypto from 'crypto';
import { env, isProd } from './env';

const COOKIE_NAME = 'personal_session';
const SESSION_TTL_MS = 7 * 24 * 60 * 60 * 1000;

function sign(payload: string): string {
    const h = crypto.createHmac('sha256', env.SESSION_SECRET).update(payload).digest('base64url');
    return `${payload}.${h}`;
}

function verify(token: string | undefined): boolean {
    if (!token) return false;
    const lastDot = token.lastIndexOf('.');
    if (lastDot < 0) return false;
    const payload = token.slice(0, lastDot);
    const provided = token.slice(lastDot + 1);
    const expected = crypto.createHmac('sha256', env.SESSION_SECRET).update(payload).digest('base64url');
    const a = Buffer.from(provided);
    const b = Buffer.from(expected);
    if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return false;
    const issuedAt = parseInt(payload, 10);
    if (!Number.isFinite(issuedAt)) return false;
    return Date.now() - issuedAt < SESSION_TTL_MS;
}

function setSession(res: Response): void {
    const token = sign(String(Date.now()));
    res.cookie(COOKIE_NAME, token, {
        httpOnly: true,
        secure: isProd,
        sameSite: 'lax',
        maxAge: SESSION_TTL_MS,
        path: '/',
    });
}

export function isAuthenticated(req: Request): boolean {
    return verify(req.cookies?.[COOKIE_NAME]);
}

export function requireAuth(req: Request, res: Response, next: NextFunction): void {
    if (isAuthenticated(req)) {
        next();
        return;
    }
    res.status(401).json({ error: 'unauthorized' });
}

export const authRouter = Router();

authRouter.get('/me', (req, res) => {
    res.json({ authenticated: isAuthenticated(req) });
});

authRouter.post('/login', async (req, res) => {
    const password = typeof req.body?.password === 'string' ? req.body.password : '';
    if (!password) {
        res.status(400).json({ error: 'password required' });
        return;
    }
    const ok = await bcrypt.compare(password, env.SIDE_PASSWORD_HASH);
    if (!ok) {
        res.status(401).json({ error: 'invalid password' });
        return;
    }
    setSession(res);
    res.json({ authenticated: true });
});

authRouter.post('/logout', (_req, res) => {
    res.clearCookie(COOKIE_NAME, { path: '/' });
    res.json({ authenticated: false });
});

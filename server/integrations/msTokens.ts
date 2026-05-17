import fs from 'fs';
import path from 'path';
import { env } from '../env';

let cachedAccess: { token: string; expiresAt: number } | null = null;
let currentRefresh: string | null = null;

const TOKEN_URL =
    'https://login.microsoftonline.com/consumers/oauth2/v2.0/token';

function loadRefresh(): string {
    // Prefer the rotated token persisted to disk; fall back to env on first run.
    try {
        const onDisk = fs.readFileSync(env.MS_TOKEN_FILE, 'utf8').trim();
        if (onDisk) return onDisk;
    } catch {
        /* file doesn't exist yet */
    }
    if (env.MS_REFRESH_TOKEN) return env.MS_REFRESH_TOKEN;
    throw new Error(
        'Microsoft refresh token missing — set MS_REFRESH_TOKEN or seed the token file'
    );
}

function persistRefresh(token: string): void {
    try {
        fs.mkdirSync(path.dirname(env.MS_TOKEN_FILE), { recursive: true });
        fs.writeFileSync(env.MS_TOKEN_FILE, token, { mode: 0o600 });
    } catch (e) {
        console.error('failed to persist rotated refresh token', e);
    }
}

export async function getAccessToken(): Promise<string> {
    if (cachedAccess && Date.now() < cachedAccess.expiresAt - 60_000) {
        return cachedAccess.token;
    }
    if (!env.MS_CLIENT_ID || !env.MS_CLIENT_SECRET) {
        throw new Error(
            'Microsoft credentials missing — set MS_CLIENT_ID and MS_CLIENT_SECRET'
        );
    }
    if (!currentRefresh) currentRefresh = loadRefresh();

    const body = new URLSearchParams({
        client_id: env.MS_CLIENT_ID,
        client_secret: env.MS_CLIENT_SECRET,
        grant_type: 'refresh_token',
        refresh_token: currentRefresh,
        scope: 'Files.ReadWrite offline_access User.Read',
    });
    const res = await fetch(TOKEN_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body,
    });
    if (!res.ok) {
        throw new Error(
            `token refresh failed: ${res.status} ${await res.text()}`
        );
    }
    const json = (await res.json()) as {
        access_token: string;
        expires_in: number;
        refresh_token?: string;
    };

    cachedAccess = {
        token: json.access_token,
        expiresAt: Date.now() + json.expires_in * 1000,
    };
    // Microsoft consumer accounts rotate refresh tokens — capture the new one
    // and persist it so the next server start can use it.
    if (json.refresh_token && json.refresh_token !== currentRefresh) {
        currentRefresh = json.refresh_token;
        persistRefresh(json.refresh_token);
    }
    return cachedAccess.token;
}

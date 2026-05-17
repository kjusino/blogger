/**
 * One-time helper: walk a delegated OAuth flow with kennethjusino@hotmail.com
 * and print a refresh token for the Microsoft Graph Files.ReadWrite scope.
 *
 * Run from the blogger repo root:
 *   npx tsx scripts/connect-excel.ts
 */
import http from 'http';
import { exec } from 'child_process';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve(__dirname, '..', '.env.local') });

const PORT = 3500;
const REDIRECT = `http://localhost:${PORT}/callback`;
const AUTH_URL =
    'https://login.microsoftonline.com/consumers/oauth2/v2.0/authorize';
const TOKEN_URL =
    'https://login.microsoftonline.com/consumers/oauth2/v2.0/token';
const SCOPES = 'offline_access Files.ReadWrite User.Read';

const clientId = process.env.MS_CLIENT_ID;
const clientSecret = process.env.MS_CLIENT_SECRET;

if (!clientId || !clientSecret) {
    console.error(
        'MS_CLIENT_ID and MS_CLIENT_SECRET must be set in .env.local before running this.'
    );
    process.exit(1);
}

const authUrl =
    `${AUTH_URL}?client_id=${clientId}` +
    `&response_type=code` +
    `&redirect_uri=${encodeURIComponent(REDIRECT)}` +
    `&response_mode=query` +
    `&scope=${encodeURIComponent(SCOPES)}` +
    `&prompt=consent`;

async function exchange(code: string): Promise<string> {
    const body = new URLSearchParams({
        client_id: clientId!,
        client_secret: clientSecret!,
        code,
        redirect_uri: REDIRECT,
        grant_type: 'authorization_code',
    });
    const res = await fetch(TOKEN_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body,
    });
    if (!res.ok) {
        throw new Error(`token exchange failed: ${res.status} ${await res.text()}`);
    }
    const json = (await res.json()) as { refresh_token?: string };
    if (!json.refresh_token) {
        throw new Error(
            'No refresh_token in response — did you include the offline_access scope?'
        );
    }
    return json.refresh_token;
}

const server = http.createServer(async (req, res) => {
    if (!req.url?.startsWith('/callback')) {
        res.statusCode = 404;
        res.end();
        return;
    }
    const url = new URL(req.url, `http://localhost:${PORT}`);
    const code = url.searchParams.get('code');
    const err = url.searchParams.get('error');
    if (err) {
        res.end(`Auth error: ${err}. Check the terminal and try again.`);
        console.error('auth error from Microsoft:', err);
        server.close();
        process.exit(1);
    }
    if (!code) {
        res.end('No code in callback — try again.');
        return;
    }
    try {
        const refreshToken = await exchange(code);
        res.end(
            'Refresh token captured. You can close this tab — the terminal has the value.'
        );
        console.log('\n\n=== MS_REFRESH_TOKEN ===\n');
        console.log(refreshToken);
        console.log('\n=========================\n');
        console.log('Push to Azure with:');
        console.log(
            `  az webapp config appsettings set -n kennethjusino-client -g 'kennethjusino.com' --settings MS_REFRESH_TOKEN='${refreshToken}'`
        );
        console.log(
            'Or paste it into .env.local as `MS_REFRESH_TOKEN=...` for local dev.\n'
        );
    } catch (e) {
        res.end(`Token exchange failed — see terminal.`);
        console.error(e);
    } finally {
        server.close();
    }
});

server.listen(PORT, () => {
    console.log(`Listening on ${REDIRECT}`);
    console.log(`Opening browser to authorize as kennethjusino@hotmail.com…`);
    exec(`open "${authUrl}"`);
});

import dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve(__dirname, '..', '.env.local') });
dotenv.config();

function required(name: string): string {
    const v = process.env[name];
    if (!v || v.length === 0) {
        if (process.env.NODE_ENV === 'production') {
            throw new Error(`Missing required env var: ${name}`);
        }
        console.warn(`[env] ${name} is not set — using dev placeholder`);
        return `dev-placeholder-${name}`;
    }
    return v;
}

function optional(name: string): string {
    return process.env[name] ?? '';
}

export const env = {
    NODE_ENV: process.env.NODE_ENV ?? 'development',
    PORT: parseInt(process.env.PORT ?? '3001', 10),
    SIDE_PASSWORD_HASH: required('SIDE_PASSWORD_HASH'),
    SESSION_SECRET: required('SESSION_SECRET'),
    MS_CLIENT_ID: optional('MS_CLIENT_ID'),
    MS_CLIENT_SECRET: optional('MS_CLIENT_SECRET'),
    MS_REFRESH_TOKEN: optional('MS_REFRESH_TOKEN'),
    MS_TOKEN_FILE:
        process.env.MS_TOKEN_FILE ??
        path.resolve(__dirname, '..', 'data', 'ms-refresh.txt'),
};

export const isProd = env.NODE_ENV === 'production';

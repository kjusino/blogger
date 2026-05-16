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

export const env = {
    NODE_ENV: process.env.NODE_ENV ?? 'development',
    PORT: parseInt(process.env.PORT ?? '3001', 10),
    SIDE_PASSWORD_HASH: required('SIDE_PASSWORD_HASH'),
    SESSION_SECRET: required('SESSION_SECRET'),
};

export const isProd = env.NODE_ENV === 'production';

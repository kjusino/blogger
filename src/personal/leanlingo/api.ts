import type { Attempt, QuestionTree } from './types';

const TREE_KEY = 'leanlingo:tree:v1';
const ATTEMPTS_KEY = 'leanlingo:attempts:v1';
const PENDING_KEY = 'leanlingo:pending:v1';

function safeGet<T>(key: string): T | null {
    try {
        const raw = localStorage.getItem(key);
        return raw ? (JSON.parse(raw) as T) : null;
    } catch {
        return null;
    }
}
function safeSet(key: string, value: unknown): void {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
        console.error('leanlingo: localStorage set failed', e);
    }
}

export function getCachedTree(): QuestionTree | null {
    return safeGet<QuestionTree>(TREE_KEY);
}

export async function fetchTree(): Promise<QuestionTree | null> {
    try {
        const res = await fetch('/api/personal/leanlingo/questions', {
            credentials: 'same-origin',
        });
        if (!res.ok) return null;
        const tree = (await res.json()) as QuestionTree;
        safeSet(TREE_KEY, tree);
        return tree;
    } catch (e) {
        console.error('leanlingo: fetchTree failed', e);
        return null;
    }
}

export function getCachedAttempts(): Attempt[] {
    return safeGet<Attempt[]>(ATTEMPTS_KEY) ?? [];
}

export async function fetchAttempts(): Promise<Attempt[] | null> {
    try {
        const res = await fetch('/api/personal/leanlingo/progress', {
            credentials: 'same-origin',
        });
        if (!res.ok) return null;
        const { rows } = (await res.json()) as { rows: Attempt[] };
        safeSet(ATTEMPTS_KEY, rows);
        return rows;
    } catch (e) {
        console.error('leanlingo: fetchAttempts failed', e);
        return null;
    }
}

/** Record a new attempt: append to local cache + buffer for sync. */
export function recordAttempt(a: Attempt): Attempt[] {
    const all = getCachedAttempts();
    all.push(a);
    safeSet(ATTEMPTS_KEY, all);
    const pending = safeGet<Attempt[]>(PENDING_KEY) ?? [];
    pending.push(a);
    safeSet(PENDING_KEY, pending);
    return all;
}

let flushing: Promise<number> | null = null;

async function doFlush(): Promise<number> {
    const pending = safeGet<Attempt[]>(PENDING_KEY) ?? [];
    if (pending.length === 0) return 0;
    // Pop atomically: take the snapshot and clear localStorage immediately.
    // If the send fails, prepend the snapshot back so it gets retried next time.
    safeSet(PENDING_KEY, []);
    try {
        const res = await fetch('/api/personal/leanlingo/attempt', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rows: pending }),
        });
        const j = (await res.json().catch(() => ({}))) as { ok?: boolean; error?: string };
        if (res.ok && j.ok !== false) return pending.length;
        throw new Error(j.error ?? `HTTP ${res.status}`);
    } catch (e) {
        const current = safeGet<Attempt[]>(PENDING_KEY) ?? [];
        safeSet(PENDING_KEY, [...pending, ...current]);
        return 0;
    }
}

/** Flush pending attempts to server. Returns # successfully flushed.
 * Serialized: concurrent calls share the in-flight promise. */
export function flushPending(): Promise<number> {
    if (flushing) return flushing;
    const p = doFlush();
    flushing = p;
    p.finally(() => { if (flushing === p) flushing = null; });
    return p;
}

export function pendingCount(): number {
    return (safeGet<Attempt[]>(PENDING_KEY) ?? []).length;
}

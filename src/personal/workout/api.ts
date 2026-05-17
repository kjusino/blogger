import { WorkoutData, HistoryEntry } from './types';

const KEY = 'personal:workouts:v1';

export type WorkoutRow = {
    date: string;
    program: string;
    exercise: string;
    set_idx: number;
    weight: number;
    pr_at_time: number;
};

export type Session = {
    date: string;
    program: string;
    exercises: { name: string; key: string; sets: number[]; max: number }[];
    /** Position in the rows array — newer sessions have higher idx. Used for sort. */
    rowIdx: number;
};

export async function loadData(): Promise<WorkoutData> {
    try {
        const raw = localStorage.getItem(KEY);
        return raw ? (JSON.parse(raw) as WorkoutData) : {};
    } catch (e) {
        console.error('workout load failed', e);
        return {};
    }
}

export async function saveData(data: WorkoutData): Promise<void> {
    try {
        localStorage.setItem(KEY, JSON.stringify(data));
    } catch (e) {
        console.error('workout save failed', e);
    }
}

export function getExerciseKey(name: string): string {
    return name.toLowerCase().replace(/[^a-z0-9]/g, '-');
}

export type HistoryPayload = {
    data: WorkoutData;
    rows: WorkoutRow[];
};

export async function fetchHistory(): Promise<HistoryPayload | null> {
    try {
        const res = await fetch('/api/personal/workout/history', {
            credentials: 'same-origin',
        });
        if (!res.ok) return null;
        const { rows } = (await res.json()) as { rows: WorkoutRow[] };
        return { data: rowsToWorkoutData(rows), rows };
    } catch (e) {
        console.error('history fetch failed', e);
        return null;
    }
}

export async function syncRows(
    rows: WorkoutRow[]
): Promise<{ ok: boolean; error?: string }> {
    try {
        const res = await fetch('/api/personal/workout/sync', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rows }),
        });
        const json = (await res.json().catch(() => ({}))) as {
            ok?: boolean;
            error?: string;
        };
        if (!res.ok || json.ok === false) {
            return { ok: false, error: json.error ?? `HTTP ${res.status}` };
        }
        return { ok: true };
    } catch (e) {
        return { ok: false, error: e instanceof Error ? e.message : String(e) };
    }
}

function rowsToWorkoutData(rows: WorkoutRow[]): WorkoutData {
    const data: WorkoutData = {};
    for (const r of rows) {
        const key = getExerciseKey(r.exercise);
        const entry = data[key] ?? {
            name: r.exercise,
            pr: 0,
            sets: {},
            history: [] as HistoryEntry[],
        };
        if (r.weight > entry.pr) entry.pr = r.weight;
        entry.history.push({ weight: r.weight, date: r.date });
        entry.name = r.exercise;
        data[key] = entry;
    }
    return data;
}

/**
 * Group flat WorkoutRow log into sessions. A session = one (date, program)
 * combination. Rows preserve their insertion order in Excel — later rowIdx =
 * more recent. We don't try to parse "May 17" into a real Date object
 * (no year), so chronological sort uses rowIdx as a stand-in for recency.
 */
export function groupRowsBySession(rows: WorkoutRow[]): Session[] {
    const map = new Map<string, Session>();
    rows.forEach((r, idx) => {
        const sessionKey = `${r.date}|${r.program}`;
        let s = map.get(sessionKey);
        if (!s) {
            s = {
                date: r.date,
                program: r.program,
                exercises: [],
                rowIdx: idx,
            };
            map.set(sessionKey, s);
        }
        // Track the latest row idx that touches this session
        s.rowIdx = Math.max(s.rowIdx, idx);

        const exKey = getExerciseKey(r.exercise);
        let ex = s.exercises.find((e) => e.key === exKey);
        if (!ex) {
            ex = { name: r.exercise, key: exKey, sets: [], max: 0 };
            s.exercises.push(ex);
        }
        ex.sets.push(r.weight);
        if (r.weight > ex.max) ex.max = r.weight;
    });
    // Most recent first
    return Array.from(map.values()).sort((a, b) => b.rowIdx - a.rowIdx);
}

/**
 * Best-effort parse of our short date format (e.g. "May 17"). The year
 * isn't stored. Assume current year; if that puts the date in the future,
 * walk it back a year.
 */
export function parseLogDate(s: string): Date | null {
    if (!s) return null;
    const d = new Date(`${s}, ${new Date().getFullYear()}`);
    if (isNaN(d.getTime())) return null;
    if (d.getTime() > Date.now() + 86_400_000) {
        d.setFullYear(d.getFullYear() - 1);
    }
    return d;
}

export function daysSince(s: string): number | null {
    const d = parseLogDate(s);
    if (!d) return null;
    return Math.floor((Date.now() - d.getTime()) / 86_400_000);
}

/** Count sessions whose date is within the last `days` days. */
export function countSessionsInLastDays(
    sessions: Session[],
    days: number
): number {
    const cutoff = Date.now() - days * 86_400_000;
    let n = 0;
    for (const s of sessions) {
        const d = parseLogDate(s.date);
        if (d && d.getTime() >= cutoff) n++;
    }
    return n;
}

/** Most recent { max weight, date } for a given exercise key. */
export function lastTimeFor(
    rows: WorkoutRow[],
    exerciseKey: string
): { max: number; date: string } | null {
    let mostRecent: { max: number; date: string; idx: number } | null = null;
    let curSessionDate: string | null = null;
    let curMax = 0;
    let curIdx = -1;
    // Walk rows; group consecutive rows of same (date, exercise) into a session-max
    for (let i = 0; i < rows.length; i++) {
        const r = rows[i];
        if (getExerciseKey(r.exercise) !== exerciseKey) continue;
        if (curSessionDate !== r.date) {
            curSessionDate = r.date;
            curMax = r.weight;
            curIdx = i;
        } else if (r.weight > curMax) {
            curMax = r.weight;
        }
        if (!mostRecent || i >= mostRecent.idx) {
            mostRecent = { max: curMax, date: r.date, idx: i };
        }
    }
    if (!mostRecent) return null;
    return { max: mostRecent.max, date: mostRecent.date };
}

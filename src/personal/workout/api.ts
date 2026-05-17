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

export async function fetchHistory(): Promise<WorkoutData | null> {
    try {
        const res = await fetch('/api/personal/workout/history', {
            credentials: 'same-origin',
        });
        if (!res.ok) return null;
        const { rows } = (await res.json()) as { rows: WorkoutRow[] };
        return rowsToWorkoutData(rows);
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

import { getAccessToken } from '../integrations/msTokens';
import type { AttemptRow, QuestionRow, QuestionType } from './types';

const WORKBOOK_PATH = '/PersonalApps/leanlingo.xlsx';
const GRAPH = 'https://graph.microsoft.com/v1.0';

function tableUrl(name: string): string {
    return `${GRAPH}/me/drive/root:${WORKBOOK_PATH}:/workbook/tables('${name}')`;
}

async function graphFetch(url: string, init: RequestInit = {}): Promise<Response> {
    const token = await getAccessToken();
    const headers: Record<string, string> = {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
        ...((init.headers as Record<string, string>) || {}),
    };
    return fetch(url, { ...init, headers });
}

function parseJsonArray(raw: unknown): string[] {
    if (typeof raw !== 'string' || !raw.trim()) return [];
    try {
        const v = JSON.parse(raw);
        return Array.isArray(v) ? v.map(String) : [];
    } catch {
        return [];
    }
}

// Excel may store strings that originally started with =/+/-/@ with a leading
// apostrophe (the literal char). The Graph API typically strips it on read,
// but be defensive in case it slips through.
function unescape(s: unknown): string {
    const str = String(s ?? '');
    return str.startsWith("'") ? str.slice(1) : str;
}

export async function readAllQuestions(): Promise<QuestionRow[]> {
    const out: QuestionRow[] = [];
    // Graph drops @odata.nextLink even when the page is full, so request
    // generously instead of relying on pagination. 5000 covers ~16x our seed.
    let next: string | null = `${tableUrl('Questions')}/rows?$top=5000`;
    while (next) {
        const res = await graphFetch(next);
        if (!res.ok) {
            throw new Error(`Questions read failed: ${res.status} ${await res.text()}`);
        }
        const json = (await res.json()) as {
            value: { values: unknown[][] }[];
            '@odata.nextLink'?: string;
        };
        for (const row of json.value) {
            const v = row.values?.[0];
            if (!v) continue;
            const id = unescape(v[0]);
            if (!id) continue;
            out.push({
                id,
                world: unescape(v[1]),
                unit: unescape(v[2]),
                lesson: unescape(v[3]),
                q_index: unescape(v[4]),
                type: unescape(v[5]) as QuestionType,
                prompt: unescape(v[6]),
                code: unescape(v[7]),
                options: parseJsonArray(v[8]),
                answer: unescape(v[9]),
                explanation: unescape(v[10]),
                ord_items: parseJsonArray(v[11]),
                book_ref: unescape(v[12]),
                lesson_title: unescape(v[13]),
            });
        }
        next = json['@odata.nextLink'] ?? null;
    }
    return out;
}

const ATTEMPT_COLS: (keyof AttemptRow)[] = [
    'timestamp',
    'lesson_id',
    'question_id',
    'attempts',
    'correct',
    'xp_awarded',
];

export async function appendAttempts(rows: AttemptRow[]): Promise<void> {
    if (rows.length === 0) return;
    const values = rows.map((r) =>
        ATTEMPT_COLS.map((c) => {
            const v = r[c];
            // Excel formula-prefix escape for any string starting with = + - @
            if (typeof v === 'string' && /^[=+\-@]/.test(v)) return "'" + v;
            return v;
        })
    );
    const res = await graphFetch(`${tableUrl('Progress')}/rows`, {
        method: 'POST',
        body: JSON.stringify({ index: null, values }),
    });
    if (!res.ok) {
        throw new Error(`Progress append failed: ${res.status} ${await res.text()}`);
    }
}

export async function readAllAttempts(): Promise<AttemptRow[]> {
    const out: AttemptRow[] = [];
    let next: string | null = `${tableUrl('Progress')}/rows?$top=500`;
    while (next) {
        const res = await graphFetch(next);
        if (!res.ok) {
            throw new Error(`Progress read failed: ${res.status} ${await res.text()}`);
        }
        const json = (await res.json()) as {
            value: { values: unknown[][] }[];
            '@odata.nextLink'?: string;
        };
        for (const row of json.value) {
            const v = row.values?.[0];
            if (!v) continue;
            const ts = unescape(v[0]);
            if (!ts) continue;
            out.push({
                timestamp: ts,
                lesson_id: unescape(v[1]),
                question_id: unescape(v[2]),
                attempts: Number(v[3] ?? 0),
                correct: !!v[4] && String(v[4]).toLowerCase() !== 'false',
                xp_awarded: Number(v[5] ?? 0),
            });
        }
        next = json['@odata.nextLink'] ?? null;
    }
    return out;
}

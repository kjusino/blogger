import { getAccessToken } from './msTokens';

export type WorkoutRow = {
    date: string;
    program: string;
    exercise: string;
    set_idx: number;
    weight: number;
    pr_at_time: number;
};

const WORKBOOK_PATH = '/PersonalApps/workout-journal.xlsx';
const TABLE = 'Table1';
const GRAPH = 'https://graph.microsoft.com/v1.0';

function tableUrl(): string {
    return `${GRAPH}/me/drive/root:${WORKBOOK_PATH}:/workbook/tables('${TABLE}')`;
}

async function graphFetch(
    path: string,
    init: RequestInit = {}
): Promise<Response> {
    const token = await getAccessToken();
    const headers: Record<string, string> = {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
        ...((init.headers as Record<string, string>) || {}),
    };
    return fetch(path, { ...init, headers });
}

const COLS: (keyof WorkoutRow)[] = [
    'date',
    'program',
    'exercise',
    'set_idx',
    'weight',
    'pr_at_time',
];

// Excel auto-converts "May 17" into a serial number. Convert it back on read.
// Excel's epoch is 1899-12-30 (the 1900 leap-year bug means days==serial after that).
function excelSerialToDateString(n: number): string {
    const ms = Date.UTC(1899, 11, 30) + n * 86_400_000;
    return new Date(ms).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        timeZone: 'UTC',
    });
}

function parseDateCell(raw: unknown): string {
    if (typeof raw === 'number') return excelSerialToDateString(raw);
    return String(raw ?? '');
}

export async function appendRows(rows: WorkoutRow[]): Promise<void> {
    if (rows.length === 0) return;
    const values = rows.map((r) => COLS.map((c) => r[c]));
    const res = await graphFetch(`${tableUrl()}/rows`, {
        method: 'POST',
        body: JSON.stringify({ values }),
    });
    if (!res.ok) {
        throw new Error(
            `Excel append failed: ${res.status} ${await res.text()}`
        );
    }
}

export async function readAllRows(): Promise<WorkoutRow[]> {
    const out: WorkoutRow[] = [];
    let next: string | null = `${tableUrl()}/rows?$top=200`;
    while (next) {
        const res = await graphFetch(next);
        if (!res.ok) {
            throw new Error(
                `Excel read failed: ${res.status} ${await res.text()}`
            );
        }
        const json = (await res.json()) as {
            value: { values: unknown[][] }[];
            '@odata.nextLink'?: string;
        };
        for (const row of json.value) {
            const v = row.values?.[0];
            if (!v) continue;
            out.push({
                date: parseDateCell(v[0]),
                program: String(v[1] ?? ''),
                exercise: String(v[2] ?? ''),
                set_idx: Number(v[3] ?? 0),
                weight: Number(v[4] ?? 0),
                pr_at_time: Number(v[5] ?? 0),
            });
        }
        next = json['@odata.nextLink'] ?? null;
    }
    return out;
}

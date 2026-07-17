import { getAccessToken } from '../integrations/msTokens';

export type EventRow = {
    timestamp: string;
    session_id: string;
    event: string;
    route: string;
    referrer: string;
    device: string;
    read_seconds: number;
    ip_hash: string;
};

// Overridable so local/CI can target a throwaway workbook and never touch
// the real analytics data. Defaults to the production workbook.
const WORKBOOK_PATH =
    process.env.ANALYTICS_WORKBOOK_PATH || '/PersonalApps/blog-analytics.xlsx';
const TABLE = 'Events';
const GRAPH = 'https://graph.microsoft.com/v1.0';

function tableUrl(): string {
    return `${GRAPH}/me/drive/root:${WORKBOOK_PATH}:/workbook/tables('${TABLE}')`;
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

const COLS: (keyof EventRow)[] = [
    'timestamp',
    'session_id',
    'event',
    'route',
    'referrer',
    'device',
    'read_seconds',
    'ip_hash',
];

function escapeFormula(v: unknown): unknown {
    if (typeof v === 'string' && /^[=+\-@]/.test(v)) return "'" + v;
    return v;
}

function unescape(s: unknown): string {
    const str = String(s ?? '');
    return str.startsWith("'") ? str.slice(1) : str;
}

export async function appendEvents(rows: EventRow[]): Promise<void> {
    if (rows.length === 0) return;
    const values = rows.map((r) => COLS.map((c) => escapeFormula(r[c])));
    const res = await graphFetch(`${tableUrl()}/rows`, {
        method: 'POST',
        body: JSON.stringify({ index: null, values }),
    });
    if (!res.ok) {
        throw new Error(`Analytics append failed: ${res.status} ${await res.text()}`);
    }
}

export async function readAllEvents(): Promise<EventRow[]> {
    const out: EventRow[] = [];
    let next: string | null = `${tableUrl()}/rows?$top=500`;
    while (next) {
        const res = await graphFetch(next);
        if (!res.ok) {
            throw new Error(`Analytics read failed: ${res.status} ${await res.text()}`);
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
                session_id: unescape(v[1]),
                event: unescape(v[2]),
                route: unescape(v[3]),
                referrer: unescape(v[4]),
                device: unescape(v[5]),
                read_seconds: Number(v[6] ?? 0),
                ip_hash: unescape(v[7]),
            });
        }
        next = json['@odata.nextLink'] ?? null;
    }
    return out;
}

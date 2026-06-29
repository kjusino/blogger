import { getAccessToken } from '../integrations/msTokens';

const WORKBOOK_PATH = '/PersonalApps/newsletter.xlsx';
const TABLE = 'Subscribers';
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

function unescape(s: unknown): string {
    const str = String(s ?? '');
    return str.startsWith("'") ? str.slice(1) : str;
}

function escapeFormula(v: unknown): unknown {
    if (typeof v === 'string' && /^[=+\-@]/.test(v)) return "'" + v;
    return v;
}

export async function emailExists(email: string): Promise<boolean> {
    const normalised = email.toLowerCase().trim();
    let next: string | null = `${tableUrl()}/rows?$top=500`;
    while (next) {
        const res = await graphFetch(next);
        if (!res.ok) {
            throw new Error(`Newsletter read failed: ${res.status} ${await res.text()}`);
        }
        const json = (await res.json()) as {
            value: { values: unknown[][] }[];
            '@odata.nextLink'?: string;
        };
        for (const row of json.value) {
            const v = row.values?.[0];
            if (!v) continue;
            if (unescape(v[0]).toLowerCase().trim() === normalised) return true;
        }
        next = json['@odata.nextLink'] ?? null;
    }
    return false;
}

export async function addSubscriber(email: string, route: string): Promise<void> {
    const values = [[
        escapeFormula(email.toLowerCase().trim()),
        new Date().toISOString(),
        escapeFormula(route),
    ]];
    const res = await graphFetch(`${tableUrl()}/rows`, {
        method: 'POST',
        body: JSON.stringify({ index: null, values }),
    });
    if (!res.ok) {
        throw new Error(`Newsletter append failed: ${res.status} ${await res.text()}`);
    }
}

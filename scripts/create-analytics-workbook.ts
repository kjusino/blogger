/**
 * One-time setup: create /PersonalApps/blog-analytics.xlsx in OneDrive with an
 * `Events` table whose columns match server/analytics/excel.ts (EventRow).
 *
 * Idempotent — safe to re-run. Reuses the app's Microsoft Graph credentials.
 *
 * Run from the blogger repo root:
 *   npx tsx scripts/create-analytics-workbook.ts
 */
import { getAccessToken } from '../server/integrations/msTokens';

const GRAPH = 'https://graph.microsoft.com/v1.0';
const WORKBOOK_PATH =
    process.env.ANALYTICS_WORKBOOK_PATH || '/PersonalApps/blog-analytics.xlsx';
const TABLE = 'Events';
const HEADERS = [
    'timestamp',
    'session_id',
    'event',
    'route',
    'referrer',
    'device',
    'read_seconds',
    'ip_hash',
];
const END_COL = String.fromCharCode('A'.charCodeAt(0) + HEADERS.length - 1); // 8 cols -> 'H'
const XLSX_MIME =
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';

async function g(
    pathOrUrl: string,
    init: RequestInit = {},
    rawBody = false
): Promise<Response> {
    const token = await getAccessToken();
    const headers: Record<string, string> = {
        Authorization: `Bearer ${token}`,
        ...((init.headers as Record<string, string>) || {}),
    };
    if (!rawBody) headers['Content-Type'] = 'application/json';
    const url = pathOrUrl.startsWith('http') ? pathOrUrl : `${GRAPH}${pathOrUrl}`;
    return fetch(url, { ...init, headers });
}

const item = (p: string) => `/me/drive/root:${p}`;
const wb = (p: string) => `${item(p)}:/workbook`;

async function ensureWorkbookFile(): Promise<void> {
    const exists = await g(item(WORKBOOK_PATH));
    if (exists.ok) {
        console.log('• workbook file already exists');
        return;
    }
    console.log('• creating empty workbook file…');
    const put = await g(
        `${item(WORKBOOK_PATH)}:/content`,
        { method: 'PUT', body: new Uint8Array(0), headers: { 'Content-Type': XLSX_MIME } },
        true
    );
    if (!put.ok) throw new Error(`create file failed: ${put.status} ${await put.text()}`);
    console.log('  created.');
}

async function firstSheetName(): Promise<string> {
    const res = await g(`${wb(WORKBOOK_PATH)}/worksheets`);
    if (!res.ok) throw new Error(`worksheets read failed: ${res.status} ${await res.text()}`);
    const json = (await res.json()) as { value: { name: string }[] };
    if (!json.value?.length) throw new Error('workbook has no worksheets');
    return json.value[0].name;
}

async function tableExists(): Promise<boolean> {
    const res = await g(`${wb(WORKBOOK_PATH)}/tables('${TABLE}')`);
    return res.ok;
}

async function createEventsTable(sheet: string): Promise<void> {
    const range = `A1:${END_COL}1`;
    console.log(`• writing headers to ${sheet}!${range}…`);
    const hdr = await g(
        `${wb(WORKBOOK_PATH)}/worksheets('${sheet}')/range(address='${range}')`,
        { method: 'PATCH', body: JSON.stringify({ values: [HEADERS] }) }
    );
    if (!hdr.ok) throw new Error(`header write failed: ${hdr.status} ${await hdr.text()}`);

    console.log(`• adding table over ${range}…`);
    const add = await g(`${wb(WORKBOOK_PATH)}/tables/add`, {
        method: 'POST',
        body: JSON.stringify({ address: `${sheet}!${range}`, hasHeaders: true }),
    });
    if (!add.ok) throw new Error(`table add failed: ${add.status} ${await add.text()}`);
    const created = (await add.json()) as { name: string };

    if (created.name !== TABLE) {
        console.log(`• renaming table '${created.name}' -> '${TABLE}'…`);
        const ren = await g(`${wb(WORKBOOK_PATH)}/tables('${created.name}')`, {
            method: 'PATCH',
            body: JSON.stringify({ name: TABLE }),
        });
        if (!ren.ok) throw new Error(`rename failed: ${ren.status} ${await ren.text()}`);
    }
}

async function currentColumns(): Promise<string[]> {
    const res = await g(`${wb(WORKBOOK_PATH)}/tables('${TABLE}')/columns`);
    if (!res.ok) throw new Error(`columns read failed: ${res.status} ${await res.text()}`);
    const json = (await res.json()) as { value: { name: string }[] };
    return json.value.map((c) => c.name);
}

// Non-destructive migration: append any HEADERS the table is missing (in order),
// leaving existing rows and data untouched. New columns get blank cells.
async function reconcileColumns(): Promise<void> {
    const existing = await currentColumns();
    const missing = HEADERS.filter((h) => !existing.includes(h));
    if (missing.length === 0) {
        console.log('• columns already up to date');
        return;
    }
    for (const name of missing) {
        console.log(`• adding missing column '${name}'…`);
        const res = await g(`${wb(WORKBOOK_PATH)}/tables('${TABLE}')/columns`, {
            method: 'POST',
            body: JSON.stringify({ name }),
        });
        if (!res.ok) throw new Error(`add column '${name}' failed: ${res.status} ${await res.text()}`);
    }
}

async function verify(): Promise<void> {
    const names = await currentColumns();
    const ok = JSON.stringify(names) === JSON.stringify(HEADERS);
    console.log('• verify columns:', names, ok ? '✅ match' : '❌ MISMATCH');
    if (!ok) throw new Error('column headers do not match EventRow');
}

async function main() {
    console.log(`• workbook: ${WORKBOOK_PATH}`);
    await ensureWorkbookFile();
    if (await tableExists()) {
        console.log('• Events table exists — reconciling columns');
        await reconcileColumns();
    } else {
        const sheet = await firstSheetName();
        await createEventsTable(sheet);
    }
    await verify();
    console.log('\n✅ blog-analytics.xlsx ready with Events table.');
}

main().catch((e) => {
    console.error('FAILED:', e);
    process.exit(1);
});

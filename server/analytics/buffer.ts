import { appendEvents, EventRow } from './excel';

const buffer: EventRow[] = [];
const FLUSH_THRESHOLD = 50;
const FLUSH_INTERVAL_MS = 2 * 60 * 1000;

export function enqueue(row: EventRow): void {
    buffer.push(row);
    if (buffer.length >= FLUSH_THRESHOLD) {
        flush();
    }
}

export async function flush(): Promise<void> {
    if (buffer.length === 0) return;
    const batch = buffer.splice(0);
    try {
        await appendEvents(batch);
    } catch (e) {
        console.error('[analytics] flush failed, re-queuing', e);
        buffer.unshift(...batch);
    }
}

export function startFlusher(): void {
    setInterval(() => {
        flush().catch((e) => console.error('[analytics] flush error', e));
    }, FLUSH_INTERVAL_MS);
}

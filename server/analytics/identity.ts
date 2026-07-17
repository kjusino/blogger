import { createHmac } from 'crypto';

/**
 * Pseudonymous per-visitor id: a keyed one-way hash of the client IP.
 *
 * The raw IP is never stored, so the analytics workbook holds no PII. Keyed
 * with a secret (SESSION_SECRET) so it can't be brute-forced from the small
 * IPv4 space. Deterministic: the same IP always maps to the same hash, which
 * is what lets us treat one IP as one "user" across visits.
 */
export function hashIp(ip: string, secret: string): string {
    return createHmac('sha256', secret).update(ip).digest('hex').slice(0, 32);
}

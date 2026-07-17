import { createHmac } from 'crypto';

/**
 * Reduce a client IP to a stable per-subscriber network key before hashing.
 *
 * IPv4: kept whole — one address ≈ one household/NAT ≈ one "user".
 * IPv6: reduced to the /64 network prefix. Modern clients use IPv6 Privacy
 *   Extensions (RFC 4941), rotating the 64-bit host part every few minutes, so
 *   the full address is NOT stable per person. The /64 prefix is the subscriber
 *   network (the IPv6 analogue of a single IPv4 address) and stays constant, so
 *   this keeps one person as one user across their rotating addresses.
 * IPv4-mapped IPv6 (::ffff:a.b.c.d): unwrapped to the underlying IPv4.
 */
export function networkKey(ip: string): string {
    const mapped = /^::ffff:(\d+\.\d+\.\d+\.\d+)$/i.exec(ip);
    if (mapped) return mapped[1];

    if (ip.includes(':')) {
        // Expand any :: compression, then keep the first 4 hextets (/64).
        const [head, tail = ''] = ip.split('::');
        const headParts = head ? head.split(':') : [];
        const tailParts = tail ? tail.split(':') : [];
        const fill = Math.max(0, 8 - headParts.length - tailParts.length);
        const full = [...headParts, ...Array(fill).fill('0'), ...tailParts];
        return (
            full
                .slice(0, 4)
                .map((h) => (h === '' ? '0' : h.toLowerCase()))
                .join(':') + '::/64'
        );
    }

    return ip; // IPv4, or anything else, unchanged
}

/**
 * Pseudonymous per-visitor id: a keyed one-way hash of the client's network key.
 *
 * The raw IP is never stored, so the analytics workbook holds no PII. Keyed
 * with a secret (SESSION_SECRET) so it can't be brute-forced from the small
 * address space. Deterministic: the same network always maps to the same hash,
 * which is what lets us treat one subscriber as one "user" across visits.
 */
export function hashIp(ip: string, secret: string): string {
    return createHmac('sha256', secret)
        .update(networkKey(ip))
        .digest('hex')
        .slice(0, 32);
}

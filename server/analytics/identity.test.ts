import assert from 'node:assert';
import { networkKey, hashIp } from './identity';

let passed = 0;
function check(name: string, fn: () => void) {
    fn();
    passed++;
    console.log(`  ✓ ${name}`);
}

console.log('networkKey:');

check('IPv4 is kept whole', () => {
    assert.strictEqual(networkKey('24.1.62.66'), '24.1.62.66');
});

check('two rotating IPv6 temp addresses in one /64 collapse to same key', () => {
    const a = networkKey('2601:240:c600:e880:1116:4052:6074:b27b');
    const b = networkKey('2601:240:c600:e880:1818:cc6c:2f58:b202');
    assert.strictEqual(a, b);
    assert.strictEqual(a, '2601:240:c600:e880::/64');
});

check('different /64 gives different key', () => {
    assert.notStrictEqual(
        networkKey('2601:240:c600:e880::1'),
        networkKey('2601:240:c600:e881::1')
    );
});

check(':: compression is expanded correctly', () => {
    // 2601:240:: -> first four hextets are 2601:240:0:0
    assert.strictEqual(networkKey('2601:240::abcd:1'), '2601:240:0:0::/64');
});

check('IPv4-mapped IPv6 unwraps to the IPv4', () => {
    assert.strictEqual(networkKey('::ffff:24.1.62.66'), '24.1.62.66');
});

// The prod bug: Azure XFF is "IP:port" and the port rotates every connection.
check('IPv4:port strips the port (same IP, diff ports -> same key)', () => {
    assert.strictEqual(networkKey('24.1.62.66:54321'), '24.1.62.66');
    assert.strictEqual(networkKey('24.1.62.66:54321'), networkKey('24.1.62.66:12345'));
});

check('[IPv6]:port strips brackets+port and still /64-collapses', () => {
    assert.strictEqual(
        networkKey('[2601:240:c600:e880:1116:4052:6074:b27b]:5555'),
        '2601:240:c600:e880::/64'
    );
    assert.strictEqual(
        networkKey('[2601:240:c600:e880:aaaa:bbbb:cccc:dddd]:9999'),
        networkKey('[2601:240:c600:e880:1116:4052:6074:b27b]:5555')
    );
});

check('bare IPv6 still works (colons not mistaken for a port)', () => {
    assert.strictEqual(
        networkKey('2601:240:c600:e880:1116:4052:6074:b27b'),
        '2601:240:c600:e880::/64'
    );
});

console.log('\nhashIp:');

check('same network -> same hash (rotating IPv6 = one user)', () => {
    const s = 'test-secret';
    assert.strictEqual(
        hashIp('2601:240:c600:e880:1116:4052:6074:b27b', s),
        hashIp('2601:240:c600:e880:1818:cc6c:2f58:b202', s)
    );
});

check('different IPv4 -> different hash', () => {
    const s = 'test-secret';
    assert.notStrictEqual(hashIp('24.1.62.66', s), hashIp('24.1.62.67', s));
});

check('hash is 32 hex chars and stores no raw IP', () => {
    const h = hashIp('24.1.62.66', 'test-secret');
    assert.match(h, /^[0-9a-f]{32}$/);
    assert.ok(!h.includes('24.1'));
});

check('secret matters (different key -> different hash)', () => {
    assert.notStrictEqual(hashIp('24.1.62.66', 'secretA'), hashIp('24.1.62.66', 'secretB'));
});

console.log(`\n${passed} passed`);

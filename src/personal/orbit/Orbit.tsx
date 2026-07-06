import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './orbit.css';

const KEY = 'personal:orbit:v1';
const MAX_PEOPLE = 500;
const DAY_MS = 24 * 60 * 60 * 1000;

type OrbitKey = 'inner' | 'close' | 'steady' | 'wide' | 'distant';

type Person = {
    id: string;
    name: string;
    orbit: OrbitKey;
    note: string;
    lastContact: number;
    createdAt: number;
};

type State = {
    people: Person[];
};

const ORBITS: { value: OrbitKey; label: string; days: number }[] = [
    { value: 'inner', label: 'Inner circle', days: 14 },
    { value: 'close', label: 'Close', days: 30 },
    { value: 'steady', label: 'Steady', days: 90 },
    { value: 'wide', label: 'Wide', days: 180 },
    { value: 'distant', label: 'Distant', days: 365 },
];

function orbitOf(key: OrbitKey) {
    return ORBITS.find((o) => o.value === key) ?? ORBITS[2];
}

function loadState(): State {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { people: [] };
        const parsed = JSON.parse(raw);
        return {
            people: Array.isArray(parsed?.people) ? parsed.people.slice(-MAX_PEOPLE) : [],
        };
    } catch {
        return { people: [] };
    }
}

function saveState(s: State): void {
    try {
        localStorage.setItem(KEY, JSON.stringify(s));
    } catch {
        /* localStorage full or denied — non-fatal */
    }
}

function makeId(): string {
    return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatDate(ts: number): string {
    return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function daysSince(ts: number): number {
    return Math.floor((Date.now() - ts) / DAY_MS);
}

type Bucket = 'overdue' | 'soon' | 'ontrack';

function bucketFor(ratio: number): Bucket {
    if (ratio >= 1) return 'overdue';
    if (ratio >= 0.7) return 'soon';
    return 'ontrack';
}

const BUCKET_LABEL: Record<Bucket, string> = {
    overdue: 'Overdue',
    soon: 'Due soon',
    ontrack: 'On track',
};

export default function Orbit() {
    const [state, setState] = useState<State>(() => loadState());
    const [name, setName] = useState('');
    const [orbit, setOrbit] = useState<OrbitKey>('close');
    const [note, setNote] = useState('');
    const [copied, setCopied] = useState(false);
    const { people } = state;

    const addPerson = useCallback(() => {
        const trimmed = name.trim();
        if (!trimmed) return;
        const person: Person = {
            id: makeId(),
            name: trimmed,
            orbit,
            note: note.trim(),
            lastContact: Date.now(),
            createdAt: Date.now(),
        };
        setState((prev) => {
            const next = { people: [...prev.people, person].slice(-MAX_PEOPLE) };
            saveState(next);
            return next;
        });
        setName('');
        setNote('');
    }, [name, orbit, note]);

    const markContacted = useCallback((id: string) => {
        setState((prev) => {
            const next = {
                people: prev.people.map((p) => (p.id === id ? { ...p, lastContact: Date.now() } : p)),
            };
            saveState(next);
            return next;
        });
    }, []);

    const removePerson = useCallback((id: string) => {
        setState((prev) => {
            const next = { people: prev.people.filter((p) => p.id !== id) };
            saveState(next);
            return next;
        });
    }, []);

    const withStatus = useMemo(
        () =>
            people.map((p) => {
                const cadenceDays = orbitOf(p.orbit).days;
                const since = daysSince(p.lastContact);
                const ratio = since / cadenceDays;
                return { person: p, since, ratio, bucket: bucketFor(ratio) };
            }),
        [people]
    );

    const sorted = useMemo(() => [...withStatus].sort((a, b) => b.ratio - a.ratio), [withStatus]);

    const overdueCount = useMemo(() => withStatus.filter((w) => w.bucket === 'overdue').length, [withStatus]);

    const overallMessage = useMemo(() => {
        if (people.length === 0) return "⚪ add the people worth staying close to — start with who you'd regret losing touch with";
        if (overdueCount === 0) return "🟢 you're caught up — nobody's gone quiet";
        if (overdueCount === 1) return '🟡 1 person is overdue for a check-in';
        return `🔴 ${overdueCount} people are overdue for a check-in`;
    }, [people.length, overdueCount]);

    const orbitCounts = useMemo(
        () =>
            ORBITS.map((o) => ({
                ...o,
                count: people.filter((p) => p.orbit === o.value).length,
            })).filter((o) => o.count > 0),
        [people]
    );
    const maxOrbitCount = Math.max(1, ...orbitCounts.map((o) => o.count));

    const grouped = useMemo(() => {
        const order: Bucket[] = ['overdue', 'soon', 'ontrack'];
        return order
            .map((b) => ({ bucket: b, items: sorted.filter((w) => w.bucket === b) }))
            .filter((g) => g.items.length > 0);
    }, [sorted]);

    const copyOutreachList = useCallback(async () => {
        const source = sorted.filter((w) => w.bucket !== 'ontrack').slice(0, 10);
        const text = source
            .map((w) => {
                const bits = [`- ${w.person.name}`];
                if (w.person.note) bits.push(`(${w.person.note})`);
                return bits.join(' ');
            })
            .join('\n');
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            /* clipboard denied — non-fatal */
        }
    }, [sorted]);

    return (
        <div className="orbit">
            <div className="orbit-header">
                <Link to="/personal" className="orbit-back">← personal</Link>
                <span className="orbit-title">Orbit</span>
                <span className="orbit-badge">{people.length}</span>
            </div>

            <div className="orbit-form">
                <label className="orbit-field">
                    <span>Who</span>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="e.g. Sam — old manager, now at Stripe"
                    />
                </label>
                <label className="orbit-field">
                    <span>How close should they stay</span>
                    <select value={orbit} onChange={(e) => setOrbit(e.target.value as OrbitKey)}>
                        {ORBITS.map((o) => (
                            <option key={o.value} value={o.value}>
                                {o.label} — every {o.days}d
                            </option>
                        ))}
                    </select>
                </label>
                <label className="orbit-field">
                    <span>What to bring up next time (optional)</span>
                    <input
                        type="text"
                        value={note}
                        onChange={(e) => setNote(e.target.value)}
                        placeholder="ask how the move went"
                    />
                </label>
                <button type="button" className="orbit-btn orbit-btn-primary" onClick={addPerson}>
                    Add to orbit
                </button>
            </div>

            <div className="orbit-status">{overallMessage}</div>

            {people.some((p) => bucketFor(daysSince(p.lastContact) / orbitOf(p.orbit).days) !== 'ontrack') && (
                <button type="button" className="orbit-btn orbit-btn-copy" onClick={copyOutreachList}>
                    {copied ? 'Copied ✓' : 'Copy outreach list'}
                </button>
            )}

            {orbitCounts.length > 0 && (
                <div className="orbit-section">
                    <div className="orbit-section-title">By orbit</div>
                    <div className="orbit-buckets">
                        {orbitCounts.map((o) => (
                            <div key={o.value} className="orbit-bucket-row">
                                <span className="orbit-bucket-label">{o.label}</span>
                                <div className="orbit-bar-track">
                                    <div
                                        className="orbit-bar-fill"
                                        style={{ width: `${(o.count / maxOrbitCount) * 100}%` }}
                                    />
                                </div>
                                <span className="orbit-bucket-count">{o.count}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {grouped.map(({ bucket, items }) => (
                <div key={bucket} className="orbit-section">
                    <div className="orbit-section-title">
                        {BUCKET_LABEL[bucket]} ({items.length})
                    </div>
                    <ul className="orbit-list">
                        {items.map(({ person, since }) => (
                            <li key={person.id} className={`orbit-item orbit-item-${bucket}`}>
                                <div className="orbit-item-body">
                                    <div className="orbit-item-title">{person.name}</div>
                                    <div className="orbit-item-meta">
                                        <span className="orbit-tag">{orbitOf(person.orbit).label}</span>
                                        <span>last {formatDate(person.lastContact)}</span>
                                        <span>{since === 0 ? 'today' : `${since}d ago`}</span>
                                    </div>
                                    {person.note && <div className="orbit-item-note">{person.note}</div>}
                                </div>
                                <div className="orbit-item-actions">
                                    <button
                                        type="button"
                                        className="orbit-chip orbit-chip-contact"
                                        onClick={() => markContacted(person.id)}
                                    >
                                        Reached out
                                    </button>
                                    <button
                                        type="button"
                                        className="orbit-chip orbit-chip-remove"
                                        onClick={() => removePerson(person.id)}
                                        aria-label="Remove person"
                                    >
                                        ✕
                                    </button>
                                </div>
                            </li>
                        ))}
                    </ul>
                </div>
            ))}

            <div className="orbit-tip">
                The people who change your career and your life rarely stay close by accident — a two-minute
                message beats a year of silence. Everything stays on this device — nothing is sent anywhere.
            </div>
        </div>
    );
}

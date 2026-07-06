import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import './depth.css';

const KEY = 'personal:depth:v1';
const MAX_SESSIONS = 1000;
const DAY_MS = 24 * 60 * 60 * 1000;
const DAILY_TARGET_MINUTES = 60; // one solid uninterrupted block counts as a "deep work day"

type Domain = 'career' | 'craft' | 'creative' | 'financial' | 'health' | 'other';

type Session = {
    id: string;
    domain: Domain;
    note: string;
    minutes: number;
    distractions: number;
    createdAt: number;
};

type State = {
    sessions: Session[];
};

const DOMAINS: { value: Domain; label: string }[] = [
    { value: 'career', label: 'Career' },
    { value: 'craft', label: 'Skill-building' },
    { value: 'creative', label: 'Creative' },
    { value: 'financial', label: 'Financial' },
    { value: 'health', label: 'Health' },
    { value: 'other', label: 'Other' },
];

function loadState(): State {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { sessions: [] };
        const parsed = JSON.parse(raw);
        return {
            sessions: Array.isArray(parsed?.sessions) ? parsed.sessions.slice(-MAX_SESSIONS) : [],
        };
    } catch {
        return { sessions: [] };
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

function dayKey(ts: number): string {
    const d = new Date(ts);
    return `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
}

function formatElapsed(ms: number): string {
    const totalSec = Math.floor(ms / 1000);
    const m = Math.floor(totalSec / 60);
    const s = totalSec % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function computeStreak(sessions: Session[]): number {
    const totals = new Map<string, number>();
    for (const s of sessions) {
        const k = dayKey(s.createdAt);
        totals.set(k, (totals.get(k) ?? 0) + s.minutes);
    }
    let streak = 0;
    let cursor = Date.now();
    for (let i = 0; i < 3650; i++) {
        const total = totals.get(dayKey(cursor)) ?? 0;
        if (total >= DAILY_TARGET_MINUTES) {
            streak++;
            cursor -= DAY_MS;
        } else if (i === 0) {
            cursor -= DAY_MS; // today still in progress — don't break the streak yet
        } else {
            break;
        }
    }
    return streak;
}

export default function Depth() {
    const [state, setState] = useState<State>(() => loadState());
    const [running, setRunning] = useState(false);
    const [startedAt, setStartedAt] = useState<number | null>(null);
    const [elapsedMs, setElapsedMs] = useState(0);
    const [distractions, setDistractions] = useState(0);
    const [domain, setDomain] = useState<Domain>('craft');
    const [note, setNote] = useState('');
    const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const { sessions } = state;

    useEffect(() => {
        if (!running || startedAt === null) return undefined;
        tickRef.current = setInterval(() => setElapsedMs(Date.now() - startedAt), 1000);
        return () => {
            if (tickRef.current) clearInterval(tickRef.current);
        };
    }, [running, startedAt]);

    const startSession = useCallback(() => {
        setStartedAt(Date.now());
        setElapsedMs(0);
        setDistractions(0);
        setRunning(true);
    }, []);

    const markDistraction = useCallback(() => setDistractions((d) => d + 1), []);

    const endSession = useCallback(
        (discard: boolean) => {
            if (startedAt === null) return;
            const finalElapsed = Date.now() - startedAt;
            if (!discard) {
                const minutes = Math.max(1, Math.round(finalElapsed / 60000));
                const session: Session = {
                    id: makeId(),
                    domain,
                    note: note.trim(),
                    minutes,
                    distractions,
                    createdAt: startedAt,
                };
                setState((prev) => {
                    const next = { sessions: [...prev.sessions, session].slice(-MAX_SESSIONS) };
                    saveState(next);
                    return next;
                });
            }
            setRunning(false);
            setStartedAt(null);
            setElapsedMs(0);
            setDistractions(0);
            setNote('');
        },
        [startedAt, domain, note, distractions]
    );

    const removeSession = useCallback((id: string) => {
        setState((prev) => {
            const next = { sessions: prev.sessions.filter((s) => s.id !== id) };
            saveState(next);
            return next;
        });
    }, []);

    const sorted = useMemo(() => [...sessions].sort((a, b) => b.createdAt - a.createdAt), [sessions]);

    const streak = useMemo(() => computeStreak(sessions), [sessions]);

    const todayMinutes = useMemo(() => {
        const key = dayKey(Date.now());
        return sessions.filter((s) => dayKey(s.createdAt) === key).reduce((sum, s) => sum + s.minutes, 0);
    }, [sessions]);

    const weekly = useMemo(() => {
        const days = [];
        for (let i = 6; i >= 0; i--) {
            const ts = Date.now() - i * DAY_MS;
            const key = dayKey(ts);
            const minutes = sessions
                .filter((s) => dayKey(s.createdAt) === key)
                .reduce((sum, s) => sum + s.minutes, 0);
            days.push({
                key,
                label: new Date(ts).toLocaleDateString(undefined, { weekday: 'short' }),
                minutes,
            });
        }
        return days;
    }, [sessions]);
    const maxWeekly = Math.max(DAILY_TARGET_MINUTES, ...weekly.map((d) => d.minutes));

    const domainTotals = useMemo(
        () =>
            DOMAINS.map((d) => ({
                ...d,
                minutes: sessions.filter((s) => s.domain === d.value).reduce((sum, s) => sum + s.minutes, 0),
            })).filter((d) => d.minutes > 0),
        [sessions]
    );
    const maxDomain = Math.max(1, ...domainTotals.map((d) => d.minutes));

    const distractionRate = useMemo(() => {
        const totalMinutes = sessions.reduce((sum, s) => sum + s.minutes, 0);
        const totalDistractions = sessions.reduce((sum, s) => sum + s.distractions, 0);
        if (totalMinutes === 0) return null;
        return totalDistractions / (totalMinutes / 60);
    }, [sessions]);

    const grouped = useMemo(() => {
        const todayK = dayKey(Date.now());
        const yesterdayK = dayKey(Date.now() - DAY_MS);
        const map = new Map<string, Session[]>();
        for (const s of sorted) {
            const key = dayKey(s.createdAt);
            const list = map.get(key) ?? [];
            list.push(s);
            map.set(key, list);
        }
        return Array.from(map.entries()).map(([key, list]) => {
            let label: string;
            if (key === todayK) label = 'Today';
            else if (key === yesterdayK) label = 'Yesterday';
            else label = new Date(list[0].createdAt).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
            return { label, list };
        });
    }, [sorted]);

    return (
        <div className="depth">
            <div className="depth-header">
                <Link to="/personal" className="depth-back">← personal</Link>
                <span className="depth-title">Depth</span>
                <span className="depth-badge">{sessions.length}</span>
            </div>

            {!running && (
                <div className="depth-form">
                    <label className="depth-field">
                        <span>Domain</span>
                        <select value={domain} onChange={(e) => setDomain(e.target.value as Domain)}>
                            {DOMAINS.map((d) => (
                                <option key={d.value} value={d.value}>
                                    {d.label}
                                </option>
                            ))}
                        </select>
                    </label>
                    <label className="depth-field">
                        <span>What are you focusing on (optional)</span>
                        <input
                            type="text"
                            value={note}
                            onChange={(e) => setNote(e.target.value)}
                            placeholder="e.g. Ship the analytics dashboard"
                        />
                    </label>
                    <button type="button" className="depth-btn depth-btn-primary" onClick={startSession}>
                        Start focus session
                    </button>
                </div>
            )}

            {running && (
                <div className="depth-live">
                    <div className="depth-live-domain">{DOMAINS.find((d) => d.value === domain)?.label}</div>
                    <div className="depth-live-clock">{formatElapsed(elapsedMs)}</div>
                    {note && <div className="depth-live-note">{note}</div>}
                    <button type="button" className="depth-btn depth-btn-distract" onClick={markDistraction}>
                        I got distracted {distractions > 0 ? `(${distractions})` : ''}
                    </button>
                    <div className="depth-live-actions">
                        <button type="button" className="depth-btn depth-btn-discard" onClick={() => endSession(true)}>
                            Discard
                        </button>
                        <button type="button" className="depth-btn depth-btn-primary" onClick={() => endSession(false)}>
                            Stop &amp; save
                        </button>
                    </div>
                </div>
            )}

            <div className="depth-section">
                <div className="depth-section-title">Today</div>
                <div className="depth-stat-row">
                    <div className="depth-stat">
                        <span className="depth-stat-value">{todayMinutes}m</span>
                        <span className="depth-stat-label">of {DAILY_TARGET_MINUTES}m target</span>
                    </div>
                    <div className="depth-stat">
                        <span className="depth-stat-value">{streak}</span>
                        <span className="depth-stat-label">day streak</span>
                    </div>
                    {distractionRate !== null && (
                        <div className="depth-stat">
                            <span className="depth-stat-value">{distractionRate.toFixed(1)}</span>
                            <span className="depth-stat-label">distractions/hr</span>
                        </div>
                    )}
                </div>
            </div>

            <div className="depth-section">
                <div className="depth-section-title">Last 7 days</div>
                <div className="depth-week">
                    {weekly.map((d) => (
                        <div key={d.key} className="depth-week-col">
                            <div className="depth-week-track">
                                <div
                                    className="depth-week-fill"
                                    style={{ height: `${Math.max(2, (d.minutes / maxWeekly) * 100)}%` }}
                                />
                            </div>
                            <span className="depth-week-label">{d.label}</span>
                        </div>
                    ))}
                </div>
            </div>

            {domainTotals.length > 0 && (
                <div className="depth-section">
                    <div className="depth-section-title">By domain</div>
                    <div className="depth-buckets">
                        {domainTotals.map((d) => (
                            <div key={d.value} className="depth-bucket-row">
                                <span className="depth-bucket-label">{d.label}</span>
                                <div className="depth-bar-track">
                                    <div
                                        className="depth-bar-fill"
                                        style={{ width: `${(d.minutes / maxDomain) * 100}%` }}
                                    />
                                </div>
                                <span className="depth-bucket-count">{d.minutes}m</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {grouped.map(({ label, list }) => (
                <div key={label} className="depth-section">
                    <div className="depth-section-title">{label}</div>
                    <ul className="depth-list">
                        {list.map((s) => (
                            <li key={s.id} className="depth-item">
                                <div className="depth-item-body">
                                    <div className="depth-item-title">
                                        {DOMAINS.find((d) => d.value === s.domain)?.label} · {s.minutes}m
                                        {s.distractions > 0 && ` · ${s.distractions} distraction${s.distractions === 1 ? '' : 's'}`}
                                    </div>
                                    {s.note && <div className="depth-item-detail">{s.note}</div>}
                                </div>
                                <button
                                    type="button"
                                    className="depth-chip depth-chip-remove"
                                    onClick={() => removeSession(s.id)}
                                    aria-label="Delete session"
                                >
                                    ✕
                                </button>
                            </li>
                        ))}
                    </ul>
                </div>
            ))}

            <div className="depth-tip">
                Depth compounds — an hour of uninterrupted focus outproduces a scattered day. Log the
                distraction the moment it happens; the count is the point, not a judgment. Everything
                stays on this device — nothing is sent anywhere.
            </div>
        </div>
    );
}

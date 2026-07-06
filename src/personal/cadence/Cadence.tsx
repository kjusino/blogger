import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './cadence.css';

const KEY = 'personal:cadence:v1';
const MAX_ENTRIES = 1000;
const DAY_MS = 24 * 60 * 60 * 1000;
const TREND_DAYS = 14;
const LOW_SLEEP_THRESHOLD = 6;
const HIGH_SLEEP_THRESHOLD = 7;
const MIN_SAMPLE = 3;

type Entry = {
    id: string;
    day: string;
    sleepHours: number;
    energy: number;
    stress: number;
    note: string;
    createdAt: number;
};

type State = {
    entries: Entry[];
};

function loadState(): State {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { entries: [] };
        const parsed = JSON.parse(raw);
        return {
            entries: Array.isArray(parsed?.entries) ? parsed.entries.slice(-MAX_ENTRIES) : [],
        };
    } catch {
        return { entries: [] };
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

function energyColor(energy: number): string {
    if (energy >= 4) return '#34d399';
    if (energy >= 3) return '#38bdf8';
    if (energy >= 2) return '#f5b942';
    return '#f87171';
}

export default function Cadence() {
    const [state, setState] = useState<State>(() => loadState());
    const todayKey = dayKey(Date.now());
    const existingToday = state.entries.find((e) => e.day === todayKey);

    const [sleepHours, setSleepHours] = useState(existingToday ? String(existingToday.sleepHours) : '');
    const [energy, setEnergy] = useState(existingToday?.energy ?? 3);
    const [stress, setStress] = useState(existingToday?.stress ?? 3);
    const [note, setNote] = useState(existingToday?.note ?? '');

    const { entries } = state;

    const save = useCallback(() => {
        const hours = parseFloat(sleepHours);
        if (Number.isNaN(hours) || hours < 0) return;
        setState((prev) => {
            const withoutToday = prev.entries.filter((e) => e.day !== todayKey);
            const entry: Entry = {
                id: existingToday?.id ?? makeId(),
                day: todayKey,
                sleepHours: hours,
                energy,
                stress,
                note: note.trim(),
                createdAt: existingToday?.createdAt ?? Date.now(),
            };
            const next = { entries: [...withoutToday, entry].sort((a, b) => a.createdAt - b.createdAt).slice(-MAX_ENTRIES) };
            saveState(next);
            return next;
        });
    }, [sleepHours, energy, stress, note, todayKey, existingToday]);

    const removeEntry = useCallback((id: string) => {
        setState((prev) => {
            const next = { entries: prev.entries.filter((e) => e.id !== id) };
            saveState(next);
            return next;
        });
    }, []);

    const sorted = useMemo(() => [...entries].sort((a, b) => b.createdAt - a.createdAt), [entries]);

    const streak = useMemo(() => {
        const days = new Set(entries.map((e) => e.day));
        let count = 0;
        let cursor = Date.now();
        for (let i = 0; i < 3650; i++) {
            if (days.has(dayKey(cursor))) {
                count++;
                cursor -= DAY_MS;
            } else if (i === 0) {
                cursor -= DAY_MS;
            } else {
                break;
            }
        }
        return count;
    }, [entries]);

    const trend = useMemo(() => {
        const days = [];
        for (let i = TREND_DAYS - 1; i >= 0; i--) {
            const ts = Date.now() - i * DAY_MS;
            const key = dayKey(ts);
            const entry = entries.find((e) => e.day === key);
            days.push({
                key,
                label: new Date(ts).toLocaleDateString(undefined, { weekday: 'narrow' }),
                sleepHours: entry?.sleepHours ?? 0,
                energy: entry?.energy ?? null,
            });
        }
        return days;
    }, [entries]);
    const maxSleep = Math.max(8, ...trend.map((d) => d.sleepHours));

    const insight = useMemo(() => {
        const low = entries.filter((e) => e.sleepHours < LOW_SLEEP_THRESHOLD);
        const high = entries.filter((e) => e.sleepHours >= HIGH_SLEEP_THRESHOLD);
        if (low.length < MIN_SAMPLE || high.length < MIN_SAMPLE) return null;
        const avgLow = low.reduce((sum, e) => sum + e.energy, 0) / low.length;
        const avgHigh = high.reduce((sum, e) => sum + e.energy, 0) / high.length;
        const diff = avgHigh - avgLow;
        if (Math.abs(diff) < 0.2) return null;
        return {
            diff,
            avgLow,
            avgHigh,
            nLow: low.length,
            nHigh: high.length,
        };
    }, [entries]);

    const avgEnergy = useMemo(() => {
        if (entries.length === 0) return null;
        return entries.reduce((sum, e) => sum + e.energy, 0) / entries.length;
    }, [entries]);

    const grouped = useMemo(() => {
        const todayK = todayKey;
        const yesterdayK = dayKey(Date.now() - DAY_MS);
        return sorted.map((e) => {
            let label: string;
            if (e.day === todayK) label = 'Today';
            else if (e.day === yesterdayK) label = 'Yesterday';
            else label = new Date(e.createdAt).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
            return { label, entry: e };
        });
    }, [sorted, todayKey]);

    return (
        <div className="cadence">
            <div className="cadence-header">
                <Link to="/personal" className="cadence-back">← personal</Link>
                <span className="cadence-title">Cadence</span>
                <span className="cadence-badge">{entries.length}</span>
            </div>

            <div className="cadence-form">
                <label className="cadence-field">
                    <span>Hours slept last night</span>
                    <input
                        type="number"
                        inputMode="decimal"
                        step="0.5"
                        min="0"
                        value={sleepHours}
                        onChange={(e) => setSleepHours(e.target.value)}
                        placeholder="e.g. 7"
                    />
                </label>
                <label className="cadence-field">
                    <span>Energy today: {energy}/5</span>
                    <input
                        type="range"
                        min="1"
                        max="5"
                        value={energy}
                        onChange={(e) => setEnergy(Number(e.target.value))}
                    />
                </label>
                <label className="cadence-field">
                    <span>Stress today: {stress}/5</span>
                    <input
                        type="range"
                        min="1"
                        max="5"
                        value={stress}
                        onChange={(e) => setStress(Number(e.target.value))}
                    />
                </label>
                <label className="cadence-field">
                    <span>Note (optional)</span>
                    <input
                        type="text"
                        value={note}
                        onChange={(e) => setNote(e.target.value)}
                        placeholder="e.g. Late night, big presentation tomorrow"
                    />
                </label>
                <button type="button" className="cadence-btn cadence-btn-primary" onClick={save} disabled={sleepHours === ''}>
                    {existingToday ? 'Update today' : 'Log today'}
                </button>
            </div>

            <div className="cadence-section">
                <div className="cadence-section-title">Overview</div>
                <div className="cadence-stat-row">
                    <div className="cadence-stat">
                        <span className="cadence-stat-value">{streak}</span>
                        <span className="cadence-stat-label">day streak</span>
                    </div>
                    {avgEnergy !== null && (
                        <div className="cadence-stat">
                            <span className="cadence-stat-value">{avgEnergy.toFixed(1)}</span>
                            <span className="cadence-stat-label">avg energy</span>
                        </div>
                    )}
                    <div className="cadence-stat">
                        <span className="cadence-stat-value">{entries.length}</span>
                        <span className="cadence-stat-label">check-ins</span>
                    </div>
                </div>
            </div>

            <div className="cadence-section">
                <div className="cadence-section-title">Last {TREND_DAYS} days</div>
                <div className="cadence-trend">
                    {trend.map((d) => (
                        <div key={d.key} className="cadence-trend-col">
                            <div className="cadence-trend-track">
                                <div
                                    className="cadence-trend-fill"
                                    style={{
                                        height: `${Math.max(2, (d.sleepHours / maxSleep) * 100)}%`,
                                        background: d.energy !== null ? energyColor(d.energy) : '#1e293b',
                                    }}
                                />
                            </div>
                            <span className="cadence-trend-label">{d.label}</span>
                        </div>
                    ))}
                </div>
                <div className="cadence-trend-legend">Bar height = hours slept · color = energy that day</div>
            </div>

            {insight && (
                <div className="cadence-section">
                    <div className="cadence-section-title">Insight</div>
                    <div className="cadence-insight">
                        Your energy runs <strong>{Math.abs(insight.diff).toFixed(1)} points {insight.diff > 0 ? 'higher' : 'lower'}</strong> on
                        {' '}{HIGH_SLEEP_THRESHOLD}+ hour nights than sub-{LOW_SLEEP_THRESHOLD} hour nights
                        {' '}({insight.avgHigh.toFixed(1)} vs {insight.avgLow.toFixed(1)}, n={insight.nHigh} vs n={insight.nLow}).
                    </div>
                </div>
            )}

            {grouped.length > 0 && (
                <div className="cadence-section">
                    <div className="cadence-section-title">History</div>
                    <ul className="cadence-list">
                        {grouped.map(({ label, entry }) => (
                            <li key={entry.id} className="cadence-item">
                                <div className="cadence-item-body">
                                    <div className="cadence-item-title">
                                        {label} · {entry.sleepHours}h sleep · energy {entry.energy}/5 · stress {entry.stress}/5
                                    </div>
                                    {entry.note && <div className="cadence-item-detail">{entry.note}</div>}
                                </div>
                                <button
                                    type="button"
                                    className="cadence-chip cadence-chip-remove"
                                    onClick={() => removeEntry(entry.id)}
                                    aria-label="Delete entry"
                                >
                                    ✕
                                </button>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            <div className="cadence-tip">
                Recovery is the input every other tool assumes you already have. Log it for a couple of weeks
                and the sleep/energy link stops being a guess. Everything stays on this device — nothing is
                sent anywhere.
            </div>
        </div>
    );
}

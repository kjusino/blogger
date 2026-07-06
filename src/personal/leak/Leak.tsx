import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './leak.css';

const KEY = 'personal:leak:v1';
const MAX_ENTRIES = 300;
const DAY_MS = 24 * 60 * 60 * 1000;
const LEAK_THRESHOLD_DAYS = 60;

type Cadence = 'weekly' | 'monthly' | 'annual';

type Entry = {
    id: string;
    name: string;
    amount: number;
    cadence: Cadence;
    category: string;
    lastUsedAt: number;
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

function monthlyCost(entry: Pick<Entry, 'amount' | 'cadence'>): number {
    if (entry.cadence === 'annual') return entry.amount / 12;
    if (entry.cadence === 'weekly') return (entry.amount * 52) / 12;
    return entry.amount;
}

function daysSince(ts: number): number {
    return Math.max(0, Math.floor((Date.now() - ts) / DAY_MS));
}

function formatMoney(n: number): string {
    return n.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });
}

const CADENCE_LABEL: Record<Cadence, string> = {
    weekly: '/wk',
    monthly: '/mo',
    annual: '/yr',
};

export default function Leak() {
    const [state, setState] = useState<State>(() => loadState());
    const { entries } = state;

    const [name, setName] = useState('');
    const [amount, setAmount] = useState('');
    const [cadence, setCadence] = useState<Cadence>('monthly');
    const [category, setCategory] = useState('');

    const add = useCallback(() => {
        const value = parseFloat(amount);
        if (!name.trim() || Number.isNaN(value) || value <= 0) return;
        const now = Date.now();
        const entry: Entry = {
            id: makeId(),
            name: name.trim(),
            amount: value,
            cadence,
            category: category.trim(),
            lastUsedAt: now,
            createdAt: now,
        };
        setState((prev) => {
            const next = { entries: [...prev.entries, entry].slice(-MAX_ENTRIES) };
            saveState(next);
            return next;
        });
        setName('');
        setAmount('');
        setCategory('');
    }, [name, amount, cadence, category]);

    const markUsed = useCallback((id: string) => {
        setState((prev) => {
            const next = {
                entries: prev.entries.map((e) => (e.id === id ? { ...e, lastUsedAt: Date.now() } : e)),
            };
            saveState(next);
            return next;
        });
    }, []);

    const removeEntry = useCallback((id: string) => {
        setState((prev) => {
            const next = { entries: prev.entries.filter((e) => e.id !== id) };
            saveState(next);
            return next;
        });
    }, []);

    const sorted = useMemo(
        () => [...entries].sort((a, b) => monthlyCost(b) - monthlyCost(a)),
        [entries],
    );

    const totalMonthly = useMemo(
        () => entries.reduce((sum, e) => sum + monthlyCost(e), 0),
        [entries],
    );

    const leaks = useMemo(
        () => entries.filter((e) => daysSince(e.lastUsedAt) >= LEAK_THRESHOLD_DAYS),
        [entries],
    );

    const leakMonthly = useMemo(
        () => leaks.reduce((sum, e) => sum + monthlyCost(e), 0),
        [leaks],
    );

    return (
        <div className="leak">
            <div className="leak-header">
                <Link to="/personal" className="leak-back">← personal</Link>
                <span className="leak-title">Leak</span>
                <span className="leak-badge">{entries.length}</span>
            </div>

            <div className="leak-form">
                <label className="leak-field">
                    <span>Subscription or recurring expense</span>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="e.g. Streaming service"
                    />
                </label>
                <div className="leak-field-row">
                    <label className="leak-field">
                        <span>Amount</span>
                        <input
                            type="number"
                            inputMode="decimal"
                            step="0.01"
                            min="0"
                            value={amount}
                            onChange={(e) => setAmount(e.target.value)}
                            placeholder="9.99"
                        />
                    </label>
                    <label className="leak-field">
                        <span>Billed</span>
                        <select value={cadence} onChange={(e) => setCadence(e.target.value as Cadence)}>
                            <option value="weekly">Weekly</option>
                            <option value="monthly">Monthly</option>
                            <option value="annual">Annually</option>
                        </select>
                    </label>
                </div>
                <label className="leak-field">
                    <span>Category (optional)</span>
                    <input
                        type="text"
                        value={category}
                        onChange={(e) => setCategory(e.target.value)}
                        placeholder="e.g. Entertainment, Software, Fitness"
                    />
                </label>
                <button
                    type="button"
                    className="leak-btn leak-btn-primary"
                    onClick={add}
                    disabled={!name.trim() || amount === ''}
                >
                    Add subscription
                </button>
            </div>

            <div className="leak-section">
                <div className="leak-section-title">Overview</div>
                <div className="leak-stat-row">
                    <div className="leak-stat">
                        <span className="leak-stat-value">{formatMoney(totalMonthly)}</span>
                        <span className="leak-stat-label">per month</span>
                    </div>
                    <div className="leak-stat">
                        <span className="leak-stat-value">{formatMoney(totalMonthly * 12)}</span>
                        <span className="leak-stat-label">per year</span>
                    </div>
                    <div className="leak-stat leak-stat-warn">
                        <span className="leak-stat-value">{leaks.length}</span>
                        <span className="leak-stat-label">leaking</span>
                    </div>
                </div>
            </div>

            {leaks.length > 0 && (
                <div className="leak-section">
                    <div className="leak-section-title">Insight</div>
                    <div className="leak-insight">
                        <strong>{leaks.length} subscription{leaks.length === 1 ? '' : 's'}</strong> unused for {LEAK_THRESHOLD_DAYS}+ days
                        {' '}are quietly costing <strong>{formatMoney(leakMonthly)}/mo</strong> ({formatMoney(leakMonthly * 12)}/yr)
                        {totalMonthly > 0 && ` — ${Math.round((leakMonthly / totalMonthly) * 100)}% of your recurring spend`}.
                    </div>
                </div>
            )}

            {sorted.length > 0 && (
                <div className="leak-section">
                    <div className="leak-section-title">Subscriptions</div>
                    <ul className="leak-list">
                        {sorted.map((e) => {
                            const idle = daysSince(e.lastUsedAt);
                            const isLeak = idle >= LEAK_THRESHOLD_DAYS;
                            return (
                                <li key={e.id} className={`leak-item${isLeak ? ' leak-item-leak' : ''}`}>
                                    <div className="leak-item-body">
                                        <div className="leak-item-title">
                                            {e.name} · {formatMoney(monthlyCost(e))}/mo
                                            {isLeak && <span className="leak-chip leak-chip-warn">leak</span>}
                                        </div>
                                        <div className="leak-item-detail">
                                            {formatMoney(e.amount)}{CADENCE_LABEL[e.cadence]}
                                            {e.category && ` · ${e.category}`}
                                            {' · '}
                                            {idle === 0 ? 'used today' : `used ${idle}d ago`}
                                        </div>
                                    </div>
                                    <div className="leak-item-actions">
                                        <button type="button" className="leak-chip leak-chip-action" onClick={() => markUsed(e.id)}>
                                            Used today
                                        </button>
                                        <button
                                            type="button"
                                            className="leak-chip leak-chip-remove"
                                            onClick={() => removeEntry(e.id)}
                                            aria-label="Delete entry"
                                        >
                                            ✕
                                        </button>
                                    </div>
                                </li>
                            );
                        })}
                    </ul>
                </div>
            )}

            <div className="leak-tip">
                Every other personal tool tracks something you're building. Leak tracks something draining you —
                money committed on autopilot long after the value stopped. Mark what you still use; cancel what
                you don't. Everything stays on this device — nothing is sent anywhere.
            </div>
        </div>
    );
}

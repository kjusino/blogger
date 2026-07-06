import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './nerve.css';

const KEY = 'personal:nerve:v1';
const MAX_ENTRIES = 500;
const DAY_MS = 24 * 60 * 60 * 1000;
const WEEK_MS = 7 * DAY_MS;

type Outcome = 'pending' | 'yes' | 'partial' | 'no';

type Ask = {
    id: string;
    what: string;
    category: string;
    value: number | null;
    outcome: Outcome;
    createdAt: number;
    resolvedAt: number | null;
};

type State = {
    asks: Ask[];
};

function loadState(): State {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { asks: [] };
        const parsed = JSON.parse(raw);
        return {
            asks: Array.isArray(parsed?.asks) ? parsed.asks.slice(-MAX_ENTRIES) : [],
        };
    } catch {
        return { asks: [] };
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

function formatMoney(n: number): string {
    return n.toLocaleString(undefined, { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
}

const OUTCOME_LABEL: Record<Outcome, string> = {
    pending: 'pending',
    yes: 'yes',
    partial: 'partial',
    no: 'no',
};

const CATEGORY_SUGGESTIONS = ['Raise', 'Discount', 'Referral', 'Extension', 'Favor', 'Opportunity'];

export default function Nerve() {
    const [state, setState] = useState<State>(() => loadState());
    const { asks } = state;

    const [what, setWhat] = useState('');
    const [category, setCategory] = useState('');
    const [value, setValue] = useState('');

    const add = useCallback(() => {
        if (!what.trim()) return;
        const parsedValue = value.trim() === '' ? null : parseFloat(value);
        const ask: Ask = {
            id: makeId(),
            what: what.trim(),
            category: category.trim(),
            value: parsedValue !== null && !Number.isNaN(parsedValue) ? parsedValue : null,
            outcome: 'pending',
            createdAt: Date.now(),
            resolvedAt: null,
        };
        setState((prev) => {
            const next = { asks: [...prev.asks, ask].slice(-MAX_ENTRIES) };
            saveState(next);
            return next;
        });
        setWhat('');
        setCategory('');
        setValue('');
    }, [what, category, value]);

    const resolve = useCallback((id: string, outcome: Outcome) => {
        setState((prev) => {
            const next = {
                asks: prev.asks.map((a) =>
                    a.id === id ? { ...a, outcome, resolvedAt: Date.now() } : a,
                ),
            };
            saveState(next);
            return next;
        });
    }, []);

    const removeAsk = useCallback((id: string) => {
        setState((prev) => {
            const next = { asks: prev.asks.filter((a) => a.id !== id) };
            saveState(next);
            return next;
        });
    }, []);

    const pending = useMemo(
        () => [...asks].filter((a) => a.outcome === 'pending').sort((a, b) => b.createdAt - a.createdAt),
        [asks],
    );

    const resolved = useMemo(
        () => [...asks].filter((a) => a.outcome !== 'pending').sort((a, b) => (b.resolvedAt ?? 0) - (a.resolvedAt ?? 0)),
        [asks],
    );

    const resolvedCount = resolved.length;

    const hitRate = useMemo(() => {
        if (resolvedCount === 0) return null;
        const hits = resolved.filter((a) => a.outcome === 'yes' || a.outcome === 'partial').length;
        return hits / resolvedCount;
    }, [resolved, resolvedCount]);

    const valueCaptured = useMemo(
        () =>
            resolved.reduce((sum, a) => {
                if (a.value === null) return sum;
                if (a.outcome === 'yes') return sum + a.value;
                if (a.outcome === 'partial') return sum + a.value * 0.5;
                return sum;
            }, 0),
        [resolved],
    );

    const now = Date.now();
    const thisWeekCount = useMemo(
        () => asks.filter((a) => now - a.createdAt < WEEK_MS).length,
        [asks, now],
    );
    const lastWeekCount = useMemo(
        () => asks.filter((a) => now - a.createdAt >= WEEK_MS && now - a.createdAt < 2 * WEEK_MS).length,
        [asks, now],
    );

    const insight = useMemo(() => {
        if (asks.length === 0) return null;
        if (thisWeekCount === 0 && lastWeekCount === 0) {
            return "No asks logged in the last two weeks. Value doesn't arrive without a request attached to it — log the next one, whatever it is.";
        }
        if (lastWeekCount > 0 && thisWeekCount < lastWeekCount) {
            return `You made ${lastWeekCount} ask${lastWeekCount === 1 ? '' : 's'} last week and only ${thisWeekCount} this week. Your hit rate${hitRate !== null ? ` (${Math.round(hitRate * 100)}%)` : ''} isn't the bottleneck — volume dropped.`;
        }
        if (hitRate !== null) {
            return `${Math.round(hitRate * 100)}% hit rate across ${resolvedCount} resolved ask${resolvedCount === 1 ? '' : 's'}. Doubling your ask volume at this same rate roughly doubles what you capture — the lever is frequency, not finesse.`;
        }
        return `${thisWeekCount} ask${thisWeekCount === 1 ? '' : 's'} logged this week. Resolve them as answers come in to start seeing your hit rate.`;
    }, [asks.length, thisWeekCount, lastWeekCount, hitRate, resolvedCount]);

    return (
        <div className="nerve">
            <div className="nerve-header">
                <Link to="/personal" className="nerve-back">← personal</Link>
                <span className="nerve-title">Nerve</span>
                <span className="nerve-badge">{asks.length}</span>
            </div>

            <div className="nerve-form">
                <label className="nerve-field">
                    <span>What did you ask for?</span>
                    <input
                        type="text"
                        value={what}
                        onChange={(e) => setWhat(e.target.value)}
                        placeholder="e.g. Raise to match market rate"
                    />
                </label>
                <div className="nerve-field-row">
                    <label className="nerve-field">
                        <span>Category (optional)</span>
                        <input
                            type="text"
                            list="nerve-categories"
                            value={category}
                            onChange={(e) => setCategory(e.target.value)}
                            placeholder="Raise, Discount, Referral…"
                        />
                        <datalist id="nerve-categories">
                            {CATEGORY_SUGGESTIONS.map((c) => (
                                <option key={c} value={c} />
                            ))}
                        </datalist>
                    </label>
                    <label className="nerve-field">
                        <span>Value if granted (optional)</span>
                        <input
                            type="number"
                            inputMode="decimal"
                            step="1"
                            min="0"
                            value={value}
                            onChange={(e) => setValue(e.target.value)}
                            placeholder="500"
                        />
                    </label>
                </div>
                <button
                    type="button"
                    className="nerve-btn nerve-btn-primary"
                    onClick={add}
                    disabled={!what.trim()}
                >
                    Log the ask
                </button>
            </div>

            <div className="nerve-section">
                <div className="nerve-section-title">Overview</div>
                <div className="nerve-stat-row">
                    <div className="nerve-stat">
                        <span className="nerve-stat-value">{thisWeekCount}</span>
                        <span className="nerve-stat-label">this week</span>
                    </div>
                    <div className="nerve-stat">
                        <span className="nerve-stat-value">{hitRate !== null ? `${Math.round(hitRate * 100)}%` : '—'}</span>
                        <span className="nerve-stat-label">hit rate</span>
                    </div>
                    <div className="nerve-stat">
                        <span className="nerve-stat-value">{formatMoney(valueCaptured)}</span>
                        <span className="nerve-stat-label">captured</span>
                    </div>
                </div>
            </div>

            {insight && (
                <div className="nerve-section">
                    <div className="nerve-section-title">Insight</div>
                    <div className="nerve-insight">{insight}</div>
                </div>
            )}

            {pending.length > 0 && (
                <div className="nerve-section">
                    <div className="nerve-section-title">Awaiting an answer</div>
                    <ul className="nerve-list">
                        {pending.map((a) => (
                            <li key={a.id} className="nerve-item">
                                <div className="nerve-item-body">
                                    <div className="nerve-item-title">{a.what}</div>
                                    <div className="nerve-item-detail">
                                        {a.category && `${a.category} · `}
                                        {a.value !== null && `${formatMoney(a.value)} · `}
                                        asked {Math.max(0, Math.floor((now - a.createdAt) / DAY_MS))}d ago
                                    </div>
                                </div>
                                <div className="nerve-item-actions">
                                    <button type="button" className="nerve-chip nerve-chip-yes" onClick={() => resolve(a.id, 'yes')}>
                                        Yes
                                    </button>
                                    <button type="button" className="nerve-chip nerve-chip-partial" onClick={() => resolve(a.id, 'partial')}>
                                        Partial
                                    </button>
                                    <button type="button" className="nerve-chip nerve-chip-no" onClick={() => resolve(a.id, 'no')}>
                                        No
                                    </button>
                                </div>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {resolved.length > 0 && (
                <div className="nerve-section">
                    <div className="nerve-section-title">Resolved</div>
                    <ul className="nerve-list">
                        {resolved.map((a) => (
                            <li key={a.id} className={`nerve-item nerve-item-${a.outcome}`}>
                                <div className="nerve-item-body">
                                    <div className="nerve-item-title">
                                        {a.what}
                                        <span className={`nerve-chip nerve-chip-outcome nerve-chip-${a.outcome}`}>
                                            {OUTCOME_LABEL[a.outcome]}
                                        </span>
                                    </div>
                                    <div className="nerve-item-detail">
                                        {a.category && `${a.category} · `}
                                        {a.value !== null && `${formatMoney(a.value)} · `}
                                        resolved {Math.max(0, Math.floor((now - (a.resolvedAt ?? now)) / DAY_MS))}d ago
                                    </div>
                                </div>
                                <div className="nerve-item-actions">
                                    <button
                                        type="button"
                                        className="nerve-chip nerve-chip-remove"
                                        onClick={() => removeAsk(a.id)}
                                        aria-label="Delete entry"
                                    >
                                        ✕
                                    </button>
                                </div>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            <div className="nerve-tip">
                Every other personal tool here optimizes something you already do. Nerve tracks something most
                people avoid doing at all: asking directly for what they want. Outcomes scale with how often you
                ask more reliably than with how good you are at asking — log the ask before you know the answer.
                Everything stays on this device — nothing is sent anywhere.
            </div>
        </div>
    );
}

import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './calibrate.css';

const KEY = 'personal:calibrate:v1';
const MAX_PREDICTIONS = 500;
const DAY_MS = 24 * 60 * 60 * 1000;

type Category = 'career' | 'financial' | 'health' | 'relationships' | 'other';

type Prediction = {
    id: string;
    statement: string;
    category: Category;
    confidencePct: number; // probability (50-99) the statement comes true
    createdAt: number;
    resolveBy: number | null;
    status: 'open' | 'yes' | 'no';
    resolvedAt: number | null;
};

type State = {
    predictions: Prediction[];
};

const CATEGORIES: { value: Category; label: string }[] = [
    { value: 'career', label: 'Career' },
    { value: 'financial', label: 'Financial' },
    { value: 'health', label: 'Health' },
    { value: 'relationships', label: 'Relationships' },
    { value: 'other', label: 'Other' },
];

const BUCKETS = [
    { label: '50–60%', min: 50, max: 60 },
    { label: '60–70%', min: 60, max: 70 },
    { label: '70–80%', min: 70, max: 80 },
    { label: '80–90%', min: 80, max: 90 },
    { label: '90–100%', min: 90, max: 101 },
];

function loadState(): State {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { predictions: [] };
        const parsed = JSON.parse(raw);
        return {
            predictions: Array.isArray(parsed?.predictions) ? parsed.predictions.slice(-MAX_PREDICTIONS) : [],
        };
    } catch {
        return { predictions: [] };
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

function formatDue(ts: number | null): string {
    if (ts === null) return 'no target date';
    const diffDays = Math.round((ts - Date.now()) / DAY_MS);
    if (diffDays > 0) return `due in ${diffDays}d`;
    if (diffDays === 0) return 'due today';
    return `${Math.abs(diffDays)}d overdue`;
}

function calibrationRating(brier: number | null, resolvedCount: number): string {
    if (resolvedCount < 5) return `⚪ resolve at least 5 predictions to see your calibration (${resolvedCount}/5)`;
    if (brier === null) return '⚪ not enough data yet';
    if (brier <= 0.1) return '🟢 excellent calibration — your confidence tracks reality';
    if (brier <= 0.2) return '🟡 decent — check the buckets below for over/under-confidence';
    return '🔴 poorly calibrated — your stated confidence is diverging from outcomes';
}

export default function Calibrate() {
    const [state, setState] = useState<State>(() => loadState());
    const [statement, setStatement] = useState('');
    const [category, setCategory] = useState<Category>('career');
    const [confidencePct, setConfidencePct] = useState(70);
    const [resolveBy, setResolveBy] = useState('');
    const { predictions } = state;

    const addPrediction = useCallback(() => {
        const trimmed = statement.trim();
        if (!trimmed) return;
        const prediction: Prediction = {
            id: makeId(),
            statement: trimmed,
            category,
            confidencePct,
            createdAt: Date.now(),
            resolveBy: resolveBy ? new Date(resolveBy).getTime() : null,
            status: 'open',
            resolvedAt: null,
        };
        setState((prev) => {
            const next = { predictions: [...prev.predictions, prediction].slice(-MAX_PREDICTIONS) };
            saveState(next);
            return next;
        });
        setStatement('');
        setResolveBy('');
        setConfidencePct(70);
    }, [statement, category, confidencePct, resolveBy]);

    const resolvePrediction = useCallback((id: string, outcome: 'yes' | 'no') => {
        setState((prev) => {
            const next = {
                predictions: prev.predictions.map((p) =>
                    p.id === id ? { ...p, status: outcome, resolvedAt: Date.now() } : p
                ),
            };
            saveState(next);
            return next;
        });
    }, []);

    const removePrediction = useCallback((id: string) => {
        setState((prev) => {
            const next = { predictions: prev.predictions.filter((p) => p.id !== id) };
            saveState(next);
            return next;
        });
    }, []);

    const open = useMemo(
        () =>
            predictions
                .filter((p) => p.status === 'open')
                .sort((a, b) => (a.resolveBy ?? Infinity) - (b.resolveBy ?? Infinity)),
        [predictions]
    );

    const resolved = useMemo(() => predictions.filter((p) => p.status !== 'open'), [predictions]);

    const brier = useMemo(() => {
        if (resolved.length === 0) return null;
        const total = resolved.reduce((sum, p) => {
            const outcome = p.status === 'yes' ? 1 : 0;
            const err = p.confidencePct / 100 - outcome;
            return sum + err * err;
        }, 0);
        return total / resolved.length;
    }, [resolved]);

    const buckets = useMemo(
        () =>
            BUCKETS.map((b) => {
                const inBucket = resolved.filter((p) => p.confidencePct >= b.min && p.confidencePct < b.max);
                if (inBucket.length === 0) return { ...b, count: 0, avgConfidence: 0, actualRate: 0 };
                const avgConfidence = inBucket.reduce((s, p) => s + p.confidencePct, 0) / inBucket.length;
                const actualRate = (inBucket.filter((p) => p.status === 'yes').length / inBucket.length) * 100;
                return { ...b, count: inBucket.length, avgConfidence, actualRate };
            }),
        [resolved]
    );

    return (
        <div className="calibrate">
            <div className="calibrate-header">
                <Link to="/personal" className="calibrate-back">← personal</Link>
                <span className="calibrate-title">Calibrate</span>
                <span className="calibrate-badge">{brier === null ? '—' : brier.toFixed(2)}</span>
            </div>

            <div className="calibrate-form">
                <label className="calibrate-field">
                    <span>Prediction</span>
                    <textarea
                        value={statement}
                        onChange={(e) => setStatement(e.target.value)}
                        placeholder="e.g. I'll land a new role by Q3"
                        rows={2}
                    />
                </label>
                <div className="calibrate-form-row">
                    <label className="calibrate-field">
                        <span>Category</span>
                        <select value={category} onChange={(e) => setCategory(e.target.value as Category)}>
                            {CATEGORIES.map((c) => (
                                <option key={c.value} value={c.value}>
                                    {c.label}
                                </option>
                            ))}
                        </select>
                    </label>
                    <label className="calibrate-field">
                        <span>Confidence: {confidencePct}%</span>
                        <input
                            type="range"
                            min={50}
                            max={99}
                            step={1}
                            value={confidencePct}
                            onChange={(e) => setConfidencePct(Number(e.target.value))}
                        />
                    </label>
                </div>
                <label className="calibrate-field">
                    <span>Review by (optional)</span>
                    <input type="date" value={resolveBy} onChange={(e) => setResolveBy(e.target.value)} />
                </label>
                <button type="button" className="calibrate-btn calibrate-btn-primary" onClick={addPrediction}>
                    Log prediction
                </button>
            </div>

            <div className="calibrate-status">{calibrationRating(brier, resolved.length)}</div>

            {open.length > 0 && (
                <div className="calibrate-section">
                    <div className="calibrate-section-title">Awaiting resolution ({open.length})</div>
                    <ul className="calibrate-list">
                        {open.map((p) => {
                            const overdue = p.resolveBy !== null && p.resolveBy < Date.now();
                            return (
                                <li key={p.id} className={`calibrate-item${overdue ? ' calibrate-item-overdue' : ''}`}>
                                    <div className="calibrate-item-body">
                                        <div className="calibrate-item-statement">{p.statement}</div>
                                        <div className="calibrate-item-meta">
                                            <span className="calibrate-tag">{p.category}</span>
                                            <span>{p.confidencePct}% confident</span>
                                            <span>{formatDue(p.resolveBy)}</span>
                                        </div>
                                    </div>
                                    <div className="calibrate-item-actions">
                                        <button
                                            type="button"
                                            className="calibrate-chip calibrate-chip-yes"
                                            onClick={() => resolvePrediction(p.id, 'yes')}
                                        >
                                            Happened
                                        </button>
                                        <button
                                            type="button"
                                            className="calibrate-chip calibrate-chip-no"
                                            onClick={() => resolvePrediction(p.id, 'no')}
                                        >
                                            Didn't
                                        </button>
                                        <button
                                            type="button"
                                            className="calibrate-chip calibrate-chip-remove"
                                            onClick={() => removePrediction(p.id)}
                                            aria-label="Delete prediction"
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

            {resolved.length > 0 && (
                <div className="calibrate-section">
                    <div className="calibrate-section-title">Calibration ({resolved.length} resolved)</div>
                    <div className="calibrate-buckets">
                        {buckets
                            .filter((b) => b.count > 0)
                            .map((b) => (
                                <div key={b.label} className="calibrate-bucket">
                                    <div className="calibrate-bucket-label">
                                        <span>{b.label}</span>
                                        <span className="calibrate-bucket-count">{b.count} calls</span>
                                    </div>
                                    <div className="calibrate-bucket-bars">
                                        <div className="calibrate-bar-row">
                                            <span className="calibrate-bar-tag">predicted</span>
                                            <div className="calibrate-bar-track">
                                                <div
                                                    className="calibrate-bar-fill calibrate-bar-predicted"
                                                    style={{ width: `${b.avgConfidence}%` }}
                                                />
                                            </div>
                                            <span className="calibrate-bar-value">{b.avgConfidence.toFixed(0)}%</span>
                                        </div>
                                        <div className="calibrate-bar-row">
                                            <span className="calibrate-bar-tag">actual</span>
                                            <div className="calibrate-bar-track">
                                                <div
                                                    className="calibrate-bar-fill calibrate-bar-actual"
                                                    style={{ width: `${b.actualRate}%` }}
                                                />
                                            </div>
                                            <span className="calibrate-bar-value">{b.actualRate.toFixed(0)}%</span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                    </div>
                </div>
            )}

            <div className="calibrate-tip">
                Brier score = mean squared error between stated confidence and outcome. 0.0 is a perfect
                forecaster, 0.25 is a coin flip. Predicted vs. actual bars should roughly match if you're
                well-calibrated. Everything stays on this device — nothing is sent anywhere.
            </div>
        </div>
    );
}

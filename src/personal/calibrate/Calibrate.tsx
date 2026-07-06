import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './calibrate.css';

type Status = 'pending' | 'true' | 'false';

type Prediction = {
    id: string;
    text: string;
    confidence: number; // 50-99, stated probability the statement resolves true
    createdAt: number;
    resolveBy: string | null; // yyyy-mm-dd
    status: Status;
    resolvedAt: number | null;
};

const KEY = 'personal:calibrate:v1';
const BUCKETS: [number, number][] = [
    [50, 60],
    [60, 70],
    [70, 80],
    [80, 90],
    [90, 100],
];

function uid(): string {
    return `${Math.random().toString(36).slice(2)}${Math.random().toString(36).slice(2)}`;
}

function load(): Prediction[] {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return [];
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? parsed.slice(-500) : [];
    } catch {
        return [];
    }
}

function save(predictions: Prediction[]): void {
    try {
        localStorage.setItem(KEY, JSON.stringify(predictions.slice(-500)));
    } catch {
        /* localStorage full or denied — non-fatal */
    }
}

function todayStr(): string {
    return new Date().toISOString().slice(0, 10);
}

function brierLabel(score: number): string {
    if (score < 0.1) return 'excellent calibration';
    if (score < 0.18) return 'well calibrated';
    if (score < 0.25) return 'room to tighten up';
    return 'overconfident or underconfident';
}

export default function Calibrate() {
    const [predictions, setPredictions] = useState<Prediction[]>(() => load());
    const [text, setText] = useState('');
    const [confidence, setConfidence] = useState(70);
    const [resolveBy, setResolveBy] = useState('');

    function persist(next: Prediction[]) {
        setPredictions(next);
        save(next);
    }

    function addPrediction() {
        const trimmed = text.trim();
        if (!trimmed) return;
        const p: Prediction = {
            id: uid(),
            text: trimmed,
            confidence,
            createdAt: Date.now(),
            resolveBy: resolveBy || null,
            status: 'pending',
            resolvedAt: null,
        };
        persist([...predictions, p]);
        setText('');
        setResolveBy('');
        setConfidence(70);
    }

    function resolve(id: string, outcome: 'true' | 'false') {
        persist(
            predictions.map((p) =>
                p.id === id ? { ...p, status: outcome, resolvedAt: Date.now() } : p
            )
        );
    }

    function remove(id: string) {
        persist(predictions.filter((p) => p.id !== id));
    }

    const pending = useMemo(
        () => predictions.filter((p) => p.status === 'pending').sort((a, b) => a.createdAt - b.createdAt),
        [predictions]
    );
    const resolved = useMemo(
        () => predictions.filter((p) => p.status !== 'pending').sort((a, b) => (b.resolvedAt ?? 0) - (a.resolvedAt ?? 0)),
        [predictions]
    );

    const brier = useMemo(() => {
        if (resolved.length === 0) return null;
        const sum = resolved.reduce((acc, p) => {
            const outcome = p.status === 'true' ? 1 : 0;
            const c = p.confidence / 100;
            return acc + (c - outcome) ** 2;
        }, 0);
        return sum / resolved.length;
    }, [resolved]);

    const buckets = useMemo(() => {
        return BUCKETS.map(([lo, hi]) => {
            const inBucket = resolved.filter((p) => p.confidence >= lo && p.confidence < hi + (hi === 100 ? 1 : 0));
            if (inBucket.length === 0) return { lo, hi, count: 0, avgConfidence: 0, actualRate: 0 };
            const avgConfidence = inBucket.reduce((s, p) => s + p.confidence, 0) / inBucket.length;
            const actualRate =
                (inBucket.filter((p) => p.status === 'true').length / inBucket.length) * 100;
            return { lo, hi, count: inBucket.length, avgConfidence, actualRate };
        });
    }, [resolved]);

    const today = todayStr();

    return (
        <div className="calib">
            <div className="calib-header">
                <Link to="/personal" className="calib-back">← personal</Link>
                <span className="calib-title">Calibrate</span>
                <span className="calib-brier">
                    {brier !== null ? `brier ${brier.toFixed(3)}` : 'no data yet'}
                </span>
            </div>

            <div className="calib-intro">
                Log a prediction with your honest confidence (50–99%). Resolve it later.
                Over time this shows whether your gut feel matches reality — the core
                skill behind every good bet, forecast, and business decision.
            </div>

            <div className="calib-form">
                <textarea
                    className="calib-input"
                    placeholder="What are you predicting? e.g. 'This feature ships by Friday'"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    rows={2}
                />
                <div className="calib-form-row">
                    <label className="calib-confidence-label">
                        confidence
                        <input
                            type="range"
                            min={50}
                            max={99}
                            value={confidence}
                            onChange={(e) => setConfidence(Number(e.target.value))}
                        />
                        <span className="calib-confidence-value">{confidence}%</span>
                    </label>
                    <input
                        type="date"
                        className="calib-date-input"
                        value={resolveBy}
                        min={today}
                        onChange={(e) => setResolveBy(e.target.value)}
                    />
                </div>
                <button type="button" className="calib-btn calib-btn-primary" onClick={addPrediction}>
                    Log prediction
                </button>
            </div>

            {pending.length > 0 && (
                <div className="calib-section">
                    <div className="calib-section-title">Pending ({pending.length})</div>
                    <ul className="calib-list">
                        {pending.map((p) => {
                            const overdue = p.resolveBy !== null && p.resolveBy < today;
                            return (
                                <li key={p.id} className={`calib-item${overdue ? ' calib-item-overdue' : ''}`}>
                                    <div className="calib-item-main">
                                        <span className="calib-item-text">{p.text}</span>
                                        <span className="calib-item-conf">{p.confidence}%</span>
                                    </div>
                                    <div className="calib-item-meta">
                                        {p.resolveBy && (
                                            <span className={overdue ? 'calib-overdue-tag' : ''}>
                                                {overdue ? 'overdue since' : 'check by'} {p.resolveBy}
                                            </span>
                                        )}
                                        <div className="calib-item-actions">
                                            <button type="button" className="calib-chip calib-chip-true" onClick={() => resolve(p.id, 'true')}>
                                                ✓ true
                                            </button>
                                            <button type="button" className="calib-chip calib-chip-false" onClick={() => resolve(p.id, 'false')}>
                                                ✗ false
                                            </button>
                                            <button type="button" className="calib-chip" onClick={() => remove(p.id)}>
                                                remove
                                            </button>
                                        </div>
                                    </div>
                                </li>
                            );
                        })}
                    </ul>
                </div>
            )}

            {resolved.length > 0 && (
                <>
                    <div className="calib-section">
                        <div className="calib-section-title">Calibration by confidence</div>
                        <div className="calib-buckets">
                            {buckets.map((b) => (
                                <div key={b.lo} className="calib-bucket">
                                    <span className="calib-bucket-range">{b.lo}-{b.hi}%</span>
                                    <div className="calib-bucket-bar-track">
                                        <div
                                            className="calib-bucket-bar-predicted"
                                            style={{ width: `${b.count ? (b.lo + b.hi) / 2 : 0}%` }}
                                        />
                                        <div
                                            className="calib-bucket-bar-actual"
                                            style={{ width: `${b.actualRate}%` }}
                                        />
                                    </div>
                                    <span className="calib-bucket-count">
                                        {b.count > 0 ? `${Math.round(b.actualRate)}% actual (n=${b.count})` : 'n=0'}
                                    </span>
                                </div>
                            ))}
                        </div>
                        <div className="calib-bucket-legend">
                            <span><i className="calib-swatch calib-swatch-predicted" /> stated confidence</span>
                            <span><i className="calib-swatch calib-swatch-actual" /> actual outcome rate</span>
                        </div>
                    </div>

                    {brier !== null && (
                        <div className="calib-stats">
                            <div className="calib-stat">
                                <span className="calib-stat-label">brier score</span>
                                <span className="calib-stat-value">{brier.toFixed(3)}</span>
                            </div>
                            <div className="calib-stat">
                                <span className="calib-stat-label">resolved</span>
                                <span className="calib-stat-value">{resolved.length}</span>
                            </div>
                            <div className="calib-stat">
                                <span className="calib-stat-label">read</span>
                                <span className="calib-stat-value calib-stat-value-text">{brierLabel(brier)}</span>
                            </div>
                        </div>
                    )}

                    <div className="calib-section">
                        <div className="calib-section-title">History</div>
                        <ul className="calib-list">
                            {resolved.slice(0, 20).map((p) => (
                                <li key={p.id} className="calib-item calib-item-resolved">
                                    <div className="calib-item-main">
                                        <span className="calib-item-text">{p.text}</span>
                                        <span className="calib-item-conf">{p.confidence}%</span>
                                    </div>
                                    <span className={`calib-outcome-tag calib-outcome-${p.status}`}>
                                        {p.status === 'true' ? 'happened' : "didn't happen"}
                                    </span>
                                </li>
                            ))}
                        </ul>
                    </div>
                </>
            )}

            {predictions.length === 0 && (
                <div className="calib-empty">
                    No predictions yet. Log your first one above — confidence should reflect
                    what you'd bet real money on, not what you want to be true.
                </div>
            )}
        </div>
    );
}

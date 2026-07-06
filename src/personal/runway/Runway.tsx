import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './runway.css';

const KEY = 'personal:runway:v1';
const MAX_SNAPSHOTS = 36; // 3 years of monthly logs
const MONTHLY_STEP_CAP = 600; // 50 years, for the years-to-FI search

type Inputs = {
    liquidSavings: number;
    investments: number;
    monthlyIncome: number;
    monthlyExpenses: number;
    annualReturnPct: number;
};

type Snapshot = {
    ts: number;
    netWorth: number;
    runwayMonths: number;
    savingsRatePct: number;
};

type State = {
    inputs: Inputs;
    history: Snapshot[];
};

const DEFAULT_INPUTS: Inputs = {
    liquidSavings: 0,
    investments: 0,
    monthlyIncome: 0,
    monthlyExpenses: 0,
    annualReturnPct: 7,
};

function loadState(): State {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { inputs: DEFAULT_INPUTS, history: [] };
        const parsed = JSON.parse(raw);
        return {
            inputs: { ...DEFAULT_INPUTS, ...(parsed?.inputs ?? {}) },
            history: Array.isArray(parsed?.history) ? parsed.history.slice(-MAX_SNAPSHOTS) : [],
        };
    } catch {
        return { inputs: DEFAULT_INPUTS, history: [] };
    }
}

function saveState(s: State): void {
    try {
        localStorage.setItem(KEY, JSON.stringify(s));
    } catch {
        /* localStorage full or denied — non-fatal */
    }
}

function money(n: number): string {
    if (!Number.isFinite(n)) return '$0';
    return n.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
}

function runwayRating(months: number): string {
    if (months >= 6) return '🟢 solid buffer — room to take calculated risks';
    if (months >= 3) return '🟡 building — keep stacking cash';
    if (months > 0) return '🔴 thin — prioritize the buffer before anything else';
    return '⚪ add your numbers to see where you stand';
}

// Iterates month-by-month rather than a closed-form annuity formula —
// simpler to keep correct when income/expenses/investments interact.
function yearsToFI(investments: number, monthlySavings: number, annualReturnPct: number, fiNumber: number): number | null {
    if (investments >= fiNumber) return 0;
    if (monthlySavings <= 0 && annualReturnPct <= 0) return null;
    const monthlyRate = annualReturnPct / 100 / 12;
    let balance = investments;
    for (let month = 1; month <= MONTHLY_STEP_CAP; month++) {
        balance = balance * (1 + monthlyRate) + monthlySavings;
        if (balance >= fiNumber) return month / 12;
    }
    return null; // won't reach FI within 50 years at this rate
}

export default function Runway() {
    const [state, setState] = useState<State>(() => loadState());
    const [note, setNote] = useState<string | null>(null);
    const { inputs, history } = state;

    const setField = useCallback((field: keyof Inputs, value: number) => {
        setState((prev) => {
            const next = { ...prev, inputs: { ...prev.inputs, [field]: value } };
            saveState(next);
            return next;
        });
    }, []);

    const stats = useMemo(() => {
        const netWorth = inputs.liquidSavings + inputs.investments;
        const monthlySavings = inputs.monthlyIncome - inputs.monthlyExpenses;
        const savingsRatePct = inputs.monthlyIncome > 0 ? (monthlySavings / inputs.monthlyIncome) * 100 : 0;
        const runwayMonths = inputs.monthlyExpenses > 0 ? inputs.liquidSavings / inputs.monthlyExpenses : 0;
        const fiNumber = inputs.monthlyExpenses * 12 * 25;
        const toFI = yearsToFI(inputs.investments, monthlySavings, inputs.annualReturnPct, fiNumber);
        return { netWorth, monthlySavings, savingsRatePct, runwayMonths, fiNumber, toFI };
    }, [inputs]);

    const logSnapshot = useCallback(() => {
        setState((prev) => {
            const snapshot: Snapshot = {
                ts: Date.now(),
                netWorth: stats.netWorth,
                runwayMonths: stats.runwayMonths,
                savingsRatePct: stats.savingsRatePct,
            };
            const nextHistory = [...prev.history, snapshot].slice(-MAX_SNAPSHOTS);
            const next = { ...prev, history: nextHistory };
            saveState(next);
            return next;
        });
        setNote('snapshot logged');
        window.setTimeout(() => setNote(null), 2000);
    }, [stats]);

    const clearHistory = useCallback(() => {
        setState((prev) => {
            const next = { ...prev, history: [] };
            saveState(next);
            return next;
        });
    }, []);

    const chartPoints = useMemo(() => {
        if (history.length < 2) return null;
        const values = history.map((h) => h.netWorth);
        const min = Math.min(...values, 0);
        const max = Math.max(...values, 1);
        const range = max - min || 1;
        const w = 100;
        const h = 32;
        return history
            .map((point, i) => {
                const x = (i / (history.length - 1)) * w;
                const y = h - ((point.netWorth - min) / range) * h;
                return `${x.toFixed(2)},${y.toFixed(2)}`;
            })
            .join(' ');
    }, [history]);

    return (
        <div className="runway">
            <div className="runway-header">
                <Link to="/personal" className="runway-back">← personal</Link>
                <span className="runway-title">Runway</span>
                <span className="runway-badge">{money(stats.netWorth)}</span>
            </div>

            <div className="runway-form">
                <label className="runway-field">
                    <span>Liquid savings</span>
                    <input
                        type="number"
                        inputMode="decimal"
                        value={inputs.liquidSavings || ''}
                        onChange={(e) => setField('liquidSavings', Number(e.target.value) || 0)}
                        placeholder="0"
                    />
                </label>
                <label className="runway-field">
                    <span>Investments</span>
                    <input
                        type="number"
                        inputMode="decimal"
                        value={inputs.investments || ''}
                        onChange={(e) => setField('investments', Number(e.target.value) || 0)}
                        placeholder="0"
                    />
                </label>
                <label className="runway-field">
                    <span>Monthly income</span>
                    <input
                        type="number"
                        inputMode="decimal"
                        value={inputs.monthlyIncome || ''}
                        onChange={(e) => setField('monthlyIncome', Number(e.target.value) || 0)}
                        placeholder="0"
                    />
                </label>
                <label className="runway-field">
                    <span>Monthly expenses</span>
                    <input
                        type="number"
                        inputMode="decimal"
                        value={inputs.monthlyExpenses || ''}
                        onChange={(e) => setField('monthlyExpenses', Number(e.target.value) || 0)}
                        placeholder="0"
                    />
                </label>
                <label className="runway-field">
                    <span>Assumed return (annual %)</span>
                    <input
                        type="number"
                        inputMode="decimal"
                        value={inputs.annualReturnPct}
                        onChange={(e) => setField('annualReturnPct', Number(e.target.value) || 0)}
                        placeholder="7"
                    />
                </label>
            </div>

            <div className="runway-status">{runwayRating(stats.runwayMonths)}</div>

            <div className="runway-stats">
                <div className="runway-stat">
                    <span className="runway-stat-label">runway</span>
                    <span className="runway-stat-value">
                        {stats.runwayMonths > 0 ? `${stats.runwayMonths.toFixed(1)}mo` : '—'}
                    </span>
                </div>
                <div className="runway-stat">
                    <span className="runway-stat-label">savings rate</span>
                    <span className="runway-stat-value">{stats.savingsRatePct.toFixed(0)}%</span>
                </div>
                <div className="runway-stat">
                    <span className="runway-stat-label">FI number</span>
                    <span className="runway-stat-value">{money(stats.fiNumber)}</span>
                </div>
                <div className="runway-stat">
                    <span className="runway-stat-label">years to FI</span>
                    <span className="runway-stat-value">
                        {stats.toFI === null ? '50+' : stats.toFI.toFixed(1)}
                    </span>
                </div>
            </div>

            {chartPoints && (
                <div className="runway-chart">
                    <svg viewBox="0 0 100 32" preserveAspectRatio="none">
                        <polyline points={chartPoints} fill="none" stroke="#0ea5e9" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
                    </svg>
                    <div className="runway-chart-caption">net worth · last {history.length} snapshots</div>
                </div>
            )}

            <div className="runway-actions">
                <button type="button" className="runway-btn runway-btn-primary" onClick={logSnapshot}>
                    Log this month's snapshot
                </button>
                {history.length > 0 && (
                    <button type="button" className="runway-btn" onClick={clearHistory}>
                        Clear history
                    </button>
                )}
                {note && <div className="runway-note">{note}</div>}
            </div>

            <div className="runway-tip">
                runway = liquid savings ÷ monthly expenses. FI number = annual expenses × 25.
                everything stays on this device — nothing is sent anywhere.
            </div>
        </div>
    );
}

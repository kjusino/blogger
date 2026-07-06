import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './parity.css';

const KEY = 'personal:parity:v1';
const MAX_OFFERS = 50;

type VestSchedule = 'even' | 'front' | 'back';

type Offer = {
    id: string;
    company: string;
    role: string;
    location: string;
    colIndex: number;
    baseSalary: number;
    signingBonus: number;
    annualBonusPct: number;
    equityValue: number;
    vestYears: number;
    vestSchedule: VestSchedule;
    benefitsValue: number;
    createdAt: number;
};

type State = {
    offers: Offer[];
};

const SCHEDULES: { value: VestSchedule; label: string }[] = [
    { value: 'even', label: 'Even (standard)' },
    { value: 'front', label: 'Front-loaded' },
    { value: 'back', label: 'Back-loaded (Amazon-style)' },
];

function loadState(): State {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { offers: [] };
        const parsed = JSON.parse(raw);
        return { offers: Array.isArray(parsed?.offers) ? parsed.offers.slice(-MAX_OFFERS) : [] };
    } catch {
        return { offers: [] };
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

function money(n: number): string {
    if (!Number.isFinite(n)) return '$0';
    return n.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 });
}

// Annual vesting weights (sum to 1). "even" mirrors a standard grant; "front"/"back" model
// employers that skew payout toward the first or final years of a multi-year grant.
function vestWeights(years: number, schedule: VestSchedule): number[] {
    const n = Math.max(1, Math.round(years) || 1);
    let raw: number[];
    if (schedule === 'front') {
        raw = Array.from({ length: n }, (_, i) => (n - i) ** 2);
    } else if (schedule === 'back') {
        raw = Array.from({ length: n }, (_, i) => (i + 1) ** 2);
    } else {
        raw = Array(n).fill(1);
    }
    const sum = raw.reduce((a, b) => a + b, 0) || 1;
    return raw.map((w) => w / sum);
}

type Metrics = {
    year1: number;
    avgAnnual: number;
    colAdjustedAvg: number;
    totalComp: number;
    advertisedAnnual: number;
    gap: number;
};

function computeMetrics(o: Offer): Metrics {
    const years = Math.max(1, Math.round(o.vestYears) || 1);
    const weights = vestWeights(years, o.vestSchedule);
    const bonus = (o.baseSalary * o.annualBonusPct) / 100;
    const perYear = weights.map(
        (w, i) => o.baseSalary + bonus + o.benefitsValue + o.equityValue * w + (i === 0 ? o.signingBonus : 0)
    );
    const totalComp = perYear.reduce((a, b) => a + b, 0);
    const avgAnnual = totalComp / years;
    const colAdjustedAvg = avgAnnual / ((o.colIndex || 100) / 100);
    const advertisedAnnual = o.baseSalary + bonus + o.equityValue / years + o.benefitsValue;
    const year1 = perYear[0];
    return { year1, avgAnnual, colAdjustedAvg, totalComp, advertisedAnnual, gap: advertisedAnnual - year1 };
}

export default function Parity() {
    const [state, setState] = useState<State>(() => loadState());
    const [company, setCompany] = useState('');
    const [role, setRole] = useState('');
    const [location, setLocation] = useState('');
    const [colIndex, setColIndex] = useState(100);
    const [baseSalary, setBaseSalary] = useState(0);
    const [signingBonus, setSigningBonus] = useState(0);
    const [annualBonusPct, setAnnualBonusPct] = useState(0);
    const [equityValue, setEquityValue] = useState(0);
    const [vestYears, setVestYears] = useState(4);
    const [vestSchedule, setVestSchedule] = useState<VestSchedule>('even');
    const [benefitsValue, setBenefitsValue] = useState(0);
    const { offers } = state;

    const addOffer = useCallback(() => {
        const trimmedCompany = company.trim();
        const trimmedRole = role.trim();
        if (!trimmedCompany || !trimmedRole || baseSalary <= 0) return;
        const offer: Offer = {
            id: makeId(),
            company: trimmedCompany,
            role: trimmedRole,
            location: location.trim(),
            colIndex: colIndex || 100,
            baseSalary,
            signingBonus,
            annualBonusPct,
            equityValue,
            vestYears: vestYears || 4,
            vestSchedule,
            benefitsValue,
            createdAt: Date.now(),
        };
        setState((prev) => {
            const next = { offers: [...prev.offers, offer].slice(-MAX_OFFERS) };
            saveState(next);
            return next;
        });
        setCompany('');
        setRole('');
        setLocation('');
        setColIndex(100);
        setBaseSalary(0);
        setSigningBonus(0);
        setAnnualBonusPct(0);
        setEquityValue(0);
        setVestYears(4);
        setVestSchedule('even');
        setBenefitsValue(0);
    }, [company, role, location, colIndex, baseSalary, signingBonus, annualBonusPct, equityValue, vestYears, vestSchedule, benefitsValue]);

    const removeOffer = useCallback((id: string) => {
        setState((prev) => {
            const next = { offers: prev.offers.filter((o) => o.id !== id) };
            saveState(next);
            return next;
        });
    }, []);

    const ranked = useMemo(
        () =>
            offers
                .map((o) => ({ offer: o, metrics: computeMetrics(o) }))
                .sort((a, b) => b.metrics.colAdjustedAvg - a.metrics.colAdjustedAvg),
        [offers]
    );

    const overallMessage = useMemo(() => {
        if (ranked.length === 0) {
            return "⚪ add offers to compare — most people compare sticker numbers, not what actually lands in year one";
        }
        if (ranked.length === 1) return 'add a second offer to see how it stacks up';
        const best = ranked[0];
        const worstGap = ranked.reduce((max, r) => Math.max(max, r.metrics.gap), 0);
        if (worstGap > 2000) {
            return `🟡 ${best.offer.company} leads on cost-of-living-adjusted comp — but check the vesting gap flagged below`;
        }
        return `🟢 ${best.offer.company} leads on cost-of-living-adjusted average comp`;
    }, [ranked]);

    return (
        <div className="parity">
            <div className="parity-header">
                <Link to="/personal" className="parity-back">← personal</Link>
                <span className="parity-title">Parity</span>
                <span className="parity-badge">{offers.length}</span>
            </div>

            <div className="parity-form">
                <div className="parity-form-row">
                    <label className="parity-field">
                        <span>Company</span>
                        <input type="text" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="e.g. Stripe" />
                    </label>
                    <label className="parity-field">
                        <span>Role</span>
                        <input type="text" value={role} onChange={(e) => setRole(e.target.value)} placeholder="e.g. Senior Engineer" />
                    </label>
                </div>
                <div className="parity-form-row">
                    <label className="parity-field">
                        <span>Location (optional)</span>
                        <input type="text" value={location} onChange={(e) => setLocation(e.target.value)} placeholder="e.g. Austin" />
                    </label>
                    <label className="parity-field">
                        <span>Cost-of-living index</span>
                        <input
                            type="number"
                            inputMode="decimal"
                            value={colIndex || ''}
                            onChange={(e) => setColIndex(Number(e.target.value) || 0)}
                            placeholder="100 = your baseline"
                        />
                    </label>
                </div>
                <div className="parity-form-row">
                    <label className="parity-field">
                        <span>Base salary</span>
                        <input
                            type="number"
                            inputMode="decimal"
                            value={baseSalary || ''}
                            onChange={(e) => setBaseSalary(Number(e.target.value) || 0)}
                            placeholder="0"
                        />
                    </label>
                    <label className="parity-field">
                        <span>Signing bonus</span>
                        <input
                            type="number"
                            inputMode="decimal"
                            value={signingBonus || ''}
                            onChange={(e) => setSigningBonus(Number(e.target.value) || 0)}
                            placeholder="0"
                        />
                    </label>
                </div>
                <div className="parity-form-row">
                    <label className="parity-field">
                        <span>Target bonus (% of base)</span>
                        <input
                            type="number"
                            inputMode="decimal"
                            value={annualBonusPct || ''}
                            onChange={(e) => setAnnualBonusPct(Number(e.target.value) || 0)}
                            placeholder="0"
                        />
                    </label>
                    <label className="parity-field">
                        <span>Benefits value (annual, est.)</span>
                        <input
                            type="number"
                            inputMode="decimal"
                            value={benefitsValue || ''}
                            onChange={(e) => setBenefitsValue(Number(e.target.value) || 0)}
                            placeholder="401k match, healthcare"
                        />
                    </label>
                </div>
                <div className="parity-form-row">
                    <label className="parity-field">
                        <span>Total equity grant</span>
                        <input
                            type="number"
                            inputMode="decimal"
                            value={equityValue || ''}
                            onChange={(e) => setEquityValue(Number(e.target.value) || 0)}
                            placeholder="0"
                        />
                    </label>
                    <label className="parity-field">
                        <span>Vest years</span>
                        <input
                            type="number"
                            inputMode="decimal"
                            value={vestYears || ''}
                            onChange={(e) => setVestYears(Number(e.target.value) || 0)}
                            placeholder="4"
                        />
                    </label>
                </div>
                <label className="parity-field">
                    <span>Vesting schedule</span>
                    <select value={vestSchedule} onChange={(e) => setVestSchedule(e.target.value as VestSchedule)}>
                        {SCHEDULES.map((s) => (
                            <option key={s.value} value={s.value}>
                                {s.label}
                            </option>
                        ))}
                    </select>
                </label>
                <button type="button" className="parity-btn parity-btn-primary" onClick={addOffer}>
                    Add offer
                </button>
            </div>

            <div className="parity-status">{overallMessage}</div>

            {ranked.map(({ offer, metrics }, i) => (
                <div key={offer.id} className="parity-card">
                    <div className="parity-card-header">
                        <span className="parity-rank">#{i + 1}</span>
                        <div className="parity-card-title">
                            <div className="parity-company">
                                {offer.company} <span className="parity-role">— {offer.role}</span>
                            </div>
                            {offer.location && (
                                <div className="parity-location">
                                    {offer.location} · COL {offer.colIndex}
                                </div>
                            )}
                        </div>
                        <button
                            type="button"
                            className="parity-chip parity-chip-remove"
                            onClick={() => removeOffer(offer.id)}
                            aria-label="Remove offer"
                        >
                            ✕
                        </button>
                    </div>
                    <div className="parity-stats">
                        <div className="parity-stat">
                            <span className="parity-stat-label">year 1</span>
                            <span className="parity-stat-value">{money(metrics.year1)}</span>
                        </div>
                        <div className="parity-stat">
                            <span className="parity-stat-label">avg / yr</span>
                            <span className="parity-stat-value">{money(metrics.avgAnnual)}</span>
                        </div>
                        <div className="parity-stat">
                            <span className="parity-stat-label">COL-adj avg</span>
                            <span className="parity-stat-value">{money(metrics.colAdjustedAvg)}</span>
                        </div>
                        <div className="parity-stat">
                            <span className="parity-stat-label">{offer.vestYears}-yr total</span>
                            <span className="parity-stat-value">{money(metrics.totalComp)}</span>
                        </div>
                    </div>
                    {metrics.gap > 2000 && (
                        <div className="parity-gap-warning">
                            recruiter math says ~{money(metrics.advertisedAnnual)}/yr, but this vesting schedule really
                            pays {money(metrics.year1)} in year one — a {money(metrics.gap)} gap
                        </div>
                    )}
                </div>
            ))}

            <div className="parity-tip">
                COL-adj avg normalizes total comp against your baseline (index 100), so a lower sticker number in a
                cheaper city can still win. Everything stays on this device — nothing is sent anywhere.
            </div>
        </div>
    );
}

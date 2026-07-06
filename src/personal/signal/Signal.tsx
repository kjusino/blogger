import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './signal.css';

const KEY = 'personal:signal:v1';
const MAX_OPPS = 500;
const DAY_MS = 24 * 60 * 60 * 1000;

type Stage = 'applied' | 'screen' | 'interview' | 'offer' | 'rejected' | 'withdrawn';

type Opportunity = {
    id: string;
    company: string;
    role: string;
    stage: Stage;
    source: string;
    notes: string;
    followUpAt: number | null;
    createdAt: number;
};

type State = {
    opportunities: Opportunity[];
};

const STAGES: { value: Stage; label: string }[] = [
    { value: 'applied', label: 'Applied' },
    { value: 'screen', label: 'Screen' },
    { value: 'interview', label: 'Interview' },
    { value: 'offer', label: 'Offer' },
    { value: 'rejected', label: 'Rejected' },
    { value: 'withdrawn', label: 'Withdrawn' },
];

const CLOSED_STAGES: Stage[] = ['rejected', 'withdrawn'];

const NEXT_STAGE: Partial<Record<Stage, Stage>> = {
    applied: 'screen',
    screen: 'interview',
    interview: 'offer',
};

function stageLabel(stage: Stage): string {
    return STAGES.find((s) => s.value === stage)?.label ?? stage;
}

function loadState(): State {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { opportunities: [] };
        const parsed = JSON.parse(raw);
        return {
            opportunities: Array.isArray(parsed?.opportunities) ? parsed.opportunities.slice(-MAX_OPPS) : [],
        };
    } catch {
        return { opportunities: [] };
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

function parseDateInput(value: string): number | null {
    if (!value) return null;
    const ts = new Date(`${value}T09:00:00`).getTime();
    return Number.isNaN(ts) ? null : ts;
}

type Bucket = 'overdue' | 'soon' | 'ontrack' | 'nodate';

function bucketFor(followUpAt: number | null): Bucket {
    if (followUpAt == null) return 'nodate';
    const diffDays = Math.ceil((followUpAt - Date.now()) / DAY_MS);
    if (diffDays <= 0) return 'overdue';
    if (diffDays <= 2) return 'soon';
    return 'ontrack';
}

const BUCKET_LABEL: Record<Bucket, string> = {
    overdue: 'Follow up overdue',
    soon: 'Follow up soon',
    ontrack: 'On track',
    nodate: 'No follow-up set',
};

export default function Signal() {
    const [state, setState] = useState<State>(() => loadState());
    const [company, setCompany] = useState('');
    const [role, setRole] = useState('');
    const [source, setSource] = useState('');
    const [followUpInput, setFollowUpInput] = useState('');
    const [copied, setCopied] = useState(false);
    const { opportunities } = state;

    const addOpportunity = useCallback(() => {
        const trimmedCompany = company.trim();
        const trimmedRole = role.trim();
        if (!trimmedCompany || !trimmedRole) return;
        const opp: Opportunity = {
            id: makeId(),
            company: trimmedCompany,
            role: trimmedRole,
            stage: 'applied',
            source: source.trim(),
            notes: '',
            followUpAt: parseDateInput(followUpInput),
            createdAt: Date.now(),
        };
        setState((prev) => {
            const next = { opportunities: [...prev.opportunities, opp].slice(-MAX_OPPS) };
            saveState(next);
            return next;
        });
        setCompany('');
        setRole('');
        setSource('');
        setFollowUpInput('');
    }, [company, role, source, followUpInput]);

    const advanceStage = useCallback((id: string) => {
        setState((prev) => {
            const next = {
                opportunities: prev.opportunities.map((o) => {
                    if (o.id !== id) return o;
                    const nextStage = NEXT_STAGE[o.stage];
                    if (!nextStage) return o;
                    return { ...o, stage: nextStage, followUpAt: null };
                }),
            };
            saveState(next);
            return next;
        });
    }, []);

    const setStage = useCallback((id: string, stage: Stage) => {
        setState((prev) => {
            const next = {
                opportunities: prev.opportunities.map((o) => (o.id === id ? { ...o, stage } : o)),
            };
            saveState(next);
            return next;
        });
    }, []);

    const snoozeFollowUp = useCallback((id: string, days: number) => {
        setState((prev) => {
            const next = {
                opportunities: prev.opportunities.map((o) =>
                    o.id === id ? { ...o, followUpAt: Date.now() + days * DAY_MS } : o
                ),
            };
            saveState(next);
            return next;
        });
    }, []);

    const removeOpportunity = useCallback((id: string) => {
        setState((prev) => {
            const next = { opportunities: prev.opportunities.filter((o) => o.id !== id) };
            saveState(next);
            return next;
        });
    }, []);

    const active = useMemo(
        () => opportunities.filter((o) => !CLOSED_STAGES.includes(o.stage)),
        [opportunities]
    );
    const closed = useMemo(
        () => opportunities.filter((o) => CLOSED_STAGES.includes(o.stage)),
        [opportunities]
    );

    const withBucket = useMemo(
        () => active.map((o) => ({ opp: o, bucket: bucketFor(o.followUpAt) })),
        [active]
    );

    const overdueCount = useMemo(() => withBucket.filter((w) => w.bucket === 'overdue').length, [withBucket]);

    const overallMessage = useMemo(() => {
        if (active.length === 0) return "⚪ log the roles you're pursuing — the follow-up is what most people skip";
        if (overdueCount === 0) return '🟢 pipeline is current — no follow-ups overdue';
        if (overdueCount === 1) return '🟡 1 follow-up is overdue';
        return `🔴 ${overdueCount} follow-ups are overdue`;
    }, [active.length, overdueCount]);

    const stageCounts = useMemo(
        () =>
            STAGES.filter((s) => !CLOSED_STAGES.includes(s.value)).map((s) => ({
                ...s,
                count: active.filter((o) => o.stage === s.value).length,
            })).filter((s) => s.count > 0),
        [active]
    );
    const maxStageCount = Math.max(1, ...stageCounts.map((s) => s.count));

    const grouped = useMemo(() => {
        const order: Bucket[] = ['overdue', 'soon', 'ontrack', 'nodate'];
        return order
            .map((b) => ({
                bucket: b,
                items: withBucket
                    .filter((w) => w.bucket === b)
                    .sort((a, b2) => (a.opp.followUpAt ?? Infinity) - (b2.opp.followUpAt ?? Infinity)),
            }))
            .filter((g) => g.items.length > 0);
    }, [withBucket]);

    const copyFollowUps = useCallback(async () => {
        const source2 = withBucket
            .filter((w) => w.bucket === 'overdue' || w.bucket === 'soon')
            .slice(0, 10);
        const text = source2
            .map(({ opp }) => `- ${opp.company} — ${opp.role} (${stageLabel(opp.stage)})`)
            .join('\n');
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            /* clipboard denied — non-fatal */
        }
    }, [withBucket]);

    return (
        <div className="signal">
            <div className="signal-header">
                <Link to="/personal" className="signal-back">← personal</Link>
                <span className="signal-title">Signal</span>
                <span className="signal-badge">{active.length}</span>
            </div>

            <div className="signal-form">
                <div className="signal-form-row">
                    <label className="signal-field">
                        <span>Company</span>
                        <input
                            type="text"
                            value={company}
                            onChange={(e) => setCompany(e.target.value)}
                            placeholder="e.g. Stripe"
                        />
                    </label>
                    <label className="signal-field">
                        <span>Role</span>
                        <input
                            type="text"
                            value={role}
                            onChange={(e) => setRole(e.target.value)}
                            placeholder="e.g. Senior Engineer"
                        />
                    </label>
                </div>
                <div className="signal-form-row">
                    <label className="signal-field">
                        <span>Source (optional)</span>
                        <input
                            type="text"
                            value={source}
                            onChange={(e) => setSource(e.target.value)}
                            placeholder="referral, recruiter, cold"
                        />
                    </label>
                    <label className="signal-field">
                        <span>Next follow-up (optional)</span>
                        <input
                            type="date"
                            value={followUpInput}
                            onChange={(e) => setFollowUpInput(e.target.value)}
                        />
                    </label>
                </div>
                <button type="button" className="signal-btn signal-btn-primary" onClick={addOpportunity}>
                    Add opportunity
                </button>
            </div>

            <div className="signal-status">{overallMessage}</div>

            {overdueCount + withBucket.filter((w) => w.bucket === 'soon').length > 0 && (
                <button type="button" className="signal-btn signal-btn-copy" onClick={copyFollowUps}>
                    {copied ? 'Copied ✓' : 'Copy follow-ups'}
                </button>
            )}

            {stageCounts.length > 0 && (
                <div className="signal-section">
                    <div className="signal-section-title">By stage</div>
                    <div className="signal-buckets">
                        {stageCounts.map((s) => (
                            <div key={s.value} className="signal-bucket-row">
                                <span className="signal-bucket-label">{s.label}</span>
                                <div className="signal-bar-track">
                                    <div
                                        className="signal-bar-fill"
                                        style={{ width: `${(s.count / maxStageCount) * 100}%` }}
                                    />
                                </div>
                                <span className="signal-bucket-count">{s.count}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {grouped.map(({ bucket, items }) => (
                <div key={bucket} className="signal-section">
                    <div className="signal-section-title">
                        {BUCKET_LABEL[bucket]} ({items.length})
                    </div>
                    <ul className="signal-list">
                        {items.map(({ opp }) => (
                            <li key={opp.id} className={`signal-item signal-item-${bucket}`}>
                                <div className="signal-item-body">
                                    <div className="signal-item-title">
                                        {opp.company} <span className="signal-item-role">— {opp.role}</span>
                                    </div>
                                    <div className="signal-item-meta">
                                        <span className="signal-tag">{stageLabel(opp.stage)}</span>
                                        {opp.source && <span>{opp.source}</span>}
                                        {opp.followUpAt != null && <span>follow up {formatDate(opp.followUpAt)}</span>}
                                    </div>
                                </div>
                                <div className="signal-item-actions">
                                    <select
                                        className="signal-stage-select"
                                        value={opp.stage}
                                        onChange={(e) => setStage(opp.id, e.target.value as Stage)}
                                    >
                                        {STAGES.map((s) => (
                                            <option key={s.value} value={s.value}>
                                                {s.label}
                                            </option>
                                        ))}
                                    </select>
                                    {NEXT_STAGE[opp.stage] && (
                                        <button
                                            type="button"
                                            className="signal-chip signal-chip-advance"
                                            onClick={() => advanceStage(opp.id)}
                                        >
                                            Advance
                                        </button>
                                    )}
                                    <button
                                        type="button"
                                        className="signal-chip signal-chip-snooze"
                                        onClick={() => snoozeFollowUp(opp.id, 7)}
                                    >
                                        +7d
                                    </button>
                                    <button
                                        type="button"
                                        className="signal-chip signal-chip-remove"
                                        onClick={() => removeOpportunity(opp.id)}
                                        aria-label="Remove opportunity"
                                    >
                                        ✕
                                    </button>
                                </div>
                            </li>
                        ))}
                    </ul>
                </div>
            ))}

            {closed.length > 0 && (
                <div className="signal-section">
                    <div className="signal-section-title">Closed ({closed.length})</div>
                    <ul className="signal-list">
                        {closed.map((opp) => (
                            <li key={opp.id} className="signal-item signal-item-closed">
                                <div className="signal-item-body">
                                    <div className="signal-item-title">
                                        {opp.company} <span className="signal-item-role">— {opp.role}</span>
                                    </div>
                                    <div className="signal-item-meta">
                                        <span className="signal-tag">{stageLabel(opp.stage)}</span>
                                    </div>
                                </div>
                                <button
                                    type="button"
                                    className="signal-chip signal-chip-remove"
                                    onClick={() => removeOpportunity(opp.id)}
                                    aria-label="Remove opportunity"
                                >
                                    ✕
                                </button>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            <div className="signal-tip">
                A pipeline decays without follow-up — the offer usually goes to whoever stayed top of mind, not
                whoever was most qualified. Everything stays on this device — nothing is sent anywhere.
            </div>
        </div>
    );
}

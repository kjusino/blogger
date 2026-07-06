import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './ledger.css';

const KEY = 'personal:ledger:v1';
const MAX_WINS = 1000;
const DAY_MS = 24 * 60 * 60 * 1000;

type Category = 'revenue' | 'efficiency' | 'leadership' | 'technical' | 'growth' | 'other';

type Win = {
    id: string;
    title: string;
    category: Category;
    impact: string;
    detail: string;
    skills: string[];
    createdAt: number;
};

type State = {
    wins: Win[];
};

const CATEGORIES: { value: Category; label: string }[] = [
    { value: 'revenue', label: 'Revenue' },
    { value: 'efficiency', label: 'Efficiency' },
    { value: 'leadership', label: 'Leadership' },
    { value: 'technical', label: 'Technical' },
    { value: 'growth', label: 'Growth' },
    { value: 'other', label: 'Other' },
];

function loadState(): State {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { wins: [] };
        const parsed = JSON.parse(raw);
        return {
            wins: Array.isArray(parsed?.wins) ? parsed.wins.slice(-MAX_WINS) : [],
        };
    } catch {
        return { wins: [] };
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

function quarterLabel(ts: number): string {
    const d = new Date(ts);
    return `${d.getFullYear()} Q${Math.floor(d.getMonth() / 3) + 1}`;
}

function formatDate(ts: number): string {
    return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

function stalenessMessage(daysSinceLast: number | null): string {
    if (daysSinceLast === null) return "⚪ log your first win — do it the day it happens, while the details are sharp";
    if (daysSinceLast === 0) return '🟢 logged today — this is the habit that wins reviews';
    if (daysSinceLast <= 7) return `🟢 last win ${daysSinceLast}d ago — still fresh`;
    if (daysSinceLast <= 14) return `🟡 last win ${daysSinceLast}d ago — log the next one before it fades`;
    return `🔴 last win ${daysSinceLast}d ago — you're almost certainly forgetting things worth claiming credit for`;
}

export default function Ledger() {
    const [state, setState] = useState<State>(() => loadState());
    const [title, setTitle] = useState('');
    const [category, setCategory] = useState<Category>('technical');
    const [impact, setImpact] = useState('');
    const [detail, setDetail] = useState('');
    const [skillsInput, setSkillsInput] = useState('');
    const [copied, setCopied] = useState(false);
    const { wins } = state;

    const addWin = useCallback(() => {
        const trimmed = title.trim();
        if (!trimmed) return;
        const win: Win = {
            id: makeId(),
            title: trimmed,
            category,
            impact: impact.trim(),
            detail: detail.trim(),
            skills: skillsInput
                .split(',')
                .map((s) => s.trim())
                .filter(Boolean),
            createdAt: Date.now(),
        };
        setState((prev) => {
            const next = { wins: [...prev.wins, win].slice(-MAX_WINS) };
            saveState(next);
            return next;
        });
        setTitle('');
        setImpact('');
        setDetail('');
        setSkillsInput('');
    }, [title, category, impact, detail, skillsInput]);

    const removeWin = useCallback((id: string) => {
        setState((prev) => {
            const next = { wins: prev.wins.filter((w) => w.id !== id) };
            saveState(next);
            return next;
        });
    }, []);

    const sorted = useMemo(() => [...wins].sort((a, b) => b.createdAt - a.createdAt), [wins]);

    const daysSinceLast = useMemo(() => {
        if (sorted.length === 0) return null;
        return Math.floor((Date.now() - sorted[0].createdAt) / DAY_MS);
    }, [sorted]);

    const currentQuarter = quarterLabel(Date.now());
    const currentQuarterWins = useMemo(
        () => sorted.filter((w) => quarterLabel(w.createdAt) === currentQuarter),
        [sorted, currentQuarter]
    );

    const categoryCounts = useMemo(
        () =>
            CATEGORIES.map((c) => ({
                ...c,
                count: wins.filter((w) => w.category === c.value).length,
            })).filter((c) => c.count > 0),
        [wins]
    );
    const maxCategoryCount = Math.max(1, ...categoryCounts.map((c) => c.count));

    const grouped = useMemo(() => {
        const map = new Map<string, Win[]>();
        for (const w of sorted) {
            const key = quarterLabel(w.createdAt);
            const list = map.get(key) ?? [];
            list.push(w);
            map.set(key, list);
        }
        return Array.from(map.entries());
    }, [sorted]);

    const copyForReview = useCallback(async () => {
        const source = currentQuarterWins.length > 0 ? currentQuarterWins : sorted.slice(0, 10);
        const text = source
            .map((w) => {
                const bits = [`- ${w.title}`];
                if (w.impact) bits.push(`(${w.impact})`);
                const line1 = bits.join(' ');
                const lines = [line1];
                if (w.detail) lines.push(`  ${w.detail}`);
                if (w.skills.length > 0) lines.push(`  skills: ${w.skills.join(', ')}`);
                return lines.join('\n');
            })
            .join('\n\n');
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            /* clipboard denied — non-fatal */
        }
    }, [currentQuarterWins, sorted]);

    return (
        <div className="ledger">
            <div className="ledger-header">
                <Link to="/personal" className="ledger-back">← personal</Link>
                <span className="ledger-title">Ledger</span>
                <span className="ledger-badge">{wins.length}</span>
            </div>

            <div className="ledger-form">
                <label className="ledger-field">
                    <span>What did you do</span>
                    <input
                        type="text"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        placeholder="e.g. Cut deploy time from 40min to 6min"
                    />
                </label>
                <div className="ledger-form-row">
                    <label className="ledger-field">
                        <span>Category</span>
                        <select value={category} onChange={(e) => setCategory(e.target.value as Category)}>
                            {CATEGORIES.map((c) => (
                                <option key={c.value} value={c.value}>
                                    {c.label}
                                </option>
                            ))}
                        </select>
                    </label>
                    <label className="ledger-field">
                        <span>Quantified impact (optional)</span>
                        <input
                            type="text"
                            value={impact}
                            onChange={(e) => setImpact(e.target.value)}
                            placeholder="$40k saved, 30% faster"
                        />
                    </label>
                </div>
                <label className="ledger-field">
                    <span>Situation / action / result (optional)</span>
                    <textarea
                        value={detail}
                        onChange={(e) => setDetail(e.target.value)}
                        placeholder="What was the problem, what did you do, what changed"
                        rows={2}
                    />
                </label>
                <label className="ledger-field">
                    <span>Skills (comma separated, optional)</span>
                    <input
                        type="text"
                        value={skillsInput}
                        onChange={(e) => setSkillsInput(e.target.value)}
                        placeholder="CI/CD, negotiation, React"
                    />
                </label>
                <button type="button" className="ledger-btn ledger-btn-primary" onClick={addWin}>
                    Log win
                </button>
            </div>

            <div className="ledger-status">{stalenessMessage(daysSinceLast)}</div>

            {wins.length > 0 && (
                <button type="button" className="ledger-btn ledger-btn-copy" onClick={copyForReview}>
                    {copied ? 'Copied ✓' : `Copy ${currentQuarterWins.length > 0 ? currentQuarter : 'recent'} wins for review`}
                </button>
            )}

            {categoryCounts.length > 0 && (
                <div className="ledger-section">
                    <div className="ledger-section-title">By category</div>
                    <div className="ledger-buckets">
                        {categoryCounts.map((c) => (
                            <div key={c.value} className="ledger-bucket-row">
                                <span className="ledger-bucket-label">{c.label}</span>
                                <div className="ledger-bar-track">
                                    <div
                                        className="ledger-bar-fill"
                                        style={{ width: `${(c.count / maxCategoryCount) * 100}%` }}
                                    />
                                </div>
                                <span className="ledger-bucket-count">{c.count}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {grouped.map(([label, groupWins]) => (
                <div key={label} className="ledger-section">
                    <div className="ledger-section-title">
                        {label} ({groupWins.length})
                    </div>
                    <ul className="ledger-list">
                        {groupWins.map((w) => (
                            <li key={w.id} className="ledger-item">
                                <div className="ledger-item-body">
                                    <div className="ledger-item-title">{w.title}</div>
                                    <div className="ledger-item-meta">
                                        <span className="ledger-tag">{CATEGORIES.find((c) => c.value === w.category)?.label}</span>
                                        {w.impact && <span className="ledger-impact">{w.impact}</span>}
                                        <span>{formatDate(w.createdAt)}</span>
                                    </div>
                                    {w.detail && <div className="ledger-item-detail">{w.detail}</div>}
                                    {w.skills.length > 0 && (
                                        <div className="ledger-item-skills">
                                            {w.skills.map((s) => (
                                                <span key={s} className="ledger-skill-chip">
                                                    {s}
                                                </span>
                                            ))}
                                        </div>
                                    )}
                                </div>
                                <button
                                    type="button"
                                    className="ledger-chip ledger-chip-remove"
                                    onClick={() => removeWin(w.id)}
                                    aria-label="Delete win"
                                >
                                    ✕
                                </button>
                            </li>
                        ))}
                    </ul>
                </div>
            ))}

            <div className="ledger-tip">
                Log wins the day they happen — memory of impact decays fast, and self-reviews, promo packets,
                and interview answers all run on specifics you'll otherwise lose. Everything stays on this
                device — nothing is sent anywhere.
            </div>
        </div>
    );
}

import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './ascent.css';

const KEY = 'personal:ascent:v1';
const MAX_SKILLS = 200;
const MIN_LEVEL = 1;
const MAX_LEVEL = 5;

const LEVEL_LABEL: Record<number, string> = {
    1: 'Novice',
    2: 'Developing',
    3: 'Competent',
    4: 'Proficient',
    5: 'Expert',
};

type Skill = {
    id: string;
    name: string;
    target: number;
    current: number;
    note: string;
    createdAt: number;
};

type State = {
    role: string;
    skills: Skill[];
};

function loadState(): State {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { role: '', skills: [] };
        const parsed = JSON.parse(raw);
        return {
            role: typeof parsed?.role === 'string' ? parsed.role : '',
            skills: Array.isArray(parsed?.skills) ? parsed.skills.slice(-MAX_SKILLS) : [],
        };
    } catch {
        return { role: '', skills: [] };
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

function clampLevel(n: number): number {
    return Math.min(MAX_LEVEL, Math.max(MIN_LEVEL, n));
}

type Bucket = 'gap' | 'near' | 'met';

function bucketFor(skill: Skill): Bucket {
    const gap = skill.target - skill.current;
    if (gap >= 2) return 'gap';
    if (gap === 1) return 'near';
    return 'met';
}

const BUCKET_LABEL: Record<Bucket, string> = {
    gap: 'Biggest gaps',
    near: 'Almost there',
    met: 'Bar cleared',
};

export default function Ascent() {
    const [state, setState] = useState<State>(() => loadState());
    const [roleInput, setRoleInput] = useState(state.role);
    const [name, setName] = useState('');
    const [target, setTarget] = useState(3);
    const [current, setCurrent] = useState(1);
    const [note, setNote] = useState('');
    const [copied, setCopied] = useState(false);
    const { skills } = state;

    const saveRole = useCallback(() => {
        setState((prev) => {
            const next = { ...prev, role: roleInput.trim() };
            saveState(next);
            return next;
        });
    }, [roleInput]);

    const addSkill = useCallback(() => {
        const trimmed = name.trim();
        if (!trimmed) return;
        const skill: Skill = {
            id: makeId(),
            name: trimmed,
            target: clampLevel(target),
            current: clampLevel(current),
            note: note.trim(),
            createdAt: Date.now(),
        };
        setState((prev) => {
            const next = { ...prev, skills: [...prev.skills, skill].slice(-MAX_SKILLS) };
            saveState(next);
            return next;
        });
        setName('');
        setNote('');
        setTarget(3);
        setCurrent(1);
    }, [name, target, current, note]);

    const bumpCurrent = useCallback((id: string) => {
        setState((prev) => {
            const next = {
                ...prev,
                skills: prev.skills.map((s) =>
                    s.id === id ? { ...s, current: clampLevel(s.current + 1) } : s
                ),
            };
            saveState(next);
            return next;
        });
    }, []);

    const removeSkill = useCallback((id: string) => {
        setState((prev) => {
            const next = { ...prev, skills: prev.skills.filter((s) => s.id !== id) };
            saveState(next);
            return next;
        });
    }, []);

    const withRatio = useMemo(
        () =>
            skills.map((s) => ({
                skill: s,
                ratio: Math.min(1, s.current / s.target),
                bucket: bucketFor(s),
            })),
        [skills]
    );

    const sorted = useMemo(
        () => [...withRatio].sort((a, b) => b.skill.target - b.skill.current - (a.skill.target - a.skill.current)),
        [withRatio]
    );

    const readiness = useMemo(() => {
        if (withRatio.length === 0) return 0;
        const sum = withRatio.reduce((acc, w) => acc + w.ratio, 0);
        return Math.round((sum / withRatio.length) * 100);
    }, [withRatio]);

    const gapCount = useMemo(() => withRatio.filter((w) => w.bucket === 'gap').length, [withRatio]);

    const overallMessage = useMemo(() => {
        if (skills.length === 0) return '⚪ list what the role actually requires — that list is the plan';
        if (readiness >= 90) return `🟢 ${readiness}% ready — you're closer than the doubt says`;
        if (gapCount === 0) return `🟡 ${readiness}% ready — no wide gaps, just polish left`;
        return `🔴 ${readiness}% ready — ${gapCount} skill${gapCount === 1 ? '' : 's'} with real gaps`;
    }, [skills.length, readiness, gapCount]);

    const grouped = useMemo(() => {
        const order: Bucket[] = ['gap', 'near', 'met'];
        return order
            .map((b) => ({ bucket: b, items: sorted.filter((w) => w.bucket === b) }))
            .filter((g) => g.items.length > 0);
    }, [sorted]);

    const copyPriorityList = useCallback(async () => {
        const source = sorted.filter((w) => w.bucket !== 'met').slice(0, 10);
        const text = source
            .map((w) => {
                const bits = [
                    `- ${w.skill.name}: ${LEVEL_LABEL[w.skill.current]} → ${LEVEL_LABEL[w.skill.target]}`,
                ];
                if (w.skill.note) bits.push(`(${w.skill.note})`);
                return bits.join(' ');
            })
            .join('\n');
        try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        } catch {
            /* clipboard denied — non-fatal */
        }
    }, [sorted]);

    return (
        <div className="ascent">
            <div className="ascent-header">
                <Link to="/personal" className="ascent-back">← personal</Link>
                <span className="ascent-title">Ascent</span>
                <span className="ascent-badge">{skills.length}</span>
            </div>

            <div className="ascent-role">
                <label className="ascent-field">
                    <span>Target role</span>
                    <input
                        type="text"
                        value={roleInput}
                        onChange={(e) => setRoleInput(e.target.value)}
                        onBlur={saveRole}
                        placeholder="e.g. Staff Engineer"
                    />
                </label>
            </div>

            <div className="ascent-form">
                <label className="ascent-field">
                    <span>Skill or competency</span>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="e.g. Systems design"
                    />
                </label>
                <div className="ascent-form-row">
                    <label className="ascent-field">
                        <span>Where you are</span>
                        <select value={current} onChange={(e) => setCurrent(Number(e.target.value))}>
                            {[1, 2, 3, 4, 5].map((lvl) => (
                                <option key={lvl} value={lvl}>
                                    {lvl} — {LEVEL_LABEL[lvl]}
                                </option>
                            ))}
                        </select>
                    </label>
                    <label className="ascent-field">
                        <span>Where the role needs you</span>
                        <select value={target} onChange={(e) => setTarget(Number(e.target.value))}>
                            {[1, 2, 3, 4, 5].map((lvl) => (
                                <option key={lvl} value={lvl}>
                                    {lvl} — {LEVEL_LABEL[lvl]}
                                </option>
                            ))}
                        </select>
                    </label>
                </div>
                <label className="ascent-field">
                    <span>What closes the gap (optional)</span>
                    <input
                        type="text"
                        value={note}
                        onChange={(e) => setNote(e.target.value)}
                        placeholder="lead the next migration RFC"
                    />
                </label>
                <button type="button" className="ascent-btn ascent-btn-primary" onClick={addSkill}>
                    Add skill
                </button>
            </div>

            <div className="ascent-status">{overallMessage}</div>

            {gapCount + withRatio.filter((w) => w.bucket === 'near').length > 0 && (
                <button type="button" className="ascent-btn ascent-btn-copy" onClick={copyPriorityList}>
                    {copied ? 'Copied ✓' : 'Copy priority list'}
                </button>
            )}

            {sorted.length > 0 && (
                <div className="ascent-section">
                    <div className="ascent-section-title">By skill</div>
                    <div className="ascent-bars">
                        {sorted.map((w) => (
                            <div key={w.skill.id} className="ascent-bar-row">
                                <span className="ascent-bar-label">{w.skill.name}</span>
                                <div className="ascent-bar-track">
                                    <div
                                        className="ascent-bar-fill"
                                        style={{ width: `${w.ratio * 100}%` }}
                                    />
                                </div>
                                <span className="ascent-bar-count">
                                    {w.skill.current}/{w.skill.target}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {grouped.map(({ bucket, items }) => (
                <div key={bucket} className="ascent-section">
                    <div className="ascent-section-title">
                        {BUCKET_LABEL[bucket]} ({items.length})
                    </div>
                    <ul className="ascent-list">
                        {items.map(({ skill }) => (
                            <li key={skill.id} className={`ascent-item ascent-item-${bucketFor(skill)}`}>
                                <div className="ascent-item-body">
                                    <div className="ascent-item-title">{skill.name}</div>
                                    <div className="ascent-item-meta">
                                        <span className="ascent-tag">
                                            {LEVEL_LABEL[skill.current]} → {LEVEL_LABEL[skill.target]}
                                        </span>
                                    </div>
                                    {skill.note && <div className="ascent-item-note">{skill.note}</div>}
                                </div>
                                <div className="ascent-item-actions">
                                    {skill.current < skill.target && (
                                        <button
                                            type="button"
                                            className="ascent-chip ascent-chip-bump"
                                            onClick={() => bumpCurrent(skill.id)}
                                        >
                                            Grew +1
                                        </button>
                                    )}
                                    <button
                                        type="button"
                                        className="ascent-chip ascent-chip-remove"
                                        onClick={() => removeSkill(skill.id)}
                                        aria-label="Remove skill"
                                    >
                                        ✕
                                    </button>
                                </div>
                            </li>
                        ))}
                    </ul>
                </div>
            ))}

            <div className="ascent-tip">
                Vague ambition stalls; a named gap gets closed. Everything stays on this device — nothing is
                sent anywhere.
            </div>
        </div>
    );
}

import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import './forge.css';

const KEY = 'personal:forge:v1';
const DAY_MS = 24 * 60 * 60 * 1000;
const MAX_SKILLS = 200;
const COMPETENCE_HOURS = 100; // rough proficiency benchmark, not the 10,000-hour myth

type Category = 'technical' | 'creative' | 'physical' | 'domain' | 'other';
type Grade = 'again' | 'hard' | 'good' | 'easy';

type Skill = {
    id: string;
    name: string;
    category: Category;
    createdAt: number;
    totalMinutes: number;
    sessionCount: number;
    intervalDays: number;
    easeFactor: number;
    nextReviewAt: number;
    lastPracticedAt: number | null;
    streak: number;
};

type State = {
    skills: Skill[];
};

const CATEGORIES: { value: Category; label: string }[] = [
    { value: 'technical', label: 'Technical' },
    { value: 'creative', label: 'Creative' },
    { value: 'physical', label: 'Physical' },
    { value: 'domain', label: 'Domain' },
    { value: 'other', label: 'Other' },
];

const GRADES: { value: Grade; label: string }[] = [
    { value: 'again', label: 'Struggled' },
    { value: 'hard', label: 'Hard' },
    { value: 'good', label: 'Good' },
    { value: 'easy', label: 'Easy' },
];

function loadState(): State {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { skills: [] };
        const parsed = JSON.parse(raw);
        return { skills: Array.isArray(parsed?.skills) ? parsed.skills.slice(-MAX_SKILLS) : [] };
    } catch {
        return { skills: [] };
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

function clamp(n: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, n));
}

// Simplified SM-2: grade drives interval growth and ease-factor drift.
// "again" resets the skill to daily practice instead of failing it outright —
// the goal is skill retention, not flashcard memorization.
function applyGrade(skill: Skill, grade: Grade, minutes: number, now: number): Skill {
    let easeFactor = skill.easeFactor;
    let intervalDays: number;
    let streak = skill.streak;

    if (grade === 'again') {
        intervalDays = 1;
        easeFactor = clamp(easeFactor - 0.2, 1.3, 3.0);
        streak = 0;
    } else {
        const delta = grade === 'hard' ? -0.15 : grade === 'easy' ? 0.15 : 0;
        easeFactor = clamp(easeFactor + delta, 1.3, 3.0);
        if (skill.sessionCount === 0 || skill.intervalDays <= 1) {
            intervalDays = grade === 'hard' ? 2 : grade === 'good' ? 3 : 5;
        } else {
            const multiplier = grade === 'hard' ? 1.2 : grade === 'easy' ? easeFactor * 1.3 : easeFactor;
            intervalDays = Math.max(1, Math.round(skill.intervalDays * multiplier));
        }
        streak += 1;
    }

    return {
        ...skill,
        easeFactor,
        intervalDays,
        streak,
        totalMinutes: skill.totalMinutes + minutes,
        sessionCount: skill.sessionCount + 1,
        lastPracticedAt: now,
        nextReviewAt: now + intervalDays * DAY_MS,
    };
}

function formatDue(ts: number, now: number): string {
    const diffDays = Math.floor((ts - now) / DAY_MS);
    if (diffDays < 0) return `${Math.abs(diffDays)}d overdue`;
    if (diffDays === 0) return 'due today';
    if (diffDays === 1) return 'due tomorrow';
    return `due in ${diffDays}d`;
}

function queueStatus(dueCount: number, totalSkills: number): string {
    if (totalSkills === 0) return '⚪ add a skill you want to keep sharp';
    if (dueCount === 0) return '🟢 all caught up — nothing due';
    if (dueCount <= 2) return `🟡 ${dueCount} skill${dueCount === 1 ? '' : 's'} due — quick session keeps it warm`;
    return `🔴 ${dueCount} skills overdue — practice is falling behind the schedule`;
}

export default function Forge() {
    const [state, setState] = useState<State>(() => loadState());
    const [name, setName] = useState('');
    const [category, setCategory] = useState<Category>('technical');
    const [minutesDraft, setMinutesDraft] = useState<Record<string, string>>({});
    const now = Date.now();
    const { skills } = state;

    const addSkill = useCallback(() => {
        const trimmed = name.trim();
        if (!trimmed) return;
        const skill: Skill = {
            id: makeId(),
            name: trimmed,
            category,
            createdAt: now,
            totalMinutes: 0,
            sessionCount: 0,
            intervalDays: 1,
            easeFactor: 2.5,
            nextReviewAt: now,
            lastPracticedAt: null,
            streak: 0,
        };
        setState((prev) => {
            const next = { skills: [...prev.skills, skill].slice(-MAX_SKILLS) };
            saveState(next);
            return next;
        });
        setName('');
    }, [name, category, now]);

    const logSession = useCallback(
        (id: string, grade: Grade) => {
            const minutes = Math.max(1, Number(minutesDraft[id]) || 15);
            setState((prev) => {
                const next = {
                    skills: prev.skills.map((s) => (s.id === id ? applyGrade(s, grade, minutes, now) : s)),
                };
                saveState(next);
                return next;
            });
            setMinutesDraft((prev) => ({ ...prev, [id]: '' }));
        },
        [minutesDraft, now]
    );

    const removeSkill = useCallback((id: string) => {
        setState((prev) => {
            const next = { skills: prev.skills.filter((s) => s.id !== id) };
            saveState(next);
            return next;
        });
    }, []);

    const due = useMemo(
        () => skills.filter((s) => s.nextReviewAt <= now).sort((a, b) => a.nextReviewAt - b.nextReviewAt),
        [skills, now]
    );
    const upcoming = useMemo(
        () => skills.filter((s) => s.nextReviewAt > now).sort((a, b) => a.nextReviewAt - b.nextReviewAt),
        [skills, now]
    );

    const totalHours = useMemo(() => skills.reduce((sum, s) => sum + s.totalMinutes, 0) / 60, [skills]);

    return (
        <div className="forge">
            <div className="forge-header">
                <Link to="/personal" className="forge-back">← personal</Link>
                <span className="forge-title">Forge</span>
                <span className="forge-badge">{due.length > 0 ? `${due.length} due` : '—'}</span>
            </div>

            <div className="forge-form">
                <label className="forge-field">
                    <span>Skill</span>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="e.g. Public speaking, SQL query tuning, Spanish subjunctive"
                    />
                </label>
                <div className="forge-form-row">
                    <label className="forge-field">
                        <span>Category</span>
                        <select value={category} onChange={(e) => setCategory(e.target.value as Category)}>
                            {CATEGORIES.map((c) => (
                                <option key={c.value} value={c.value}>
                                    {c.label}
                                </option>
                            ))}
                        </select>
                    </label>
                    <button type="button" className="forge-btn forge-btn-primary" onClick={addSkill}>
                        Add skill
                    </button>
                </div>
            </div>

            <div className="forge-status">{queueStatus(due.length, skills.length)}</div>

            {due.length > 0 && (
                <div className="forge-section">
                    <div className="forge-section-title">Due for practice ({due.length})</div>
                    <ul className="forge-list">
                        {due.map((s) => (
                            <li key={s.id} className="forge-item forge-item-due">
                                <div className="forge-item-body">
                                    <div className="forge-item-name">{s.name}</div>
                                    <div className="forge-item-meta">
                                        <span className="forge-tag">{s.category}</span>
                                        <span>{formatDue(s.nextReviewAt, now)}</span>
                                        {s.streak > 0 && <span>🔥 {s.streak} streak</span>}
                                    </div>
                                </div>
                                <div className="forge-item-log">
                                    <input
                                        type="number"
                                        inputMode="numeric"
                                        className="forge-minutes"
                                        placeholder="15"
                                        value={minutesDraft[s.id] ?? ''}
                                        onChange={(e) =>
                                            setMinutesDraft((prev) => ({ ...prev, [s.id]: e.target.value }))
                                        }
                                    />
                                    <span className="forge-minutes-label">min</span>
                                    <div className="forge-grades">
                                        {GRADES.map((g) => (
                                            <button
                                                key={g.value}
                                                type="button"
                                                className={`forge-chip forge-chip-${g.value}`}
                                                onClick={() => logSession(s.id, g.value)}
                                            >
                                                {g.label}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {upcoming.length > 0 && (
                <div className="forge-section">
                    <div className="forge-section-title">On schedule ({upcoming.length})</div>
                    <ul className="forge-list">
                        {upcoming.map((s) => {
                            const hours = s.totalMinutes / 60;
                            const pct = clamp((hours / COMPETENCE_HOURS) * 100, 0, 100);
                            return (
                                <li key={s.id} className="forge-item">
                                    <div className="forge-item-body">
                                        <div className="forge-item-name">{s.name}</div>
                                        <div className="forge-item-meta">
                                            <span className="forge-tag">{s.category}</span>
                                            <span>{formatDue(s.nextReviewAt, now)}</span>
                                            <span>{hours.toFixed(1)}h logged</span>
                                        </div>
                                        <div className="forge-progress-track">
                                            <div className="forge-progress-fill" style={{ width: `${pct}%` }} />
                                        </div>
                                    </div>
                                    <button
                                        type="button"
                                        className="forge-chip forge-chip-remove"
                                        onClick={() => removeSkill(s.id)}
                                        aria-label={`Delete ${s.name}`}
                                    >
                                        ✕
                                    </button>
                                </li>
                            );
                        })}
                    </ul>
                </div>
            )}

            {skills.length > 0 && (
                <div className="forge-summary">
                    {totalHours.toFixed(1)}h total deliberate practice logged across {skills.length} skill
                    {skills.length === 1 ? '' : 's'}.
                </div>
            )}

            <div className="forge-tip">
                each "Good" push roughly doubles the gap until the next review, like spaced repetition for
                flashcards — but applied to skills instead. "Struggled" resets the interval to daily so the
                skill doesn't quietly decay. everything stays on this device — nothing is sent anywhere.
            </div>
        </div>
    );
}

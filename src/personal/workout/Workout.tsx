import {
    useState,
    useEffect,
    useCallback,
    useRef,
    FormEvent,
} from 'react';
import { Link } from 'react-router-dom';
import { PROGRAMS } from './programs';
import { FORM_VIDEOS } from './formVideos';
import {
    loadData,
    saveData,
    getExerciseKey,
    fetchHistory,
    syncRows,
    groupRowsBySession,
    lastTimeFor,
    parseLogDate,
    daysSince,
    countSessionsInLastDays,
    WorkoutRow,
    Session,
} from './api';
import {
    WorkoutData,
    Block,
    Pair,
    Program,
    ProgramKey,
    BlockType,
} from './types';
import './workout.css';

const REST_BY_TYPE: Record<BlockType, number> = {
    compound: 90,
    isolation: 45,
    core: 30,
};

function FormLink({ exercise }: { exercise: string }) {
    const url = FORM_VIDEOS[exercise];
    if (!url) return null;
    return (
        <a
            className="wj-form-badge"
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            title={`Form: ${exercise}`}
            aria-label={`Form guide for ${exercise}`}
        >
            ▶
        </a>
    );
}

function FormGuides({ exercises }: { exercises: string[] }) {
    const [open, setOpen] = useState(false);
    const unique = Array.from(new Set(exercises)).filter((e) => FORM_VIDEOS[e]);
    if (unique.length === 0) return null;
    return (
        <div>
            <button
                type="button"
                className="wj-form-toggle"
                onClick={() => setOpen((v) => !v)}
                aria-expanded={open}
            >
                <span>📖 Form Guides ({unique.length})</span>
                <span>{open ? '▾' : '▸'}</span>
            </button>
            {open && (
                <div className="wj-form-list">
                    {unique.map((ex) => (
                        <a
                            key={ex}
                            className="wj-form-link"
                            href={FORM_VIDEOS[ex]}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            <span className="wj-form-badge">▶</span>
                            <span>{ex}</span>
                        </a>
                    ))}
                </div>
            )}
        </div>
    );
}

type RestTimerProps = { seconds: number; onDone: () => void };

function RestTimer({ seconds, onDone }: RestTimerProps) {
    const [left, setLeft] = useState(seconds);
    const onDoneRef = useRef(onDone);
    onDoneRef.current = onDone;

    useEffect(() => {
        setLeft(seconds);
        const iv = setInterval(() => {
            setLeft((l) => {
                if (l <= 1) {
                    clearInterval(iv);
                    onDoneRef.current();
                    return 0;
                }
                return l - 1;
            });
        }, 1000);
        return () => clearInterval(iv);
    }, [seconds]);

    const pct = ((seconds - left) / seconds) * 100;
    const mins = Math.floor(Math.max(0, left) / 60);
    const secs = Math.max(0, left) % 60;
    const isLow = left <= 5;

    return (
        <div className={`wj-rest-timer${isLow ? ' is-low' : ''}`}>
            <div className="wj-rest-value">
                {mins}:{secs.toString().padStart(2, '0')}
            </div>
            <div className="wj-rest-bar">
                <div
                    className="wj-rest-bar-fill"
                    style={{ width: `${pct}%` }}
                />
            </div>
            <button
                type="button"
                className="wj-rest-skip"
                onClick={() => onDoneRef.current()}
            >
                SKIP
            </button>
        </div>
    );
}

type WeightInputProps = {
    exercise: string;
    setIdx: number;
    data: WorkoutData;
    onChange: (key: string, setIdx: number, value: string) => void;
};

function WeightInput({ exercise, setIdx, data, onChange }: WeightInputProps) {
    const key = getExerciseKey(exercise);
    const entry = data[key];
    const val = entry?.sets?.[setIdx]?.weight ?? '';
    const pr = entry?.pr ?? 0;
    const isPR = val !== '' && pr > 0 && Number(val) >= pr;

    return (
        <div className="wj-weight">
            <input
                type="number"
                inputMode="decimal"
                placeholder="lbs"
                value={val}
                aria-label={`${exercise} set ${setIdx + 1} weight`}
                className={`wj-weight-input${isPR ? ' is-pr' : ''}`}
                onChange={(e: FormEvent<HTMLInputElement>) =>
                    onChange(key, setIdx, e.currentTarget.value)
                }
            />
            {isPR && <span className="wj-pr-badge">PR!</span>}
        </div>
    );
}

function LastTimeHint({
    rows,
    exerciseKey,
}: {
    rows: WorkoutRow[];
    exerciseKey: string;
}) {
    const last = lastTimeFor(rows, exerciseKey);
    if (!last) return <div className="wj-last-time wj-last-time--empty">First time</div>;
    return (
        <div className="wj-last-time">
            Last: <strong>{last.max}</strong> · {last.date}
        </div>
    );
}

type SupersetBlockProps = {
    block: Block;
    data: WorkoutData;
    rows: WorkoutRow[];
    onWeightChange: (key: string, setIdx: number, value: string) => void;
};

function SupersetBlock({ block, data, rows, onWeightChange }: SupersetBlockProps) {
    const [activeTimer, setActiveTimer] = useState<{
        pairIdx: number;
        setIdx: number;
    } | null>(null);
    const [done, setDone] = useState<Record<string, boolean>>({});
    const type = block.pairs[0]?.type ?? 'compound';
    const restSeconds = REST_BY_TYPE[type];

    function toggleSet(pairIdx: number, setIdx: number) {
        const k = `${pairIdx}-${setIdx}`;
        setDone((prev) => {
            const next = { ...prev };
            if (next[k]) {
                delete next[k];
            } else {
                next[k] = true;
                setActiveTimer({ pairIdx, setIdx });
            }
            return next;
        });
    }

    const allExercises = block.pairs.flatMap((p) => [p.a, p.b]);

    return (
        <div className={`wj-block wj-block--${type}`}>
            <div className="wj-block-tag">{block.name}</div>
            <p className="wj-block-note">{block.note}</p>

            {block.pairs.map((pair, pi) => (
                <PairRows
                    key={pi}
                    pair={pair}
                    pairIdx={pi}
                    data={data}
                    rows={rows}
                    onWeightChange={onWeightChange}
                    done={done}
                    onToggleSet={toggleSet}
                    activeTimerPairIdx={
                        activeTimer?.pairIdx === pi ? pi : undefined
                    }
                    restSeconds={restSeconds}
                    onTimerDone={() => setActiveTimer(null)}
                />
            ))}

            <FormGuides exercises={allExercises} />
        </div>
    );
}

type PairRowsProps = {
    pair: Pair;
    pairIdx: number;
    data: WorkoutData;
    rows: WorkoutRow[];
    onWeightChange: (key: string, setIdx: number, value: string) => void;
    done: Record<string, boolean>;
    onToggleSet: (pi: number, si: number) => void;
    activeTimerPairIdx: number | undefined;
    restSeconds: number;
    onTimerDone: () => void;
};

function PairRows({
    pair,
    pairIdx,
    data,
    rows,
    onWeightChange,
    done,
    onToggleSet,
    activeTimerPairIdx,
    restSeconds,
    onTimerDone,
}: PairRowsProps) {
    const keyA = getExerciseKey(pair.a);
    const keyB = getExerciseKey(pair.b);
    const prA = data[keyA]?.pr ?? 0;
    const prB = data[keyB]?.pr ?? 0;

    return (
        <>
            <div className="wj-pair-headers">
                <div />
                <div className="wj-pair-header">
                    <span className="wj-pair-letter">A</span>
                    <span>{pair.a}</span>
                    <FormLink exercise={pair.a} />
                    <span className="wj-pair-reps">×{pair.repsA}</span>
                    {prA > 0 && (
                        <span className="wj-pair-pr">PR {prA}</span>
                    )}
                </div>
                <div className="wj-pair-header">
                    <span className="wj-pair-letter">B</span>
                    <span>{pair.b}</span>
                    <FormLink exercise={pair.b} />
                    <span className="wj-pair-reps">×{pair.repsB}</span>
                    {prB > 0 && (
                        <span className="wj-pair-pr">PR {prB}</span>
                    )}
                </div>
            </div>

            <div className="wj-last-row">
                <div />
                <LastTimeHint rows={rows} exerciseKey={keyA} />
                <LastTimeHint rows={rows} exerciseKey={keyB} />
            </div>

            {Array.from({ length: pair.sets }).map((_, si) => {
                const isDone = !!done[`${pairIdx}-${si}`];
                return (
                    <div
                        key={si}
                        className={`wj-set-row${isDone ? ' is-done' : ''}`}
                    >
                        <button
                            type="button"
                            className={`wj-set-check${
                                isDone ? ' is-done' : ''
                            }`}
                            onClick={() => onToggleSet(pairIdx, si)}
                            aria-label={`Toggle set ${si + 1}`}
                        >
                            {isDone ? '✓' : si + 1}
                        </button>
                        <WeightInput
                            exercise={pair.a}
                            setIdx={si}
                            data={data}
                            onChange={onWeightChange}
                        />
                        <WeightInput
                            exercise={pair.b}
                            setIdx={si}
                            data={data}
                            onChange={onWeightChange}
                        />
                    </div>
                );
            })}

            {activeTimerPairIdx === pairIdx && (
                <RestTimer seconds={restSeconds} onDone={onTimerDone} />
            )}
        </>
    );
}

function Dashboard({ sessions }: { sessions: Session[] }) {
    const last7 = countSessionsInLastDays(sessions, 7);
    const last30 = countSessionsInLastDays(sessions, 30);
    const lastSession = sessions[0];
    const sinceLast = lastSession ? daysSince(lastSession.date) : null;

    let coachLine: string;
    if (sessions.length === 0) {
        coachLine = 'Day 1 starts whenever you want.';
    } else if (sinceLast !== null && sinceLast >= 7) {
        coachLine = 'Welcome back. Show up today and the rest follows.';
    } else if (last7 >= 4) {
        coachLine = 'Strong week. Recovery matters too — listen to your body.';
    } else if (last7 >= 3) {
        coachLine = "You're locked in. Keep going.";
    } else if (last7 >= 1) {
        coachLine = "You're on the board this week. One more.";
    } else {
        coachLine = "New week, fresh page. Let's get one in.";
    }

    return (
        <div className="wj-dashboard">
            <div className="wj-eyebrow">Workout Journal</div>
            <h1 className="wj-h1">{coachLine}</h1>
            <div className="wj-stats">
                <div className="wj-stat">
                    <div className="wj-stat-num">{last7}</div>
                    <div className="wj-stat-label">this week</div>
                </div>
                <div className="wj-stat">
                    <div className="wj-stat-num">{last30}</div>
                    <div className="wj-stat-label">last 30 days</div>
                </div>
                <div className="wj-stat">
                    <div className="wj-stat-num">
                        {sinceLast === null
                            ? '—'
                            : sinceLast === 0
                              ? 'today'
                              : sinceLast === 1
                                ? '1d'
                                : `${sinceLast}d`}
                    </div>
                    <div className="wj-stat-label">since last</div>
                </div>
            </div>
            {lastSession && (
                <div className="wj-last-session-strip">
                    Last lift:{' '}
                    <strong>
                        {(lastSession.program || '').toUpperCase()}
                    </strong>{' '}
                    · {lastSession.date} ·{' '}
                    {lastSession.exercises.length} exercises
                </div>
            )}
        </div>
    );
}

function Celebration({
    celebration,
    onDismiss,
}: {
    celebration: NonNullable<Celebration>;
    onDismiss: () => void;
}) {
    return (
        <div className="wj-celebration" role="status">
            <div className="wj-celebration-title">
                {celebration.newPRs.length === 1
                    ? '🎉 New PR'
                    : `🎉 ${celebration.newPRs.length} new PRs`}
            </div>
            <ul className="wj-celebration-list">
                {celebration.newPRs.map((p) => (
                    <li key={p.name}>
                        <strong>{p.name}</strong> · {p.weight} lb
                        {p.delta > 0 && (
                            <span className="wj-celebration-delta">
                                {' '}
                                (+{p.delta})
                            </span>
                        )}
                    </li>
                ))}
            </ul>
            <button
                type="button"
                className="wj-celebration-dismiss"
                onClick={onDismiss}
            >
                Nice
            </button>
        </div>
    );
}

function TrainingLog({
    data,
    rows,
}: {
    data: WorkoutData;
    rows: WorkoutRow[];
}) {
    const [tab, setTab] = useState<'sessions' | 'prs'>('sessions');
    const sessions = groupRowsBySession(rows);

    if (sessions.length === 0) {
        return (
            <div className="wj-history-empty">
                No workouts yet. Finish your first session and it'll show up here.
            </div>
        );
    }

    return (
        <div>
            <div className="wj-tabs" role="tablist">
                <button
                    type="button"
                    role="tab"
                    aria-selected={tab === 'sessions'}
                    className={`wj-tab${tab === 'sessions' ? ' is-active' : ''}`}
                    onClick={() => setTab('sessions')}
                >
                    Sessions
                </button>
                <button
                    type="button"
                    role="tab"
                    aria-selected={tab === 'prs'}
                    className={`wj-tab${tab === 'prs' ? ' is-active' : ''}`}
                    onClick={() => setTab('prs')}
                >
                    PRs
                </button>
            </div>
            {tab === 'sessions' ? (
                <SessionsList sessions={sessions} data={data} />
            ) : (
                <PRList data={data} />
            )}
        </div>
    );
}

function SessionsList({
    sessions,
    data,
}: {
    sessions: Session[];
    data: WorkoutData;
}) {
    return (
        <div className="wj-sessions">
            {sessions.map((s, idx) => (
                <div key={idx} className="wj-session-card">
                    <div className="wj-session-card-head">
                        <span className="wj-session-card-program">
                            {(s.program || '').toUpperCase()}
                        </span>
                        <span className="wj-session-card-date">{s.date}</span>
                    </div>
                    <ul className="wj-session-card-exercises">
                        {s.exercises.map((ex) => {
                            const overallPR = data[ex.key]?.pr ?? 0;
                            return (
                                <li
                                    key={ex.key}
                                    className="wj-session-card-exercise"
                                >
                                    <div className="wj-session-exercise-name">
                                        {ex.name}
                                    </div>
                                    <div className="wj-session-exercise-sets">
                                        {ex.sets.map((w, i) => {
                                            const isPR =
                                                overallPR > 0 &&
                                                w >= overallPR &&
                                                w === ex.max;
                                            return (
                                                <span
                                                    key={i}
                                                    className={`wj-session-set${isPR ? ' is-pr' : ''}`}
                                                >
                                                    {w}
                                                    {isPR && ' ★'}
                                                </span>
                                            );
                                        })}
                                    </div>
                                </li>
                            );
                        })}
                    </ul>
                </div>
            ))}
        </div>
    );
}

function PRList({ data }: { data: WorkoutData }) {
    const exercises = Object.entries(data)
        .filter(([, v]) => v.history && v.history.length > 0 && v.pr > 0)
        .sort((a, b) => (b[1].pr || 0) - (a[1].pr || 0));

    if (exercises.length === 0) {
        return (
            <div className="wj-history-empty">No PRs logged yet.</div>
        );
    }

    return (
        <div className="wj-prs">
            {exercises.map(([key, val]) => (
                <div key={key} className="wj-pr-row">
                    <div className="wj-pr-name">
                        {val.name || key}
                        <FormLink exercise={val.name || ''} />
                    </div>
                    <div className="wj-pr-value">{val.pr} lb</div>
                </div>
            ))}
        </div>
    );
}

/**
 * Suggest the program key the user has done LEAST recently. If they've never
 * done any, suggest push. Helps drive habit-building rotation.
 */
function isSuggestedNextSplit(
    key: ProgramKey,
    sessions: Session[]
): boolean {
    const programs: ProgramKey[] = ['push', 'pull', 'legs'];
    if (sessions.length === 0) return key === 'push';
    const lastByProgram = new Map<string, number>();
    sessions.forEach((s, idx) => {
        if (!lastByProgram.has(s.program)) lastByProgram.set(s.program, idx);
    });
    let stalest: ProgramKey = 'push';
    let stalestIdx = -1;
    for (const p of programs) {
        // Programs never done → infinite staleness → win immediately
        const idx = lastByProgram.has(p)
            ? (lastByProgram.get(p) as number)
            : Number.POSITIVE_INFINITY;
        if (idx > stalestIdx) {
            stalestIdx = idx;
            stalest = p;
        }
    }
    return key === stalest;
}

type View = 'picker' | 'session' | 'history';
type Celebration = {
    newPRs: { name: string; weight: number; delta: number }[];
} | null;

export default function Workout() {
    const [view, setView] = useState<View>('picker');
    const [day, setDay] = useState<ProgramKey | null>(null);
    const [data, setData] = useState<WorkoutData>({});
    const [rows, setRows] = useState<WorkoutRow[]>([]);
    const [sessionStart, setSessionStart] = useState<number | null>(null);
    const [elapsed, setElapsed] = useState(0);
    const [loading, setLoading] = useState(true);
    const [syncStatus, setSyncStatus] = useState<
        { state: 'syncing' } | { state: 'ok' } | { state: 'err'; error: string } | null
    >(null);
    const [celebration, setCelebration] = useState<Celebration>(null);

    useEffect(() => {
        loadData().then((local) => {
            setData(local);
            setLoading(false);
        });
        fetchHistory().then((cloud) => {
            if (!cloud) return;
            setData((prev) => mergeCloudIntoLocal(prev, cloud.data));
            setRows(cloud.rows);
        });
    }, []);

    useEffect(() => {
        if (!sessionStart) return;
        const iv = setInterval(() => {
            setElapsed(Math.floor((Date.now() - sessionStart) / 1000));
        }, 1000);
        return () => clearInterval(iv);
    }, [sessionStart]);

    const handleWeightChange = useCallback(
        (key: string, setIdx: number, value: string) => {
            setData((prev) => {
                const next: WorkoutData = { ...prev };
                const entry = next[key]
                    ? { ...next[key] }
                    : { pr: 0, sets: {}, history: [] };
                entry.sets = { ...entry.sets, [setIdx]: { weight: value } };
                const w = Number(value);
                if (Number.isFinite(w) && w > entry.pr) entry.pr = w;
                next[key] = entry;
                saveData(next);
                return next;
            });
        },
        []
    );

    function startDay(key: ProgramKey) {
        setDay(key);
        setView('session');
        setSessionStart(Date.now());
        setElapsed(0);
    }

    function cancelSession() {
        setDay(null);
        setView('picker');
        setSessionStart(null);
        setElapsed(0);
    }

    function finishWorkout() {
        const today = new Date().toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric',
        });
        const newRows: WorkoutRow[] = [];
        const next: WorkoutData = {};
        const newPRs: { name: string; weight: number; delta: number }[] = [];
        for (const [key, val] of Object.entries(data)) {
            const e = { ...val, sets: { ...val.sets } };
            const setEntries = Object.entries(val.sets).filter(
                ([, s]) => Number(s.weight) > 0
            );
            if (setEntries.length > 0) {
                const weights = setEntries.map(([, s]) => Number(s.weight));
                const max = Math.max(...weights);
                e.name = e.name || displayNameFor(key);
                // Compare against the previously-known max (from cloud rows,
                // not the local in-progress data which already includes this session)
                const prevMax = lastTimeFor(rows, key)?.max ?? 0;
                if (max > prevMax) {
                    newPRs.push({
                        name: e.name,
                        weight: max,
                        delta: prevMax > 0 ? max - prevMax : 0,
                    });
                }
                e.history = [
                    ...(e.history || []),
                    { weight: max, date: today },
                ];
                if (day) {
                    for (const [siStr, s] of setEntries) {
                        newRows.push({
                            date: today,
                            program: day,
                            exercise: e.name,
                            set_idx: Number(siStr) + 1,
                            weight: Number(s.weight),
                            pr_at_time: e.pr,
                        });
                    }
                }
                e.sets = {};
            }
            next[key] = e;
        }
        setData(next);
        saveData(next);

        if (newRows.length > 0) {
            setSyncStatus({ state: 'syncing' });
            syncRows(newRows).then((result) => {
                if (result.ok) {
                    setSyncStatus({ state: 'ok' });
                    setRows((prev) => [...prev, ...newRows]);
                    if (newPRs.length > 0) {
                        setCelebration({ newPRs });
                    }
                } else {
                    setSyncStatus({
                        state: 'err',
                        error: result.error ?? 'unknown',
                    });
                }
                setTimeout(() => setSyncStatus(null), 4000);
            });
        }

        cancelSession();
    }

    if (loading) {
        return (
            <div className="wj-root">
                <div className="wj-loading">LOADING…</div>
            </div>
        );
    }

    if (view === 'history') {
        return (
            <div className="wj-root">
                <Link to="/personal" className="wj-back">
                    ← Personal
                </Link>
                <div className="wj-picker-header">
                    <div className="wj-eyebrow">Workout Journal</div>
                    <h1 className="wj-h1">Training log</h1>
                </div>
                <button
                    type="button"
                    className="wj-toggle-history"
                    onClick={() => setView('picker')}
                >
                    ← Back
                </button>
                <TrainingLog data={data} rows={rows} />
            </div>
        );
    }

    if (view === 'picker' || !day) {
        const sessions = groupRowsBySession(rows);
        return (
            <div className="wj-root">
                <Link to="/personal" className="wj-back">
                    ← Personal
                </Link>
                <Dashboard sessions={sessions} />
                {celebration && (
                    <Celebration
                        celebration={celebration}
                        onDismiss={() => setCelebration(null)}
                    />
                )}
                {syncStatus && !celebration && (
                    <div className={`wj-sync wj-sync--${syncStatus.state}`}>
                        {syncStatus.state === 'syncing' && 'Syncing to Excel…'}
                        {syncStatus.state === 'ok' && 'Synced ✓'}
                        {syncStatus.state === 'err' &&
                            `Sync failed: ${syncStatus.error} — saved locally`}
                    </div>
                )}
                <div className="wj-section-label">Today, pick a split</div>
                <div className="wj-program-list">
                    {(Object.entries(PROGRAMS) as [ProgramKey, Program][]).map(
                        ([key, prog]) => {
                            const totalSets = prog.blocks.reduce(
                                (n, b) =>
                                    n +
                                    b.pairs.reduce((m, p) => m + p.sets, 0),
                                0
                            );
                            const suggested = isSuggestedNextSplit(
                                key,
                                sessions
                            );
                            return (
                                <button
                                    key={key}
                                    type="button"
                                    className={`wj-program-card${suggested ? ' is-suggested' : ''}`}
                                    onClick={() => startDay(key)}
                                >
                                    <span className="wj-program-icon">
                                        {prog.icon}
                                    </span>
                                    <span className="wj-program-meta">
                                        <span className="wj-program-label">
                                            {prog.label}
                                            {suggested && (
                                                <span className="wj-suggested-pill">
                                                    next up
                                                </span>
                                            )}
                                        </span>
                                        <span className="wj-program-sub">
                                            {prog.subtitle}
                                        </span>
                                        <span className="wj-program-stats">
                                            {prog.blocks.length} blocks ·{' '}
                                            {totalSets} sets
                                        </span>
                                    </span>
                                </button>
                            );
                        }
                    )}
                </div>
                <button
                    type="button"
                    className="wj-toggle-history"
                    onClick={() => setView('history')}
                >
                    Open training log →
                </button>
            </div>
        );
    }

    const program = PROGRAMS[day];
    const elapsedStr = `${Math.floor(elapsed / 60)}:${(elapsed % 60)
        .toString()
        .padStart(2, '0')}`;
    const isOver = elapsed > 3600;

    return (
        <div className="wj-root">
            <div className="wj-session-header">
                <div className="wj-session-title">
                    <div className="wj-session-label">
                        {program.icon} {program.label} DAY
                    </div>
                    <div className="wj-session-sub">{program.subtitle}</div>
                </div>
                <div className="wj-elapsed">
                    <div
                        className={`wj-elapsed-value${
                            isOver ? ' is-over' : ''
                        }`}
                    >
                        {elapsedStr}
                    </div>
                    <div className="wj-elapsed-tag">ELAPSED</div>
                </div>
            </div>

            <div className="wj-pacing">
                <span>A 0–18m</span>
                <span>B 18–34m</span>
                <span>C 34–48m</span>
                <span>D 48–56m</span>
            </div>

            {program.blocks.map((block, bi) => (
                <SupersetBlock
                    key={bi}
                    block={block}
                    data={data}
                    rows={rows}
                    onWeightChange={handleWeightChange}
                />
            ))}

            <div className="wj-actions">
                <div className="wj-actions-inner">
                    <button
                        type="button"
                        className="wj-btn"
                        onClick={cancelSession}
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        className="wj-btn wj-btn--primary"
                        onClick={finishWorkout}
                    >
                        Finish &amp; Save
                    </button>
                </div>
            </div>
        </div>
    );
}

function displayNameFor(key: string): string {
    for (const program of Object.values(PROGRAMS)) {
        for (const block of program.blocks) {
            for (const pair of block.pairs) {
                if (getExerciseKey(pair.a) === key) return pair.a;
                if (getExerciseKey(pair.b) === key) return pair.b;
            }
        }
    }
    return key;
}

function mergeCloudIntoLocal(
    local: WorkoutData,
    cloud: WorkoutData
): WorkoutData {
    // Cloud (Excel) is the absolute source of truth for history + PRs.
    // Local only contributes in-progress `sets` for exercises with non-empty sets.
    const out: WorkoutData = {};
    for (const [k, c] of Object.entries(cloud)) {
        const l = local[k];
        out[k] = {
            ...c,
            sets:
                l && l.sets && Object.keys(l.sets).length > 0 ? l.sets : {},
        };
    }
    // Preserve local-only entries that have in-progress sets (haven't synced yet).
    for (const [k, l] of Object.entries(local)) {
        if (!out[k] && l.sets && Object.keys(l.sets).length > 0) {
            out[k] = {
                name: l.name,
                pr: 0,
                history: [],
                sets: l.sets,
            };
        }
    }
    return out;
}

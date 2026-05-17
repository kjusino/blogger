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
    WorkoutRow,
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

type SupersetBlockProps = {
    block: Block;
    data: WorkoutData;
    onWeightChange: (key: string, setIdx: number, value: string) => void;
};

function SupersetBlock({ block, data, onWeightChange }: SupersetBlockProps) {
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

function HistoryView({ data }: { data: WorkoutData }) {
    const exercises = Object.entries(data)
        .filter(([, v]) => v.history && v.history.length > 0)
        .sort((a, b) => (b[1].pr || 0) - (a[1].pr || 0));

    if (exercises.length === 0) {
        return (
            <div className="wj-history-empty">
                No history yet. Finish a workout to see your progress here.
            </div>
        );
    }

    return (
        <div className="wj-history">
            {exercises.map(([key, val]) => (
                <div key={key} className="wj-history-row">
                    <div className="wj-history-top">
                        <div className="wj-history-name">
                            <span>{val.name || key}</span>
                            <FormLink exercise={val.name || ''} />
                        </div>
                        <span className="wj-history-pr">PR {val.pr} lb</span>
                    </div>
                    <div className="wj-history-entries">
                        {val.history.slice(-10).map((h, i) => (
                            <div key={i} className="wj-history-entry">
                                <div
                                    className={`wj-history-entry-weight${
                                        h.weight >= val.pr ? ' is-pr' : ''
                                    }`}
                                >
                                    {h.weight}
                                </div>
                                <div>{h.date}</div>
                            </div>
                        ))}
                    </div>
                </div>
            ))}
        </div>
    );
}

type View = 'picker' | 'session' | 'history';

export default function Workout() {
    const [view, setView] = useState<View>('picker');
    const [day, setDay] = useState<ProgramKey | null>(null);
    const [data, setData] = useState<WorkoutData>({});
    const [sessionStart, setSessionStart] = useState<number | null>(null);
    const [elapsed, setElapsed] = useState(0);
    const [loading, setLoading] = useState(true);
    const [syncStatus, setSyncStatus] = useState<
        { state: 'syncing' } | { state: 'ok' } | { state: 'err'; error: string } | null
    >(null);

    useEffect(() => {
        loadData().then((local) => {
            setData(local);
            setLoading(false);
        });
        fetchHistory().then((cloud) => {
            if (!cloud) return;
            setData((prev) => mergeCloudIntoLocal(prev, cloud));
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
        const rows: WorkoutRow[] = [];
        const next: WorkoutData = {};
        for (const [key, val] of Object.entries(data)) {
            const e = { ...val, sets: { ...val.sets } };
            const setEntries = Object.entries(val.sets).filter(
                ([, s]) => Number(s.weight) > 0
            );
            if (setEntries.length > 0) {
                const weights = setEntries.map(([, s]) => Number(s.weight));
                const max = Math.max(...weights);
                e.name = e.name || displayNameFor(key);
                e.history = [
                    ...(e.history || []),
                    { weight: max, date: today },
                ];
                if (day) {
                    for (const [siStr, s] of setEntries) {
                        rows.push({
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

        if (rows.length > 0) {
            setSyncStatus({ state: 'syncing' });
            syncRows(rows).then((result) => {
                if (result.ok) {
                    setSyncStatus({ state: 'ok' });
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
                    <h1 className="wj-h1">PR History</h1>
                </div>
                <button
                    type="button"
                    className="wj-toggle-history"
                    onClick={() => setView('picker')}
                >
                    ← Back to picker
                </button>
                <HistoryView data={data} />
            </div>
        );
    }

    if (view === 'picker' || !day) {
        return (
            <div className="wj-root">
                <Link to="/personal" className="wj-back">
                    ← Personal
                </Link>
                <div className="wj-picker-header">
                    <div className="wj-eyebrow">Workout Journal</div>
                    <h1 className="wj-h1">Pick your split</h1>
                    <p className="wj-sub">
                        60 min · compound-first · superset-optimized
                    </p>
                </div>
                {syncStatus && (
                    <div className={`wj-sync wj-sync--${syncStatus.state}`}>
                        {syncStatus.state === 'syncing' && 'Syncing to Excel…'}
                        {syncStatus.state === 'ok' && 'Synced ✓'}
                        {syncStatus.state === 'err' &&
                            `Sync failed: ${syncStatus.error} — saved locally`}
                    </div>
                )}
                <div className="wj-program-list">
                    {(Object.entries(PROGRAMS) as [ProgramKey, Program][]).map(
                        ([key, prog]) => {
                            const totalSets = prog.blocks.reduce(
                                (n, b) =>
                                    n +
                                    b.pairs.reduce((m, p) => m + p.sets, 0),
                                0
                            );
                            return (
                                <button
                                    key={key}
                                    type="button"
                                    className="wj-program-card"
                                    onClick={() => startDay(key)}
                                >
                                    <span className="wj-program-icon">
                                        {prog.icon}
                                    </span>
                                    <span className="wj-program-meta">
                                        <span className="wj-program-label">
                                            {prog.label}
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
                    View PR history →
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

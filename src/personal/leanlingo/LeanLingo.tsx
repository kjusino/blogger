import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import LessonRunner from './LessonRunner';
import { fetchAttempts, fetchTree, flushPending, getCachedAttempts, getCachedTree } from './api';
import { deriveProgress, lessonNumber, unitNumber, worldNumber, worldsSorted } from './progress';
import type { Attempt, Lesson, ProgressState, QuestionTree, Unit, World } from './types';
import './leanlingo.css';

type View =
    | { kind: 'worlds' }
    | { kind: 'units'; world: World }
    | { kind: 'lesson'; world: World; unit: Unit; lesson: Lesson };

export default function LeanLingo() {
    const [tree, setTree] = useState<QuestionTree | null>(() => getCachedTree());
    const [attempts, setAttempts] = useState<Attempt[]>(() => getCachedAttempts());
    const [view, setView] = useState<View>({ kind: 'worlds' });
    const [loading, setLoading] = useState(!getCachedTree());
    const [toast, setToast] = useState<string | null>(null);

    // Initial load — fetch fresh data, but render cached immediately if present
    useEffect(() => {
        (async () => {
            const [t, a] = await Promise.all([fetchTree(), fetchAttempts()]);
            if (t) setTree(t);
            if (a) setAttempts(a);
            setLoading(false);
        })();
    }, []);

    // Flush any pending writes on mount + when network returns
    useEffect(() => {
        const tryFlush = () => {
            flushPending().then((n) => {
                if (n > 0) setToast(`synced ${n} pending answer${n === 1 ? '' : 's'}`);
            });
        };
        tryFlush();
        window.addEventListener('online', tryFlush);
        return () => window.removeEventListener('online', tryFlush);
    }, []);

    const progress: ProgressState = useMemo(
        () => deriveProgress(tree ?? { worlds: [] }, attempts),
        [tree, attempts]
    );

    const onLessonExit = useCallback(
        (_xp: number, _perfect: boolean) => {
            // Local cache is the truth — Excel has eventual-consistency lag.
            // Pull from local immediately, then merge any server updates in the background.
            setAttempts(getCachedAttempts());
            fetchAttempts().then((a) => {
                if (!a) return;
                // Merge: server rows + any local rows not yet on server (by timestamp+question_id).
                const seen = new Set(a.map((r) => `${r.timestamp}|${r.question_id}`));
                const local = getCachedAttempts();
                const merged = a.slice();
                for (const r of local) {
                    if (!seen.has(`${r.timestamp}|${r.question_id}`)) merged.push(r);
                }
                setAttempts(merged);
            });
            setView((v) =>
                v.kind === 'lesson' ? { kind: 'units', world: v.world } : v
            );
        },
        []
    );

    // Auto-dismiss toast
    useEffect(() => {
        if (!toast) return;
        const t = setTimeout(() => setToast(null), 2500);
        return () => clearTimeout(t);
    }, [toast]);

    if (loading && !tree) {
        return (
            <div className="leanlingo">
                <Header progress={progress} onBack={null} />
                <div className="leanlingo-loading">Loading lessons…</div>
            </div>
        );
    }
    if (!tree) {
        return (
            <div className="leanlingo">
                <Header progress={progress} onBack={null} />
                <div className="leanlingo-loading">Couldn't load lessons. Try again later.</div>
            </div>
        );
    }

    return (
        <div className="leanlingo">
            <Header
                progress={progress}
                onBack={
                    view.kind === 'worlds'
                        ? null
                        : view.kind === 'units'
                            ? () => setView({ kind: 'worlds' })
                            : () => setView({ kind: 'units', world: view.world })
                }
            />

            {view.kind === 'worlds' && (
                <WorldList tree={tree} progress={progress} onPick={(w) => setView({ kind: 'units', world: w })} />
            )}

            {view.kind === 'units' && (
                <UnitList
                    world={view.world}
                    progress={progress}
                    onLesson={(unit, lesson) => setView({ kind: 'lesson', world: view.world, unit, lesson })}
                />
            )}

            {view.kind === 'lesson' && (
                <LessonRunner lesson={view.lesson} onExit={onLessonExit} />
            )}

            {toast && <div className="leanlingo-toast">{toast}</div>}
        </div>
    );
}

function Header({ progress, onBack }: { progress: ProgressState; onBack: (() => void) | null }) {
    return (
        <div className="leanlingo-header">
            {onBack ? (
                <button className="leanlingo-back" onClick={onBack}>← back</button>
            ) : (
                <Link to="/personal" className="leanlingo-back">← personal</Link>
            )}
            <div className="leanlingo-stats">
                <span className="leanlingo-stat leanlingo-stat-streak">🔥 {progress.streak}</span>
                <span className="leanlingo-stat leanlingo-stat-xp">★ {progress.xp}</span>
            </div>
        </div>
    );
}

function WorldList({
    tree,
    progress,
    onPick,
}: {
    tree: QuestionTree;
    progress: ProgressState;
    onPick: (w: World) => void;
}) {
    const worlds = worldsSorted(tree);
    return (
        <div>
            <h1 className="leanlingo-h1">LeanLingo</h1>
            <p className="leanlingo-sub">Lean 4 in micro-doses. {worldsSorted(tree).length} worlds · {totalLessons(tree)} lessons.</p>
            <div className="leanlingo-worlds">
                {worlds.map((w) => {
                    const unlocked = progress.unlockedWorlds.has(w.id);
                    const done = w.units.reduce(
                        (s, u) => s + u.lessons.filter((l) => progress.completedLessons.has(l.id)).length,
                        0
                    );
                    const total = w.units.reduce((s, u) => s + u.lessons.length, 0);
                    const cls = `leanlingo-world ${unlocked ? '' : 'locked'}`;
                    return (
                        <button
                            key={w.id}
                            className={cls}
                            disabled={!unlocked}
                            onClick={() => unlocked && onPick(w)}
                        >
                            <span className="leanlingo-world-number">World {worldNumber(w.id)}</span>
                            <span className="leanlingo-world-title">{worldTitle(w)}</span>
                            <span className="leanlingo-world-progress">
                                {unlocked ? `${done}/${total} lessons` : '🔒 locked'}
                            </span>
                        </button>
                    );
                })}
            </div>
        </div>
    );
}

function UnitList({
    world,
    progress,
    onLesson,
}: {
    world: World;
    progress: ProgressState;
    onLesson: (u: Unit, l: Lesson) => void;
}) {
    return (
        <div>
            <h1 className="leanlingo-h1">World {worldNumber(world.id)}</h1>
            <p className="leanlingo-sub">{worldTitle(world)}</p>
            <div className="leanlingo-units">
                {world.units.map((u) => {
                    const unlocked = progress.unlockedUnits.has(u.id);
                    return (
                        <div key={u.id} className={`leanlingo-unit ${unlocked ? '' : 'locked'}`}>
                            <div className="leanlingo-unit-header">Unit {unitNumber(u.id)}</div>
                            <div className="leanlingo-lessons">
                                {u.lessons.map((l) => {
                                    const done = progress.completedLessons.has(l.id);
                                    const perfect = progress.perfectLessons.has(l.id);
                                    return (
                                        <button
                                            key={l.id}
                                            className="leanlingo-lesson"
                                            disabled={!unlocked}
                                            onClick={() => unlocked && onLesson(u, l)}
                                        >
                                            <div className="leanlingo-lesson-info">
                                                <span className="leanlingo-lesson-title">
                                                    {lessonNumber(l.id)}. {l.title}
                                                </span>
                                                <span className="leanlingo-lesson-ref">{l.book_ref}</span>
                                            </div>
                                            <span className="leanlingo-lesson-status">
                                                {perfect ? '★' : done ? '✓' : unlocked ? '›' : '🔒'}
                                            </span>
                                        </button>
                                    );
                                })}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

const WORLD_TITLES: Record<string, string> = {
    w1: 'Expressions, Types & Functions',
    w2: 'Dependent Type Theory',
    w3: 'Structures, Datatypes & Patterns',
    w4: 'Propositions and Proofs',
    w5: 'Hello, World!',
    w6: 'Quantifiers and Equality',
    w7: 'Tactics',
    w8: 'Type Classes',
    w9: 'Interacting with Lean',
    w10: 'Monads',
    w11: 'Inductive Types',
    w12: 'Functors, Applicatives & Monads',
    w13: 'Induction & Recursion',
};

function worldTitle(w: World): string {
    return WORLD_TITLES[w.id] ?? w.id;
}

function totalLessons(tree: QuestionTree): number {
    let n = 0;
    for (const w of tree.worlds) for (const u of w.units) n += u.lessons.length;
    return n;
}

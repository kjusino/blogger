import type { Attempt, ProgressState, QuestionTree, World } from './types';

function localDate(iso: string): string {
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    return d.toLocaleDateString('en-CA');
}

function todayLocal(): string {
    return new Date().toLocaleDateString('en-CA');
}

function dayDiff(a: string, b: string): number {
    const da = new Date(a + 'T00:00:00');
    const db = new Date(b + 'T00:00:00');
    return Math.round((db.getTime() - da.getTime()) / 86_400_000);
}

export function deriveProgress(
    tree: QuestionTree,
    attempts: Attempt[]
): ProgressState {
    const baseXp = attempts.reduce((s, a) => s + (a.xp_awarded || 0), 0);

    // A lesson is "completed" if every question in it has a correct attempt.
    // A lesson is "perfect" if every question was correct on the first try.
    const correctByQuestion = new Map<string, { firstTryCorrect: boolean; everCorrect: boolean }>();
    for (const a of attempts) {
        const cur = correctByQuestion.get(a.question_id) ?? { firstTryCorrect: false, everCorrect: false };
        if (a.correct) {
            cur.everCorrect = true;
            if (a.attempts === 1) cur.firstTryCorrect = true;
        }
        correctByQuestion.set(a.question_id, cur);
    }

    const completedLessons = new Set<string>();
    const perfectLessons = new Set<string>();
    for (const world of tree.worlds) {
        for (const unit of world.units) {
            for (const lesson of unit.lessons) {
                if (lesson.questions.length === 0) continue;
                const everyCorrect = lesson.questions.every(
                    (q) => correctByQuestion.get(q.id)?.everCorrect
                );
                if (everyCorrect) completedLessons.add(lesson.id);
                const everyFirstTry = lesson.questions.every(
                    (q) => correctByQuestion.get(q.id)?.firstTryCorrect
                );
                if (everyFirstTry && everyCorrect) perfectLessons.add(lesson.id);
            }
        }
    }

    const unlockedWorlds = new Set<string>();
    const unlockedUnits = new Set<string>();
    let priorWorldComplete = true;
    for (const world of tree.worlds) {
        if (priorWorldComplete) unlockedWorlds.add(world.id);
        let allUnitsComplete = true;
        let priorUnitComplete = true;
        for (const unit of world.units) {
            if (priorUnitComplete && unlockedWorlds.has(world.id)) {
                unlockedUnits.add(unit.id);
            }
            const allLessonsComplete = unit.lessons.every((l) => completedLessons.has(l.id));
            if (!allLessonsComplete) {
                priorUnitComplete = false;
                allUnitsComplete = false;
            }
        }
        if (!allUnitsComplete) priorWorldComplete = false;
    }

    // Streak: count distinct local-dates that have ≥1 correct attempt, walking
    // backward from today (or the most recent active day) until a gap appears.
    const activeDays = new Set<string>();
    let mostRecent = '';
    for (const a of attempts) {
        if (!a.correct) continue;
        const d = localDate(a.timestamp);
        if (!d) continue;
        activeDays.add(d);
        if (d > mostRecent) mostRecent = d;
    }
    const today = todayLocal();
    let streak = 0;
    let lastActiveDate = mostRecent;
    if (activeDays.size > 0) {
        // Start from today if active today, else from most recent active day
        let cursor = activeDays.has(today) ? today : mostRecent;
        // Only count the streak if the most recent active day is today or yesterday.
        const gapFromToday = cursor === today ? 0 : dayDiff(cursor, today);
        if (gapFromToday <= 1) {
            while (activeDays.has(cursor)) {
                streak++;
                const d = new Date(cursor + 'T00:00:00');
                d.setDate(d.getDate() - 1);
                cursor = d.toLocaleDateString('en-CA');
            }
        }
    }

    const xp = baseXp + PERFECT_BONUS * perfectLessons.size;

    return {
        xp,
        streak,
        lastActiveDate,
        completedLessons,
        perfectLessons,
        unlockedWorlds,
        unlockedUnits,
    };
}

export function xpForAttempt(attemptNumber: number, correct: boolean): number {
    if (!correct) return 0;
    if (attemptNumber === 1) return 10;
    if (attemptNumber === 2) return 5;
    return 0;
}

export const PERFECT_BONUS = 25;

export function worldNumber(id: string): number {
    return Number(id.replace(/^w/, '')) || 0;
}

export function unitNumber(id: string): number {
    return Number(id.split('-u')[1]) || 0;
}

export function lessonNumber(id: string): number {
    return Number(id.split('-l')[1]) || 0;
}

export function worldsSorted(tree: QuestionTree): World[] {
    return tree.worlds.slice().sort((a, b) => worldNumber(a.id) - worldNumber(b.id));
}

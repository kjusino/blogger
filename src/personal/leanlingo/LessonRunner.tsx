import { useMemo, useState } from 'react';
import QuestionCard, { Outcome } from './QuestionCard';
import type { Attempt, Lesson } from './types';
import { PERFECT_BONUS, xpForAttempt } from './progress';
import { recordAttempt, flushPending } from './api';

type Props = {
    lesson: Lesson;
    onExit: (gainedXp: number, perfect: boolean) => void;
};

export default function LessonRunner({ lesson, onExit }: Props) {
    const [idx, setIdx] = useState(0);
    const [gainedXp, setGainedXp] = useState(0);
    const [allFirstTry, setAllFirstTry] = useState(true);
    const [summary, setSummary] = useState<null | { perfect: boolean; xp: number }>(null);

    const total = lesson.questions.length;
    const current = lesson.questions[idx];
    const progressPct = useMemo(() => Math.round((idx / total) * 100), [idx, total]);

    function onAnswered(o: Outcome) {
        const xp = xpForAttempt(o.attemptNumber, o.correct);
        const attempt: Attempt = {
            timestamp: new Date().toISOString(),
            lesson_id: lesson.id,
            question_id: current.id,
            attempts: o.attemptNumber,
            correct: o.correct,
            xp_awarded: xp,
        };
        recordAttempt(attempt);
        const nextXp = gainedXp + xp;
        const stillFirstTry = allFirstTry && o.correct && o.attemptNumber === 1;
        setGainedXp(nextXp);
        setAllFirstTry(stillFirstTry);

        // Fire-and-forget sync; offline state is buffered.
        flushPending().catch(() => {});

        if (idx + 1 >= total) {
            const perfect = stillFirstTry;
            const final = nextXp + (perfect ? PERFECT_BONUS : 0);
            setSummary({ perfect, xp: final });
        } else {
            setIdx(idx + 1);
        }
    }

    if (summary) {
        return (
            <div className="leanlingo-summary">
                <div className="leanlingo-h2">{lesson.title}</div>
                <div className="leanlingo-sub">complete</div>
                <div className="leanlingo-summary-xp">+{summary.xp} XP</div>
                {summary.perfect && <div className="leanlingo-summary-perfect">★ Perfect lesson — +{PERFECT_BONUS} bonus</div>}
                <div className="leanlingo-actions" style={{ marginTop: 24 }}>
                    <button
                        className="leanlingo-btn"
                        onClick={() => onExit(summary.xp, summary.perfect)}
                    >
                        Continue
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div>
            <div className="leanlingo-h2">{lesson.title}</div>
            <div className="leanlingo-sub">{lesson.book_ref} · question {idx + 1} of {total}</div>
            <QuestionCard
                key={current.id}
                question={current}
                onAnswered={onAnswered}
                progressPct={progressPct}
            />
        </div>
    );
}

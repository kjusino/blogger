import { useState } from 'react';
import LeanCode from './LeanCode';
import type { Question } from './types';

export type Outcome = { correct: boolean; attemptNumber: number };

type Props = {
    question: Question;
    onAnswered: (o: Outcome) => void;
    progressPct: number;
};

function normalize(s: string): string {
    return s.trim().replace(/\s+/g, ' ').toLowerCase();
}

function answersMatch(input: string, answer: string): boolean {
    return normalize(input) === normalize(answer);
}

export default function QuestionCard({ question, onAnswered, progressPct }: Props) {
    const [attempts, setAttempts] = useState(0);
    const [done, setDone] = useState(false);
    const [feedback, setFeedback] = useState<'correct' | 'wrong' | null>(null);
    const [pendingOutcome, setPendingOutcome] = useState<Outcome | null>(null);

    function judge(correct: boolean) {
        const next = attempts + 1;
        setAttempts(next);
        setFeedback(correct ? 'correct' : 'wrong');
        if (correct || next >= 3) {
            setDone(true);
            setPendingOutcome({ correct, attemptNumber: next });
        }
    }

    return (
        <div className="leanlingo-question">
            <div className="leanlingo-progress-bar">
                <div className="leanlingo-progress-fill" style={{ width: `${progressPct}%` }} />
            </div>
            <p className="leanlingo-prompt">{question.prompt}</p>
            <LeanCode code={question.code} />
            <Body question={question} judge={judge} done={done} feedback={feedback} />
            {feedback && (
                <div className={`leanlingo-feedback ${feedback}`}>
                    {feedback === 'correct' ? '✓ Correct! ' : '✗ Not quite. '}
                    {question.explanation}
                </div>
            )}
            {feedback === 'wrong' && (question.quote || question.source_url) && (
                <div className="leanlingo-source">
                    {question.quote && (
                        <blockquote className="leanlingo-source-quote">
                            {question.quote}
                            <cite className="leanlingo-source-cite">— {question.book_ref}</cite>
                        </blockquote>
                    )}
                    {question.source_url && (
                        <a
                            className="leanlingo-source-link"
                            href={question.source_url}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Read more in the book ↗
                        </a>
                    )}
                </div>
            )}
            {done && pendingOutcome && (
                <div className="leanlingo-actions">
                    <button
                        className="leanlingo-btn"
                        onClick={() => onAnswered(pendingOutcome)}
                        autoFocus
                    >
                        Next →
                    </button>
                </div>
            )}
        </div>
    );
}

function Body({
    question,
    judge,
    done,
    feedback,
}: {
    question: Question;
    judge: (correct: boolean) => void;
    done: boolean;
    feedback: 'correct' | 'wrong' | null;
}) {
    switch (question.type) {
        case 'MC':
        case 'SE':
            return <MC question={question} judge={judge} done={done} feedback={feedback} />;
        case 'FIB':
        case 'PO':
            return <TextInput question={question} judge={judge} done={done} />;
        case 'ORD':
            return <Ordering question={question} judge={judge} done={done} />;
        default:
            return null;
    }
}

function MC({
    question,
    judge,
    done,
    feedback,
}: {
    question: Question;
    judge: (correct: boolean) => void;
    done: boolean;
    feedback: 'correct' | 'wrong' | null;
}) {
    const [picked, setPicked] = useState<string | null>(null);
    const [wrongPicks, setWrongPicks] = useState<Set<string>>(new Set());

    function pick(opt: string) {
        if (done) return;
        setPicked(opt);
        const correct = opt === question.answer;
        if (!correct) {
            setWrongPicks((s) => new Set(s).add(opt));
            // allow another try
            setTimeout(() => setPicked(null), 600);
        }
        judge(correct);
    }

    return (
        <div className="leanlingo-options">
            {question.options.map((opt) => {
                let cls = 'leanlingo-option';
                if (done && opt === question.answer) cls += ' correct';
                else if (wrongPicks.has(opt)) cls += ' wrong';
                else if (picked === opt && feedback === 'wrong') cls += ' wrong';
                return (
                    <button
                        key={opt}
                        className={cls}
                        disabled={done || wrongPicks.has(opt)}
                        onClick={() => pick(opt)}
                    >
                        {opt}
                    </button>
                );
            })}
        </div>
    );
}

function TextInput({
    question,
    judge,
    done,
}: {
    question: Question;
    judge: (correct: boolean) => void;
    done: boolean;
}) {
    const [value, setValue] = useState('');

    function submit() {
        if (done || !value.trim()) return;
        const correct = answersMatch(value, question.answer);
        if (!correct) {
            // keep input but flag wrong; clear after a moment for the next try
            setTimeout(() => setValue(''), 700);
        }
        judge(correct);
    }

    return (
        <div>
            <input
                className="leanlingo-input"
                value={done ? question.answer : value}
                onChange={(e) => setValue(e.target.value)}
                onKeyDown={(e) => {
                    if (e.key === 'Enter') submit();
                }}
                disabled={done}
                placeholder="type your answer…"
                autoFocus
                spellCheck={false}
                autoCapitalize="off"
                autoComplete="off"
            />
            <div className="leanlingo-actions">
                <button className="leanlingo-btn" onClick={submit} disabled={done || !value.trim()}>
                    Check
                </button>
            </div>
        </div>
    );
}

function Ordering({
    question,
    judge,
    done,
}: {
    question: Question;
    judge: (correct: boolean) => void;
    done: boolean;
}) {
    // Answer is stored as 'item1 | item2 | item3' — split to get the canonical order.
    const correctOrder = question.answer.split('|').map((s) => s.trim());
    const [items, setItems] = useState<string[]>(() => question.ord_items.slice());

    function move(i: number, dir: -1 | 1) {
        if (done) return;
        const j = i + dir;
        if (j < 0 || j >= items.length) return;
        const next = items.slice();
        const tmp = next[i];
        next[i] = next[j];
        next[j] = tmp;
        setItems(next);
    }

    function submit() {
        if (done) return;
        const correct = items.length === correctOrder.length &&
            items.every((v, idx) => normalize(v) === normalize(correctOrder[idx]));
        if (!correct) setItems(question.ord_items.slice());
        judge(correct);
    }

    const display = done ? correctOrder : items;

    return (
        <div>
            <div className="leanlingo-ord-list">
                {display.map((text, i) => (
                    <div key={`${text}-${i}`} className="leanlingo-ord-item">
                        <span className="leanlingo-ord-text">{text}</span>
                        <div className="leanlingo-ord-controls">
                            <button
                                className="leanlingo-ord-arrow"
                                disabled={done || i === 0}
                                onClick={() => move(i, -1)}
                                aria-label="move up"
                            >↑</button>
                            <button
                                className="leanlingo-ord-arrow"
                                disabled={done || i === display.length - 1}
                                onClick={() => move(i, 1)}
                                aria-label="move down"
                            >↓</button>
                        </div>
                    </div>
                ))}
            </div>
            <div className="leanlingo-actions">
                <button className="leanlingo-btn" onClick={submit} disabled={done}>
                    Check order
                </button>
            </div>
        </div>
    );
}

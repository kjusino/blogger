import { useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import './reflex.css';

type Phase = 'idle' | 'waiting' | 'ready' | 'tooSoon' | 'reacted' | 'done';

const KEY = 'personal:reflex:v1';
const TRIALS = 5;
const MIN_DELAY_MS = 1200;
const MAX_DELAY_MS = 4200;

type Stats = {
    bestMs: number;
    history: number[]; // most recent 50
    runs: number;
};

function loadStats(): Stats {
    try {
        const raw = localStorage.getItem(KEY);
        if (!raw) return { bestMs: 0, history: [], runs: 0 };
        const parsed = JSON.parse(raw);
        return {
            bestMs: Number(parsed?.bestMs ?? 0),
            history: Array.isArray(parsed?.history) ? parsed.history.slice(-50) : [],
            runs: Number(parsed?.runs ?? 0),
        };
    } catch {
        return { bestMs: 0, history: [], runs: 0 };
    }
}

function saveStats(s: Stats): void {
    try {
        localStorage.setItem(KEY, JSON.stringify(s));
    } catch {
        /* localStorage full or denied — non-fatal */
    }
}

function avg(xs: number[]): number {
    if (xs.length === 0) return 0;
    return Math.round(xs.reduce((s, x) => s + x, 0) / xs.length);
}

function rating(ms: number): string {
    if (ms < 180) return '⚡ inhuman';
    if (ms < 220) return '🚀 elite';
    if (ms < 260) return '🔥 sharp';
    if (ms < 310) return '👍 solid';
    if (ms < 380) return '🙂 normal';
    return '🐢 sleepy';
}

export default function Reflex() {
    const [phase, setPhase] = useState<Phase>('idle');
    const [stats, setStats] = useState<Stats>(() => loadStats());
    const [trialTimes, setTrialTimes] = useState<number[]>([]);
    const [lastMs, setLastMs] = useState<number | null>(null);
    const [shareNote, setShareNote] = useState<string | null>(null);

    const goAtRef = useRef<number>(0);
    const timerRef = useRef<number | null>(null);

    const clearTimer = useCallback(() => {
        if (timerRef.current !== null) {
            window.clearTimeout(timerRef.current);
            timerRef.current = null;
        }
    }, []);

    useEffect(() => () => clearTimer(), [clearTimer]);

    const startTrial = useCallback(() => {
        clearTimer();
        setPhase('waiting');
        const delay = MIN_DELAY_MS + Math.random() * (MAX_DELAY_MS - MIN_DELAY_MS);
        timerRef.current = window.setTimeout(() => {
            goAtRef.current = performance.now();
            setPhase('ready');
        }, delay);
    }, [clearTimer]);

    const handleTap = useCallback(() => {
        if (phase === 'idle' || phase === 'done' || phase === 'tooSoon' || phase === 'reacted') {
            // start (or restart) the next trial
            setLastMs(null);
            startTrial();
            return;
        }
        if (phase === 'waiting') {
            // tapped before green — count as a fail (no time), restart prompt
            clearTimer();
            setPhase('tooSoon');
            return;
        }
        if (phase === 'ready') {
            const ms = Math.max(0, Math.round(performance.now() - goAtRef.current));
            setLastMs(ms);
            const nextTrialTimes = [...trialTimes, ms];
            setTrialTimes(nextTrialTimes);
            if (nextTrialTimes.length >= TRIALS) {
                const runAvg = avg(nextTrialTimes);
                const newBest = stats.bestMs === 0 ? runAvg : Math.min(stats.bestMs, runAvg);
                const nextStats: Stats = {
                    bestMs: newBest,
                    history: [...stats.history, runAvg].slice(-50),
                    runs: stats.runs + 1,
                };
                setStats(nextStats);
                saveStats(nextStats);
                setPhase('done');
            } else {
                setPhase('reacted');
            }
        }
    }, [phase, trialTimes, stats, startTrial, clearTimer]);

    const reset = useCallback(() => {
        clearTimer();
        setTrialTimes([]);
        setLastMs(null);
        setPhase('idle');
        setShareNote(null);
    }, [clearTimer]);

    const onShare = useCallback(async () => {
        if (phase !== 'done') return;
        const runAvg = avg(trialTimes);
        const text = `🏃 reflex: ${runAvg}ms avg (best: ${runAvg}ms — ${rating(runAvg)})\nbeat me at https://kennethjusino.com/personal/reflex`;
        try {
            if (typeof navigator !== 'undefined' && (navigator as { share?: (data: { text: string }) => Promise<void> }).share) {
                await (navigator as { share: (data: { text: string }) => Promise<void> }).share({ text });
                setShareNote('shared');
                return;
            }
        } catch {
            /* user dismissed share sheet */
        }
        try {
            await navigator.clipboard.writeText(text);
            setShareNote('copied to clipboard');
        } catch {
            setShareNote('share unavailable — manual copy: ' + text);
        }
    }, [phase, trialTimes]);

    const runAvg = avg(trialTimes);
    const trialNumber = Math.min(trialTimes.length + 1, TRIALS);

    return (
        <div className="reflex">
            <div className="reflex-header">
                <Link to="/personal" className="reflex-back">← personal</Link>
                <span className="reflex-title">Reflex</span>
                <span className="reflex-best">
                    {stats.bestMs > 0 ? `best ${stats.bestMs}ms` : 'no best yet'}
                </span>
            </div>

            <button
                type="button"
                className={`reflex-stage reflex-${phase}`}
                onClick={handleTap}
                aria-label="reflex stage — tap to interact"
            >
                {phase === 'idle' && (
                    <>
                        <div className="reflex-stage-main">Tap to start</div>
                        <div className="reflex-stage-sub">{TRIALS} trials · avg wins</div>
                    </>
                )}
                {phase === 'waiting' && (
                    <>
                        <div className="reflex-stage-main">Wait…</div>
                        <div className="reflex-stage-sub">tap on green ({trialNumber}/{TRIALS})</div>
                    </>
                )}
                {phase === 'ready' && (
                    <>
                        <div className="reflex-stage-main">TAP!</div>
                    </>
                )}
                {phase === 'reacted' && (
                    <>
                        <div className="reflex-stage-main">{lastMs}ms</div>
                        <div className="reflex-stage-sub">tap for trial {trialNumber}/{TRIALS}</div>
                    </>
                )}
                {phase === 'tooSoon' && (
                    <>
                        <div className="reflex-stage-main">Too soon</div>
                        <div className="reflex-stage-sub">tap to retry</div>
                    </>
                )}
                {phase === 'done' && (
                    <>
                        <div className="reflex-stage-main">{runAvg}ms</div>
                        <div className="reflex-stage-sub">{rating(runAvg)}</div>
                    </>
                )}
            </button>

            {trialTimes.length > 0 && (
                <div className="reflex-trials">
                    {trialTimes.map((t, i) => (
                        <span key={i} className="reflex-trial">{t}ms</span>
                    ))}
                </div>
            )}

            {phase === 'done' && (
                <div className="reflex-done-actions">
                    <button type="button" className="reflex-btn reflex-btn-primary" onClick={onShare}>
                        Send to Genesis
                    </button>
                    <button type="button" className="reflex-btn" onClick={reset}>
                        Run again
                    </button>
                    {shareNote && <div className="reflex-share-note">{shareNote}</div>}
                </div>
            )}

            <div className="reflex-stats">
                <div className="reflex-stat">
                    <span className="reflex-stat-label">runs</span>
                    <span className="reflex-stat-value">{stats.runs}</span>
                </div>
                <div className="reflex-stat">
                    <span className="reflex-stat-label">best avg</span>
                    <span className="reflex-stat-value">{stats.bestMs > 0 ? `${stats.bestMs}ms` : '—'}</span>
                </div>
                <div className="reflex-stat">
                    <span className="reflex-stat-label">last 5 avg</span>
                    <span className="reflex-stat-value">
                        {stats.history.length > 0 ? `${avg(stats.history.slice(-5))}ms` : '—'}
                    </span>
                </div>
            </div>

            <div className="reflex-tip">
                tap when the screen turns green. tap too early and the trial resets.
                {' '}your run average is the score to beat. share with Genesis when you crush it.
            </div>
        </div>
    );
}

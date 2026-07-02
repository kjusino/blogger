import { useState, useRef, useEffect, useCallback } from 'react';
import '../audioplayer.css';
import { trackEvent } from '../analytics/tracker';

const SPEEDS = [1, 1.25, 1.5, 2];

function formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function AudioPlayer({ src, title, route, label }: { src: string; title: string; route: string; label?: string }) {
    const audioRef = useRef<HTMLAudioElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [speedIndex, setSpeedIndex] = useState(0);

    useEffect(() => {
        const audio = audioRef.current;
        if (!audio) return;

        const onTimeUpdate = () => setCurrentTime(audio.currentTime);
        const onLoadedMetadata = () => setDuration(audio.duration);
        const onEnded = () => { setIsPlaying(false); trackEvent('audio_complete', route); };
        const onPlay = () => { setIsPlaying(true); trackEvent('audio_play', route); };
        const onPause = () => setIsPlaying(false);

        audio.addEventListener('timeupdate', onTimeUpdate);
        audio.addEventListener('loadedmetadata', onLoadedMetadata);
        audio.addEventListener('ended', onEnded);
        audio.addEventListener('play', onPlay);
        audio.addEventListener('pause', onPause);

        return () => {
            audio.removeEventListener('timeupdate', onTimeUpdate);
            audio.removeEventListener('loadedmetadata', onLoadedMetadata);
            audio.removeEventListener('ended', onEnded);
            audio.removeEventListener('play', onPlay);
            audio.removeEventListener('pause', onPause);
        };
    }, []);

    const togglePlay = useCallback(() => {
        const audio = audioRef.current;
        if (!audio) return;
        if (audio.paused) {
            audio.play();
        } else {
            audio.pause();
        }
    }, []);

    const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const audio = audioRef.current;
        if (!audio) return;
        const time = Number(e.target.value);
        audio.currentTime = time;
        setCurrentTime(time);
    }, []);

    const cycleSpeed = useCallback(() => {
        const next = (speedIndex + 1) % SPEEDS.length;
        setSpeedIndex(next);
        if (audioRef.current) {
            audioRef.current.playbackRate = SPEEDS[next];
        }
    }, [speedIndex]);

    const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

    return (
        <div className="audio-player">
            <audio ref={audioRef} src={src} preload="metadata" />
            <div className="audio-player-controls">
                <button
                    className="audio-player-play"
                    onClick={togglePlay}
                    aria-label={isPlaying ? `Pause ${title}` : `Play ${title}`}
                >
                    {isPlaying ? (
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                            <rect x="2" y="1" width="3.5" height="12" rx="1" />
                            <rect x="8.5" y="1" width="3.5" height="12" rx="1" />
                        </svg>
                    ) : (
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                            <path d="M3 1.5v11l9-5.5z" />
                        </svg>
                    )}
                </button>
                <input
                    type="range"
                    className="audio-player-progress"
                    min={0}
                    max={duration || 0}
                    step={0.1}
                    value={currentTime}
                    onChange={handleSeek}
                    aria-label="Seek"
                    style={{ '--progress': `${progress}%` } as React.CSSProperties}
                />
                <span className="audio-player-time">
                    {formatTime(currentTime)} / {formatTime(duration)}
                </span>
                <button
                    className="audio-player-speed"
                    onClick={cycleSpeed}
                    aria-label={`Playback speed ${SPEEDS[speedIndex]}x`}
                >
                    {SPEEDS[speedIndex]}x
                </button>
            </div>
            <span className="audio-player-label">{label ?? 'Listen to this article'}</span>
        </div>
    );
}

export default AudioPlayer;

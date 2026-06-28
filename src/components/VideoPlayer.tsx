import { useState, useRef, useEffect, useCallback } from 'react';
import '../videoplayer.css';
import { trackEvent } from '../analytics/tracker';

function formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function VideoPlayer({ src, title, route }: { src: string; title: string; route: string }) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const wrapperRef = useRef<HTMLDivElement>(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [hasStarted, setHasStarted] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [isFullscreen, setIsFullscreen] = useState(false);

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const onTimeUpdate = () => setCurrentTime(video.currentTime);
        const onLoadedMetadata = () => setDuration(video.duration);
        const onEnded = () => { setIsPlaying(false); setHasStarted(false); trackEvent('video_complete', route); };
        const onPlay = () => { setIsPlaying(true); setHasStarted(true); trackEvent('video_play', route); };
        const onPause = () => setIsPlaying(false);

        video.addEventListener('timeupdate', onTimeUpdate);
        video.addEventListener('loadedmetadata', onLoadedMetadata);
        video.addEventListener('ended', onEnded);
        video.addEventListener('play', onPlay);
        video.addEventListener('pause', onPause);

        const onFullscreenChange = () => {
            setIsFullscreen(!!document.fullscreenElement);
        };
        document.addEventListener('fullscreenchange', onFullscreenChange);

        return () => {
            video.removeEventListener('timeupdate', onTimeUpdate);
            video.removeEventListener('loadedmetadata', onLoadedMetadata);
            video.removeEventListener('ended', onEnded);
            video.removeEventListener('play', onPlay);
            video.removeEventListener('pause', onPause);
            document.removeEventListener('fullscreenchange', onFullscreenChange);
        };
    }, []);

    const togglePlay = useCallback(() => {
        const video = videoRef.current;
        if (!video) return;
        if (video.paused) {
            video.play();
        } else {
            video.pause();
        }
    }, []);

    const handleSeek = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const video = videoRef.current;
        if (!video) return;
        const time = Number(e.target.value);
        video.currentTime = time;
        setCurrentTime(time);
    }, []);

    const toggleFullscreen = useCallback(() => {
        const wrapper = wrapperRef.current;
        if (!wrapper) return;
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            wrapper.requestFullscreen();
        }
    }, []);

    const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

    return (
        <div className="video-player">
            <div
                className="video-player-wrapper"
                ref={wrapperRef}
                onClick={togglePlay}
            >
                <video
                    ref={videoRef}
                    src={src}
                    preload="metadata"
                    playsInline
                    controls={isFullscreen}
                />
                {!isPlaying && !hasStarted && (
                    <div className="video-player-overlay" aria-label={`Play ${title}`}>
                        <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
                            <circle cx="32" cy="32" r="31" stroke="white" strokeWidth="2" opacity="0.85" />
                            <path d="M26 20v24l20-12z" fill="white" opacity="0.9" />
                        </svg>
                    </div>
                )}
            </div>
            <div className="video-player-controls">
                <button
                    className="video-player-play"
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
                    className="video-player-progress"
                    min={0}
                    max={duration || 0}
                    step={0.1}
                    value={currentTime}
                    onChange={handleSeek}
                    aria-label="Seek"
                    style={{ '--progress': `${progress}%` } as React.CSSProperties}
                />
                <span className="video-player-time">
                    {formatTime(currentTime)} / {formatTime(duration)}
                </span>
                <button
                    className="video-player-fullscreen"
                    onClick={toggleFullscreen}
                    aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                >
                    {isFullscreen ? (
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                            <path d="M1 9h2v2h2v2H1zM9 1v2h2v2h2V1zM1 5h2V3h2V1H1zM9 13h2v-2h2V9h-4z" />
                        </svg>
                    ) : (
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="currentColor">
                            <path d="M1 5V1h4v2H3v2zM9 1h4v4h-2V3h-2zM13 9v4H9v-2h2v-2zM5 13H1V9h2v2h2z" />
                        </svg>
                    )}
                </button>
            </div>
            <span className="video-player-label">Watch</span>
        </div>
    );
}

export default VideoPlayer;

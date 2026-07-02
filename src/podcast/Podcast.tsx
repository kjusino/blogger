import './podcast.css';
import { episodes, PODCAST_TITLE, PODCAST_DESCRIPTION } from './episodes';
import AudioPlayer from '../components/AudioPlayer';
import usePageTracking from '../analytics/usePageTracking';

function formatDate(dateStr: string): string {
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(year, month - 1, day);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
    });
}

function Podcast() {
    usePageTracking('/podcast');

    return (
        <div className="podcast-page">
            <header className="podcast-header">
                <div className="podcast-label">Podcast</div>
                <h1>{PODCAST_TITLE}</h1>
                <p className="podcast-tagline">{PODCAST_DESCRIPTION}</p>
                <div className="podcast-subscribe">
                    <a href="/feed.xml" target="_blank" rel="noreferrer">
                        RSS Feed
                    </a>
                </div>
            </header>

            {episodes.length === 0 ? (
                <p className="podcast-empty">
                    Episodes coming soon. Subscribe via RSS to get notified.
                </p>
            ) : (
                <div className="podcast-episodes">
                    {episodes.map((ep) => (
                        <div className="podcast-episode" key={ep.guid}>
                            <div className="episode-meta">
                                <span className="episode-number">
                                    Episode {ep.episodeNumber}
                                </span>
                                <span className="episode-dot">&middot;</span>
                                <time
                                    className="episode-date"
                                    dateTime={ep.pubDate}
                                >
                                    {formatDate(ep.pubDate)}
                                </time>
                            </div>
                            <h2 className="episode-title">{ep.title}</h2>
                            <p className="episode-description">
                                {ep.description}
                            </p>
                            <AudioPlayer
                                src={ep.audioUrl}
                                title={ep.title}
                                route={`/podcast/ep-${ep.episodeNumber}`}
                                label={`Episode ${ep.episodeNumber} · ${ep.duration}`}
                            />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export default Podcast;

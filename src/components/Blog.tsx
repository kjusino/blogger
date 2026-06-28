import '../blog.css';
import { Tags } from '../resources/enums/Tags';
import ThemeBadge from './ThemeBadge';
import AudioPlayer from './AudioPlayer';
import VideoPlayer from './VideoPlayer';
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

function Blog({
    route,
    title,
    abstract,
    pics,
    caption,
    content,
    tags,
    isBlogPost,
    createdDate,
    audioSrc,
    videoSrc,
}: {
    route: string;
    title: string;
    abstract: string;
    pics: string[];
    caption: string;
    content: JSX.Element[];
    tags?: Tags[];
    isBlogPost?: boolean;
    createdDate?: string;
    audioSrc?: string;
    videoSrc?: string;
}) {
    usePageTracking(route);

    const img = require(`../articles/pics/${pics[0]}`);

    const img2 =
        pics.length > 1 ? require(`../articles/pics/${pics[1]}`) : undefined;

    const articleClass = isBlogPost ? 'Article blog-page' : 'Article';

    return (
        <div className={articleClass}>
            <header className="Article-header">
                <h1>{title}</h1>
                {tags && tags.length > 0 && (
                    <div
                        className="ThemeTags"
                        style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            gap: 8,
                        }}
                    >
                        <ThemeBadge tags={tags} size={18} />
                        <span>{tags.join(' · ')}</span>
                    </div>
                )}
                <em className="Abstract">{abstract}</em>
                <figure>
                    <a
                        href={img}
                        target="_blank"
                        rel="noreferrer"
                        className="hero-zoom"
                        title="Click to open full size"
                    >
                        <img src={img} className="App-logo" alt={caption} />
                    </a>
                    <figcaption className="Caption">{caption}</figcaption>
                </figure>
                {isBlogPost && createdDate && (
                    <time className="publish-date" dateTime={createdDate}>
                        {formatDate(createdDate)}
                    </time>
                )}
            </header>
            {audioSrc && <AudioPlayer src={audioSrc} title={title} route={route} />}
            {videoSrc && <VideoPlayer src={videoSrc} title={title} route={route} />}
            <article className="Content">{content}</article>
            {img2 && (
                <figure>
                    <img src={img2} className="App-logo" />
                </figure>
            )}
        </div>
    );
}

export default Blog;

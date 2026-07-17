import { Link } from 'react-router-dom';
import allData from '../articles/allData';
import { relatedArticles } from '../resources/relatedArticles';
import { formatDate } from '../resources/formatDate';
import ThemeBadge from './ThemeBadge';

// End-of-post "Keep reading" footer. Surfaces the two most relevant other
// posts (by shared theme, newest breaking ties) as small thumbnail cards.
// Renders nothing off blog posts or when no candidates exist.
function RelatedReading({ route }: { route: string }) {
    const current = allData.find((a) => a.route === route);
    if (!current) return null;

    const related = relatedArticles(current, allData, 2);
    if (related.length === 0) return null;

    return (
        <aside className="RelatedReading">
            <h2 className="RelatedReading-heading">Keep reading</h2>
            <div className="RelatedReading-cards">
                {related.map((a) => {
                    const thumb = require(`../articles/pics/${a.pics[0]}`);
                    return (
                        <Link
                            key={a.route}
                            to={a.route}
                            className="RelatedReading-card"
                        >
                            <img
                                src={thumb}
                                alt=""
                                className="RelatedReading-thumb"
                                loading="lazy"
                            />
                            <div className="RelatedReading-body">
                                <span className="RelatedReading-title">
                                    {a.title}
                                </span>
                                {a.createdDate && (
                                    <span className="RelatedReading-date">
                                        {formatDate(a.createdDate)}
                                    </span>
                                )}
                                {a.tags && a.tags.length > 0 && (
                                    <span className="RelatedReading-tags">
                                        <ThemeBadge tags={a.tags} size={12} />
                                        <span>{a.tags.join(' · ')}</span>
                                    </span>
                                )}
                            </div>
                        </Link>
                    );
                })}
            </div>
        </aside>
    );
}

export default RelatedReading;

import { useLocation } from 'react-router-dom';
import allData from '../articles/allData';
import ThemeBadge from './ThemeBadge';

const Articles = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const tagForThisPage = queryParams.get('tag') ?? '';
    const filteredArticles = allData
        .filter((article) => {
            return article.tags?.toLocaleString().includes(tagForThisPage);
        })
        .reverse();

    return (
        <div className="articles-page">
            {filteredArticles.length === 0 ? (
                <p>No articles found for this tag.</p>
            ) : (
                <div>
                    <h2>
                        <u>{tagForThisPage} Articles</u>
                    </h2>
                    <ul className="article-list">
                        {filteredArticles.map(
                            ({ title, abstract, route, tags }) => (
                                <div>
                                    <h3
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: 8,
                                        }}
                                    >
                                        <ThemeBadge tags={tags} size={14} />
                                        <a href={route}>{title}</a>
                                    </h3>
                                    <p>{abstract}</p>
                                </div>
                            ),
                        )}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default Articles;

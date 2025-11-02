import { useLocation } from 'react-router-dom';
import { loadAllArticles } from '../utils/articleLoader';

const Articles = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const tagForThisPage = queryParams.get('tag') ?? '';

    // Load all articles from auto-discovery system
    const allArticles = loadAllArticles();

    const filteredArticles = allArticles.filter((article) => {
        return article.tags?.toLocaleString().includes(tagForThisPage);
    });

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
                        {filteredArticles.map(({ title, abstract, route }) => (
                            <div key={route}>
                                <h3>
                                    <a href={route}>{title}</a>
                                </h3>
                                <p>{abstract}</p>
                            </div>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
};

export default Articles;

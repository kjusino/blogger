import { useLocation } from 'react-router-dom';
import allData from '../articles/allData';

const Articles = () => {
    const location = useLocation();
    const queryParams = new URLSearchParams(location.search);
    const tagForThisPage = queryParams.get('tag') ?? '';
    const filteredArticles = allData.filter((article) => {
        return article.tags?.toLocaleString().includes(tagForThisPage);
    });

    return (
        <div className="articles-page">
            <h2>{tagForThisPage}</h2>
            {filteredArticles.length === 0 ? (
                <p>No articles found for this tag.</p>
            ) : (
                <ul className="article-list">
                    {filteredArticles.map(({ title, abstract, route }) => (
                        <div>
                            <h3>
                                <a href={route}>{title}</a>
                            </h3>
                            <p>{abstract}</p>
                        </div>
                    ))}
                </ul>
            )}
        </div>
    );
};

export default Articles;

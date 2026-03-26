import allData from '../articles/allData';

const EXCLUDED_ROUTES = ['/', '/cv'];

const Blogs = () => {
    const blogPosts = allData
        .filter((article) => !EXCLUDED_ROUTES.includes(article.route))
        .reverse();

    return (
        <div className="articles-page">
            <div>
                <h2>
                    <u>All Posts</u>
                </h2>
                <ul className="article-list">
                    {blogPosts.map(({ title, abstract, route }) => (
                        <div key={route}>
                            <h3>
                                <a href={route}>{title}</a>
                            </h3>
                            <p>{abstract}</p>
                        </div>
                    ))}
                </ul>
            </div>
        </div>
    );
};

export default Blogs;

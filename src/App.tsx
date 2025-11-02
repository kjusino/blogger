import './index.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import NavBar from './components/NavBar';
import Articles from './components/Articles';
import { loadAllArticles } from './utils/articleLoader';

// Lazy load the Blog component for better performance
const Blog = lazy(() => import('./components/Blog'));

function App() {
    // Auto-discover all markdown articles from generated JSON
    const articles = loadAllArticles();

    return (
        <header className="App-header">
            <BrowserRouter>
                <NavBar />
                <Suspense fallback={<div>Loading...</div>}>
                    <Routes>
                        {/* Dynamically generate routes from markdown articles */}
                        {articles.map((article) => (
                            <Route
                                key={article.route}
                                path={article.route}
                                element={
                                    <Blog
                                        route={article.route}
                                        title={article.title}
                                        abstract={article.abstract}
                                        pics={article.pics}
                                        caption={article.caption}
                                        content={article.content}
                                        tags={article.tags}
                                    />
                                }
                            />
                        ))}
                        {/* Articles listing page */}
                        <Route path="articles" element={<Articles />} />
                    </Routes>
                </Suspense>
            </BrowserRouter>
        </header>
    );
}

export default App;

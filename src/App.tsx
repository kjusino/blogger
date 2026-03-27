import './index.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Blog from '../src/components/Blog';
import allData from './articles/allData';
import NavBar from './components/NavBar';
import Articles from './components/Articles';
import Blogs from './components/Blogs';

const NON_BLOG_ROUTES = new Set(['/', '/cv']);

function App() {
    let routes = [];
    for (let blog of allData) {
        const isBlogPost = !NON_BLOG_ROUTES.has(blog.route);
        const component = (
            <Blog
                route={blog.route}
                title={blog.title}
                abstract={blog.abstract ?? ''}
                pics={blog.pics}
                caption={blog.caption}
                content={blog.content}
                isBlogPost={isBlogPost}
            />
        );
        routes.push(<Route path={blog.route} element={component} />);
    }
    routes.push(<Route path="articles" element={<Articles />} />);
    routes.push(<Route path="blogs" element={<Blogs />} />);
    return (
        <header className="App-header">
            <BrowserRouter>
                <NavBar />
                <Routes>{routes}</Routes>
            </BrowserRouter>
        </header>
    );
}

export default App;

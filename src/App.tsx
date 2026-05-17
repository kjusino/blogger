import './index.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Blog from '../src/components/Blog';
import allData from './articles/allData';
import NavBar from './components/NavBar';
import Articles from './components/Articles';
import Blogs from './components/Blogs';
import PersonalLayout from './personal/PersonalLayout';
import PersonalIndex from './personal/Index';
import Workout from './personal/workout/Workout';
import LeanLingo from './personal/leanlingo/LeanLingo';

const NON_BLOG_ROUTES = new Set(['/', '/cv', '/personal']);

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
        routes.push(<Route key={blog.route} path={blog.route} element={component} />);
    }
    routes.push(<Route key="articles" path="articles" element={<Articles />} />);
    routes.push(<Route key="blogs" path="blogs" element={<Blogs />} />);
    return (
        <header className="App-header">
            <BrowserRouter>
                <NavBar />
                <Routes>
                    {routes}
                    <Route path="/personal" element={<PersonalLayout />}>
                        <Route index element={<PersonalIndex />} />
                        <Route path="workout" element={<Workout />} />
                        <Route path="leanlingo" element={<LeanLingo />} />
                    </Route>
                </Routes>
            </BrowserRouter>
        </header>
    );
}

export default App;

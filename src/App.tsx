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
import Projects from './components/Projects';
import Reflex from './personal/reflex/Reflex';
import Analytics from './personal/analytics/Analytics';
import Runway from './personal/runway/Runway';
import Calibrate from './personal/calibrate/Calibrate';
import Forge from './personal/forge/Forge';

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
                tags={blog.tags}
                isBlogPost={isBlogPost}
                createdDate={blog.createdDate}
                audioSrc={blog.audioSrc}
                videoSrc={blog.videoSrc}
            />
        );
        routes.push(<Route key={blog.route} path={blog.route} element={component} />);
    }
    routes.push(<Route key="articles" path="articles" element={<Articles />} />);
    routes.push(<Route key="blogs" path="blogs" element={<Blogs />} />);
    routes.push(<Route key="projects" path="projects" element={<Projects />} />);
    return (
        <header className="App-header">
            <BrowserRouter>
                <NavBar />
                <Routes>
                    {routes}
                    <Route path="/personal" element={<PersonalLayout />}>
                        <Route index element={<PersonalIndex />} />
                        <Route path="workout" element={<Workout />} />
                        <Route path="reflex" element={<Reflex />} />
                        <Route path="analytics" element={<Analytics />} />
                        <Route path="runway" element={<Runway />} />
                        <Route path="calibrate" element={<Calibrate />} />
                        <Route path="forge" element={<Forge />} />
                    </Route>
                </Routes>
            </BrowserRouter>
        </header>
    );
}

export default App;

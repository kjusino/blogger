import './index.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Blog from '../src/components/Blog';
import allData from './articles/allData';
import NavBar from './components/NavBar';

function App() {
    let routes = [];
    for (let blog of allData) {
        const component = (
            <Blog
                route={blog.route}
                title={blog.title}
                abstract={blog.abstract ?? ''}
                pics={blog.pics}
                caption={blog.caption}
                content={blog.content}
            />
        );
        routes.push(<Route path={blog.route} element={component} />);
    }
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

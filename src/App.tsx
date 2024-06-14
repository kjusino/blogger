import './index.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Blog from '../src/components/Blog';
import allData from './articles/allData';

function App() {
    let routes = [];
    for (let blog of allData) {
        const component = (
            <Blog
                route={blog.route}
                title={blog.title}
                abstract={blog.abstract}
                pic={blog.pic}
                caption={blog.caption}
                content={blog.content}
            />
        );
        routes.push(<Route path={blog.route} element={component} />);
    }
    return (
        <header className="App-header">
            <BrowserRouter>
                <Routes>{routes}</Routes>
            </BrowserRouter>
        </header>
    );
}

export default App;

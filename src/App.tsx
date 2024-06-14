import './index.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Blog from '../src/components/Blog';
import data from './content/data';

function App() {
    let routes = [];
    for (let blog of data) {
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

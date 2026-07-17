import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

// React Router keeps the previous scroll position across route changes, so
// navigating (e.g. via the Keep-reading footer or nav bar) can leave a new
// article scrolled to the middle/bottom. Reset to the top on every path change.
function ScrollToTop() {
    const { pathname } = useLocation();

    useEffect(() => {
        window.scrollTo(0, 0);
    }, [pathname]);

    return null;
}

export default ScrollToTop;

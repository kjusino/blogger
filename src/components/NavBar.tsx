import { useState } from 'react';
import { useTheme } from '../context/ThemeContext';

const NavBar = () => {
    const [menuOpen, setMenuOpen] = useState(false);
    const { isDark, toggleTheme } = useTheme();

    const toggleMenu = () => setMenuOpen((prev) => !prev);
    const closeMenu = () => setMenuOpen(false);

    return (
        <header className="navbar">
            <div className="logo">
                <a href="/">Kasike Kéne</a>
            </div>

            <div className="nav-right">
                <button
                    className="theme-toggle"
                    onClick={toggleTheme}
                    aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
                    title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                    {isDark ? '☀️' : '🌙'}
                </button>

                <nav className={`nav-links ${menuOpen ? 'open' : ''}`}>
                    <a href="/blogs" onClick={closeMenu}>
                        Blogs
                    </a>
                    <a href="/cv" onClick={closeMenu}>
                        CV
                    </a>
                </nav>
            </div>

            <button
                className={`hamburger ${menuOpen ? 'open' : ''}`}
                onClick={toggleMenu}
                aria-label="Toggle navigation"
                aria-expanded={menuOpen}
            >
                <span className="bar top" />
                <span className="bar middle" />
                <span className="bar bottom" />
            </button>
        </header>
    );
};

export default NavBar;

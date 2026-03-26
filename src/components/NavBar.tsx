import { useState } from 'react';

const NavBar = () => {
    const [menuOpen, setMenuOpen] = useState(false);

    const toggleMenu = () => setMenuOpen((prev) => !prev);
    const closeMenu = () => setMenuOpen(false);

    return (
        <header className="navbar">
            <div className="logo">
                <a href="/">Kasike Kéne</a>
            </div>

            <nav className={`nav-links ${menuOpen ? 'open' : ''}`}>
                <a href="/blogs" onClick={closeMenu}>
                    Blogs
                </a>
                <a href="/cv" onClick={closeMenu}>
                    CV
                </a>
            </nav>

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

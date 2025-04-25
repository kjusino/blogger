import { useState } from 'react';
import { Tags } from '../resources/enums/Tags';

const NavBar = () => {
    const [menuOpen, setMenuOpen] = useState(false);

    const toggleMenu = () => setMenuOpen((prev) => !prev);
    const closeMenu = () => setMenuOpen(false);
    const mathRoute = `/articles?tag=${Tags.Math}`;
    const physicsRoute = `/articles?tag=${Tags.Physics}`;
    const computationRoute = `/articles?tag=${Tags.Computation}`;
    const cultureRoute = `/articles?tag=${Tags.Culture}`;

    return (
        <header className="navbar">
            <div className="logo">
                <a href="/">Kasike Kéne</a>
            </div>

            <nav className={`nav-links ${menuOpen ? 'open' : ''}`}>
                <a href={mathRoute} onClick={closeMenu}>
                    Math
                </a>
                <a href={physicsRoute} onClick={closeMenu}>
                    Physics
                </a>
                <a href={computationRoute} onClick={closeMenu}>
                    Computation
                </a>
                <a href={cultureRoute} onClick={closeMenu}>
                    Culture
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

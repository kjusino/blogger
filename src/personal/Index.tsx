import { Link, useNavigate } from 'react-router-dom';
import { logout } from './authClient';
import './personal.css';

const TOOLS = [
    { to: '/personal/workout', label: 'Workout Tracker' },
    { to: '/personal/reflex', label: 'Reflex' },
    { to: '/personal/analytics', label: 'Analytics' },
    { to: '/personal/calibrate', label: 'Calibrate' },
];

export default function PersonalIndex() {
    const navigate = useNavigate();

    async function onLogout() {
        await logout();
        navigate(0);
    }

    return (
        <div className="personal-page">
            <div className="personal-page-header">
                <h2>Personal</h2>
                <button className="personal-btn" onClick={onLogout}>
                    Log out
                </button>
            </div>
            <ul className="personal-tools">
                {TOOLS.map((t) => (
                    <li key={t.to}>
                        <Link to={t.to}>{t.label}</Link>
                    </li>
                ))}
            </ul>
        </div>
    );
}

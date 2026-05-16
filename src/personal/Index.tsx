import { Link, useNavigate } from 'react-router-dom';
import { logout } from './authClient';

export default function PersonalIndex() {
    const navigate = useNavigate();

    async function onLogout() {
        await logout();
        navigate(0);
    }

    return (
        <div style={{ maxWidth: 600, margin: '3rem auto', padding: '1rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2>Personal</h2>
                <button onClick={onLogout} style={{ padding: '0.4rem 0.8rem' }}>
                    Log out
                </button>
            </div>
            <ul style={{ marginTop: '1.5rem', lineHeight: 2 }}>
                <li>
                    <Link to="/personal/workout">Workout Tracker</Link>
                </li>
            </ul>
        </div>
    );
}

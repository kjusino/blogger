import { Link } from 'react-router-dom';

export default function Workout() {
    return (
        <div style={{ maxWidth: 600, margin: '3rem auto', padding: '1rem' }}>
            <Link to="/personal">← Personal</Link>
            <h2>Workout Tracker</h2>
            <p>Coming soon.</p>
        </div>
    );
}

import { Link } from 'react-router-dom';
import '../personal.css';

export default function Workout() {
    return (
        <div className="personal-page">
            <Link to="/personal" className="personal-back">
                ← Personal
            </Link>
            <div className="personal-page-header">
                <h2>Workout Tracker</h2>
            </div>
            <p className="personal-muted">Coming soon.</p>
        </div>
    );
}

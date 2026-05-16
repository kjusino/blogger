import { useState, FormEvent } from 'react';
import { login } from './authClient';
import './personal.css';

type Props = { onSuccess: () => void };

export default function Login({ onSuccess }: Props) {
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);

    async function onSubmit(e: FormEvent) {
        e.preventDefault();
        setBusy(true);
        setError(null);
        const result = await login(password);
        setBusy(false);
        if (result.authenticated) {
            onSuccess();
        } else {
            setError('Incorrect password');
            setPassword('');
        }
    }

    return (
        <div className="personal-login">
            <h2>Personal</h2>
            <form onSubmit={onSubmit}>
                <input
                    type="password"
                    className="personal-input"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Password"
                    autoFocus
                    disabled={busy}
                />
                <button
                    type="submit"
                    className="personal-btn personal-btn-primary"
                    disabled={busy || !password}
                >
                    {busy ? 'Checking…' : 'Enter'}
                </button>
                {error && <div className="personal-error">{error}</div>}
            </form>
        </div>
    );
}

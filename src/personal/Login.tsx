import { useState, FormEvent } from 'react';
import { login } from './authClient';

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
        <div style={{ maxWidth: 360, margin: '4rem auto', padding: '1rem' }}>
            <h2>Personal</h2>
            <form onSubmit={onSubmit}>
                <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Password"
                    autoFocus
                    style={{ width: '100%', padding: '0.5rem', fontSize: '1rem' }}
                    disabled={busy}
                />
                <button
                    type="submit"
                    disabled={busy || !password}
                    style={{ marginTop: '0.75rem', padding: '0.5rem 1rem' }}
                >
                    {busy ? 'Checking…' : 'Enter'}
                </button>
                {error && <div style={{ color: 'crimson', marginTop: '0.5rem' }}>{error}</div>}
            </form>
        </div>
    );
}

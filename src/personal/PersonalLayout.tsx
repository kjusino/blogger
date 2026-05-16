import { useEffect, useState, useCallback } from 'react';
import { Outlet } from 'react-router-dom';
import { getAuthState } from './authClient';
import Login from './Login';
import './personal.css';

type Status = 'loading' | 'in' | 'out';

export default function PersonalLayout() {
    const [status, setStatus] = useState<Status>('loading');

    const refresh = useCallback(async () => {
        const { authenticated } = await getAuthState();
        setStatus(authenticated ? 'in' : 'out');
    }, []);

    useEffect(() => {
        refresh();
    }, [refresh]);

    if (status === 'loading') {
        return <div className="personal-loading">Loading…</div>;
    }
    if (status === 'out') {
        return <Login onSuccess={refresh} />;
    }
    return <Outlet />;
}

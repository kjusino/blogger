import { useEffect, useState, useCallback } from 'react';
import { Outlet } from 'react-router-dom';
import { getAuthState } from './authClient';
import Login from './Login';

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
        return <div style={{ padding: '2rem' }}>Loading…</div>;
    }
    if (status === 'out') {
        return <Login onSuccess={refresh} />;
    }
    return <Outlet />;
}

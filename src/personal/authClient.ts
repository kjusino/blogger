export type AuthState = { authenticated: boolean };

export async function getAuthState(): Promise<AuthState> {
    const res = await fetch('/api/personal/me', { credentials: 'same-origin' });
    if (!res.ok) return { authenticated: false };
    return res.json();
}

export async function login(password: string): Promise<AuthState> {
    const res = await fetch('/api/personal/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ password }),
    });
    if (!res.ok) return { authenticated: false };
    return res.json();
}

export async function logout(): Promise<void> {
    await fetch('/api/personal/logout', {
        method: 'POST',
        credentials: 'same-origin',
    });
}

import { useState, FormEvent } from 'react';
import './newsletter.css';

function NewsletterSubscribe({ route }: { route: string }) {
    const [email, setEmail] = useState('');
    const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
    const [errorMsg, setErrorMsg] = useState('');

    async function handleSubmit(e: FormEvent) {
        e.preventDefault();
        if (!email.trim()) return;
        setStatus('loading');
        setErrorMsg('');
        try {
            const res = await fetch('/api/newsletter/subscribe', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email: email.trim(), route }),
            });
            if (res.status === 429) {
                setErrorMsg('Too many attempts. Please try again later.');
                setStatus('error');
                return;
            }
            const data = await res.json();
            if (data.ok) {
                setStatus('success');
            } else {
                setErrorMsg(data.error || 'Something went wrong.');
                setStatus('error');
            }
        } catch {
            setErrorMsg('Network error. Please try again.');
            setStatus('error');
        }
    }

    if (status === 'success') {
        return (
            <div className="newsletter-card">
                <p className="newsletter-success">Thanks for subscribing!</p>
            </div>
        );
    }

    return (
        <div className="newsletter-card">
            <h3 className="newsletter-heading">Stay updated</h3>
            <p className="newsletter-subtext">Get notified when I publish new articles.</p>
            <form onSubmit={handleSubmit} className="newsletter-form">
                <input
                    type="email"
                    className="personal-input newsletter-input"
                    placeholder="your@email.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    disabled={status === 'loading'}
                />
                <button
                    type="submit"
                    className="personal-btn personal-btn-primary newsletter-btn"
                    disabled={status === 'loading'}
                >
                    {status === 'loading' ? 'Subscribing...' : 'Subscribe'}
                </button>
            </form>
            {errorMsg && <p className="personal-error">{errorMsg}</p>}
        </div>
    );
}

export default NewsletterSubscribe;

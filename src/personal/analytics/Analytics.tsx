import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { fetchAnalyticsSummary, refreshAnalytics, AnalyticsSummary } from './api';
import '../personal.css';
import './analytics.css';

function formatReadTime(seconds: number): string {
    if (seconds < 60) return `${seconds}s`;
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function pct(num: number, denom: number): string {
    if (denom === 0) return '—';
    return `${Math.round((num / denom) * 100)}%`;
}

export default function Analytics() {
    const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    const load = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const data = await fetchAnalyticsSummary();
            setSummary(data);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { load(); }, [load]);

    async function onRefresh() {
        await refreshAnalytics();
        load();
    }

    if (loading && !summary) {
        return <p className="personal-loading">Loading analytics…</p>;
    }

    return (
        <div className="personal-page">
            <Link to="/personal" className="personal-back">&larr; Back</Link>
            <div className="personal-page-header">
                <h2>Analytics</h2>
                <div className="analytics-actions">
                    <button className="personal-btn" onClick={onRefresh} disabled={loading}>
                        {loading ? 'Refreshing…' : 'Refresh'}
                    </button>
                </div>
            </div>

            {error && <p className="personal-error">{error}</p>}

            {summary && (
                <>
                    <div className="analytics-cards">
                        <div className="analytics-card">
                            <div className="analytics-card-value">{summary.totalViews}</div>
                            <div className="analytics-card-label">Total Views</div>
                        </div>
                        <div className="analytics-card">
                            <div className="analytics-card-value">{summary.uniqueSessions}</div>
                            <div className="analytics-card-label">Unique Sessions</div>
                        </div>
                        <div className="analytics-card">
                            <div className="analytics-card-value">{summary.totalAudioPlays}</div>
                            <div className="analytics-card-label">Audio Plays</div>
                        </div>
                        <div className="analytics-card">
                            <div className="analytics-card-value">{summary.totalVideoPlays}</div>
                            <div className="analytics-card-label">Video Plays</div>
                        </div>
                    </div>

                    <div className="analytics-section">
                        <h3>Per Article</h3>
                        <div className="analytics-table-wrap">
                            <table className="analytics-table">
                                <thead>
                                    <tr>
                                        <th>Article</th>
                                        <th>Views</th>
                                        <th>Unique</th>
                                        <th>Listens</th>
                                        <th>Listen %</th>
                                        <th>Watches</th>
                                        <th>Watch %</th>
                                        <th>Avg Read</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {summary.articles.map((a) => (
                                        <tr key={a.route}>
                                            <td>{a.route}</td>
                                            <td>{a.views}</td>
                                            <td>{a.uniqueSessions}</td>
                                            <td>{a.audioPlays}</td>
                                            <td>{pct(a.audioCompletes, a.audioPlays)}</td>
                                            <td>{a.videoPlays}</td>
                                            <td>{pct(a.videoCompletes, a.videoPlays)}</td>
                                            <td>{formatReadTime(a.avgReadSeconds)}</td>
                                        </tr>
                                    ))}
                                    {summary.articles.length === 0 && (
                                        <tr>
                                            <td colSpan={8} style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                                                No data yet
                                            </td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {summary.topReferrers.length > 0 && (
                        <div className="analytics-section">
                            <h3>Top Referrers</h3>
                            <ul className="analytics-referrers">
                                {summary.topReferrers.map((r) => (
                                    <li key={r.referrer}>
                                        <span>{r.referrer}</span>
                                        <span>{r.count}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {summary.devices.length > 0 && (
                        <div className="analytics-section">
                            <h3>Devices</h3>
                            <div className="analytics-devices">
                                {summary.devices.map((d) => (
                                    <span key={d.device} className="analytics-device">
                                        {d.device}{' '}
                                        <span className="analytics-device-pct">{d.percent}%</span>
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

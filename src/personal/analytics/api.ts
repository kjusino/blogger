export interface ArticleStats {
    route: string;
    views: number;
    uniqueSessions: number;
    audioPlays: number;
    audioCompletes: number;
    videoPlays: number;
    videoCompletes: number;
    avgReadSeconds: number;
}

export interface AnalyticsSummary {
    totalViews: number;
    uniqueSessions: number;
    totalAudioPlays: number;
    totalVideoPlays: number;
    articles: ArticleStats[];
    topReferrers: { referrer: string; count: number }[];
    devices: { device: string; count: number; percent: number }[];
}

export async function fetchAnalyticsSummary(): Promise<AnalyticsSummary> {
    const res = await fetch('/api/personal/analytics/summary', {
        credentials: 'same-origin',
    });
    if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
    }
    return res.json();
}

export async function refreshAnalytics(): Promise<void> {
    await fetch('/api/personal/analytics/refresh', {
        method: 'POST',
        credentials: 'same-origin',
    });
}

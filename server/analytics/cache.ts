import { readAllEvents, EventRow } from './excel';

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

const TTL_MS = 5 * 60 * 1000;

let cached: { summary: AnalyticsSummary; expiresAt: number } | null = null;
let inflight: Promise<AnalyticsSummary> | null = null;

function buildSummary(rows: EventRow[]): AnalyticsSummary {
    const allSessions = new Set<string>();
    const routeMap = new Map<string, EventRow[]>();
    const referrerCounts = new Map<string, number>();
    const deviceCounts = new Map<string, number>();

    for (const r of rows) {
        allSessions.add(r.session_id);

        let bucket = routeMap.get(r.route);
        if (!bucket) { bucket = []; routeMap.set(r.route, bucket); }
        bucket.push(r);

        if (r.referrer) {
            referrerCounts.set(r.referrer, (referrerCounts.get(r.referrer) ?? 0) + 1);
        }

        if (r.event === 'view') {
            deviceCounts.set(r.device, (deviceCounts.get(r.device) ?? 0) + 1);
        }
    }

    let totalViews = 0;
    let totalAudioPlays = 0;
    let totalVideoPlays = 0;

    const articles: ArticleStats[] = [];
    for (const [route, events] of routeMap) {
        const sessions = new Set<string>();
        let views = 0, audioPlays = 0, audioCompletes = 0;
        let videoPlays = 0, videoCompletes = 0;
        let readSecondsSum = 0, readSecondsCount = 0;

        for (const e of events) {
            sessions.add(e.session_id);
            switch (e.event) {
                case 'view':
                    views++;
                    if (e.read_seconds > 0) {
                        readSecondsSum += e.read_seconds;
                        readSecondsCount++;
                    }
                    break;
                case 'audio_play': audioPlays++; break;
                case 'audio_complete': audioCompletes++; break;
                case 'video_play': videoPlays++; break;
                case 'video_complete': videoCompletes++; break;
            }
        }

        totalViews += views;
        totalAudioPlays += audioPlays;
        totalVideoPlays += videoPlays;

        articles.push({
            route,
            views,
            uniqueSessions: sessions.size,
            audioPlays,
            audioCompletes,
            videoPlays,
            videoCompletes,
            avgReadSeconds: readSecondsCount > 0 ? Math.round(readSecondsSum / readSecondsCount) : 0,
        });
    }

    articles.sort((a, b) => b.views - a.views);

    const topReferrers = Array.from(referrerCounts.entries())
        .map(([referrer, count]) => ({ referrer, count }))
        .sort((a, b) => b.count - a.count)
        .slice(0, 20);

    const totalDeviceViews = Array.from(deviceCounts.values()).reduce((a, b) => a + b, 0);
    const devices = Array.from(deviceCounts.entries())
        .map(([device, count]) => ({
            device,
            count,
            percent: totalDeviceViews > 0 ? Math.round((count / totalDeviceViews) * 100) : 0,
        }))
        .sort((a, b) => b.count - a.count);

    return {
        totalViews,
        uniqueSessions: allSessions.size,
        totalAudioPlays,
        totalVideoPlays,
        articles,
        topReferrers,
        devices,
    };
}

export async function getAnalyticsSummary(): Promise<AnalyticsSummary> {
    if (cached && Date.now() < cached.expiresAt) return cached.summary;
    if (inflight) return inflight;
    inflight = (async () => {
        try {
            const rows = await readAllEvents();
            const summary = buildSummary(rows);
            cached = { summary, expiresAt: Date.now() + TTL_MS };
            return summary;
        } finally {
            inflight = null;
        }
    })();
    return inflight;
}

export function bustAnalyticsCache(): void {
    cached = null;
}

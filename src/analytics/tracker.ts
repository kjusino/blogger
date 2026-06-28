const SID_KEY = 'analytics:sid';

function getSessionId(): string {
    let sid = sessionStorage.getItem(SID_KEY);
    if (!sid) {
        sid = crypto.randomUUID();
        sessionStorage.setItem(SID_KEY, sid);
    }
    return sid;
}

function getDevice(): 'mobile' | 'tablet' | 'desktop' {
    const ua = navigator.userAgent;
    if (/Tablet|iPad/i.test(ua)) return 'tablet';
    if (/Mobile|Android|iPhone/i.test(ua)) return 'mobile';
    return 'desktop';
}

function getReferrer(): string {
    try {
        const ref = document.referrer;
        if (!ref) return '';
        const url = new URL(ref);
        if (url.hostname === window.location.hostname) return '';
        return url.hostname;
    } catch {
        return '';
    }
}

const sent = new Set<string>();

function dedupKey(event: string, route: string): string {
    return `${event}::${route}`;
}

export function trackEvent(
    event: 'view' | 'audio_play' | 'audio_complete' | 'video_play' | 'video_complete',
    route: string,
    extra?: { read_seconds?: number; skipDedup?: boolean }
): void {
    const key = dedupKey(event, route);
    if (!extra?.skipDedup && sent.has(key)) return;
    sent.add(key);

    const payload = JSON.stringify({
        event,
        route,
        session_id: getSessionId(),
        referrer: getReferrer(),
        device: getDevice(),
        read_seconds: extra?.read_seconds ?? 0,
    });

    if (navigator.sendBeacon) {
        const blob = new Blob([payload], { type: 'application/json' });
        navigator.sendBeacon('/api/analytics/event', blob);
    } else {
        fetch('/api/analytics/event', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: payload,
            keepalive: true,
        }).catch(() => {});
    }
}

import { useEffect, useRef } from 'react';
import { trackEvent } from './tracker';

export default function usePageTracking(route: string): void {
    const startRef = useRef(Date.now());

    useEffect(() => {
        startRef.current = Date.now();
        trackEvent('view', route);

        function sendReadTime() {
            const seconds = Math.round((Date.now() - startRef.current) / 1000);
            if (seconds > 2) {
                trackEvent('view', route, { read_seconds: seconds, skipDedup: true });
            }
        }

        function onVisibilityChange() {
            if (document.visibilityState === 'hidden') {
                sendReadTime();
            }
        }

        document.addEventListener('visibilitychange', onVisibilityChange);

        return () => {
            document.removeEventListener('visibilitychange', onVisibilityChange);
            sendReadTime();
        };
    }, [route]);
}

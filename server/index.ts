import express from 'express';
import cookieParser from 'cookie-parser';
import helmet from 'helmet';
import fs from 'fs';
import path from 'path';
import { env } from './env';
import { authRouter, requireAuth } from './auth';
import { workoutRouter } from './routes/workout';
import { leanlingoRouter } from './routes/leanlingo';
import { analyticsPublicRouter, analyticsPrivateRouter } from './routes/analytics';
import { startFlusher } from './analytics/buffer';
import { newsletterRouter } from './routes/newsletter';
import { generateRss } from './rss';
import { injectMetaTags } from './ogTags';

const app = express();

// Trust X-Forwarded-* from Azure App Service so req.ip is the real client
// IP (used by express-rate-limit). `true` trusts all forwarders; Azure
// strips client-supplied X-Forwarded-For on ingress so spoofing isn't a
// concern. `1` was insufficient — Azure adds >1 hop.
app.set('trust proxy', true);
app.disable('x-powered-by');

app.use(
    helmet({
        contentSecurityPolicy: false, // CRA inline-scripts would break with default CSP; revisit on Vite migration
        crossOriginEmbedderPolicy: false,
        strictTransportSecurity: { maxAge: 31_536_000, includeSubDomains: true, preload: false },
    })
);
app.use(express.json({ limit: '256kb' }));
app.use(cookieParser());

app.use('/api/personal', authRouter);
app.use('/api/personal/workout', requireAuth, workoutRouter);
app.use('/api/personal/leanlingo', requireAuth, leanlingoRouter);
app.use('/api/analytics', analyticsPublicRouter);
app.use('/api/personal/analytics', requireAuth, analyticsPrivateRouter);
app.use('/api/newsletter', newsletterRouter);

startFlusher();

const buildDir = path.resolve(__dirname, '..', 'build');
app.use(express.static(buildDir));

const rssXml = generateRss();
app.get('/rss.xml', (_req, res) => {
    res.set('Content-Type', 'application/rss+xml; charset=utf-8');
    res.set('Cache-Control', 'public, max-age=3600');
    res.send(rssXml);
});

const indexHtmlTemplate = fs.readFileSync(path.join(buildDir, 'index.html'), 'utf8');
app.get('*', (req, res) => {
    const html = injectMetaTags(indexHtmlTemplate, req.path);
    res.set('Content-Type', 'text/html');
    res.send(html);
});

app.listen(env.PORT, () => {
    console.log(`[server] listening on :${env.PORT} (${env.NODE_ENV})`);
});

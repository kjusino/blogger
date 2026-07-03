import express from 'express';
import cookieParser from 'cookie-parser';
import helmet from 'helmet';
import path from 'path';
import fs from 'fs';
import { env } from './env';
import { authRouter, requireAuth } from './auth';
import { workoutRouter } from './routes/workout';
import { leanlingoRouter } from './routes/leanlingo';
import { analyticsPublicRouter, analyticsPrivateRouter } from './routes/analytics';
import { startFlusher } from './analytics/buffer';
import { injectOgTags } from './og';

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

startFlusher();

const buildDir = path.resolve(__dirname, '..', 'build');
app.use(express.static(buildDir, { index: false }));

const indexHtml = fs.readFileSync(path.join(buildDir, 'index.html'), 'utf-8');
const ogInjector = injectOgTags(buildDir);

app.get('*', (req, res) => {
    res.type('html').send(ogInjector(indexHtml, req.path));
});

app.listen(env.PORT, () => {
    console.log(`[server] listening on :${env.PORT} (${env.NODE_ENV})`);
});

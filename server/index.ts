import express from 'express';
import cookieParser from 'cookie-parser';
import helmet from 'helmet';
import path from 'path';
import { env } from './env';
import { authRouter, requireAuth } from './auth';
import { workoutRouter } from './routes/workout';
import { leanlingoRouter } from './routes/leanlingo';

const app = express();

// Tell Express to trust X-Forwarded-* from the Azure App Service front-end
// so req.ip reflects the real client (used by express-rate-limit).
app.set('trust proxy', 1);
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

const buildDir = path.resolve(__dirname, '..', 'build');
app.use(express.static(buildDir));

app.get('*', (_req, res) => {
    res.sendFile(path.join(buildDir, 'index.html'));
});

app.listen(env.PORT, () => {
    console.log(`[server] listening on :${env.PORT} (${env.NODE_ENV})`);
});

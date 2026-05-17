import express from 'express';
import cookieParser from 'cookie-parser';
import path from 'path';
import { env } from './env';
import { authRouter, requireAuth } from './auth';
import { workoutRouter } from './routes/workout';

const app = express();

app.use(express.json());
app.use(cookieParser());

app.use('/api/personal', authRouter);
app.use('/api/personal/workout', requireAuth, workoutRouter);

const buildDir = path.resolve(__dirname, '..', 'build');
app.use(express.static(buildDir));

app.get('*', (_req, res) => {
    res.sendFile(path.join(buildDir, 'index.html'));
});

app.listen(env.PORT, () => {
    console.log(`[server] listening on :${env.PORT} (${env.NODE_ENV})`);
});

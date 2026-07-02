# CLAUDE.md

## Project Overview

Personal blog — React 18 (CRA) + Express.js + TypeScript, deployed via Docker to Azure App Service (`kennethjusino-client`). Container registry: `blogger.azurecr.io`. CI triggers on push to `master`.

## Commands

- `npm run build` — production build (output in `build/`)
- `npm start` — CRA dev server (may need `DANGEROUSLY_DISABLE_HOST_CHECK=true`)
- `npx tsx server/index.ts` — run Express server (serves `build/` + API routes)
- `docker compose up` — local dev with Docker

## Architecture

- **Articles**: Static JSX components in `src/articles/`, each exports an `ArticleProps` object. Registered in `src/articles/allData.tsx`.
- **No database**: Personal features use Microsoft Excel via Graph API.
- **Auth**: Custom HMAC-SHA256 cookie auth for `/personal` routes.
- **Styling**: Pure CSS with custom properties (`--bg`, `--card-bg`, `--border`, `--accent`, `--text`, `--text-muted`, `--text-dim`). Dark/light theme via `data-theme` attribute.

## Media Features

### Audio (implemented)

Articles can include `audioSrc?: string` pointing to an MP3/M4A in `public/audio/`. The `AudioPlayer` component renders play/pause, seek, time, and speed toggle using native HTML5 `<audio>`.

### Video (implemented — Azure Blob Storage live)

Articles can include `videoSrc?: string` pointing to an Azure Blob Storage URL. The `VideoPlayer` component renders a 16:9 player with click-to-play overlay, seek, time, and fullscreen toggle.

**Azure Blob Storage — already provisioned (hardened).** Account `kennethjusinoblog`
(East US, StorageV2, Standard_LRS) in the `kennethjusino.com` resource group:

- HTTPS-only, min TLS 1.2.
- Container `videos` at `blob` public access — anonymous **read of known URLs only,
  no directory listing**. Nothing else in the account is public.
- CORS scoped to the real origins (`https://kennethjusino.com`,
  `https://www.kennethjusino.com`, `https://kennethjusino-client.azurewebsites.net`);
  GET/HEAD/OPTIONS; exposes `Content-Range,Content-Length,Accept-Ranges` (range
  requests for seeking verified working — 206 Partial Content).
- Blob soft-delete enabled, 7-day retention (accidental-deletion safety net).

The setup is done — you only need to upload videos and reference them. To upload
(uses your Azure login, no account key needed):

```bash
az storage blob upload \
  --account-name kennethjusinoblog --container-name videos \
  --name video-name.mp4 --file ./path/to/video.mp4 \
  --content-type video/mp4 --auth-mode login
```

Video URL pattern: `https://kennethjusinoblog.blob.core.windows.net/videos/<filename>.mp4`

Usage in an article file:
```typescript
videoSrc: 'https://kennethjusinoblog.blob.core.windows.net/videos/my-video.mp4',
```

> These are **public** blog videos (no confidentiality). For private/gated video
> you'd need a private container + server-minted SAS URLs + a VideoPlayer change —
> a different design, not what's set up here.

### Analytics (implemented)

Tracks visitor engagement across all articles and stores data in Microsoft Excel via Graph API.

**What's tracked** (all automatic, no per-article wiring needed):
- `view` — page load (deduplicated per session)
- `audio_play` / `audio_complete` — audio playback start and finish
- `video_play` / `video_complete` — video playback start and finish
- `read_seconds` — time spent on page (sent on tab switch or navigation)
- `referrer` — external hostname that linked to the page
- `device` — desktop / mobile / tablet (from user agent)
- `session_id` — random UUID per browser session (`sessionStorage`)

**Architecture:**

```
Browser (public)                    Express Server                    Microsoft Graph
─────────────────                   ──────────────                    ───────────────
Blog.tsx         ──► POST /api/analytics/event ──► in-memory buffer
AudioPlayer.tsx  ──►   (no auth, rate-limited)      │
VideoPlayer.tsx  ──►                                 ├──► flush every 2min ──► Excel append
                                                     │    or when buffer ≥ 50
Dashboard        ◄── GET /api/personal/analytics ◄──────── Excel read ◄── /PersonalApps/blog-analytics.xlsx
(requireAuth)         /summary
```

**Server files:**
- `server/analytics/excel.ts` — Graph API read/write for the analytics workbook
- `server/analytics/buffer.ts` — In-memory event buffer with timed flush (2min / 50 events)
- `server/analytics/cache.ts` — 5-min TTL cache for dashboard reads; aggregates per-route stats
- `server/routes/analytics.ts` — Public `POST /api/analytics/event` (rate-limited: 30 req/min/IP) + auth'd `GET /api/personal/analytics/summary` and `POST /api/personal/analytics/refresh`

**Client files:**
- `src/analytics/tracker.ts` — `trackEvent()` using `navigator.sendBeacon()` (fire-and-forget), with per-session dedup
- `src/analytics/usePageTracking.ts` — React hook for views + read time (called in `Blog.tsx`)

**Dashboard:**
- `src/personal/analytics/Analytics.tsx` — Summary cards (views, sessions, audio plays, video plays), per-article table, top referrers, device breakdown
- `src/personal/analytics/analytics.css` — Dashboard styles
- `src/personal/analytics/api.ts` — Fetch helpers
- Route: `/personal/analytics` (behind auth)

**Excel workbook setup (provisioned):**

`/PersonalApps/blog-analytics.xlsx` exists in OneDrive with an `Events` table and these column headers:

| timestamp | session_id | event | route | referrer | device | read_seconds |
|-----------|------------|-------|-------|----------|--------|--------------|

It was created with `npx tsx scripts/create-analytics-workbook.ts` (idempotent — re-run to recreate the workbook/table if it's ever deleted). The existing Graph API credentials already have `Files.ReadWrite` scope — no new permissions needed.

**How tracking hooks in:**
- `Blog.tsx` calls `usePageTracking(route)` — tracks view on mount, sends read time on unmount/tab-switch
- `AudioPlayer.tsx` and `VideoPlayer.tsx` accept a `route` prop and call `trackEvent()` on play/ended events
- Blog.tsx passes `route` down to both media players

**Rate limiting:** Simple in-memory Map per IP, 30 requests per 60-second window. No npm dependency (no `express-rate-limit`).

**No cookies or fingerprinting.** Session ID lives in `sessionStorage` (cleared when the tab closes). Referrer is the external hostname only (same-site navigations are ignored).

### Podcast (implemented — infrastructure ready)

Podcast hosting infrastructure for distribution via Apple Podcasts and Spotify.

**Architecture:**
- Episode metadata lives in `src/podcast/episodes.ts` (single source of truth for the SPA)
- `public/feed.xml` is the RSS 2.0 feed with iTunes/podcast namespace extensions — Apple and Spotify poll this
- Audio files are served from Azure Blob Storage (`kennethjusinoblog.blob.core.windows.net/podcast/`)
- The `/podcast` page lists all episodes with embedded audio players

**Azure Blob Storage — reuses the existing `kennethjusinoblog` account.** Requires a `podcast` container with `blob`-level public access (same as the `videos` container). CORS should be configured identically to the video setup.

```bash
# Create the podcast container (one-time)
az storage container create \
  --account-name kennethjusinoblog --name podcast \
  --public-access blob --auth-mode login

# Upload an episode
az storage blob upload \
  --account-name kennethjusinoblog --container-name podcast \
  --name episodes/ep001.mp3 --file ./path/to/episode.mp3 \
  --content-type audio/mpeg --auth-mode login

# Upload artwork (3000x3000 JPEG or PNG required by Apple)
az storage blob upload \
  --account-name kennethjusinoblog --container-name podcast \
  --name artwork/cover.jpg --file ./path/to/cover.jpg \
  --content-type image/jpeg --auth-mode login
```

**Episode publish workflow:**
1. Upload `.mp3` to Azure Blob (`podcast/episodes/epXXX.mp3`)
2. Get the blob URL and exact file size in bytes
3. Add the episode to `src/podcast/episodes.ts` (for the SPA page)
4. Add a matching `<item>` block to `public/feed.xml` (for Apple/Spotify)
5. Deploy — Apple and Spotify auto-poll and pick up the new episode

**Client files:**
- `src/podcast/episodes.ts` — `PodcastEpisode` interface and episodes array
- `src/podcast/Podcast.tsx` — Podcast listing page with embedded players
- `src/podcast/podcast.css` — Podcast page styles
- `public/feed.xml` — RSS 2.0 feed

**Route:** `/podcast` (public, in NavBar)

**Directory submission (one-time, after first episode is published):**
- Apple Podcasts: submit `https://kennethjusino.com/feed.xml` at podcastsconnect.apple.com
- Spotify: submit same URL at podcasters.spotify.com

**Constraints:**
- `enclosure` `length` in feed.xml must be exact file size in bytes
- `guid` must never change after publishing
- `pubDate` must be RFC 2822 format (e.g. `Thu, 02 Jul 2026 12:00:00 GMT`)
- Artwork must be 3000x3000px, JPEG or PNG, HTTPS URL

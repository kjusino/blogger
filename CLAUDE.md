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

### Video (implemented, needs Azure setup)

Articles can include `videoSrc?: string` pointing to an Azure Blob Storage URL. The `VideoPlayer` component renders a 16:9 player with click-to-play overlay, seek, time, and fullscreen toggle.

**Azure Blob Storage setup (run with `az` CLI):**

```bash
# Find the resource group
az webapp show --name kennethjusino-client --query resourceGroup -o tsv

# Create storage account
az storage account create \
  --name kennethjusinoblog \
  --resource-group <RESOURCE_GROUP> \
  --location eastus --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2

# Create public-read container
az storage container create \
  --name videos --account-name kennethjusinoblog --public-access blob

# Configure CORS for browser playback
az storage cors add \
  --account-name kennethjusinoblog --services b \
  --methods GET HEAD OPTIONS \
  --origins "https://kennethjusino-client.azurewebsites.net" \
  --allowed-headers "*" \
  --exposed-headers "Content-Range,Content-Length,Accept-Ranges" \
  --max-age 3600

# Upload a video
az storage blob upload \
  --account-name kennethjusinoblog --container-name videos \
  --name video-name.mp4 --file ./path/to/video.mp4 --content-type video/mp4
```

Video URL pattern: `https://kennethjusinoblog.blob.core.windows.net/videos/<filename>.mp4`

Usage in an article file:
```typescript
videoSrc: 'https://kennethjusinoblog.blob.core.windows.net/videos/my-video.mp4',
```

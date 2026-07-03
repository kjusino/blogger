import fs from 'fs';
import path from 'path';

interface OgEntry {
    title: string;
    description: string;
    image: string;
}

const BASE_URL = process.env.BASE_URL || 'https://kennethjusino.com';
const SITE_NAME = 'Your Fave Place To Learn';
const HEAD_CLOSE = '</head>';

function escapeAttr(s: string): string {
    return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function injectOgTags(buildDir: string): (html: string, reqPath: string) => string {
    const manifestPath = path.join(buildDir, 'og-manifest.json');
    const manifest: Record<string, OgEntry> = fs.existsSync(manifestPath)
        ? JSON.parse(fs.readFileSync(manifestPath, 'utf-8'))
        : {};

    return (html: string, reqPath: string): string => {
        const entry = manifest[reqPath];

        const title = entry?.title || SITE_NAME;
        const description = entry?.description || 'Math, Code, and Culture — by Kenneth Jusino';
        const url = `${BASE_URL}${reqPath}`;
        const ogType = entry && reqPath !== '/' ? 'article' : 'website';

        const tags = [
            `<meta property="og:type" content="${ogType}" />`,
            `<meta property="og:title" content="${escapeAttr(title)}" />`,
            `<meta property="og:description" content="${escapeAttr(description)}" />`,
            `<meta property="og:url" content="${escapeAttr(url)}" />`,
            `<meta property="og:site_name" content="${escapeAttr(SITE_NAME)}" />`,
        ];

        if (entry) {
            const imageUrl = `${BASE_URL}${entry.image}`;
            tags.push(`<meta property="og:image" content="${escapeAttr(imageUrl)}" />`);
            tags.push(`<meta name="twitter:card" content="summary_large_image" />`);
            tags.push(`<meta name="twitter:title" content="${escapeAttr(title)}" />`);
            tags.push(`<meta name="twitter:description" content="${escapeAttr(description)}" />`);
            tags.push(`<meta name="twitter:image" content="${escapeAttr(imageUrl)}" />`);
        }

        return html.replace(HEAD_CLOSE, tags.join('') + HEAD_CLOSE);
    };
}

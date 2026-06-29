import { articleMeta } from './shared/articleMeta';

const SITE_URL = 'https://kennethjusino.com';
const SITE_NAME = 'Kenneth Jusino';
const DEFAULT_TITLE = 'Kenneth Jusino — Math ∩ Programming ∩ Science ∩ Culture';
const DEFAULT_DESC = 'Articles on math, programming, science, and culture by Kenneth Jusino.';
const DEFAULT_IMAGE = `${SITE_URL}/og-images/muralPic.png`;

const metaByRoute = new Map(articleMeta.map((a) => [a.route, a]));

function esc(s: string): string {
    return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function injectMetaTags(html: string, requestPath: string): string {
    const article = metaByRoute.get(requestPath);

    const title = article ? `${article.title} — ${SITE_NAME}` : DEFAULT_TITLE;
    const description = article ? esc(article.description) : DEFAULT_DESC;
    const imageUrl = article ? `${SITE_URL}/og-images/${article.pic}` : DEFAULT_IMAGE;
    const url = `${SITE_URL}${requestPath}`;
    const ogType = article && article.createdDate ? 'article' : 'website';

    const metaTags = `
    <meta property="og:title" content="${esc(title)}" />
    <meta property="og:description" content="${description}" />
    <meta property="og:image" content="${imageUrl}" />
    <meta property="og:url" content="${url}" />
    <meta property="og:type" content="${ogType}" />
    <meta property="og:site_name" content="${SITE_NAME}" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="${esc(title)}" />
    <meta name="twitter:description" content="${description}" />
    <meta name="twitter:image" content="${imageUrl}" />
    <meta name="description" content="${description}" />`;

    let result = html.replace(
        /<title>[^<]*<\/title>/,
        `<title>${esc(title)}</title>`
    );

    result = result.replace(
        /<meta\s+name="description"\s+content="[^"]*"\s*\/?>/,
        ''
    );

    result = result.replace('</head>', `${metaTags}\n    </head>`);

    return result;
}

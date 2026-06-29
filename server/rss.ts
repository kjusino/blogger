import { articleMeta } from './shared/articleMeta';

const SITE_URL = 'https://kennethjusino.com';
const FEED_TITLE = 'Kenneth Jusino';
const FEED_DESCRIPTION = 'Math ∩ Programming ∩ Science ∩ Culture';

function escapeXml(s: string): string {
    return s
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&apos;');
}

export function generateRss(): string {
    const posts = articleMeta
        .filter((a) => a.createdDate)
        .sort((a, b) => (b.createdDate! > a.createdDate! ? 1 : -1));

    const items = posts.map((a) => {
        const pubDate = new Date(a.createdDate + 'T00:00:00Z').toUTCString();
        const url = `${SITE_URL}${a.route}`;
        const imageUrl = `${SITE_URL}/og-images/${a.pic}`;

        let enclosure = '';
        if (a.audioSrc) {
            enclosure = `<enclosure url="${SITE_URL}${a.audioSrc}" type="audio/mp4" />`;
        }

        return `    <item>
      <title>${escapeXml(a.title)}</title>
      <link>${url}</link>
      <guid isPermaLink="true">${url}</guid>
      <description>${escapeXml(a.description)}</description>
      <pubDate>${pubDate}</pubDate>
      ${enclosure}
      <itunes:image href="${imageUrl}" />
      <itunes:summary>${escapeXml(a.description)}</itunes:summary>
    </item>`;
    });

    return `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
     xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
     xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>${escapeXml(FEED_TITLE)}</title>
    <link>${SITE_URL}</link>
    <description>${escapeXml(FEED_DESCRIPTION)}</description>
    <language>en-us</language>
    <lastBuildDate>${new Date().toUTCString()}</lastBuildDate>
    <atom:link href="${SITE_URL}/rss.xml" rel="self" type="application/rss+xml" />
    <itunes:author>Kenneth Jusino</itunes:author>
    <itunes:image href="${SITE_URL}/og-images/muralPic.png" />
    <itunes:category text="Technology" />
${items.join('\n')}
  </channel>
</rss>`;
}

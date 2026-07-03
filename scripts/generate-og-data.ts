import fs from 'fs';
import path from 'path';

const ARTICLES_DIR = path.resolve(__dirname, '..', 'src', 'articles');
const PICS_DIR = path.join(ARTICLES_DIR, 'pics');
const BUILD_DIR = path.resolve(__dirname, '..', 'build');
const OG_DIR = path.join(BUILD_DIR, 'og');

interface OgEntry {
    title: string;
    description: string;
    image: string;
}

function extractField(source: string, field: string): string | undefined {
    // Match field: 'value' or field: "value", handling embedded quotes of the other type
    const doubleQuoted = new RegExp(`${field}:\\s*"([^"]+)"`);
    const singleQuoted = new RegExp(`${field}:\\s*'([^']+)'`);

    // Also handle multi-line: field:\n        "value"
    const doubleMulti = new RegExp(`${field}:\\s*\\n\\s*"([^"]+)"`);
    const singleMulti = new RegExp(`${field}:\\s*\\n\\s*'([^']+)'`);

    for (const re of [doubleQuoted, singleQuoted, doubleMulti, singleMulti]) {
        const m = source.match(re);
        if (m) return m[1];
    }

    return undefined;
}

function extractFirstPic(source: string): string | undefined {
    const m = source.match(/pics:\s*\[['"]([^'"]+)['"]/);
    return m ? m[1] : undefined;
}

const allDataSource = fs.readFileSync(path.join(ARTICLES_DIR, 'allData.tsx'), 'utf-8');
const importMatches = [...allDataSource.matchAll(/import\s+\w+\s+from\s+['"]\.\/([\w]+)['"]/g)];
const articleFiles = importMatches.map(m => m[1] + '.tsx');

fs.mkdirSync(OG_DIR, { recursive: true });

const manifest: Record<string, OgEntry> = {};

for (const file of articleFiles) {
    const filePath = path.join(ARTICLES_DIR, file);
    if (!fs.existsSync(filePath)) continue;

    const source = fs.readFileSync(filePath, 'utf-8');

    let route = extractField(source, 'route');
    if (!route) continue;
    if (!route.startsWith('/')) route = '/' + route;

    const title = extractField(source, 'title');
    if (!title) continue;

    const pic = extractFirstPic(source);
    if (!pic) continue;

    const description = extractField(source, 'abstract') || extractField(source, 'caption') || title;

    const srcImage = path.join(PICS_DIR, pic);
    if (!fs.existsSync(srcImage)) {
        console.warn(`[og] image not found: ${srcImage}, skipping ${route}`);
        continue;
    }

    fs.copyFileSync(srcImage, path.join(OG_DIR, pic));

    manifest[route] = {
        title,
        description,
        image: `/og/${pic}`,
    };
}

fs.writeFileSync(path.join(BUILD_DIR, 'og-manifest.json'), JSON.stringify(manifest, null, 2));
console.log(`[og] generated manifest with ${Object.keys(manifest).length} entries`);

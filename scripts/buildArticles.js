const fs = require('fs');
const path = require('path');
const matter = require('gray-matter');

/**
 * Build script that converts markdown articles to importable JSON
 * Run this before build/start to generate article data
 */

const CONTENT_DIR = path.join(__dirname, '../src/articles/content');
const OUTPUT_FILE = path.join(__dirname, '../src/articles/generatedArticles.json');

function buildArticles() {
    console.log('🔍 Discovering markdown articles...');

    // Ensure content directory exists
    if (!fs.existsSync(CONTENT_DIR)) {
        fs.mkdirSync(CONTENT_DIR, { recursive: true });
        console.log('✅ Created content directory');
    }

    const articles = [];
    const files = fs.readdirSync(CONTENT_DIR).filter(f => f.endsWith('.md'));

    console.log(`📄 Found ${files.length} markdown files`);

    files.forEach(file => {
        const filePath = path.join(CONTENT_DIR, file);
        const fileContent = fs.readFileSync(filePath, 'utf-8');

        // Parse frontmatter
        const { data, content } = matter(fileContent);

        // Validate required fields
        if (!data.route || !data.title) {
            console.warn(`⚠️  Skipping ${file}: missing required fields (route, title)`);
            return;
        }

        articles.push({
            route: data.route,
            title: data.title,
            abstract: data.abstract || '',
            pics: data.pics || [],
            caption: data.caption || '',
            backgroundColor: data.backgroundColor || '',
            textColor: data.textColor || '',
            tags: data.tags || [],
            createdDate: data.createdDate || new Date().toISOString(),
            content: content.trim(),
        });

        console.log(`✅ Loaded: ${data.title} (${data.route})`);
    });

    // Sort by creation date (newest first)
    articles.sort((a, b) => {
        return new Date(b.createdDate).getTime() - new Date(a.createdDate).getTime();
    });

    // Write to JSON file
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(articles, null, 2));
    console.log(`\n✨ Generated ${articles.length} articles → ${OUTPUT_FILE}`);
}

// Run the build
buildArticles();

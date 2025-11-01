# How to Add New Articles

Articles are now written in Markdown format with YAML frontmatter for metadata.

## Creating a New Article

1. Create a new `.md` file in this directory (`src/articles/content/`)
2. Add frontmatter with metadata at the top
3. Write your content in Markdown
4. Run `npm start` or `npm build` - articles are auto-discovered!

## Article Template

```markdown
---
route: /your-article-slug
title: Your Article Title
abstract: Short description (optional)
pics:
  - image1.png
  - image2.png
caption: Image caption
tags:
  - Math
  - Computation
  - Physics
  - Culture
createdDate: 2024-01-15
---

## Your First Heading

Your content here with **markdown formatting**...

### Subsection

- Bullet points
- Work great

You can also include [links](https://example.com) and code blocks:

\```typescript
function example() {
  console.log("Hello, world!");
}
\```
```

## Available Tags

- Math
- Computation
- Physics
- Culture

## Images

Place images in `src/articles/pics/` and reference them by filename in the `pics` array.

## Auto-Discovery

No need to manually register articles! The build script (`scripts/buildArticles.js`) automatically discovers all `.md` files in this directory and generates the article data at build time.

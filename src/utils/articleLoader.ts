import { Tags } from '../resources/enums/Tags';
import generatedArticles from '../articles/generatedArticles.json';

export interface MarkdownArticle {
    route: string;
    title: string;
    abstract?: string;
    pics: string[];
    caption: string;
    backgroundColor?: string;
    textColor?: string;
    tags?: Tags[];
    createdDate?: string;
    content: string; // Markdown content as string
}

/**
 * Load all articles from the generated JSON file
 * Articles are auto-discovered at build time via scripts/buildArticles.js
 */
export function loadAllArticles(): MarkdownArticle[] {
    return generatedArticles as MarkdownArticle[];
}

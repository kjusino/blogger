import { relatedArticles } from './relatedArticles';
import { ArticleProps } from './interfaces/ArticleProps';
import { Tags } from './enums/Tags';

// Minimal fixture — only the fields the ranker reads.
function article(p: Partial<ArticleProps> & { route: string }): ArticleProps {
    return {
        title: p.route,
        pics: ['x.jpg'],
        caption: '',
        content: [],
        ...p,
    } as ArticleProps;
}

const lean = article({ route: '/lean', tags: [Tags.Computation, Tags.Math], createdDate: '2026-06-27' });
const phd = article({ route: '/phd', tags: [Tags.Math, Tags.Computation, Tags.Physics], createdDate: '2025-01-01' });
const rust = article({ route: '/rust', tags: [Tags.Computation], createdDate: '2025-11-01' });
const ai = article({ route: '/ai-engineering', tags: [Tags.Computation], createdDate: '2025-03-01' });
const about = article({ route: '/about', tags: [Tags.Culture], createdDate: '2024-01-01' });
const home = article({ route: '/', createdDate: '2020-01-01' });
const cv = article({ route: '/cv', createdDate: '2020-01-01' });

const all = [lean, phd, rust, ai, about, home, cv];

describe('relatedArticles', () => {
    it('ranks by tag overlap, descending', () => {
        const result = relatedArticles(lean, all, 2);
        // /phd shares 2 tags (Math, Computation) — must rank first.
        expect(result[0].route).toBe('/phd');
    });

    it('breaks ties by most recent date', () => {
        // /rust and /ai-engineering both share 1 tag; /rust is newer.
        const result = relatedArticles(lean, all, 2);
        expect(result[1].route).toBe('/rust');
    });

    it('excludes the current article', () => {
        const routes = relatedArticles(lean, all, 5).map((a) => a.route);
        expect(routes).not.toContain('/lean');
    });

    it('excludes non-blog pages (home, cv)', () => {
        const routes = relatedArticles(lean, all, 10).map((a) => a.route);
        expect(routes).not.toContain('/');
        expect(routes).not.toContain('/cv');
    });

    it('excludes hidden articles', () => {
        const hidden = article({ route: '/secret', tags: [Tags.Computation, Tags.Math], hidden: true });
        const routes = relatedArticles(lean, [...all, hidden], 10).map((a) => a.route);
        expect(routes).not.toContain('/secret');
    });

    it('returns at most `count` items', () => {
        expect(relatedArticles(lean, all, 2)).toHaveLength(2);
    });

    it('fills remaining slots by recency when overlap runs out', () => {
        // current shares no tags with anything → pure recency order.
        const culture = article({ route: '/culture', tags: [Tags.Culture], createdDate: '2026-01-01' });
        const result = relatedArticles(culture, [culture, phd, rust, ai], 2);
        // rust (2025-11) newer than ai (2025-03) and phd (2025-01)
        expect(result.map((a) => a.route)).toEqual(['/rust', '/ai-engineering']);
    });
});

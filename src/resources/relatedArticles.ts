import { ArticleProps } from './interfaces/ArticleProps';
import { NON_BLOG_ROUTES } from './routes';

/**
 * Pick the articles most worth surfacing at the end of `current`.
 *
 * Ranking: most shared tags first; ties broken by most-recent `createdDate`.
 * Excludes the current article, non-blog pages (home/cv/personal), and hidden
 * articles. When tag overlap runs out, remaining slots fill by recency so the
 * footer always offers `count` reads when that many candidates exist.
 */
export function relatedArticles(
    current: ArticleProps,
    all: ArticleProps[],
    count = 2
): ArticleProps[] {
    const currentTags = new Set(current.tags ?? []);

    return all
        .filter(
            (a) =>
                a.route !== current.route &&
                !a.hidden &&
                !NON_BLOG_ROUTES.has(a.route)
        )
        .map((a) => ({
            article: a,
            overlap: (a.tags ?? []).filter((t) => currentTags.has(t)).length,
            date: a.createdDate ?? '',
        }))
        .sort((x, y) =>
            y.overlap !== x.overlap
                ? y.overlap - x.overlap
                : y.date.localeCompare(x.date)
        )
        .slice(0, count)
        .map((s) => s.article);
}

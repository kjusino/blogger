import { Tags } from './enums/Tags';

// Single source of truth for theme colors, shared across the knowledge
// graph, the article list, and the per-article header.
export const THEME_COLOR: Record<Tags, string> = {
    [Tags.Math]: '#fb7185', // red
    [Tags.Computation]: '#f59e0b', // amber
    [Tags.Physics]: '#4ade80', // green
    [Tags.Culture]: '#22d3ee', // blue
};

// Canonical render order, so a given combination of themes always produces
// the same pie (e.g. math+cs is always cyan-then-amber, never the reverse).
export const THEME_ORDER: Tags[] = [
    Tags.Math,
    Tags.Computation,
    Tags.Physics,
    Tags.Culture,
];

const FALLBACK = '#888888';

/** Theme colors for a set of tags, in canonical order. */
export function orderedThemeColors(tags?: Tags[]): string[] {
    if (!tags || tags.length === 0) return [FALLBACK];
    const set = new Set(tags);
    return THEME_ORDER.filter((t) => set.has(t)).map((t) => THEME_COLOR[t]);
}

/**
 * A CSS background value: a solid color for a single theme, or a hard-stop
 * conic-gradient (a crisp pie) for multiple. Each theme keeps its own color
 * instead of blending into an ambiguous new hue.
 */
export function conicPie(tags?: Tags[]): string {
    const colors = orderedThemeColors(tags);
    if (colors.length === 1) return colors[0];
    const slice = 100 / colors.length;
    const stops = colors
        .map((c, i) => `${c} ${(i * slice).toFixed(4)}% ${((i + 1) * slice).toFixed(4)}%`)
        .join(', ');
    return `conic-gradient(${stops})`;
}

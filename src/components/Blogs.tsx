import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useTheme } from '../context/ThemeContext';

// ── Types ────────────────────────────────────────────────────────────────────

interface GraphNode extends d3.SimulationNodeDatum {
    id: string;
    label: string;
    type: 'root' | 'theme' | 'post';
    route?: string;
    theme?: string;
    color: string;
    r: number;
    labelLines?: string[];
}

interface RawLink {
    source: string;
    target: string;
}

// ── Graph data ───────────────────────────────────────────────────────────────

const THEME_COLORS: Record<string, string> = {
    math: '#22d3ee',
    cs: '#f59e0b',
    physics: '#a78bfa',
    culture: '#fb7185',
};

const NODES: GraphNode[] = [
    // Root
    { id: 'kenneth', label: 'Kenneth', type: 'root', r: 30, color: '#ffffff' },

    // Theme nodes
    {
        id: 'math',
        label: 'Math',
        type: 'theme',
        r: 20,
        color: THEME_COLORS.math,
        theme: 'math',
        labelLines: ['MATH'],
    },
    {
        id: 'cs',
        label: 'Computer Science',
        type: 'theme',
        r: 20,
        color: THEME_COLORS.cs,
        theme: 'cs',
        labelLines: ['COMPUTER', 'SCIENCE'],
    },
    {
        id: 'physics',
        label: 'Physics',
        type: 'theme',
        r: 20,
        color: THEME_COLORS.physics,
        theme: 'physics',
        labelLines: ['PHYSICS'],
    },
    {
        id: 'culture',
        label: 'Culture',
        type: 'theme',
        r: 20,
        color: THEME_COLORS.culture,
        theme: 'culture',
        labelLines: ['CULTURE'],
    },

    // Blog post leaf nodes
    {
        id: 'phd',
        label: 'Quantum Computation',
        type: 'post',
        route: '/phd',
        r: 11,
        color: THEME_COLORS.math,
        theme: 'math',
    },
    {
        id: 'rust',
        label: 'Getting Rusty',
        type: 'post',
        route: '/rust',
        r: 11,
        color: THEME_COLORS.cs,
        theme: 'cs',
    },
    {
        id: 'ai_engineering',
        label: 'AI Engineering',
        type: 'post',
        route: '/ai-engineering',
        r: 11,
        color: THEME_COLORS.cs,
        theme: 'cs',
    },
    {
        id: 'about',
        label: 'About Me',
        type: 'post',
        route: '/about',
        r: 11,
        color: THEME_COLORS.culture,
        theme: 'culture',
    },
    {
        id: 'pa',
        label: 'Happy Fathers Day, Papi!',
        type: 'post',
        route: 'pa',
        r: 11,
        color: THEME_COLORS.culture,
        theme: 'culture',
    },
];

const RAW_LINKS: RawLink[] = [
    { source: 'kenneth', target: 'math' },
    { source: 'kenneth', target: 'cs' },
    { source: 'kenneth', target: 'physics' },
    { source: 'kenneth', target: 'culture' },
    { source: 'math', target: 'phd' },
    { source: 'physics', target: 'phd' },
    { source: 'cs', target: 'rust' },
    { source: 'cs', target: 'ai_engineering' },
    { source: 'culture', target: 'about' },
    { source: 'culture', target: 'pa' },
];

// ── Component ────────────────────────────────────────────────────────────────

const Blogs = () => {
    const svgRef = useRef<SVGSVGElement>(null);
    const { isDark, toggleTheme } = useTheme();

    useEffect(() => {
        if (!svgRef.current) return;

        // Theme-derived colors for D3 (SVG attrs don't support CSS vars)
        const linkColor = isDark ? '#2c2c2c' : '#d0d0d0';
        const rootCircleFill = isDark ? '#ffffff' : '#1a1a1a';
        const kInitialsFill = isDark ? '#0d0d0d' : '#f8f8f5';
        const kennethLabelFill = isDark ? '#e8e8e8' : '#1a1a1a';
        const postLabelFill = isDark ? '#888888' : '#555555';
        const themeIcon = isDark ? '☀️' : '🌙';

        const svgEl = svgRef.current;
        const width = window.innerWidth;
        const height = window.innerHeight - 140;

        // Clone data so simulation can mutate freely
        const nodes: GraphNode[] = NODES.map((n) => ({ ...n }));
        const links: d3.SimulationLinkDatum<GraphNode>[] = RAW_LINKS.map(
            (l) => ({
                ...l,
            }),
        );

        // ── SVG setup ──────────────────────────────────────────────────────
        const svg = d3.select(svgEl);
        svg.selectAll('*').remove();

        // Glow filter
        const defs = svg.append('defs');
        const glowFilter = defs.append('filter').attr('id', 'glow');
        glowFilter
            .append('feGaussianBlur')
            .attr('stdDeviation', '4')
            .attr('result', 'coloredBlur');
        const feMerge = glowFilter.append('feMerge');
        feMerge.append('feMergeNode').attr('in', 'coloredBlur');
        feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

        // Zoom container
        const g = svg.append('g');

        svg.call(
            d3
                .zoom<SVGSVGElement, unknown>()
                .scaleExtent([0.25, 4])
                .on('zoom', (event) => {
                    g.attr('transform', event.transform);
                }),
        );

        // ── Force simulation ───────────────────────────────────────────────
        const simulation = d3
            .forceSimulation<GraphNode>(nodes)
            .force(
                'link',
                d3
                    .forceLink<GraphNode, d3.SimulationLinkDatum<GraphNode>>(
                        links,
                    )
                    .id((d) => d.id)
                    .distance((l) => {
                        const s = l.source as GraphNode;
                        const t = l.target as GraphNode;
                        if (s.type === 'root' || t.type === 'root') return 160;
                        if (s.type === 'theme' || t.type === 'theme')
                            return 110;
                        return 80;
                    })
                    .strength(0.7),
            )
            .force(
                'charge',
                d3.forceManyBody<GraphNode>().strength((d) => {
                    if (d.type === 'root') return -700;
                    if (d.type === 'theme') return -250;
                    return -100;
                }),
            )
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force(
                'collision',
                d3.forceCollide<GraphNode>().radius((d) => d.r + 24),
            );

        // ── Links ──────────────────────────────────────────────────────────
        const linkSel = g
            .append('g')
            .selectAll<
                SVGLineElement,
                d3.SimulationLinkDatum<GraphNode>
            >('line')
            .data(links)
            .join('line')
            .attr('stroke', linkColor)
            .attr('stroke-width', 1.5)
            .attr('stroke-opacity', 0.9);

        // ── Node groups ────────────────────────────────────────────────────
        const nodeGroup = g
            .append('g')
            .selectAll<SVGGElement, GraphNode>('g')
            .data(nodes)
            .join('g')
            .attr('cursor', (d) =>
                d.type === 'post' || d.type === 'root' ? 'pointer' : 'default',
            )
            .call(
                d3
                    .drag<SVGGElement, GraphNode>()
                    .on('start', (event, d) => {
                        if (!event.active)
                            simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    })
                    .on('drag', (event, d) => {
                        d.fx = event.x;
                        d.fy = event.y;
                    })
                    .on('end', (event, d) => {
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    }),
            );

        // Circles
        nodeGroup
            .append('circle')
            .attr('r', (d) => d.r)
            .attr('fill', (d) => {
                if (d.type === 'root') return rootCircleFill;
                if (d.type === 'theme') return d.color + '1a'; // 10% opacity fill
                return d.color; // fully opaque solid fill
            })
            .attr('stroke', (d) => (d.type === 'root' ? 'none' : d.color))
            .attr('stroke-width', (d) => (d.type === 'theme' ? 2 : 1.5));

        // Initials inside Kenneth node
        nodeGroup
            .filter((d) => d.type === 'root')
            .append('text')
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'central')
            .attr('fill', kInitialsFill)
            .attr('font-size', '14px')
            .attr('font-weight', '700')
            .attr(
                'font-family',
                "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            )
            .attr('pointer-events', 'none')
            .text('KJ');

        // Labels
        nodeGroup.each(function (d) {
            const grp = d3.select<SVGGElement, GraphNode>(this);

            if (d.type === 'root') {
                // "Kenneth ☀️/🌙" below circle
                grp.append('text')
                    .attr('text-anchor', 'middle')
                    .attr('dy', d.r + 17)
                    .attr('fill', kennethLabelFill)
                    .attr('font-size', '15px')
                    .attr('font-weight', '700')
                    .attr(
                        'font-family',
                        "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    )
                    .attr('pointer-events', 'none')
                    .text(`Kenneth ${themeIcon}`);
            } else if (d.type === 'theme') {
                // Letter-spaced small-caps label below
                const lines = d.labelLines ?? [d.label.toUpperCase()];
                const textEl = grp
                    .append('text')
                    .attr('text-anchor', 'middle')
                    .attr('fill', d.color)
                    .attr('font-size', '12px')
                    .attr('font-weight', '700')
                    .attr('letter-spacing', '0.1em')
                    .attr(
                        'font-family',
                        "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    )
                    .attr('pointer-events', 'none');

                lines.forEach((line, i) => {
                    textEl
                        .append('tspan')
                        .attr('x', 0)
                        .attr('dy', i === 0 ? d.r + 16 : '1.3em')
                        .text(line);
                });
            } else {
                // Post label — short, below node
                const short =
                    d.label.length > 22 ? d.label.slice(0, 20) + '…' : d.label;
                grp.append('text')
                    .attr('text-anchor', 'middle')
                    .attr('dy', d.r + 14)
                    .attr('fill', postLabelFill)
                    .attr('font-size', '11.5px')
                    .attr(
                        'font-family',
                        "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    )
                    .attr('pointer-events', 'none')
                    .text(short);
            }
        });

        // Tooltip on all nodes
        nodeGroup
            .append('title')
            .text((d) =>
                d.type === 'root' ? `Kenneth — click to toggle theme` : d.label,
            );

        // ── Hover interaction ──────────────────────────────────────────────
        nodeGroup
            .on('mouseenter', function (_, d) {
                const connectedIds = new Set<string>([d.id]);

                links.forEach((l) => {
                    const srcId = (l.source as GraphNode).id;
                    const tgtId = (l.target as GraphNode).id;
                    if (srcId === d.id) connectedIds.add(tgtId);
                    if (tgtId === d.id) connectedIds.add(srcId);
                });

                nodeGroup
                    .transition()
                    .duration(180)
                    .style('opacity', (n) =>
                        connectedIds.has(n.id) ? 1 : 0.12,
                    );

                linkSel
                    .transition()
                    .duration(180)
                    .attr('stroke', (l) => {
                        const s = (l.source as GraphNode).id;
                        const t = (l.target as GraphNode).id;
                        return connectedIds.has(s) && connectedIds.has(t)
                            ? d.color
                            : linkColor;
                    })
                    .attr('stroke-opacity', (l) => {
                        const s = (l.source as GraphNode).id;
                        const t = (l.target as GraphNode).id;
                        return connectedIds.has(s) && connectedIds.has(t)
                            ? 1
                            : 0.08;
                    })
                    .attr('stroke-width', (l) => {
                        const s = (l.source as GraphNode).id;
                        const t = (l.target as GraphNode).id;
                        return connectedIds.has(s) && connectedIds.has(t)
                            ? 2
                            : 1.5;
                    });

                d3.select<SVGGElement, GraphNode>(this)
                    .select('circle')
                    .transition()
                    .duration(180)
                    .attr('r', d.r * 1.18)
                    .attr('filter', 'url(#glow)');
            })
            .on('mouseleave', function (_, d) {
                nodeGroup.transition().duration(280).style('opacity', 1);

                linkSel
                    .transition()
                    .duration(280)
                    .attr('stroke', linkColor)
                    .attr('stroke-opacity', 0.9)
                    .attr('stroke-width', 1.5);

                d3.select<SVGGElement, GraphNode>(this)
                    .select('circle')
                    .transition()
                    .duration(280)
                    .attr('r', d.r)
                    .attr('filter', null as unknown as string);
            });

        // ── Click interaction ──────────────────────────────────────────────
        nodeGroup.on('click', function (_, d) {
            if (d.type === 'post' && d.route) {
                window.location.href = d.route;
            } else if (d.type === 'root') {
                toggleTheme();
            }
        });

        // ── Tick ──────────────────────────────────────────────────────────
        simulation.on('tick', () => {
            linkSel
                .attr('x1', (l) => (l.source as GraphNode).x ?? 0)
                .attr('y1', (l) => (l.source as GraphNode).y ?? 0)
                .attr('x2', (l) => (l.target as GraphNode).x ?? 0)
                .attr('y2', (l) => (l.target as GraphNode).y ?? 0);

            nodeGroup.attr(
                'transform',
                (d) => `translate(${d.x ?? 0},${d.y ?? 0})`,
            );
        });

        return () => {
            simulation.stop();
        };
    }, [isDark, toggleTheme]);

    return (
        <div
            style={{
                background: 'var(--bg)',
                minHeight: 'calc(100vh - 70px)',
                display: 'flex',
                flexDirection: 'column',
                width: '100%',
                alignSelf: 'stretch',
            }}
        >
            {/* Header */}
            <div
                style={{
                    padding: '22px 40px 14px',
                    borderBottom: '1px solid var(--border-subtle)',
                    flexShrink: 0,
                }}
            >
                <div
                    style={{
                        fontSize: '10px',
                        fontWeight: 700,
                        letterSpacing: '0.14em',
                        textTransform: 'uppercase',
                        color: 'var(--accent)',
                        marginBottom: 8,
                        fontFamily:
                            "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    }}
                >
                    Knowledge Graph
                </div>
                <h1
                    style={{
                        margin: 0,
                        fontSize: '26px',
                        fontWeight: 700,
                        fontFamily:
                            "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                        color: 'var(--text-bright)',
                        letterSpacing: '-0.02em',
                    }}
                >
                    All Writing
                </h1>
                <p
                    style={{
                        margin: '7px 0 0',
                        fontSize: '12px',
                        color: 'var(--text-muted)',
                        fontFamily:
                            "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                    }}
                >
                    Drag · Scroll to zoom · Click a post to read it · Click
                    Kenneth to toggle theme
                </p>
            </div>

            {/* Graph canvas */}
            <svg
                ref={svgRef}
                style={{
                    flex: 1,
                    width: '100%',
                    height: 'calc(100vh - 200px)',
                    minHeight: '560px',
                    background: 'var(--bg)',
                    display: 'block',
                }}
            />
        </div>
    );
};

export default Blogs;

import { ArticleProps } from '../resources/interfaces/ArticleProps';

// ── shared style tokens ──────────────────────────────────────────────────────
const sh = {
    borderBottom: '2px solid #1e90ff',
    display: 'inline-block' as const,
    paddingBottom: 4,
    marginBottom: 18,
    fontSize: 21,
    fontWeight: 700,
    marginTop: 0,
};

const rowSB = {
    display: 'flex' as const,
    justifyContent: 'space-between' as const,
    alignItems: 'baseline' as const,
    flexWrap: 'wrap' as const,
    gap: 6,
};

const instName = { fontWeight: 600, fontSize: 16 };
const degree = { color: '#ccc', fontSize: 15, marginTop: 3 };
const detail = { color: '#888', fontSize: 13, marginTop: 2 };
const meta = { color: '#666', fontSize: 12 };

const resume: ArticleProps = {
    route: '/cv',
    title: 'Kenneth Jusino',
    pics: ['profilepic.png'],
    caption: 'CS PhD Candidate · NSF CSGrad4US Fellow · AI × Formal Methods',
    content: [
        <div
            style={{
                maxWidth: '700px',
                margin: '0 auto 40px auto',
                fontFamily:
                    "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif",
                fontSize: '16px',
                color: '#e0e0e0',
            }}
        >
            {/* ── Education ───────────────────────────────────────────── */}
            <section style={{ marginBottom: 40 }}>
                <h2 style={sh}>Education</h2>

                {/* PhD */}
                <div style={{ marginBottom: 22 }}>
                    <div style={rowSB}>
                        <span style={instName}>Northeastern University</span>
                        <span style={meta}>Boston, MA · Incoming Fall 2026</span>
                    </div>
                    <div style={degree}>Ph.D. in Computer Science</div>
                    <div style={detail}>
                        Advisor: Stavros Tripakis · Focus: AI-Assisted Formal Methods,
                        Formalized Mathematics, Verified Software Engineering
                    </div>
                    <div style={detail}>Funded by NSF CSGrad4US Fellowship</div>
                </div>

                {/* BU */}
                <div style={{ marginBottom: 4 }}>
                    <div style={rowSB}>
                        <span style={instName}>Boston University</span>
                        <span style={meta}>Boston, MA · 2015–2019</span>
                    </div>
                    <div style={degree}>
                        Bachelor of Arts — Pure &amp; Applied Mathematics
                    </div>
                    <div style={detail}>Minors: Physics, Computer Science</div>
                    <div style={detail}>
                        Graduate-Level Coursework: Partial Differential Equations I–II,
                        Real Analysis I–II, Logic, Differential Geometry
                    </div>
                </div>
            </section>

            {/* ── Fellowships & Awards ─────────────────────────────────── */}
            <section style={{ marginBottom: 40 }}>
                <h2 style={sh}>Fellowships &amp; Awards</h2>

                {/* NSF — typography-only callout */}
                <div
                    style={{
                        borderLeft: '3px solid #666',
                        padding: '14px 18px',
                        marginBottom: 18,
                    }}
                >
                    <div
                        style={{
                            fontSize: 10,
                            fontWeight: 700,
                            letterSpacing: '0.1em',
                            textTransform: 'uppercase',
                            color: '#888',
                            marginBottom: 6,
                        }}
                    >
                        NSF Fellowship · May 2024
                    </div>
                    <div
                        style={{
                            fontSize: 18,
                            fontWeight: 700,
                            color: '#e0e0e0',
                            marginBottom: 8,
                            letterSpacing: '-0.01em',
                        }}
                    >
                        <a
                            href="https://www.nsf.gov/funding/opportunities/dcl-computer-information-science-engineering-graduate"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            National Science Foundation CSGrad4US Fellowship
                        </a>
                    </div>
                    <p style={{ margin: '0 0 6px 0', fontSize: 14, color: '#bbb', lineHeight: 1.65 }}>
                        Selective fellowship — $159,000 stipend covering the first three
                        years of a CISE doctorate, awarded to prospective graduate students
                        with significant industry experience.
                    </p>
                    <p style={{ margin: 0, fontSize: 13, color: '#999', lineHeight: 1.65 }}>
                        Awarded for proposed research in AI-assisted formal methods,
                        trustworthy AI, formalized mathematics, and neurosymbolic software
                        verification.
                    </p>
                </div>

                {/* Perimeter */}
                <div style={{ fontSize: 13, color: '#888', lineHeight: 1.9, paddingLeft: 4, marginBottom: 6 }}>
                    <div>
                        · PSI Start &amp; PSI Bridge Scholar — Perimeter Institute for
                        Theoretical Physics | 2023
                    </div>
                </div>

                <div style={{ fontSize: 13, color: '#888', lineHeight: 1.9, paddingLeft: 4 }}>
                    <div>
                        · Boston Hacks 2018 — First Place, Home Automation Track ($500)
                    </div>
                    <div>
                        · Rhode Island Math League 2015 — Top in School, Most Points
                        ($1,000 scholarship)
                    </div>
                </div>
            </section>

            {/* ── Research Interests ───────────────────────────────────── */}
            <section style={{ marginBottom: 40 }}>
                <h2 style={sh}>Research Interests</h2>
                <p style={{ marginTop: 10, fontSize: 15, lineHeight: 1.75, color: '#ccc' }}>
                    My research sits at the intersection of trustworthy AI, scalable formal
                    methods, formalized mathematics, and provably correct software
                    engineering. I am drawn to the challenge of making mechanically verified
                    reasoning practical at scale — through AI-assisted theorem proving,
                    neurosymbolic methods, and constrained LLM decoding that produces
                    outputs checkable by proof assistants. I am particularly interested in
                    autoformalization: transforming natural-language mathematical and
                    computational artifacts into formal representations amenable to
                    machine-checkable verification. Broader themes: lattice-theoretic
                    semantics, proof assistant tooling (Lean 4), and the theoretical
                    foundations connecting language models to deductive systems.
                </p>
            </section>

            {/* ── Industry Experience ──────────────────────────────────── */}
            <section style={{ marginBottom: 40 }}>
                <h2 style={sh}>Industry Experience</h2>

                {/* Vertical career timeline — most recent → oldest */}
                <div
                    style={{
                        position: 'relative',
                        paddingLeft: 22,
                        marginBottom: 22,
                    }}
                >
                    {/* vertical spine */}
                    <div
                        style={{
                            position: 'absolute',
                            left: 5,
                            top: 8,
                            bottom: 8,
                            width: 2,
                            background: '#2e2e2e',
                        }}
                    />

                    {[
                        {
                            title: 'Senior Manager of Software Engineering, R&D',
                            org: 'Roche',
                            years: '2024–Present',
                            current: true,
                        },
                        {
                            title: 'Manager of Software Engineering, R&D',
                            org: 'Roche',
                            years: '2023–2024',
                            current: false,
                        },
                        {
                            title: 'Senior Software Engineer & Technical Lead, R&D',
                            org: 'Roche',
                            years: '2022–2023',
                            current: false,
                        },
                        {
                            title: 'Software Engineer & Technical Lead, R&D',
                            org: 'Roche',
                            years: '2020–2022',
                            current: false,
                        },
                        {
                            title: 'Software Developer, NextGen Cloud',
                            org: 'IBM',
                            years: '2019–2020',
                            current: false,
                        },
                    ].map(({ title, org, years, current }) => (
                        <div
                            key={years}
                            style={{
                                position: 'relative',
                                marginBottom: 10,
                                paddingLeft: 16,
                            }}
                        >
                            {/* dot */}
                            <div
                                style={{
                                    position: 'absolute',
                                    left: -17,
                                    top: 5,
                                    width: 10,
                                    height: 10,
                                    borderRadius: '50%',
                                    background: current ? '#e0e0e0' : '#333',
                                    border: `2px solid ${current ? '#e0e0e0' : '#555'}`,
                                }}
                            />
                            <div
                                style={{
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'baseline',
                                    flexWrap: 'wrap',
                                    gap: 6,
                                }}
                            >
                                <span
                                    style={{
                                        fontSize: 14,
                                        fontWeight: current ? 600 : 400,
                                        color: current ? '#e0e0e0' : '#bbb',
                                    }}
                                >
                                    {title}
                                </span>
                                <span
                                    style={{
                                        fontSize: 12,
                                        color: current ? '#aaa' : '#666',
                                        whiteSpace: 'nowrap',
                                    }}
                                >
                                    {org} · {years}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>

            </section>

            {/* ── Technical Skills ─────────────────────────────────────── */}
            <section style={{ marginBottom: 40 }}>
                <h2 style={sh}>Technical Skills</h2>
                <div
                    style={{
                        marginTop: 10,
                        fontSize: 14,
                        color: '#bbb',
                        lineHeight: 1.85,
                    }}
                >
                    {[
                        {
                            label: 'Formal Methods',
                            value: 'LTL, ASMs, Model Checking, Temporal Logic Specification & Verification, Formal Refinement',
                        },
                        {
                            label: 'Proof Assistants',
                            value: 'Lean 4 (learning)',
                        },
                        {
                            label: 'AI & ML',
                            value: 'LangChain, LangGraph, RAG Systems, Agentic AI Systems',
                        },
                        {
                            label: 'Programming',
                            value: 'Java, TypeScript, Python, Golang, Rust · SpringBoot, Kafka, PostgreSQL, Docker, AWS',
                        },
                        {
                            label: 'Languages',
                            value: 'English (native), Spanish (native)',
                        },
                    ].map(({ label, value }) => (
                        <div key={label} style={{ marginBottom: 6 }}>
                            <span style={{ color: '#e0e0e0', fontWeight: 500 }}>
                                {label}:
                            </span>{' '}
                            {value}
                        </div>
                    ))}
                </div>
            </section>

            {/* ── Mentorship & Outreach ────────────────────────────────── */}
            <section style={{ marginBottom: 40 }}>
                <h2 style={sh}>Mentorship &amp; Outreach</h2>
                <div
                    style={{
                        marginTop: 10,
                        fontSize: 14,
                        color: '#bbb',
                        lineHeight: 1.85,
                    }}
                >
                    <div style={{ marginBottom: 10 }}>
                        <div style={rowSB}>
                            <span style={{ color: '#e0e0e0', fontWeight: 500 }}>
                                Industry Panelist — CS Careers
                            </span>
                            <span style={meta}>Feb 2024</span>
                        </div>
                        <div>
                            Spoke to 20+ CS majors at Rhode Island College on formal methods
                            in industry
                        </div>
                    </div>
                    <div>
                        <div style={rowSB}>
                            <span style={{ color: '#e0e0e0', fontWeight: 500 }}>
                                Industry Panelist — STEM Careers
                            </span>
                            <span style={meta}>June 2023</span>
                        </div>
                        <div>
                            Spoke to 100+ high school students via Skills for Rhode Island's
                            Future
                        </div>
                    </div>
                </div>
            </section>

            {/* ── Footer ───────────────────────────────────────────────── */}
            <footer
                style={{
                    marginTop: 32,
                    paddingTop: 20,
                    borderTop: '1px solid #2a2a2a',
                    display: 'flex',
                    justifyContent: 'center',
                    gap: 28,
                    alignItems: 'center',
                }}
            >
                <a
                    href={require('./Principal-Engineer-KJ.pdf')}
                    download="Principal-Engineer-KJ.pdf"
                    aria-label="Download PDF Resume"
                    style={{
                        textDecoration: 'none',
                        display: 'flex',
                        alignItems: 'center',
                    }}
                >
                    <img
                        src={require('./pics/pdf.png')}
                        alt="PDF Resume"
                        style={{ width: 24, height: 24, display: 'block' }}
                    />
                </a>
                <a
                    href="https://github.com/kjusino"
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label="GitHub"
                    style={{ textDecoration: 'none' }}
                >
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="12" fill="#181717" />
                        <path
                            d="M12 2C6.48 2 2 6.58 2 12.26c0 4.48 2.87 8.28 6.84 9.63.5.09.68-.22.68-.48 0-.24-.01-.87-.01-1.7-2.78.62-3.37-1.36-3.37-1.36-.45-1.18-1.1-1.5-1.1-1.5-.9-.63.07-.62.07-.62 1 .07 1.53 1.05 1.53 1.05.89 1.56 2.34 1.11 2.91.85.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.7 0 0 .84-.28 2.75 1.05A9.38 9.38 0 0 1 12 6.84c.85.004 1.71.12 2.51.35 1.91-1.33 2.75-1.05 2.75-1.05.55 1.4.2 2.44.1 2.7.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.8-4.57 5.06.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.81 0 .27.18.58.69.48C19.13 20.54 22 16.74 22 12.26 22 6.58 17.52 2 12 2Z"
                            fill="#fff"
                        />
                    </svg>
                </a>
                <a
                    href="https://www.linkedin.com/in/kenneth-jusino/"
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label="LinkedIn"
                    style={{
                        textDecoration: 'none',
                        display: 'flex',
                        alignItems: 'center',
                    }}
                >
                    <img
                        src={require('./pics/LI-In-Bug.png')}
                        alt="LinkedIn"
                        style={{ width: 24, height: 24, display: 'block' }}
                    />
                </a>
                <a
                    href="mailto:kennethjusino@hotmail.com"
                    aria-label="Email"
                    style={{ textDecoration: 'none' }}
                >
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                        <rect x="2" y="4" width="20" height="16" rx="2" fill="#EA4335" />
                        <polyline
                            points="22,6 12,13 2,6"
                            fill="none"
                            stroke="#fff"
                            strokeWidth="2"
                        />
                    </svg>
                </a>
            </footer>
        </div>,
    ],
};

export default resume;

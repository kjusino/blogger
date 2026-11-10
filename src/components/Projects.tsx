import '../projects.css';

interface Project {
    name: string;
    description: string;
    url: string;
    displayUrl: string;
    image?: string;
}

const PROJECTS: Project[] = [
    {
        name: 'LeanLingo',
        description:
            'A Duolingo-style interactive learning platform for Lean 4, the programming language and theorem prover. Practice formal verification through bite-sized lessons.',
        url: 'https://leanlingo.org',
        displayUrl: 'leanlingo.org',
        image: 'https://leanlingo.org/og.png',
    },
    {
        name: 'Lean Design Patterns',
        description:
            'The 23 Gang of Four design patterns, rebuilt in Lean 4. Each one starts from a real industry case and comes with a spec, an implementation and a machine-checked proof. In development.',
        url: 'https://kjusino.github.io/lean-design-patterns/',
        displayUrl: 'kjusino.github.io/lean-design-patterns',
        image: 'https://kjusino.github.io/lean-design-patterns/og.png',
    },
];

const Projects = () => (
    <div className="projects-page">
        <div className="projects-header">
            <div className="projects-header-label">Portfolio</div>
            <h1>Projects</h1>
            <p>Things I've built and made accessible for all.</p>
        </div>
        <div className="projects-grid">
            {PROJECTS.map((p) => (
                <a
                    key={p.url}
                    className="project-card"
                    href={p.url}
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    {p.image && (
                        <img
                            className="project-card-img"
                            src={p.image}
                            alt={p.name}
                        />
                    )}
                    <div className="project-card-body">
                        <h2 className="project-card-name">{p.name}</h2>
                        <p className="project-card-desc">{p.description}</p>
                        <span className="project-card-url">
                            {p.displayUrl} &rarr;
                        </span>
                    </div>
                </a>
            ))}
        </div>
    </div>
);

export default Projects;

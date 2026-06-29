export interface ArticleMeta {
    route: string;
    title: string;
    description: string;
    pic: string;
    createdDate?: string;
    audioSrc?: string;
    videoSrc?: string;
}

export const articleMeta: ArticleMeta[] = [
    {
        route: '/',
        title: 'Math ∩ Programming ∩ Science ∩ Culture',
        description: 'Me Visiting San Juan, Puerto Rico, 2021',
        pic: 'muralPic.png',
    },
    {
        route: '/about',
        title: 'About Me',
        description: 'Young Caribbean Boy Grinding',
        pic: 'caribbeanSea.png',
    },
    {
        route: 'pa',
        title: 'Happy Fathers Day, Papi!',
        description: 'The Man, The Myth, The Legend',
        pic: 'pa.png',
    },
    {
        route: '/phd',
        title: 'Quantum Computation',
        description: 'Quantum circuit progressing through time',
        pic: 'quantumCircuit.png',
    },
    {
        route: '/cv',
        title: 'Kenneth Jusino',
        description: 'CS PhD Candidate · NSF CSGrad4US Fellow · AI × Formal Methods',
        pic: 'profilepic.png',
    },
    {
        route: '/rust',
        title: 'Getting Rusty',
        description: 'Rust, a programming language with guaranteed memory safety, speed, and high concurrency.',
        pic: 'rustLogo.png',
        createdDate: '2025-05-25',
    },
    {
        route: '/ai-engineering',
        title: 'AI Engineering',
        description: 'Building deterministic and stochastic software with LangChain, LangGraph, and Ollama.',
        pic: 'legos.png',
        createdDate: '2026-03-28',
        audioSrc: '/audio/ai-engineering.m4a',
        videoSrc: 'https://kennethjusinoblog.blob.core.windows.net/videos/img-1710.mov',
    },
    {
        route: '/lean',
        title: "Lean'in' to Grad School",
        description: "Why I'm leaving industry to pursue a PhD at the intersection of AI, formal verification, mathematics, and software engineering.",
        pic: 'researchFlywheel.jpeg',
        createdDate: '2026-06-27',
    },
];

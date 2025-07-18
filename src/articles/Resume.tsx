import { ArticleProps } from '../resources/interfaces/ArticleProps';

const resumeCaption =
    'Principal Software Engineer | Tech Lead | Senior Engineering Manager';

const resume: ArticleProps = {
    route: '/resume',
    title: 'Kenneth Jusino',
    pics: ['profilepic.png'],
    caption: resumeCaption,
    content: [
        <div
            style={{
                maxWidth: '700px',
                margin: '40px auto',
                background: 'black',
                borderRadius: '18px',
                boxShadow: '0 4px 24px rgba(0,0,0,0.08)',
                padding: '32px 32px 40px 32px',
                fontFamily: 'Segoe UI, Arial, sans-serif',
                fontSize: '20px', // increased base font size by 2pt
            }}
        >
            {/* Consistent font and sizing for all sections */}
            <section style={{ marginBottom: 32, marginTop: 0 }}>
                <h2
                    style={{
                        borderBottom: '2px solid #1976d2',
                        display: 'inline-block',
                        paddingBottom: 4,
                        marginBottom: 16,
                        fontSize: 30, // was 28
                        fontWeight: 700,
                        fontFamily: 'inherit',
                        marginTop: 0,
                    }}
                >
                    Profile
                </h2>
                <p
                    style={{
                        fontSize: 19, // was 17
                        marginTop: 12,
                        fontFamily: 'inherit',
                        fontWeight: 400,
                    }}
                >
                    Principal Software Engineer and Engineering Manager with 7+
                    years of experience solving business-critical problems by
                    developing performant, resilient, high-quality backend
                    distributed systems.
                </p>
            </section>
            <section style={{ marginBottom: 32 }}>
                <h2
                    style={{
                        borderBottom: '2px solid #1976d2',
                        display: 'inline-block',
                        paddingBottom: 4,
                        marginBottom: 16,
                        fontSize: 30, // was 28
                        fontWeight: 700,
                        fontFamily: 'inherit',
                    }}
                >
                    Experience
                </h2>
                <div style={{ marginTop: 12 }}>
                    <h4
                        style={{
                            marginBottom: 4,
                            fontSize: 22, // was 20
                            fontWeight: 600,
                            fontFamily: 'inherit',
                        }}
                    >
                        Senior Manager of Software Engineering, Roche Inc,
                        Remote{' '}
                    </h4>
                    <p
                        style={{
                            marginTop: 0,
                            marginBottom: 16,
                            fontSize: 18, // was 16
                            fontFamily: 'inherit',
                            fontWeight: 400,
                        }}
                    >
                        Saved $5M annually by automating SDLC artifact
                        verification and document generation complying with IEC
                        62304 for Software as a Medical Device
                    </p>
                    <h4
                        style={{
                            marginBottom: 4,
                            fontSize: 22, // was 20
                            fontWeight: 600,
                            fontFamily: 'inherit',
                        }}
                    >
                        Senior Software Engineer, Roche Inc, Remote{' '}
                    </h4>
                    <p
                        style={{
                            marginTop: 0,
                            marginBottom: 16,
                            fontSize: 18, // was 16
                            fontFamily: 'inherit',
                            fontWeight: 400,
                        }}
                    >
                        Manager of seven engineering direct reports in CE.
                        Designed technical strategy, executed, and drove user
                        adoption. Java, Kafka, Typescript, Prisma, PostgreSQL,
                        AWS, Docker, Jenkins, Git
                    </p>
                    <h4
                        style={{
                            marginBottom: 4,
                            fontSize: 22, // was 20
                            fontWeight: 600,
                            fontFamily: 'inherit',
                        }}
                    >
                        Software Engineer 3, Roche Inc, Remote{' '}
                    </h4>
                    <p
                        style={{
                            marginTop: 0,
                            marginBottom: 16,
                            fontSize: 18, // was 16
                            fontFamily: 'inherit',
                            fontWeight: 400,
                        }}
                    >
                        Technical lead of Compliance Engineering and senior back
                        end engineer
                    </p>
                    <h4
                        style={{
                            marginBottom: 4,
                            fontSize: 22, // was 20
                            fontWeight: 600,
                            fontFamily: 'inherit',
                        }}
                    >
                        Software Engineer 2, Roche Inc, Boston MA{' '}
                    </h4>
                    <p
                        style={{
                            marginTop: 0,
                            marginBottom: 16,
                            fontSize: 18, // was 16
                            fontFamily: 'inherit',
                            fontWeight: 400,
                        }}
                    >
                        Back end engineer for Quality Engineering building web
                        API’s
                    </p>
                    <h4
                        style={{
                            marginBottom: 4,
                            fontSize: 22, // was 20
                            fontWeight: 600,
                            fontFamily: 'inherit',
                        }}
                    >
                        Software Developer, IBM Cloud, Littleton MA{' '}
                    </h4>
                    <p
                        style={{
                            marginTop: 0,
                            marginBottom: 0,
                            fontSize: 18, // was 16
                            fontFamily: 'inherit',
                            fontWeight: 400,
                        }}
                    >
                        SDET for IBM’s Kubernetes control plane. Built novel
                        protocol for local-to-server manual testing. Python,
                        PyTest, Jenkins, IBM Cloud, Kubernetes
                    </p>
                </div>
            </section>
            <section style={{ marginBottom: 32 }}>
                <h2
                    style={{
                        borderBottom: '2px solid #1976d2',
                        display: 'inline-block',
                        paddingBottom: 4,
                        marginBottom: 16,
                        fontSize: 30, // was 28
                        fontWeight: 700,
                        fontFamily: 'inherit',
                    }}
                >
                    Education
                </h2>
                <div style={{ marginTop: 12 }}>
                    <h4
                        style={{
                            marginBottom: 4,
                            fontSize: 22, // was 20
                            fontWeight: 600,
                            fontFamily: 'inherit',
                        }}
                    >
                        Perimeter Institute for Theoretical Physics, Remote
                    </h4>
                    <p
                        style={{
                            marginTop: 0,
                            marginBottom: 16,
                            fontSize: 18, // was 16
                            fontFamily: 'inherit',
                            fontWeight: 400,
                        }}
                    >
                        Certificate of Completion - PSI Start Scholar and PSI
                        Bridge Program
                    </p>
                    <h4
                        style={{
                            marginBottom: 4,
                            fontSize: 22, // was 20
                            fontWeight: 600,
                            fontFamily: 'inherit',
                        }}
                    >
                        Boston University, MA{' '}
                    </h4>
                    <p
                        style={{
                            marginTop: 0,
                            marginBottom: 0,
                            fontSize: 18, // was 16
                            fontFamily: 'inherit',
                            fontWeight: 400,
                        }}
                    >
                        B.S. in Pure & Applied Mathematics, Minors in Computer
                        Science & Physics
                    </p>
                </div>
            </section>
            <section style={{ marginBottom: 32 }}>
                <h2
                    style={{
                        borderBottom: '2px solid #1976d2',
                        display: 'inline-block',
                        paddingBottom: 4,
                        marginBottom: 16,
                        fontSize: 30, // was 28
                        fontWeight: 700,
                        fontFamily: 'inherit',
                    }}
                >
                    Skills
                </h2>
                <ul
                    style={{
                        marginTop: 12,
                        marginBottom: 0,
                        paddingLeft: 20,
                        fontSize: 18, // was 16
                        fontFamily: 'inherit',
                        fontWeight: 400,
                        listStyle: 'disc',
                    }}
                >
                    <li>
                        <span style={{ fontWeight: 600 }}>
                            Programming Languages:
                        </span>{' '}
                        Java, Typescript, Python, Golang, Rust, Javascript,
                        Bash/Shell
                    </li>
                    <li>
                        <span style={{ fontWeight: 600 }}>Frameworks:</span>{' '}
                        Springboot, Node, FastAPI, Gin, Express, React, NestJS
                    </li>
                    <li>
                        <span style={{ fontWeight: 600 }}>
                            Databases/ Infra:
                        </span>{' '}
                        PostgreSQL, Prisma ORM, MongoDB, Cloudformation,
                        Terraform
                    </li>
                    <li>
                        <span style={{ fontWeight: 600 }}>
                            Tooling/ Platforms:
                        </span>{' '}
                        Docker, Git, Kafka, Jenkins, AWS, Jira, qTest, LogzIO,
                        TestNG
                    </li>
                    <li>
                        <span style={{ fontWeight: 600 }}>AI & Quantum:</span>{' '}
                        LangChain, LangGraph, Browseruse, Pydantic, Copilot,
                        Qiskit, Ollama
                    </li>
                </ul>
            </section>
            <section style={{ marginBottom: 16 }}>
                <h2
                    style={{
                        borderBottom: '2px solid #1976d2',
                        display: 'inline-block',
                        paddingBottom: 4,
                        marginBottom: 16,
                        fontSize: 30, // was 28
                        fontWeight: 700,
                        fontFamily: 'inherit',
                    }}
                >
                    Awards
                </h2>
                <ul
                    style={{
                        marginTop: 12,
                        marginBottom: 0,
                        paddingLeft: 20,
                        fontSize: 18, // was 16
                        fontFamily: 'inherit',
                        fontWeight: 400,
                        listStyle: 'disc',
                    }}
                >
                    <li>
                        <span style={{ fontWeight: 600 }}>
                            National Science Foundation, D.C. May 2024:
                        </span>{' '}
                        NSF CSGrad4US Fellowship Recipient- quantum software,
                        $159,000 award over 3 years
                    </li>
                    <li>
                        <span style={{ fontWeight: 600 }}>
                            Boston Hacks, MA April 2018:
                        </span>{' '}
                        First Place, $500 award, Home Automation Track
                    </li>
                    <li>
                        <span style={{ fontWeight: 600 }}>
                            Rhode Island Math League, R.I. May 2015:
                        </span>{' '}
                        Top in School - Most Points, $1000 scholarship for
                        college
                    </li>
                </ul>
            </section>
            <footer
                style={{
                    marginTop: 40,
                    paddingTop: 24,
                    borderTop: '1px solid #222',
                    display: 'flex',
                    justifyContent: 'center',
                    gap: 32,
                    alignItems: 'center',
                }}
            >
                {/* PDF Resume */}
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
                        style={{ width: 28, height: 28, display: 'block' }}
                    />
                </a>
                {/* GitHub */}
                <a
                    href="https://github.com/kjusino"
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label="GitHub"
                    style={{ textDecoration: 'none' }}
                >
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="12" fill="#181717" />
                        <path
                            d="M12 2C6.48 2 2 6.58 2 12.26c0 4.48 2.87 8.28 6.84 9.63.5.09.68-.22.68-.48 0-.24-.01-.87-.01-1.7-2.78.62-3.37-1.36-3.37-1.36-.45-1.18-1.1-1.5-1.1-1.5-.9-.63.07-.62.07-.62 1 .07 1.53 1.05 1.53 1.05.89 1.56 2.34 1.11 2.91.85.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.7 0 0 .84-.28 2.75 1.05A9.38 9.38 0 0 1 12 6.84c.85.004 1.71.12 2.51.35 1.91-1.33 2.75-1.05 2.75-1.05.55 1.4.2 2.44.1 2.7.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.8-4.57 5.06.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.81 0 .27.18.58.69.48C19.13 20.54 22 16.74 22 12.26 22 6.58 17.52 2 12 2Z"
                            fill="#fff"
                        />
                    </svg>
                </a>
                {/* LinkedIn */}
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
                        style={{ width: 28, height: 28, display: 'block' }}
                    />
                </a>
                {/* Email */}
                <a
                    href="mailto:kennethjusino@hotmail.com"
                    aria-label="Email"
                    style={{ textDecoration: 'none' }}
                >
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                        <rect
                            x="2"
                            y="4"
                            width="20"
                            height="16"
                            rx="2"
                            fill="#EA4335"
                        />
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

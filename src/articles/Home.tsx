import { ArticleProps } from '../components/ArticleProps';
const data: ArticleProps = {
    route: '/',
    title: "Kenneth Jusino's Website",
    abstract: 'Science, Mathematics, Programming, Culture',
    pic: 'profilepic.png',
    caption: 'Boston University, May 2019',
    content: [
        <h2> Hi, I'm Kenneth! 👋🏽 </h2>,
        <p>
            I'm interested in mathematics, computer science, physics, and
            culture. I could talk your {''}
            <em>ear off</em> about:
        </p>,
        <ul>
            <li> partial differential equations</li>
            <li> backend software engineering</li>
            <li> Hamiltonian mechanics</li>
            <li> hip-hop/reggaetón</li>
        </ul>,
        <p>
            So instead of just yapping to my friends and family, I created a
            simple React App using Typescript to host my thoughts on my own
            platform. This UI will act as the entrypoint for all software
            projects I build in the future. It is containerized with Docker and
            deployed using Github Actions onto the Azure App Service.{' '}
        </p>,
        <h2> Senior Software Engineer & Tech Lead</h2>,
        <h3>Software Engineer</h3>,
        <p>
            I am a professional software engineer. I specialize in building
            large-scale, real-time, resilient backend software solutions. My two
            employers have been technology giant
            <a href="https://www.ibm.com/"> IBM </a> and
            <a href="https://www.roche.com/stories/comprehensive-genomic-profiling-in-personalised-healthcare">
                {' '}
                comprehensive genomic profiling{' '}
            </a>
            company{' '}
            <a href="https://www.foundationmedicine.com/">
                Foundation Medicine
            </a>
            , a subsidiary of European health conglomarate{' '}
            <a href="https://www.roche.com/">Roche, Inc.</a>
        </p>,
        <p>
            I build software systems using various combinations of technologies,
            each more suitable for a particular problem over the others:
        </p>,
        <ul>
            <li> Java & Spring</li>
            <li> Typescript & Node</li>
            <li> Python & Django</li>
            <li> Golang & Gin</li>
            <li> SQL & Prisma </li>
        </ul>,
        <p>
            {' '}
            I've worked with Kafka for event-driven architectures, GraphQL for
            scalable backend federation, Docker for containerization, AWS and
            Azure as Cloud Services, Jenkins and Github Actions for CI/CD, and
            React for front end development.
        </p>,
        <h3>Technical Lead</h3>,
        <p>
            I've been the technical lead for FMI's Compliance Engineering team
            for over two years. CE's team of 9 SWE's and SDET's build software
            solutions used by the Technology enterprise to monitor, test, and
            verify the compliance of applications built by Engineering. The
            solutions I build leverage multithreading and asynchronous
            programming to optimize the users experience.
        </p>,
        <h2> Boston University Mathematics Alumnus </h2>,
        <p>
            I graduated from{' '}
            <a href="https://www.bu.edu/"> Boston University </a>
            in May 2019 with a Bachelors Degree in{' '}
            <a href="https://www.bu.edu/math/">Pure and Applied Mathematics</a>,
            with double minors in{' '}
            <a href="https://www.bu.edu/cs/">Computer Science</a> and{' '}
            <a href="https://www.bu.edu/physics/">Physics</a>. I have studied
            the art of problem solving in settings in both academia and industry
            for a decade, and have learned how to efficiently learn and invent
            new information and information processing systems.
        </p>,
    ],
};

export default data;

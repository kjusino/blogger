import { ArticleProps } from '../resources/interfaces/ArticleProps';
import { Tags } from '../resources/enums/Tags';
const sweData: ArticleProps = {
    route: '/swe',
    title: 'My Software Engineering Journey',
    pics: ['profilepic.png'],
    caption: 'Me Graduating Boston University, 2019',
    tags: [Tags.Computation],
    content: [
        <h2>My SWE Journey</h2>,
        <h3>
            {' '}
            SWE I -{'>'} SWE II -{'>'} SWE III -{'>'} Senior SWE -{'>'} Senior
            Mgr, Engineering{' '}
        </h3>,
        <p>
            I am a professional software engineer. I've trained in building
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
            I am the technical lead of FMI's Compliance Engineering department,
            where we are automating compliance by design to FMI's SDLC. My team
            of 9 SWE's and SDET's build software solutions used by the
            Technology enterprise to monitor, test, and verify the compliance of
            applications built by Engineering. The solutions I build leverage
            object-oriented programming, multithreading and design patterns to
            optimize the users experience and overall product quality.
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

export default sweData;

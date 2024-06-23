import { ArticleProps } from '../components/ArticleProps';
const data: ArticleProps = {
    route: '/netflix',
    title: 'I Want To Build For Netflix',
    abstract: 'Specifically, Scalable Back-End Distributed Systems in Java',
    pics: ['netflix.png'],
    caption: 'Because I Want To Learn How Its Done at Global Scale',
    content: [
        <h2> Why Netflix? </h2>,
        <p> Because I</p>,
        <ul>
            <li>
                received my first dvd in the mail in 2008, and have been a
                grateful customer since
            </li>
            <li>
                am a self-taught backend software engineer with 6 years of
                professional experience, and am always in awe of what Netflix
                Engineering is capable of accomplishing
            </li>
            <li>
                watched Netflix's first original series{' '}
                <em>"House of Cards"</em> the day it came out while I was in
                High School, with my jaw on the ground during the opening
                monologue
            </li>
            <li>
                {' '}
                want to learn about Netflix's implementation of Adaptive
                Streaming and individualized/dynamic bitrate and its effects on
                Quality of Experience (QoE)
            </li>
        </ul>,
        <h2>Why Me?</h2>,
        <p> Because I</p>,
        <ul>
            <li>
                have an undergraduate background in pure and applied mathematics
                and could think about problems all day
            </li>
            <li>
                am an incredibly curious person who is constantly searching for
                better solutions to problems
            </li>
            <li>
                will never start writing code until I fully understand the
                problem and have a provably scalable algorithm designed
            </li>
            <li>
                care immensely about code quality and testing algorithms
                rigorously for edge cases and inefficiencies
            </li>
        </ul>,
        <h2> What Have You Done To Prepare?</h2>,
        <p>
            Every piece of software I have written has been developed because of
            its necessity across engineering organizations and I have learned
            through trial and error. My software systems and solutions have
            successfully transformed all of the enterprises I have ever been a
            part of for the better. Thus far, I've saved my current engineering
            organization at Roche, Inc 64,000 human-hours in idle time per year,
            with yearly savings of over $4.5 million. At the same time,
            standardizing and enforcing higher standards of quality, security,
            and compliance across the company. At previous employers, I've
            developed new methods of securely communicating with servers to
            perform automated testing of global cloud provider IBM.
        </p>,

        <p>
            I want to apply my skill sets at Netflix, and grow as an engineer.
            If it sounds like your engineering team at Netflix would benefit
            from someone like me, please let me know, it would be my pleasure :)
        </p>,
    ],
};

export default data;

import { Tags } from '../resources/enums/Tags';
import { ArticleProps } from '../resources/interfaces/ArticleProps';
const phdData: ArticleProps = {
    route: '/phd',
    title: 'Quantum Computation',
    pics: ['quantumCircuit.png', 'quantumGap.png', 'quantumStack.png'],
    tags: [Tags.Math, Tags.Computation, Tags.Physics],
    caption: 'Quantum circuit progressing through time',
    content: [
        <h2>Quantum Computer Scientist Soon🔜</h2>,
        <p>
            I'm a prospective quantum computer scientist interested in
            researching the intersection of mathematics, computation, and
            physics in a CS PhD program.
        </p>,
        <p>
            The National Science Foundation awarded me the CSGrad4US Fellowship
            in 2024, which comes with three years financial support totaling
            $159,000 for my studies.
        </p>,
        <p>
            I am broadly interested in researching the quantum computing "stack"
            in its entirety. More specifically, I'm interested in quantum-
            <ul>
                <li>algorithms</li>
                <li>complexity theory</li>
                <li>programming languages</li>
                <li>computer architecture</li>
                <li>error correcting codes</li>{' '}
                <li>software/hardware co-design</li>
            </ul>
        </p>,
    ],
};

export default phdData;

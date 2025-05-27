import { Tags } from '../resources/enums/Tags';
import { ArticleProps } from '../resources/interfaces/ArticleProps';
const lean: ArticleProps = {
    route: 'lean',
    title: 'Lean Theorem Prover',
    pics: ['pa.png'],
    caption: 'Lets discover new mathematics with Lean!',
    createdDate: '2025-05-26',
    tags: [Tags.Computation, Tags.Math],
    content: [
        <h2>Leaning</h2>,
        <p>
            tools like the
            <a href="https://lean-lang.org/">
                {' '}
                Lean Programming Language and Theorem Prover{' '}
            </a>{' '}
            and projects using it to formalize mathematics and physics like{' '}
            <a href="https://physlean.com/">PhysLean/</a>.
        </p>,
    ],
};

export default lean;

import { ArticleProps } from '../resources/interfaces/ArticleProps';
import { Tags } from '../resources/enums/Tags';

const Verification: ArticleProps = {
    route: '/verification',
    title: 'Trust, but Verify',
    abstract:
        'Where automated software engineering, AI, and automated mathematics converge.',
    pics: ['legos.png'],
    caption: 'Coming soon.',
    createdDate: '2026-06-25',
    tags: [Tags.Computation, Tags.Math],
    content: [
        <h2>Coming soon</h2>,
        <p>This one is still being written. Check back soon.</p>,
    ],
};

export default Verification;

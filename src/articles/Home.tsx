import { ArticleProps } from '../resources/interfaces/ArticleProps';
const introData: ArticleProps = {
    route: '/',
    title: 'Math ∩ Programming ∩ Science ∩ Culture',
    pics: ['muralPic.png'],
    caption: 'Me Visiting San Juan, Puerto Rico, 2021',
    content: [
        <h2>
            👋🏽 <a href="/about">I'm Kenneth</a>
        </h2>,
        <p>
            I am interested in many topics, and plan to write short articles
            with some of my learnings here.
        </p>,
        <p>
            I'm primarily driven by a deep curiosity for the world around me,
            and my desire to communicate with others. I created this blog to
            publish some of my personal thoughts and opinions on my own
            platform.
        </p>,
        <p>
            I've studied mathematics at{' '}
            <a href="https://www.bu.edu/math/">Boston University</a>, physics
            with{' '}
            <a href="https://perimeterinstitute.ca">The Perimeter Institute </a>
            , and computer science while working as a Software Engineer at{' '}
            <a href="https://www.ibm.com/"> IBM </a> and
            <a href="https://www.foundationmedicine.com/">
                {' '}
                Foundation Medicine
            </a>
            , a subsidiary of <a href="https://www.roche.com/">Roche, Inc.</a>
        </p>,
    ],
};

export default introData;

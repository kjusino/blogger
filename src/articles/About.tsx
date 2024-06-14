import { ArticleProps } from '../components/ArticleProps';
const data: ArticleProps = {
    route: '/about-me',
    title: 'About Me',
    abstract: 'Young Caribbean Boy Grinding',
    pic: 'About/profilepic.png',
    caption: 'Me, graduating from Boston University in 2019',
    content: [
        <h2> Life Journey </h2>,
        <p>
            I would love to take the credit and say everything I've done or
            accomplished during my life has been completely intentional and
            completely planned out for maximum long-term benefits.
        </p>,
        <ul>
            <li> partial differential equations</li>
            <li> backend software engineering</li>
            <li> Hamiltonian mechanics</li>
            <li> hip-hop/reggaetón</li>
        </ul>,
        <h2> PVD Born and Raised </h2>,
        <p></p>,
        <h2> BOS 4L </h2>,
        <p>
            Let AI decide what the fastest/most optimal form of syntax is and
            lets just learn to talk to them
        </p>,
    ],
};

export default data;

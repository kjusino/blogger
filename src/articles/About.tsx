import { ArticleProps } from '../resources/interfaces/ArticleProps';
import { Tags } from '../resources/enums/Tags';
const data: ArticleProps = {
    route: '/about',
    title: 'About Me',
    abstract: 'Young Caribbean Boy Grinding',
    pics: ['caribbeanSea.png'],
    caption: '🇩🇴 Dominican Mother & 🇵🇷 Puerto Rican Father',
    tags: [Tags.Culture],
    createdDate: '2025-05-26',
    content: [
        <h2> The Beginning </h2>,
        <p>
            I was born in Providence, RI to a Dominican mother and a Puerto
            Rican father. They met in the 1990's while enrolled in
            English-as-a-Second Language night classes offered by Rhode Island
            College. After my younger brother was born, our family moved to
            Santo Domingo, Dominican Republic, so my parents could leverage
            their familial support network back in the Caribbean during my
            childhood. It was an amazing way to grow up, surrounded by{' '}
            <em>hundreds</em> of family members and friends, all gathered for
            even the smallest occasions.
        </p>,
        <p>
            I learned Spanish as my first language, and we practiced our English
            at home. After my seventh birthday and fifth year in Santo Domingo,
            my parents made the difficult decision to return to Rhode Island
            after economic hardships began to affect everyday life in DR. I
            didn't know what a coat was, let alone what awaited us in the United
            States.
        </p>,
        <h2> PVD Born and Raised </h2>,
        <p>
            Upon returning to Providence, I began attending{' '}
            <a
                href="https://www.usnews.com/education/k12/rhode-island/alfred-lima-sr-elementary-school-278884"
                target="_blank"
            >
                Alfred Lima Elementary
            </a>
            , a bilingual public school where many first generation students in
            Providence pass through. This is where I learned English as a Second
            Language. I later arrived at{' '}
            <a
                href="https://www.usnews.com/education/best-high-schools/rhode-island/districts/providence-public-schools/times2-middle-high-school-409586"
                target="_blank"
            >
                Times Squared Academy
            </a>
            , the charter school I graduated high school from before attending
            college. Growing up in Providence was a unique experience that I
            would not trade for anything in the world. My friends and I grew up
            surrounded by the children of immigrants from the Caribbean, Africa,
            Central America, and South America. We all enjoyed the benefits of
            multiculturalism early on in our lives and got to experience a true
            sense of the word "community". We learned the customs and the
            language of our new home, and partook in American traditions while
            still being nestled in Providence's global community.
        </p>,
        <p>
            I knew while I was in high school that I would not attend university
            in Rhode Island. My top choices after I applied were Boston
            University in Massachusetts and Case Western Reserve University in
            Ohio. Luckily, I followed my instincts and enrolled at BU, where I
            planned to study all things STEM.
        </p>,
        <h2> BOS 4L </h2>,
        <p>
            Attending Boston University was the smartest decision I've ever
            made. The very first day I stepped foot on campus, I met Genesis, my
            college sweetheart and 10 years later, now my wife. While she
            studied Political Science in preparation for her current career as
            an attorney, I let my wide-ranging scientific curiosity guide my
            undergraduate curriculum (for better or for worse).
        </p>,
        <p>
            My interests have always been at the intersection of math, computer
            science, and physics. I wanted my coursework to reflect that and for
            classes from the neighboring disciplines to illuminate new paths
            between each, so as to purposefully build bridges between my mental
            model. After all, there is only one Nature.
        </p>,
    ],
};

export default data;

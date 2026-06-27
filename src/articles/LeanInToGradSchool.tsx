import { ArticleProps } from '../resources/interfaces/ArticleProps';
import { Tags } from '../resources/enums/Tags';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const curryHowardIdentity = `-- As a program: a function that returns its input
def identity (x : Nat) : Nat := x

-- As a proof: "if P is true, then P is true"
theorem self_implies_self (p : P) : P := p`;

const curryHowardCompose = `-- As a program: compose two functions
def compose (f : β → γ) (g : α → β) (x : α) : γ := f (g x)

-- As a proof: if A → B and B → C, then A → C
theorem transitivity (h1 : A → B) (h2 : B → C) : A → C := fun a => h2 (h1 a)`;

const codeBlockStyle = {
    minWidth: 0,
    fontSize: '0.875rem',
    padding: '0 6%',
    margin: 0,
    boxSizing: 'border-box' as const,
    width: '100%',
    maxWidth: '90vw',
    overflowX: 'auto' as const,
};

const pullQuoteStyle = {
    borderLeft: '3px solid var(--accent)',
    margin: '2rem 0',
    padding: '0.5rem 0 0.5rem 1.5rem',
    fontStyle: 'italic' as const,
    fontSize: '1.35rem',
    lineHeight: 1.5,
    fontWeight: 600,
    color: 'var(--text-bright)',
};

const LeanInToGradSchool: ArticleProps = {
    route: '/lean',
    title: "Lean'in' to Grad School",
    abstract:
        "Why I'm leaving industry to pursue a PhD at the intersection of AI, formal verification, mathematics, and software engineering.",
    pics: ['researchFlywheel.jpeg'],
    caption:
        'The Research Flywheel: AI × Formal Methods × Mathematics × Software Engineering.',
    createdDate: '2026-06-27',
    tags: [Tags.Computation, Tags.Math],
    content: [
        <p>
            I'm resigning from my Sr. Software Engineering Manager role at
            Foundation Medicine this September, and taking a hiatus from
            full-time employment.
        </p>,
        <p>
            I will begin a Computer Science PhD program this fall at{' '}
            <a href="https://www.khoury.northeastern.edu/" target="_blank">
                Northeastern University's Khoury College for Computer Science
            </a>{' '}
            as part of the Formal Methods Group. My research is supported by the
            NSF's CSGrad4US Fellowship.
        </p>,
        <p>
            I want to work at the frontier of four distinct but overlapping
            sub-disciplines, conducting research experiments that accelerate
            (flywheel) progress in all four simultaneously. My research will be
            grounded in a powerful technology developed recently:{' '}
            <a href="https://lean-lang.org/" target="_blank">
                <strong>Lean</strong>
            </a>
            .
        </p>,
        <hr />,
        <h2>Why? My Time in Software Engineering</h2>,
        <p>
            I began my career as a Software Developer in Test at IBM's NextGen
            Cloud in Littleton, Massachusetts. I learned so much from the
            incredibly talented people there, but the most important lesson was
            this: the absolute necessity of thoroughly testing software for the
            enormous range of scenarios that real usage brings. The larger and
            wider you cast the scenario-net, the better chance of finding bugs
            and fixing them before software reaches production.
        </p>,
        <blockquote style={pullQuoteStyle}>
            But the absence of bugs does not imply that bugs are absent.
        </blockquote>,
        <p>
            No matter how big your scenario-net, there could still be bugs
            unknown to you, crouching in the corners of scenario-space waiting
            for the (im)perfect set of conditions to cause catastrophic
            failures.
        </p>,
        <p>
            Having just graduated with my Bachelor's in Mathematics, I strived to
            be as thorough with my software testing at IBM as I had been solving
            partial differential equations and calculating Fourier transforms by
            pen and paper: <em>very</em>. I always thought about how to find edge
            cases and 99th-percentile scenarios, and I took this testing mindset
            with me when I joined Foundation Medicine's Quality Engineering group
            as a Software Engineer.
        </p>,
        <p>
            FMI sells FDA-regulated Software as Medical Device products and must
            adhere to all regulatory standards when releasing new software to
            patients. Every team deploying to our globally-distributed production
            environments had to follow a strict SDLC process to prove to
            auditors that we adhered to all standards during development.
        </p>,
        <p>
            I led the Compliance Engineering team, where we built automated
            systems to formally verify regulatory adherence by other software
            teams promoting products from dev → qa → stage → prod. Our "release
            manager" system generates a software verification report that proves,
            to FMI and to auditors, that products were installed successfully
            and are behaving as expected. It pulls live data across all
            environments and synthesizes a current model of the release. If the
            release passes a comprehensive swath of verification checks, the
            system verifies the release, the report is sent for signatures, and
            the product goes live.
        </p>,
        <blockquote style={pullQuoteStyle}>
            But process correctness does not guarantee product correctness.
        </blockquote>,
        <p>
            A compliant software system can still contain bugs unknown to all.
            And this <em>really</em> bothers me. As a student of mathematics, I
            have been trained in the most extreme of mindsets: things are either
            100% correct or 0% correct. No in between. Correctness is binary, and
            software engineering is a discipline destined to cause me agony, or
            so it seemed.
        </p>,
        <hr />,
        <h2>What's Changed? Lean 4</h2>,
        <p>
            You might be thinking: "So no matter how wide we cast the testing
            net, how many verification checks we perform on a release candidate,
            there'll always be bugs in software?"
        </p>,
        <p>
            <strong>No.</strong> À la the renowned mathematician David Hilbert:{' '}
            <em>"We must not, we will not!"</em> allow bugs to exist in software.
            And Lean's creator Leonardo de Moura agrees.
        </p>,
        <p>
            He and a world-renowned group of computer scientists and
            mathematicians around the globe have been building tools to{' '}
            <strong>prove</strong> software correctness, not just test for it,
            but <em>mathematically guarantee</em> it. I want to join this global
            movement to build safer software that accelerates scientific
            discoveries in medicine, mathematics, physics, and computer science
            by constructing provably correct systems that will power tomorrow's
            world today.
        </p>,
        <p>
            So what is Lean? Lean is both a programming language{' '}
            <strong>and</strong> an{' '}
            <a
                href="https://en.wikipedia.org/wiki/Automated_theorem_proving"
                target="_blank"
            >
                automated theorem prover
            </a>{' '}
            (ATP). You've likely heard of programming languages like Python and
            Java. ATPs are software tools that help humans mechanically prove
            logical statements correct or incorrect, using the mathematics of{' '}
            <a
                href="https://en.wikipedia.org/wiki/Dependent_type"
                target="_blank"
            >
                Dependent Type Theory
            </a>
            , deep theoretical CS that I'm excited to study at Northeastern.
            Computer scientists have researched ATPs for decades, but the
            barrier to entry was so high that they were used solely by the
            academics researching them.
        </p>,
        <p>
            Leonardo changed this. Designed with a software verification-first
            mindset, Lean reduces the barrier to entry for ATPs and makes it
            practical for engineers to prove the correctness of software built in
            the Lean programming language using the Lean theorem prover. One
            language, one syntax, one stack.
        </p>,
        <p>
            What could humans achieve if we did our most ambitious software
            engineering yet, backed by a mathematical guarantee of correctness?
            This is the future I plan to contribute to as a computer scientist
            and mathematician, building provably safe, secure, and correct
            software systems for the betterment of humanity.
        </p>,
        <p>
            And this guarantee has never been more urgent, because of AI.
        </p>,
        <hr />,
        <h2>When? Right Now.</h2>,
        <p>
            In January 2026, tools like Anthropic's Claude Code and OpenAI's
            Codex changed software engineering forever, right in front of our
            eyes. These were not the autocomplete LLMs that engineers rightly
            ignored between 2022 and 2025. The early LLMs hallucinated broken
            code constantly. But the 2026 generation completely convinced me of
            two things:
        </p>,
        <p>
            <strong>
                1. Software development has undergone a phase transition
            </strong>
            , like when humans went from vacuum tubes to punch cards, punch cards
            to machine code, machine code to programming languages.
            LLM-augmented engineering is the future, and there's nothing anyone
            can do about it.
        </p>,
        <p>
            <strong>
                2. Stochastic LLMs <em>must</em> integrate with deterministic
                ATPs.
            </strong>{' '}
            If writing code by hand is walking, LLM-augmented engineering is
            flying, and Lean is what keeps us from crashing. This integration
            creates a positive feedback loop: AI generates candidate code, Lean
            verifies it, failures refine the next attempt. Guess and check. And
            mathematicians have been perfecting guess-and-check for centuries.
        </p>,
        <p>
            The systems of the future should behave like mathematicians iterating
            on ideas until correctness is proven through logic. Stochastic
            generation, constrained by deterministic verification. This feedback
            loop accelerates into a flywheel, and humans must be at the helm as
            we navigate the transition.
        </p>,
        <p>
            One important caveat worth acknowledging: Lean proves correctness{' '}
            <em>relative to a specification</em>. The spec itself can still be
            wrong. But this is exactly where human expertise remains
            irreplaceable, and where AI can help us explore the specification
            space faster than ever before.
        </p>,
        <hr />,
        <h2>How? AI × Math × Lean = Correct Software</h2>,
        <p>
            If AI × Lean makes verifying correctness easier, how do we make the
            AI better at building correct code in the first place?
        </p>,
        <p>We bake mathematics into the models.</p>,
        <p>
            Lean's official mathematics library,{' '}
            <a
                href="https://leanprover-community.github.io/mathlib-overview.html"
                target="_blank"
            >
                <strong>Mathlib</strong>
            </a>
            , is where mathematical facts are being formalized and codified for
            future
            generations to leverage as a library of truth. Instead of letting
            models randomly predict the next token, we endow them with the
            ability to reason mathematically, and therefore learn new domains
            correctly, potentially discovering solutions to earth's biggest
            challenges that no single person could achieve alone.
        </p>,
        <p>
            With Mathlib, anyone in the world using Lean can leverage formalized
            theorems to prove statements about mathematics <em>or</em> software.
            This is possible because of the{' '}
            <a
                href="https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence"
                target="_blank"
            >
                <strong>Curry-Howard Correspondence</strong>
            </a>
            .
        </p>,
        <p>Don't take my word for it, let me show you:</p>,
        <SyntaxHighlighter
            language="lean"
            style={vscDarkPlus}
            wrapLongLines={true}
            customStyle={codeBlockStyle}
        >
            {curryHowardIdentity}
        </SyntaxHighlighter>,
        <p>
            Same syntax. Same logic. The function <em>is</em> the proof.
        </p>,
        <hr />,
        <h2>
            Conclusion: Future Software Engineering = Verifying Mathematical
            Proofs
        </h2>,
        <p>
            The Curry-Howard Correspondence is a fundamental result in computer
            science and mathematics: there is an equivalence between mathematical
            proofs and computer programs. A theorem is a type. A proof is a
            program. A program's computational logic <em>is</em> a proof's
            deduction logic. At the deepest level, math proofs and computer
            programs are the same thing, and we can create one-to-one
            correspondences between them to guarantee the correctness of
            software.
        </p>,
        <p>
            And this scales. Function composition, the backbone of all
            software,{' '}
            <em>is</em> logical transitivity:
        </p>,
        <SyntaxHighlighter
            language="lean"
            style={vscDarkPlus}
            wrapLongLines={true}
            customStyle={codeBlockStyle}
        >
            {curryHowardCompose}
        </SyntaxHighlighter>,
        <p>
            The <code>→</code> symbol means "function" and "implies"
            simultaneously. These aren't analogies, they are the same thing.
            That's Curry-Howard. And that's why the future of software
            engineering is verifying mathematical proofs.
        </p>,
        <p>
            The futures of AI, Mathematics, and Computer Science are intricately
            intertwined, and those who sit at the intersections will bridge the
            disciplines and build holistic systems capable of discovering new
            mathematics, new medicine, and new science that will change the
            world. I want to contribute to this global effort and use my time in
            graduate school to push the frontiers of knowledge and help bring
            about the next scientific renaissance.
        </p>,
        <p>
            I plan to use my research to contribute to efforts like Mathlib, and
            help build better tools for software engineering and mathematical
            research at places like{' '}
            <a href="https://axiommath.ai/" target="_blank">
                Axiom
            </a>
            ,{' '}
            <a
                href="https://www.amazon.science/research-areas/automated-reasoning"
                target="_blank"
            >
                AWS ARG
            </a>
            ,{' '}
            <a href="https://harmonic.fun/" target="_blank">
                Harmonic
            </a>
            ,{' '}
            <a href="https://www.anthropic.com/" target="_blank">
                Anthropic
            </a>
            ,{' '}
            <a href="https://deepmind.google/" target="_blank">
                Google DeepMind
            </a>
            ,{' '}
            <a href="https://openai.com/" target="_blank">
                OpenAI
            </a>
            ,{' '}
            <a href="https://www.microsoft.com/en-us/research/" target="_blank">
                Microsoft Research
            </a>
            ,{' '}
            <a href="https://research.ibm.com/" target="_blank">
                IBM Research
            </a>
            , and beyond.
        </p>,
        <p>
            <strong>
                If you are interested in working together or are looking for
                graduate student interns for summer 2027, please reach out.
            </strong>{' '}
            There has never been a better time to be a builder.
        </p>,
    ],
};

export default LeanInToGradSchool;

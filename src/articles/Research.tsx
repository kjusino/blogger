import { ArticleProps } from '../resources/interfaces/ArticleProps';
import { Tags } from '../resources/enums/Tags';

const Research: ArticleProps = {
    route: '/research',
    title: 'Why Engineers In Critical Industries Should Pay Attention To CS Research',
    pics: ['rustLogo.png'],
    caption:
        'Correctness-critical industries deserve correctness-critical tools.',
    tags: [Tags.Computation],
    createdDate: '2026-02-08',
    content: [
        <h2>The Status Quo Is Not Good Enough</h2>,
        <p>
            If you're an engineer building software in a correctness-critical
            industry — pharma, medical devices, aerospace, automotive, finance —
            you know the weight of getting things right. A bug in your system
            isn't just a bad user experience. It's a patient getting the wrong
            dose. It's an aircraft behaving unpredictably. It's millions of
            dollars evaporating because a trading system failed a boundary
            condition that a manual test case didn't cover.
        </p>,
        <p>
            And yet, most of these industries rely on SDLC methodologies that
            were designed in the 90s and early 2000s. We're talking about V-model
            validation, mountains of paper-based compliance documentation,
            handwritten test protocols, and heavyweight change control processes
            that treat software like it's a physical product rolling off an
            assembly line. These approaches were state of the art once. They
            aren't anymore.
        </p>,
        <p>
            Meanwhile, the computer science research community has been producing
            tools and techniques that fundamentally change what it means to
            verify that software is correct. And the gap between what's available
            in research and what's actually used in industry is growing wider
            every year. That gap is a liability.
        </p>,
        <h2>What Has CS Research Been Up To?</h2>,
        <p>
            Theoretical computer science, programming language theory, and formal
            methods researchers have spent decades building mathematically
            rigorous tools for reasoning about software correctness. These
            aren't toys or academic curiosities — they're production-ready
            techniques that attack the exact problems correctness-critical
            industries care about. Here are a few:
        </p>,
        <p>
            <b>Formal Verification and Proof Assistants.</b> Tools like{' '}
            <a href="https://lean-lang.org/">Lean</a>,{' '}
            <a href="https://coq.inria.fr/">Coq</a>, and{' '}
            <a href="https://isabelle.in.tum.de/">Isabelle</a> allow you to
            write mathematical proofs about the behavior of your software. Not
            tests that check a handful of inputs — <i>proofs</i> that your code
            satisfies a specification for <i>all</i> possible inputs. NASA has
            used formal methods for{' '}
            <a href="https://shemesh.larc.nasa.gov/fm/fm-what.html">
                flight-critical systems
            </a>{' '}
            for years. The question is why the rest of us in regulated
            industries aren't doing the same.
        </p>,
        <p>
            <b>Type Systems That Prevent Bugs at Compile Time.</b> Rust's{' '}
            <a href="https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html">
                ownership model and borrow checker
            </a>{' '}
            are a direct product of programming language research in{' '}
            <a href="https://en.wikipedia.org/wiki/Substructural_type_system">
                substructural type systems
            </a>
            . The compiler mathematically guarantees that your program is free
            from memory safety bugs — no null pointer exceptions, no data races,
            no use-after-free. These aren't runtime checks that can be missed.
            They're compile-time rejections of incorrect programs. The
            mathematics doesn't let bad code through.
        </p>,
        <p>
            <b>Property-Based Testing and Model Checking.</b> Instead of writing
            individual test cases by hand, tools like{' '}
            <a href="https://github.com/BurntSushi/quickcheck">QuickCheck</a>{' '}
            and{' '}
            <a href="https://www.cs.cmu.edu/~modelcheck/smv.html">SMV</a>{' '}
            let you define properties your system should satisfy and then
            automatically generate thousands (or millions) of test cases to find
            violations. This is the difference between "I tested 50 scenarios
            and they passed" and "I tested 5 million randomly generated
            scenarios and found an edge case that would have shipped to
            production."
        </p>,
        <p>
            <b>Static Analysis and Abstract Interpretation.</b> Techniques
            like{' '}
            <a href="https://en.wikipedia.org/wiki/Abstract_interpretation">
                abstract interpretation
            </a>{' '}
            can automatically analyze your code and prove properties about it
            without even running it. The Astrée static analyzer, for example,
            was used to prove the{' '}
            <a href="https://www.absint.com/astree/index.htm">
                absence of runtime errors in Airbus flight control software
            </a>
            . No test suite in the world can give you that guarantee.
        </p>,
        <h2>The Gap Is a Liability</h2>,
        <p>
            Here's what frustrates me. I've spent years building software in a
            regulated industry. I've seen firsthand how much time, money, and
            human effort goes into validation processes that are fundamentally
            limited. Manual test cases can only cover the scenarios you think of.
            Document-based compliance proves you followed a process, not that
            your software is correct. Code reviews catch what human eyes happen
            to notice.
        </p>,
        <p>
            At Roche, I automated compliance to FDA standards using formal
            verification methods embedded directly into our SDLC. My system
            authorizes the final go/no-go decision for new software releases
            going out to patients around the world. That work saved the company
            over $5 million annually and eliminated thousands of hours of manual
            validation. And it didn't require inventing anything new — it
            required <i>applying</i> what researchers had already built.
        </p>,
        <p>
            That's the point. The research exists. The tools exist. The proofs
            that these techniques work at scale exist. What's missing is
            engineers in critical industries actually learning about them and
            advocating for their adoption.
        </p>,
        <h2>Why This Matters Now</h2>,
        <p>
            Software is eating the world, and regulated industries are no
            exception. Medical devices are increasingly software-defined.
            Autonomous vehicles rely on millions of lines of code. Financial
            systems execute trades in microseconds. The complexity of these
            systems is growing exponentially, and our validation methodologies
            need to keep up.
        </p>,
        <p>
            The FDA itself has started acknowledging this. Their guidance on{' '}
            <a href="https://www.fda.gov/regulatory-information/search-fda-guidance-documents/computer-software-assurance-production-and-quality-system-software">
                Computer Software Assurance
            </a>{' '}
            represents a shift toward risk-based, automated approaches and away
            from the paper-heavy Computer System Validation paradigm that has
            dominated pharma for decades. The regulatory landscape is moving.
            The question is whether engineers are moving with it.
        </p>,
        <h2>A Call To Engineers</h2>,
        <p>
            If you're building software where correctness matters — really
            matters — I'd challenge you to do three things:
        </p>,
        <ol>
            <li>
                <b>Read a paper.</b> Seriously. Pick a topic relevant to your
                domain — formal verification, type theory, property-based
                testing, static analysis — and read a recent paper or survey.{' '}
                <a href="https://arxiv.org/list/cs.SE/recent">arXiv</a> is
                free. The knowledge is there. The barrier is just taking the
                time.
            </li>
            <li>
                <b>Learn a tool.</b> Try writing a small proof in{' '}
                <a href="https://lean-lang.org/">Lean</a>. Build something
                in Rust and experience what a compiler that enforces correctness
                feels like. Run a property-based testing framework against your
                existing codebase and see what it finds. The best way to
                understand the value of these tools is to use them.
            </li>
            <li>
                <b>Advocate.</b> Bring these ideas to your team. Show your
                leadership what's possible. The engineers closest to the code are
                the ones best positioned to push for better methodologies. Don't
                wait for a mandate from above. Demonstrate the value and the
                mandate will follow.
            </li>
        </ol>,
        <p>
            The research community is producing incredible work. It's on us as
            industry engineers to close the gap. Our users — patients, pilots,
            drivers, customers — deserve software that is proven correct, not
            just tested and hoped for the best.
        </p>,
    ],
};

export default Research;

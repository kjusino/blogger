import { ArticleProps } from '../resources/interfaces/ArticleProps';
import { Tags } from '../resources/enums/Tags';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const truthCompilesString = `-- Lean checks every line. Truth compiles. Falsehood does not.

theorem two_plus_two : 2 + 2 = 4 := by rfl   -- ✓ accepted

theorem wishful_thinking : 2 + 2 = 5 := by rfl
-- ✗ rejected: type mismatch
--    the term has type 2 + 2 = 4, but is expected to prove 2 + 2 = 5`;

const reverseProofString = `-- This is not a comment promising correctness.
-- It is a proof the compiler re-checks on every single build.

theorem reverse_reverse {α : Type} (xs : List α) :
    xs.reverse.reverse = xs := by
  simp`;

const Verification: ArticleProps = {
    route: '/verification',
    title: 'Trust, but Verify',
    abstract:
        'Why the future of software and mathematics is the same problem, and why I am leaving industry to work on it.',
    pics: ['legos.png'],
    caption:
        'AI is the stochastic generator. Formal verification is the deterministic checker. The future is both.',
    createdDate: '2026-06-25',
    tags: [Tags.Computation, Tags.Math],
    content: [
        <h2>Two automations, one collision</h2>,
        <p>
            For seven years I have built software that decides whether a medical
            diagnostic can ship to patients. The systems I design at{' '}
            <a href="https://www.foundationmedicine.com/" target="_blank">
                Foundation Medicine
            </a>
            , a subsidiary of{' '}
            <a href="https://www.roche.com/" target="_blank">
                Roche
            </a>
            , encode FDA requirements as formal specifications — temporal logic
            formulas, abstract state machines — and then <i>prove</i>, not test,
            that the software conforms before it ever reaches a patient. When I
            started, the bottleneck was writing those specifications by hand.
            Today the bottleneck is something else entirely.
        </p>,
        <p>
            Two crafts that used to belong exclusively to trained humans are
            being automated at the same time. The first is{' '}
            <b>software engineering</b>. AI agents now read entire codebases,
            write features, fix bugs, and open pull requests with remarkably
            little guidance. The second is <b>mathematics</b>. Models sketch
            proofs, propose lemmas, and navigate the kind of symbolic reasoning
            that once demanded a PhD and a quiet afternoon. Both shifts are
            exhilarating. Both share the same fatal flaw: the machine doing the
            writing is, at its core, a probability engine, and a probability
            engine cannot tell you when it is wrong. That is the problem this
            post is about, and the solution is older and more beautiful than the
            AI itself: <i>verification</i>.
        </p>,
        <h2>The trust problem (again)</h2>,
        <p>
            In an{' '}
            <a href="/ai-engineering" target="_blank">
                earlier post
            </a>{' '}
            I described software as a Lego roller coaster track: every block
            rigid, every transition predictable, the whole system deterministic
            by its physical nature. Then Large Language Models arrived as a new
            kind of Lego piece, one that is inherently <b>stochastic</b>. It
            produces tokens sampled from a probability distribution, which is
            wonderful when you want a draft email and disastrous when you need a
            correct diagnosis.
        </p>,
        <p>
            Ask a model to generate a formal specification and it will hand you
            something that <i>looks</i> like one — the right keywords, plausible
            structure, confident tone — that is quietly, subtly wrong. I have
            seen this firsthand. The danger is not that AI makes mistakes. Every
            engineer makes mistakes. The danger is that AI makes mistakes{' '}
            <i>persuasively</i>, at a scale and speed no human reviewer can keep
            up with. In a domain where a malformed specification can send an
            incorrect cancer diagnostic to an oncologist, this is not a quirk. It
            is a non-starter.
        </p>,
        <h2>The missing Lego piece: a verifier</h2>,
        <p>
            Here is the insight that ties everything together. Generating an
            answer and <i>checking</i> an answer are fundamentally different
            levels of difficulty. Finding the proof of a theorem can take a
            century; confirming a proof is correct is mechanical. Writing a
            correct program is hard; checking that a program satisfies a precise
            specification can be made automatic. This asymmetry is the whole
            game.
        </p>,
        <p>
            So you let the stochastic engine do the hard, creative, exploratory
            part — generate the candidate — and you bolt a small, deterministic,
            ruthless <b>verifier</b> onto the output. The verifier does not care
            how confident the model sounded. It asks one question: does this
            actually hold? If it does not, it is rejected. No persuasion, no
            benefit of the doubt.
        </p>,
        <p>
            This is exactly what I have been building at Roche. My most recent
            project, <b>AIRE</b> (AI Requirements Engineering), uses{' '}
            <b>grammar-constrained LLM decoding</b> to synthesize formal
            specifications from natural-language requirements. Instead of hoping
            the model produces syntactically valid output, AIRE embeds a
            context-free grammar directly into the decoding process so that every
            token the model emits is <i>compliant by construction</i>. The model
            supplies creativity; the grammar supplies the guardrail. What comes
            out is not a best guess — it is a specification that is structurally
            guaranteed to be well-formed before any downstream checker even sees
            it.
        </p>,
        <p>
            For mathematics and, increasingly, for software, the most powerful
            verifier we have is a <b>proof assistant</b>. My favorite is{' '}
            <a href="https://lean-lang.org/" target="_blank">
                Lean 4
            </a>
            .
        </p>,
        <h2>Lean 4: where only truth compiles</h2>,
        <p>
            Lean is two things at once: a real programming language and a theorem
            prover. At its heart sits a tiny <i>trusted kernel</i>, a few
            thousand lines of code whose only job is to check that each proof
            step follows from the last. Everything else — the tactics, the
            automation, the AI suggestions — is untrusted scaffolding. It can be
            as clever or as reckless as it likes, because nothing reaches you
            until the kernel signs off.
        </p>,
        <p>
            The consequence is profound: in Lean, a false statement{' '}
            <b>will not compile</b>. You cannot bluff the kernel.
        </p>,
        <SyntaxHighlighter
            language="lean"
            style={vscDarkPlus}
            wrapLongLines={true}
            customStyle={{
                minWidth: 0,
                fontSize: '0.875rem',
                padding: '0 6%',
                margin: 0,
                boxSizing: 'border-box',
                width: '100%',
                maxWidth: '90vw',
                overflowX: 'auto',
            }}
        >
            {truthCompilesString}
        </SyntaxHighlighter>,
        <p>
            This is why the pairing with AI is so natural. Let a model generate a
            proof however it wants — brilliantly or by hallucinating wildly. Then
            feed it to Lean. If it compiles, it is <i>true</i>, full stop, with
            the same certainty as the mathematics it rests on. If it does not,
            you throw it away and ask again. The model becomes a tireless
            generator of candidates; the kernel becomes an incorruptible judge.
            Stochastic proposes, deterministic disposes.
        </p>,
        <h2>Automated mathematics</h2>,
        <p>
            This is already reshaping how new mathematics gets done. Researchers
            formalize results in Lean's{' '}
            <a
                href="https://leanprover-community.github.io/mathlib-overview.html"
                target="_blank"
            >
                Mathlib
            </a>{' '}
            library so that a theorem, once accepted, is mechanically certified
            forever. Projects like{' '}
            <a href="https://physlean.com/" target="_blank">
                PhysLean
            </a>{' '}
            are extending the same rigor to physics. Pair that infrastructure
            with an AI that can draft and iterate on proofs thousands of times
            faster than a person, and you get a flywheel: the AI explores the
            space of possible arguments, and Lean guarantees that whatever
            survives is real mathematics, not a confident-sounding mirage.
        </p>,
        <p>
            The bottleneck on discovery stops being "can we check this?" and
            becomes "what should we ask?"
        </p>,
        <h2>Automated software engineering</h2>,
        <p>
            The same idea bends back onto software, and this is where my career
            lives. Through the lens of the{' '}
            <a
                href="https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence"
                target="_blank"
            >
                Curry–Howard correspondence
            </a>
            , a program <i>is</i> a proof and a type <i>is</i> a theorem. Writing
            a function that type-checks against a precise enough specification is
            the same act as proving a mathematical statement. In Lean you can
            state a property about your code and prove it once, and the compiler
            will re-verify it on every build for the rest of the project's life.
        </p>,
        <SyntaxHighlighter
            language="lean"
            style={vscDarkPlus}
            wrapLongLines={true}
            customStyle={{
                minWidth: 0,
                fontSize: '0.875rem',
                padding: '0 6%',
                margin: 0,
                boxSizing: 'border-box',
                width: '100%',
                maxWidth: '90vw',
                overflowX: 'auto',
            }}
        >
            {reverseProofString}
        </SyntaxHighlighter>,
        <p>
            I have spent seven years learning this lesson the hard way. At Roche,
            I encoded FDA release criteria as linear temporal logic formulas and
            embedded them into our CI/CD pipeline so that every build is verified
            against regulatory specifications before it can proceed. The result
            was a 20× reduction in non-conformance events and $4.5 million in
            annual savings — not because we tested harder, but because we{' '}
            <i>proved</i>. Tests sample a few inputs and hope. A proof closes the
            door on entire classes of failure for good.
        </p>,
        <p>
            It is the same principle that makes me love{' '}
            <a href="/rust" target="_blank">
                Rust
            </a>
            , whose borrow checker mathematically guarantees memory safety before
            the program ever runs. And it is the principle behind AIRE: if you
            can constrain the model's output space at decode time, you do not need
            to pray for correctness after the fact. You get it by construction.
        </p>,
        <p>
            Now put an AI agent in that loop. Let it write the feature, but make
            it earn its merge by satisfying a machine-checked specification. The
            agent supplies speed and breadth; the verifier supplies the
            guarantee. You get the productivity of "the AI wrote it" without
            surrendering the certainty of "and we proved it correct."
        </p>,
        <h2>The convergence</h2>,
        <p>
            Step back and the picture is the same in both domains. A fast,
            creative, fundamentally unreliable generator on one side. A small,
            slow, absolutely reliable verifier on the other. The generator is AI.
            The verifier is formal proof. Automated software engineering and
            automated mathematics turn out to be the <i>same problem</i> wearing
            two costumes, and they share the same answer: a kernel that cannot be
            fooled.
        </p>,
        <p>
            We spent decades treating "the code is correct" and "the theorem is
            true" as separate disciplines, practiced by separate communities,
            published in separate venues. AI collapses the distance between them,
            because the moment a machine is generating both, both need the same
            defense. The researchers building AI-powered theorem provers and the
            engineers building verified software pipelines are converging on the
            same mathematics, the same tools, and the same open questions.
        </p>,
        <h2>What comes next</h2>,
        <p>
            This is why I am leaving industry. After seven years of applying
            formal methods to FDA-regulated software — from temporal logic
            specifications to grammar-constrained neural decoding — I am entering
            a{' '}
            <b>
                Computer Science PhD at{' '}
                <a
                    href="https://www.khoury.northeastern.edu/"
                    target="_blank"
                >
                    Northeastern University's Khoury College
                </a>
            </b>{' '}
            this fall, supported by the{' '}
            <a
                href="https://www.nsf.gov/funding/opportunities/dcl-computer-information-science-engineering-graduate"
                target="_blank"
            >
                NSF CSGrad4US Fellowship
            </a>
            . My advisor,{' '}
            <a href="https://stavros.tripakis.net/" target="_blank">
                Stavros Tripakis
            </a>
            , works at precisely this intersection — AI-assisted formal
            verification, neurosymbolic methods, and provably correct software
            systems — and I cannot imagine a better place to push this work
            forward.
        </p>,
        <p>
            The questions I want to answer are the ones my industry experience
            forced me to ask. How do we scale formal verification beyond the
            domains rich enough to afford it today? Can LLMs trained on
            mathematical reasoning produce outputs that are not just plausible but{' '}
            <i>checkable</i> — specifications a proof assistant can verify, code a
            type system can certify? And can we build the constraint layers, the
            grammars, the neurosymbolic architectures, that make "correct by
            construction" the default rather than the exception?
        </p>,
        <p>
            I grew up in Providence, Rhode Island, the son of a Dominican mother
            and a Puerto Rican father who met in English night classes after long
            days in the textile factories. My father was the first in his family
            to earn a college degree. Now I will be the first in mine to earn a
            doctorate. The road from Santo Domingo to a CS PhD is not a straight
            line, but formal methods taught me that the path does not need to be
            straight. It just needs to be <i>verified</i>.
        </p>,
        <p>
            The roller coaster track is still there. We have added a Lego piece
            that does not always go where you expect, and right next to it, a
            piece whose entire purpose is to catch the first one when it wanders.
            Trust the machine to generate. Verify before you believe. That is the
            work ahead of me, and there has never been a better time to build it.
        </p>,
    ],
};

export default Verification;

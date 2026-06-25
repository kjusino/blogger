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
        'Where automated software engineering, AI, and automated mathematics converge.',
    pics: ['legos.png'],
    caption:
        'AI is the stochastic generator. Formal verification is the deterministic checker. The future is both.',
    createdDate: '2026-06-25',
    tags: [Tags.Computation, Tags.Math],
    content: [
        <h2>Two automations, one collision</h2>,
        <p>
            Two things that used to be the exclusive domain of trained humans
            are being automated at the same time, and they are about to collide.
            The first is <b>software engineering</b>. AI agents now read entire
            codebases, write features, fix bugs, and open pull requests with very
            little hand-holding. The second is <b>mathematics</b>. Models can
            sketch proofs, propose lemmas, and chew through the kind of symbolic
            reasoning that once required a PhD and a quiet afternoon.
        </p>,
        <p>
            Both shifts are exhilarating. Both share the same fatal flaw. The
            machine doing the writing is, at its core, a probability engine, and
            a probability engine cannot tell you when it is wrong. That is the
            problem this post is about, and the solution is older and more
            beautiful than the AI itself: <i>verification</i>.
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
            produces tokens from a probability distribution, which is wonderful
            when you want an essay and disastrous when you want the truth.
        </p>,
        <p>
            Ask a model to count the R's in "strawberry" and it might miss one.
            Ask it for a proof and it will hand you something that <i>looks</i>{' '}
            like a proof, with confident prose and the right vocabulary, that is
            quietly, subtly false. The danger is not that AI gets things wrong.
            The danger is that it gets things wrong <i>persuasively</i>, and at a
            scale and speed no human reviewer can keep up with. In the
            correctness-critical work I do, this is not a quirk. It is a
            non-starter.
        </p>,
        <h2>The missing Lego piece: a verifier</h2>,
        <p>
            Here is the insight that ties everything together. Generating an
            answer and <i>checking</i> an answer are not the same difficulty.
            Finding the proof of a theorem can take a century; confirming a
            proof is correct is mechanical. Writing a correct program is hard;
            checking that a program satisfies a precise specification can be made
            automatic. This asymmetry is the whole game.
        </p>,
        <p>
            So you let the stochastic engine do the hard, creative,
            non-deterministic part, generate the candidate, and you bolt a small,
            deterministic, ruthless <b>verifier</b> onto the output. The verifier
            does not care how confident the model sounded. It only asks one
            question: does this actually hold? If it does not, it is rejected.
            No persuasion, no benefit of the doubt.
        </p>,
        <p>
            For mathematics and, increasingly, for software, the best verifier we
            have is a <b>proof assistant</b>. My favorite is{' '}
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
            step follows from the last. Everything else, the tactics, the
            automation, the AI suggestions, is untrusted scaffolding. It can be
            as clever or as reckless as it likes, because nothing reaches you
            until the kernel signs off.
        </p>,
        <p>
            The consequence is profound: in Lean, a false statement <b>will not
            compile</b>. You cannot bluff the kernel.
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
            proof however it wants, brilliantly or by hallucinating wildly. Then
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
            are doing the same for physics. Pair that infrastructure with an AI
            that can draft and iterate on proofs thousands of times faster than a
            person, and you get a flywheel: the AI explores the space of possible
            arguments, and Lean guarantees that whatever survives is real
            mathematics, not a confident-sounding mirage.
        </p>,
        <p>
            For someone headed into a CS PhD at the intersection of math,
            computation, and physics, this is the most exciting tooling shift in
            a generation. The bottleneck on discovery stops being "can we check
            this?" and becomes "what should we ask?"
        </p>,
        <h2>Automated software engineering</h2>,
        <p>
            The same idea bends back onto software, and this is where my day job
            lives. Through the lens of the{' '}
            <a
                href="https://en.wikipedia.org/wiki/Curry%E2%80%93Howard_correspondence"
                target="_blank"
            >
                Curry–Howard correspondence
            </a>
            , a program <i>is</i> a proof and a type <i>is</i> a theorem. Writing
            a function that type-checks against a precise enough specification is
            the same act as proving a statement. In Lean you can state a property
            about your code and prove it once, and the compiler will re-verify it
            on every build for the rest of the project's life.
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
            At{' '}
            <a href="/rust" target="_blank">
                Roche
            </a>
            , I build systems that apply formal methods to the software lifecycle
            for Software as a Medical Device, automating compliance to FDA
            standards and issuing the final go/no-go on releases that reach
            patients around the world. The lesson from that work is simple: when
            being wrong is unacceptable, you do not <i>test harder</i>, you{' '}
            <i>prove</i>. Tests sample a few inputs and hope; a proof closes the
            door on entire classes of failure for good. It is the same reason I
            love{' '}
            <a href="/rust" target="_blank">
                Rust
            </a>
            , whose borrow checker mathematically guarantees memory safety before
            the program ever runs.
        </p>,
        <p>
            Now put an AI agent in that loop. Let it write the feature, but make
            it earn its merge by satisfying a machine-checked specification. The
            agent supplies speed and breadth; the verifier supplies the
            guarantee. You get the productivity of "the AI wrote it" without
            surrendering the certainty of "and we know it's correct."
        </p>,
        <h2>The convergence</h2>,
        <p>
            Step back and the picture is the same in both domains. A fast,
            creative, fundamentally unreliable generator on one side. A small,
            slow, absolutely reliable verifier on the other. The generator is AI.
            The verifier is formal proof. Automated software engineering and
            automated mathematics turn out to be the <i>same problem</i> wearing
            two costumes, and they have the same answer.
        </p>,
        <p>
            That is the convergence I keep coming back to. We spent decades
            treating "the code is correct" and "the theorem is true" as separate
            crafts. AI collapses the distance between them, because the moment a
            machine is generating both, both need the same defense: a kernel that
            cannot be fooled.
        </p>,
        <h2>The future belongs to builders who verify</h2>,
        <p>
            I am an optimist about all of this, but it is a specific kind of
            optimism. The winners of the next decade will not be the people who
            generate the most code or the most proofs. They will be the people
            who build the <i>loops</i>, the systems where a stochastic model
            proposes and a deterministic verifier guarantees, where you get
            AI-scale output with proof-grade trust.
        </p>,
        <p>
            The roller coaster track is still there. We have added a Lego piece
            that doesn't always go where you expect, and right next to it, a
            piece whose entire purpose is to catch the first one when it
            wanders. Trust the machine to generate. Verify before you believe.
            That is the whole job now, and there has never been a better time to
            be the person who builds it. What will you prove?
        </p>,
    ],
};

export default Verification;

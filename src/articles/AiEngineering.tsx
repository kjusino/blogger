import { ArticleProps } from '../resources/interfaces/ArticleProps';
import { Tags } from '../resources/enums/Tags';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

const ollamaInstallString = `ollama pull gpt-oss:20b`;

const ollamaConnectString = `const llm = new Ollama({
  model: "gpt-oss:20b",
  baseUrl: "http://localhost:11434",
});`;

const deterministicKeywordSearch = `
const glossary = ["code", "javascript", "programming", "python", "java",...]; 
question.forEach(word => {
    if(glossary.includes(word)){
        return 'code' 
    }else{
        return 'general'
    }    
});
`;

const exampleOutputString = `// Code question: "What is a closure in JavaScript?"
{
  "node": "code",
  "answer": "[CODE NODE] A closure is the combination of a function and the lexical environment...",
  "question": "What is a closure in JavaScript?"
}

// General question: "Why is the sky blue?"
{
  "node": "general",
  "answer": "[GENERAL NODE] The sky looks blue because of Rayleigh scattering...",
  "question": "Why is the sky blue?"
}`;

const fullImplementationString = `import express, { Request, Response } from "express";
import { Ollama } from "@langchain/community/llms/ollama";
import { StateGraph, START, END } from "@langchain/langgraph";

// State

interface AgentState {
  question: string;
  classification: string;
  answer: string;
  node: string;
}

const stateChannels = {
  question: {
    value: (_a: string, b: string) => b,
    default: () => "",
  },
  classification: {
    value: (_a: string, b: string) => b,
    default: () => "",
  },
  answer: {
    value: (_a: string, b: string) => b,
    default: () => "",
  },
  node: {
    value: (_a: string, b: string) => b,
    default: () => "",
  },
};

// Ollama model

const llm = new Ollama({
  model: "gpt-oss:20b",
  baseUrl: "http://localhost:11434",
});

// Nodes

async function classifierNode(
  state: AgentState
): Promise<Partial<AgentState>> {
  const prompt = \`Classify the following question as either "code" or "general".
Only reply with one word: code or general. Nothing else.

Question: \${state.question}\`;

  const result = await llm.invoke(prompt);
  const classification = result.trim().toLowerCase().includes("code")
    ? "code"
    : "general";

  return { classification };
}

async function codeNode(state: AgentState): Promise<Partial<AgentState>> {
  const prompt = \`You are an expert programmer. Answer the following coding question clearly and concisely.

Question: \${state.question}\`;

  const result = await llm.invoke(prompt);
  return {
    node: "code",
    answer: \`[CODE NODE] \${result.trim()}\`,
  };
}

async function generalNode(state: AgentState): Promise<Partial<AgentState>> {
  const prompt = \`You are a helpful assistant. Answer the following question clearly and concisely.

Question: \${state.question}\`;

  const result = await llm.invoke(prompt);
  return {
    node: "general",
    answer: \`[GENERAL NODE] \${result.trim()}\`,
  };
}

// Routing

function routeAfterClassifier(state: AgentState): string {
  return state.classification === "code" ? "code_node" : "general_node";
}

// Graph

const graph = new StateGraph<AgentState>({ channels: stateChannels })
  .addNode("classifier", classifierNode)
  .addNode("code_node", codeNode)
  .addNode("general_node", generalNode)
  .addEdge(START, "classifier")
  .addConditionalEdges("classifier", routeAfterClassifier, {
    code_node: "code_node",
    general_node: "general_node",
  })
  .addEdge("code_node", END)
  .addEdge("general_node", END)
  .compile();

// Express API

const app = express();
app.use(express.json());

app.post("/ask", async (req: Request, res: Response) => {
  const { question } = req.body as { question?: string };

  if (!question || typeof question !== "string" || question.trim() === "") {
    res.status(400).json({ error: "Missing or invalid 'question' field" });
    return;
  }

  try {
    const result = await graph.invoke({
      question: question.trim(),
      classification: "",
      answer: "",
      node: "",
    });

    res.json({
      node: result.node,
      answer: result.answer,
      question: result.question,
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    res.status(500).json({ error: message });
  }
});

const PORT = process.env.PORT ?? 3000;
app.listen(PORT, () => {
  console.log(\`Router agent listening on http://localhost:\${PORT}\`);
  console.log(\`POST /ask  { "question": "..." }\`);
});`;

const AiEngineering: ArticleProps = {
    route: '/ai-engineering',
    title: 'AI Engineering',
    pics: ['legos.png'],
    caption:
        'Building deterministic and stochastic software with LangChain, LangGraph, and Ollama.',
    createdDate: '2026-03-28',
    tags: [Tags.Computation],
    content: [
        <h2>Deterministic vs Stochastic Software</h2>,
        <p>
            Since the beginning of software, developers could only build systems
            that are, by their physical nature, inherently deterministic. This
            means the programmer could (and should) be able to predict every
            possible state and all transitions between states that the program
            could end up in. Ultimately, a program's dynamics were completely
            "determined" by the static text that comprised it, with extremely
            rare exceptions like bit flips caused by solar rays or other random
            physical phenomena.
        </p>,
        <p>
            I think of this deterministic behavior like a Lego roller coaster
            track, each line of code a Lego block. Each piece is rigid and
            provides predictable behavior you can count on ~100% of the time. A
            user interface requests data from an API, which ingests and
            manipulates data from databases or other APIs. A completed system
            always behaves as expected, with no room for deviation from the
            "roller coaster track."
        </p>,
        <h2>A new Lego piece has been unlocked</h2>,
        <p>
            This all changed with the release of Large Language Models, a new
            class of inherently stochastic software systems. LLMs produce tokens
            based on probability distributions, which means no one can predict
            what word an LLM produces next. A completely new piece can now be
            added to software systems, one that introduces genuine randomness
            into the mix.
        </p>,
        <p>
            In some cases, that randomness is useful: writing essays, emails,
            creative text, and more. In these cases, there is no single correct
            answer, and subjective variation is a feature, not a bug. LLMs today
            excel at these tasks.
        </p>,
        <p>
            In other cases, the randomness is disastrous. Ask an LLM to count
            the R's in "strawberry," perform basic arithmetic, or write
            syntactically correct code (tasks with objectively correct answers)
            and any deviation is simply wrong. This is most important in
            correctness-critical domains, like in Software as Medical Device (
            <a
                href="https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd"
                target="_blank"
            >
                SaMD
            </a>
            ) or autonomous vehicles. The stochastic nature of LLMs is a
            liability here.
        </p>,
        <p>
            As software systems begin mixing deterministic and stochastic
            behaviors, new tools are needed to guide and constrain LLMs toward
            "the right thing" and to isolate their randomness to the parts of
            the system where it's actually helpful.
        </p>,
        <h2>Enter LangChain and LangGraph</h2>,
        <p>
            <a href="https://www.langchain.com/" target="_blank">
                LangChain
            </a>{' '}
            is a framework that gives developers a unified interface for
            interacting with LLMs. Think of it as the adapter that lets your
            deterministic code "speak" to a stochastic model, and swap out
            models without rewriting your application logic.
        </p>,
        <p>
            <a href="https://www.langchain.com/langgraph" target="_blank">
                LangGraph
            </a>{' '}
            goes further. It lets you define your AI-enhanced system as a
            directed graph: a set of nodes (discrete processing steps) and edges
            (transitions between them). Some of those edges can be{' '}
            <i>conditional</i>, meaning an LLM's output determines which path
            the program takes next. This is the key insight: you can use
            stochastic outputs to drive deterministic routing.
        </p>,
        <h2>Running models locally with Ollama</h2>,
        <p>
            Before we get to the code, it's worth talking about where the LLM
            itself lives. Most tutorials assume you're calling a cloud API like
            OpenAI, Anthropic, or Gemini. But for development, experimentation,
            and privacy-sensitive applications, running a model locally is often
            preferable.
        </p>,
        <p>
            <a href="https://ollama.com" target="_blank">
                Ollama
            </a>{' '}
            is a tool that lets you download and run open-source LLMs directly
            on your machine with a single command. Once installed, pulling a
            model is as simple as:
        </p>,
        <SyntaxHighlighter
            language="bash"
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
            {ollamaInstallString}
        </SyntaxHighlighter>,
        <p>
            Ollama then exposes a local REST API at{' '}
            <code>http://localhost:11434</code> that mimics the structure of
            cloud LLM APIs. Your application code doesn't change much; you just
            point it at localhost instead of a remote endpoint. No API keys, no
            usage costs, no data leaving your machine.
        </p>,
        <img
            src={require(`../articles/pics/ollama.png`)}
            style={{ width: '100%', height: 'auto' }}
            alt={'Ollama running locally'}
        />,
        <p>
            For our example, we're running <code>gpt-oss:20b</code>, a
            20-billion parameter model, entirely locally. LangChain connects to
            it through its Ollama integration:
        </p>,
        <SyntaxHighlighter
            language="typescript"
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
            {ollamaConnectString}
        </SyntaxHighlighter>,
        <p>
            From here, <code>llm.invoke(prompt)</code> sends a prompt to your
            local model and returns a response, just like it would with any
            cloud provider.
        </p>,
        <h2>Real Example: Code Question, or Not?</h2>,
        <div
            style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                width: '100%',
            }}
        >
            <img
                src={require(`../articles/pics/graph.png`)}
                style={{ width: '40%', height: 'auto' }}
                alt={'LangGraph visualized'}
            />
        </div>,
        <p>
            To make this concrete, here's a small API built with TypeScript,
            LangChain, LangGraph, and Ollama. It exposes a single endpoint,{' '}
            <code>POST /ask</code>, that routes any question to one of two
            specialized LLM nodes depending on what kind of question it is.
        </p>,
        <p>
            When a question arrives, the <code>classifier</code> node sends it
            to the LLM with a strict prompt: respond with one word, either{' '}
            <code>code</code> or <code>general</code>. That single token, a
            stochastic output, feeds into a deterministic <code>if</code>{' '}
            statement that routes the question to the appropriate node. The{' '}
            <code>code_node</code> answers with a programmer persona; the{' '}
            <code>general_node</code> answers as a general assistant. The
            response clearly labels which node handled it:
        </p>,
        <SyntaxHighlighter
            language="json"
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
            {exampleOutputString}
        </SyntaxHighlighter>,
        <h3>Pre-LLMs</h3>,
        <p>
            Before LLMs, fully deterministic programs would have had to perform
            a keyword search against a predetermined list of terms manually
            added to a "glossary". In this example's case, a "code glossary"
            would contain all words that might refer to a coding question. I
            would have done something like the following:
        </p>,
        <SyntaxHighlighter
            language="typescript"
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
            {deterministicKeywordSearch}
        </SyntaxHighlighter>,
        <p>
            It would have been an extremely tedious task to keep the list of
            keywords up to date. Gaps would always exist if a coding question
            did not include any of the words in the glossary but was still about
            programming.
        </p>,
        <h3>Post-LLMs</h3>,
        <p>
            Fast forward to today, and the LLMs can now <b>be the glossary</b>{' '}
            and more. They can interpret the question and, based on their
            training data, probabilistically choose whether the question is
            code-related or not. LangGraph helps with this "probabilistic
            choosing", and provides a framework for conditional logic.
        </p>,
        <p>Here's the full implementation, in 126 lines of TypeScript:</p>,
        <SyntaxHighlighter
            language="typescript"
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
            {fullImplementationString}
        </SyntaxHighlighter>,
        <p>
            Notice how <code>StateGraph</code> threads the{' '}
            <code>AgentState</code> object through every node; each node
            receives the full state and returns only the fields it wants to
            update. The graph itself is compiled once at startup and reused
            across requests, which keeps latency low.
        </p>,
        <p>
            This is AI Engineering in practice. You're not just calling an LLM
            and hoping for the best. You're building a system where LLM outputs
            become inputs to deterministic logic, which then constrains and
            directs the next LLM call. The stochastic and deterministic pieces
            interlock, each compensating for the other's weaknesses.
        </p>,
        <p>
            The roller coaster track is still there. You've just added a new
            kind of Lego piece, one that doesn't always go where you expect, but
            is useful precisely because of that.
        </p>,
        <h2>The future belongs to builders</h2>,
        <p>
            <a
                href="https://en.wikipedia.org/wiki/Artificial_intelligence_engineering"
                target="_blank"
            >
                AI Engineering
            </a>{' '}
            is not a niche specialty. It is the direction all software
            development is heading. The systems of tomorrow will not be purely
            deterministic or purely stochastic; they will be compositions of
            both, carefully designed to get the best out of each. The developers
            who understand how to wire these pieces together will have an
            enormous advantage.
        </p>,
        <p>
            There has never been a better time to be a builder. The tools are
            accessible, the models are powerful, and the barrier to entry is
            lower than ever. What matters now is imagination. The most important
            question is not "what can the model do?" but "what can I build with
            it?" Creativity and vision are the new competitive edge. The Lego
            pieces are on the table. What will you build?
        </p>,
    ],
};

export default AiEngineering;

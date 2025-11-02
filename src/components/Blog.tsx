import { Tags } from '../resources/enums/Tags';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

function Blog({
    route,
    title,
    abstract,
    pics,
    caption,
    content,
    tags,
}: {
    route: string;
    title: string;
    abstract?: string;
    pics: string[];
    caption: string;
    content: string; // Now a markdown string instead of JSX.Element[]
    tags?: Tags[];
}) {
    // Load images if they exist
    const img = pics.length > 0 ? require(`../articles/pics/${pics[0]}`) : null;
    const img2 = pics.length > 1 ? require(`../articles/pics/${pics[1]}`) : null;

    return (
        <div className="Article">
            <header className="Article-header">
                <h1>{title}</h1>
                {abstract && <em className="Abstract">{abstract}</em>}
                {img && (
                    <figure>
                        <img src={img} className="App-logo" alt={caption} />
                        <figcaption className="Caption">{caption}</figcaption>
                    </figure>
                )}
            </header>
            <article className="Content">
                <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                        code({ node, inline, className, children, ...props }: any) {
                            const match = /language-(\w+)/.exec(className || '');
                            return !inline && match ? (
                                <SyntaxHighlighter
                                    style={tomorrow}
                                    language={match[1]}
                                    PreTag="div"
                                    {...props}
                                >
                                    {String(children).replace(/\n$/, '')}
                                </SyntaxHighlighter>
                            ) : (
                                <code className={className} {...props}>
                                    {children}
                                </code>
                            );
                        },
                    }}
                >
                    {content}
                </ReactMarkdown>
            </article>
            {img2 && (
                <figure>
                    <img src={img2} className="App-logo" alt="" />
                </figure>
            )}
        </div>
    );
}

export default Blog;

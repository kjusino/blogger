function Blog({
    route,
    title,
    abstract,
    pics,
    caption,
    content,
}: {
    route: string;
    title: string;
    abstract: string;
    pics: string[];
    caption: string;
    content: JSX.Element[];
}) {
    const img = require(`../articles/pics/${pics[0]}`);

    const img2 =
        pics.length > 1 ? require(`../articles/pics/${pics[1]}`) : undefined;
    return (
        <div className="Article">
            <header className="Article-header">
                <h1>{title}</h1>
                <em className="Abstract">{abstract}</em>
                <figure>
                    <img src={img} className="App-logo" alt={caption} />
                    <figcaption className="Caption">{caption}</figcaption>
                </figure>
            </header>
            <article className="Content">{content}</article>
            {img2 && (
                <figure>
                    <img src={img2} className="App-logo" alt={caption} />
                </figure>
            )}
        </div>
    );
}

export default Blog;

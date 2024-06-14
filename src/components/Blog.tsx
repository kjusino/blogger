function Blog({
    route,
    title,
    abstract,
    pic,
    caption,
    content,
}: {
    route: string;
    title: string;
    abstract: string;
    pic: string;
    caption: string;
    content: JSX.Element[];
}) {
    const img = require(`../content/${pic}`);
    return (
        <div className="Article">
            <header className="Article-header">
                <h1>{title}</h1>
                <em className="Abstract">{abstract}</em>
                <figure>
                    <img src={img} className="App-logo" alt={caption} />
                    <figcaption>{caption}</figcaption>
                </figure>
            </header>
            <article className="Content">{content}</article>
        </div>
    );
}

export default Blog;

export interface ArticleProps {
    route: string;
    title: string;
    abstract: string;
    pics: string[];
    caption: string;
    backgroundColor?: string;
    textColor?: string;
    content: JSX.Element[];
}

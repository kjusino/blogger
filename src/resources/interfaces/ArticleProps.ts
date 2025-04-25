import { Tags } from '../enums/Tags';

export interface ArticleProps {
    route: string;
    title: string;
    abstract?: string;
    pics: string[];
    caption: string;
    backgroundColor?: string;
    textColor?: string;
    tags?: Tags[];
    content: JSX.Element[];
}

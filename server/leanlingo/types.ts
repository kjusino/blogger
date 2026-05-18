export type QuestionType = 'MC' | 'FIB' | 'PO' | 'SE' | 'ORD';

export type QuestionRow = {
    id: string;
    world: string;
    unit: string;
    lesson: string;
    q_index: string;
    type: QuestionType;
    prompt: string;
    code: string;
    options: string[];
    answer: string;
    explanation: string;
    ord_items: string[];
    book_ref: string;
    lesson_title: string;
    quote: string;
    source_url: string;
};

export type AttemptRow = {
    timestamp: string;
    lesson_id: string;
    question_id: string;
    attempts: number;
    correct: boolean;
    xp_awarded: number;
};

export type LessonNode = {
    id: string;
    title: string;
    book_ref: string;
    questions: QuestionRow[];
};

export type UnitNode = {
    id: string;
    lessons: LessonNode[];
};

export type WorldNode = {
    id: string;
    units: UnitNode[];
};

export type QuestionTree = {
    worlds: WorldNode[];
};

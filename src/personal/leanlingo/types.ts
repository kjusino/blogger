export type QuestionType = 'MC' | 'FIB' | 'PO' | 'SE' | 'ORD';

export type Question = {
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

export type Lesson = {
    id: string;
    title: string;
    book_ref: string;
    questions: Question[];
};

export type Unit = {
    id: string;
    lessons: Lesson[];
};

export type World = {
    id: string;
    units: Unit[];
};

export type QuestionTree = { worlds: World[] };

export type Attempt = {
    timestamp: string;
    lesson_id: string;
    question_id: string;
    attempts: number;
    correct: boolean;
    xp_awarded: number;
};

export type ProgressState = {
    xp: number;
    streak: number;
    lastActiveDate: string;
    completedLessons: Set<string>;
    perfectLessons: Set<string>;
    unlockedWorlds: Set<string>;
    unlockedUnits: Set<string>;
};

export type SetEntry = { weight: string };
export type HistoryEntry = { weight: number; date: string };

export type ExerciseEntry = {
    name?: string;
    pr: number;
    sets: Record<string, SetEntry>;
    history: HistoryEntry[];
};

export type WorkoutData = Record<string, ExerciseEntry>;

export type BlockType = 'compound' | 'isolation' | 'core';

export type Pair = {
    a: string;
    b: string;
    sets: number;
    repsA: string;
    repsB: string;
    type: BlockType;
};

export type Block = {
    name: string;
    note: string;
    pairs: Pair[];
};

export type ProgramKey = 'push' | 'pull' | 'legs';

export type Program = {
    label: string;
    subtitle: string;
    icon: string;
    blocks: Block[];
};

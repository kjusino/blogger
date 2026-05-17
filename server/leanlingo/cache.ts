import { readAllQuestions } from './excel';
import type { LessonNode, QuestionRow, QuestionTree, UnitNode, WorldNode } from './types';

const TTL_MS = 5 * 60 * 1000;

let cached: { tree: QuestionTree; expiresAt: number } | null = null;
let inflight: Promise<QuestionTree> | null = null;

function buildTree(rows: QuestionRow[]): QuestionTree {
    const worldMap = new Map<string, Map<string, Map<string, QuestionRow[]>>>();
    for (const r of rows) {
        let u = worldMap.get(r.world);
        if (!u) { u = new Map(); worldMap.set(r.world, u); }
        let l = u.get(r.unit);
        if (!l) { l = new Map(); u.set(r.unit, l); }
        let q = l.get(r.lesson);
        if (!q) { q = []; l.set(r.lesson, q); }
        q.push(r);
    }
    const worldIdSort = (a: string, b: string) =>
        Number(a.replace(/^w/, '')) - Number(b.replace(/^w/, ''));
    const unitIdSort = (a: string, b: string) =>
        Number(a.split('-u')[1]) - Number(b.split('-u')[1]);
    const lessonIdSort = (a: string, b: string) =>
        Number(a.split('-l')[1]) - Number(b.split('-l')[1]);
    const qSort = (a: QuestionRow, b: QuestionRow) =>
        a.q_index.localeCompare(b.q_index);

    const worlds: WorldNode[] = [];
    for (const wid of Array.from(worldMap.keys()).sort(worldIdSort)) {
        const units: UnitNode[] = [];
        const uMap = worldMap.get(wid)!;
        for (const uid of Array.from(uMap.keys()).sort(unitIdSort)) {
            const lessons: LessonNode[] = [];
            const lMap = uMap.get(uid)!;
            for (const lid of Array.from(lMap.keys()).sort(lessonIdSort)) {
                const qs = lMap.get(lid)!.slice().sort(qSort);
                lessons.push({
                    id: lid,
                    title: qs[0]?.lesson_title ?? lid,
                    book_ref: qs[0]?.book_ref ?? '',
                    questions: qs,
                });
            }
            units.push({ id: uid, lessons });
        }
        worlds.push({ id: wid, units });
    }
    return { worlds };
}

export async function getQuestionTree(): Promise<QuestionTree> {
    if (cached && Date.now() < cached.expiresAt) return cached.tree;
    if (inflight) return inflight;
    inflight = (async () => {
        try {
            const rows = await readAllQuestions();
            const tree = buildTree(rows);
            cached = { tree, expiresAt: Date.now() + TTL_MS };
            return tree;
        } finally {
            inflight = null;
        }
    })();
    return inflight;
}

export function bustCache(): void {
    cached = null;
}

import React from 'react';

const KEYWORDS = new Set([
    'def', 'theorem', 'lemma', 'example', 'instance', 'class', 'structure',
    'inductive', 'match', 'with', 'fun', 'by', 'where', 'import', 'open',
    'namespace', 'section', 'variable', 'let', 'in', 'do', 'return',
    'if', 'then', 'else', 'mutual', 'end', 'show', 'have', 'exact',
    'apply', 'intro', 'rw', 'simp', 'ring', 'omega', 'induction',
    'constructor', 'assumption', 'cases', 'unfold', 'rfl',
    'termination_by', 'decreasing_by', 'deriving', 'extends',
]);

const COMMANDS = new Set(['#eval', '#check', '#print', '#reduce']);

type Tok = { text: string; kind: 'kw' | 'cmd' | 'str' | 'num' | 'cmt' | 'sym' | 't' };

function tokenize(src: string): Tok[] {
    const out: Tok[] = [];
    let i = 0;
    while (i < src.length) {
        const c = src[i];
        // line comment
        if (c === '-' && src[i + 1] === '-') {
            let j = i;
            while (j < src.length && src[j] !== '\n') j++;
            out.push({ text: src.slice(i, j), kind: 'cmt' });
            i = j;
            continue;
        }
        // string
        if (c === '"') {
            let j = i + 1;
            while (j < src.length && src[j] !== '"') {
                if (src[j] === '\\') j += 2;
                else j++;
            }
            j++;
            out.push({ text: src.slice(i, j), kind: 'str' });
            i = j;
            continue;
        }
        // #eval, #check, etc.
        if (c === '#') {
            let j = i + 1;
            while (j < src.length && /[A-Za-z_]/.test(src[j])) j++;
            const word = src.slice(i, j);
            out.push({ text: word, kind: COMMANDS.has(word) ? 'cmd' : 't' });
            i = j;
            continue;
        }
        // word
        if (/[A-Za-z_]/.test(c)) {
            let j = i;
            while (j < src.length && /[A-Za-z0-9_']/.test(src[j])) j++;
            const word = src.slice(i, j);
            out.push({ text: word, kind: KEYWORDS.has(word) ? 'kw' : 't' });
            i = j;
            continue;
        }
        // number
        if (/[0-9]/.test(c)) {
            let j = i;
            while (j < src.length && /[0-9.]/.test(src[j])) j++;
            out.push({ text: src.slice(i, j), kind: 'num' });
            i = j;
            continue;
        }
        // symbol / whitespace
        let j = i + 1;
        out.push({ text: src.slice(i, j), kind: 't' });
        i = j;
    }
    return out;
}

const COLOR: Record<Tok['kind'], string> = {
    kw: '#c084fc',
    cmd: '#22d3ee',
    str: '#86efac',
    num: '#fbbf24',
    cmt: '#94a3b8',
    sym: '#e5e7eb',
    t: '#e5e7eb',
};

export default function LeanCode({ code }: { code: string }) {
    if (!code.trim()) return null;
    const toks = tokenize(code);
    return (
        <pre className="leanlingo-code">
            <code>
                {toks.map((t, i) => (
                    <span key={i} style={{ color: COLOR[t.kind] }}>{t.text}</span>
                ))}
            </code>
        </pre>
    );
}

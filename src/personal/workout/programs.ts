import { Program, ProgramKey } from './types';

export const PROGRAMS: Record<ProgramKey, Program> = {
    push: {
        label: 'PUSH',
        subtitle: 'Chest · Shoulders · Triceps',
        icon: '⬆',
        blocks: [
            {
                name: 'BLOCK A — Heavy Compounds',
                note: 'Chest ↔ Shoulders superset · 90s rest between rounds',
                pairs: [
                    {
                        a: 'Barbell Bench Press',
                        b: 'Shoulder Press',
                        sets: 4,
                        repsA: '6-8',
                        repsB: '6-8',
                        type: 'compound',
                    },
                ],
            },
            {
                name: 'BLOCK B — Volume Compounds',
                note: 'Incline ↔ Machine Press superset · 60s rest',
                pairs: [
                    {
                        a: 'Incline Bench Press',
                        b: 'Machine Incline Press',
                        sets: 3,
                        repsA: '8-10',
                        repsB: '10-12',
                        type: 'compound',
                    },
                ],
            },
            {
                name: 'BLOCK C — Isolation Finishers',
                note: 'Lateral delts ↔ Triceps superset · 45s rest',
                pairs: [
                    {
                        a: 'Cable Tricep Pull-Down',
                        b: 'DB Lateral Raises',
                        sets: 3,
                        repsA: '12-15',
                        repsB: '12-15',
                        type: 'isolation',
                    },
                ],
            },
            {
                name: 'BLOCK D — Core Circuit',
                note: 'No rest between exercises · 60s rest between rounds',
                pairs: [
                    {
                        a: 'Ab Roll-ups',
                        b: 'Oblique Curls',
                        sets: 2,
                        repsA: '15',
                        repsB: '15',
                        type: 'core',
                    },
                ],
            },
        ],
    },
    pull: {
        label: 'PULL',
        subtitle: 'Back · Rear Delts · Biceps',
        icon: '⬇',
        blocks: [
            {
                name: 'BLOCK A — Heavy Compounds',
                note: 'Deadlift ↔ Row superset · 90s rest between rounds',
                pairs: [
                    {
                        a: 'Deadlifts',
                        b: 'Machine Row',
                        sets: 4,
                        repsA: '5-6',
                        repsB: '6-8',
                        type: 'compound',
                    },
                ],
            },
            {
                name: 'BLOCK B — Volume Compounds',
                note: 'Vertical ↔ Horizontal pull superset · 60s rest',
                pairs: [
                    {
                        a: 'Lateral Pulldown',
                        b: 'Machine Overhead Row',
                        sets: 3,
                        repsA: '8-10',
                        repsB: '10-12',
                        type: 'compound',
                    },
                ],
            },
            {
                name: 'BLOCK C — Isolation Finishers',
                note: 'Biceps ↔ Forearms superset · 45s rest',
                pairs: [
                    {
                        a: 'E-Z Bar Bicep Curls',
                        b: 'E-Z Bar Forearm Curls',
                        sets: 3,
                        repsA: '10-12',
                        repsB: '12-15',
                        type: 'isolation',
                    },
                ],
            },
            {
                name: 'BLOCK D — Core Circuit',
                note: 'No rest between exercises · 60s rest between rounds',
                pairs: [
                    {
                        a: 'Leg Raises',
                        b: 'Weighted Scissor Kicks',
                        sets: 2,
                        repsA: '15',
                        repsB: '15',
                        type: 'core',
                    },
                ],
            },
        ],
    },
    legs: {
        label: 'LEGS',
        subtitle: 'Quads · Hamstrings · Glutes · Calves',
        icon: '🦵',
        blocks: [
            {
                name: 'BLOCK A — Heavy Compounds',
                note: 'Squat ↔ Hip Thrust superset · 90s rest between rounds',
                pairs: [
                    {
                        a: 'Barbell Squats',
                        b: 'Smith Machine Hip Thrusts',
                        sets: 4,
                        repsA: '6-8',
                        repsB: '8-10',
                        type: 'compound',
                    },
                ],
            },
            {
                name: 'BLOCK B — Volume Compounds',
                note: 'Bulgarian Split Squat ↔ Leg Press superset · 60s rest',
                pairs: [
                    {
                        a: 'DB Bulgarian Split Squats',
                        b: 'Leg Press',
                        sets: 3,
                        repsA: '10-12/leg',
                        repsB: '10-12',
                        type: 'compound',
                    },
                ],
            },
            {
                name: 'BLOCK C — Isolation Finishers',
                note: 'Quads ↔ Hamstrings superset · 45s rest',
                pairs: [
                    {
                        a: 'Leg Extensions',
                        b: 'Seated Leg Curls',
                        sets: 3,
                        repsA: '12-15',
                        repsB: '12-15',
                        type: 'isolation',
                    },
                ],
            },
            {
                name: 'BLOCK D — Calves + Core',
                note: 'Superset calves with core · 45s rest',
                pairs: [
                    {
                        a: 'Calf Raises',
                        b: 'Knee Raises',
                        sets: 2,
                        repsA: '15-20',
                        repsB: '15',
                        type: 'isolation',
                    },
                ],
            },
        ],
    },
};

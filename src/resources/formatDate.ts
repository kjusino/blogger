// Format a 'YYYY-MM-DD' article date as e.g. "June 27, 2026".
// Parsed as local (not UTC) so the day never shifts across timezones.
export function formatDate(dateStr: string): string {
    const [year, month, day] = dateStr.split('-').map(Number);
    const date = new Date(year, month - 1, day);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
    });
}

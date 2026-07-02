export interface PodcastEpisode {
    guid: string;
    episodeNumber: number;
    title: string;
    description: string;
    audioUrl: string;
    duration: string;
    pubDate: string;
    fileSize: number;
}

export const PODCAST_TITLE = 'Kasike Kéne Podcast';
export const PODCAST_DESCRIPTION =
    'Conversations and thoughts on engineering, culture, and building things.';
export const PODCAST_AUTHOR = 'Kenneth Jusino';

export const BLOB_BASE =
    'https://kennethjusinoblog.blob.core.windows.net/podcast';

export const episodes: PodcastEpisode[] = [
    // Add episodes here, newest first. Example:
    //
    // {
    //     guid: 'kkp-001',
    //     episodeNumber: 1,
    //     title: 'Pilot Episode',
    //     description: 'Welcome to the podcast.',
    //     audioUrl: `${BLOB_BASE}/episodes/ep001.mp3`,
    //     duration: '32:15',
    //     pubDate: '2026-07-02',
    //     fileSize: 30_000_000,
    // },
];

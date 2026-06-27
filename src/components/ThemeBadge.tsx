import { Tags } from '../resources/enums/Tags';
import { conicPie } from '../resources/themeColors';

// A circular pie swatch: one solid color for a single-theme article, or
// equal colored wedges for an article that spans multiple themes.
const ThemeBadge = ({ tags, size = 14 }: { tags?: Tags[]; size?: number }) => (
    <span
        title={(tags ?? []).join(' + ')}
        aria-label={(tags ?? []).join(' + ')}
        style={{
            display: 'inline-block',
            width: size,
            height: size,
            borderRadius: '50%',
            background: conicPie(tags),
            flexShrink: 0,
            verticalAlign: 'middle',
        }}
    />
);

export default ThemeBadge;

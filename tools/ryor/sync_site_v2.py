"""Copy the rescue route between site-next and its versioned copy on ryor-26oct.

By default, capture the edited site in tools/ryor/site-next. With --install, copy
that version into --site. Other site files, films and sections stay in place.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COPY = ROOT / 'tools/ryor/site-next'
FILES = ['src/app/rescue/page.tsx', 'src/components/Nav.tsx', 'src/rescue/data.ts',
         'src/rescue/Rescue.tsx', 'src/rescue/Scene.tsx', 'src/rescue/Picker.tsx',
         'src/rescue/FilmPlayer.tsx', 'src/rescue/LiveMission.tsx', 'src/rescue/missions.ts',
         'src/rescue/live.ts', 'src/sections/Mission.tsx']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=Path, default=Path('/Volumes/Tashon/opencv/showcase/site-next'))
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    source, dest = (COPY, args.site) if args.install else (args.site, COPY)
    paths = FILES + [str(p.relative_to(source)) for p in (source / 'public/rescue-data').glob('*') if p.is_file()]
    paths += [str(p.relative_to(source)) for p in (source / 'public/film/judge').glob('*') if p.is_file()]
    # The existing one-door Unity video is reused byte for byte.
    paths += ['public/evidence/run_family_ranch_5of5.mp4']
    # Remove withdrawn old-engine files from the receiving rescue-data directory.
    for old in (dest / 'public/rescue-data').glob('*__r*__easy.json'):
        if str(old.relative_to(dest)) not in paths:
            old.unlink()
    for name in paths:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    print(f'Copied {len(paths)} files from {source} to {dest}')


if __name__ == '__main__':
    main()

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
         'src/rescue/Rescue.tsx', 'src/rescue/Scene.tsx']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=Path, default=Path('/Volumes/Tashon/opencv/showcase/site-next'))
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    source, dest = (COPY, args.site) if args.install else (args.site, COPY)
    paths = FILES + [str(p.relative_to(source)) for p in (source / 'public/rescue-data').glob('*') if p.is_file()]
    assert len(paths) == 22, f'Expected 5 source files and 17 data/license files, got {len(paths)}'
    for name in paths:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    print(f'Copied {len(paths)} files from {source} to {dest}')


if __name__ == '__main__':
    main()

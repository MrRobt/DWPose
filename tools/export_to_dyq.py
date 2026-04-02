#!/usr/bin/env python3
"""Export minimal DWPose runtime files into another project (e.g. dyq)."""

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

KEY_FILES = [
    Path('ControlNet-v1-1-nightly/dwpose_server.py'),
    Path('ControlNet-v1-1-nightly/annotator/dwpose/__init__.py'),
    Path('ControlNet-v1-1-nightly/annotator/dwpose/wholebody.py'),
    Path('ControlNet-v1-1-nightly/annotator/dwpose/onnxdet.py'),
    Path('ControlNet-v1-1-nightly/annotator/dwpose/onnxpose.py'),
    Path('ControlNet-v1-1-nightly/annotator/dwpose/util.py'),
]


def copy_key_files(target_root: Path) -> None:
    copied = []
    for relative_src in KEY_FILES:
        src = REPO_ROOT / relative_src
        dst = target_root / relative_src
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(str(relative_src))

    print('Copied files:')
    for item in copied:
        print(f'  - {item}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Copy key DWPose runtime files into dyq project.')
    parser.add_argument('target', help='Target project path, e.g. /path/to/dyq')
    args = parser.parse_args()

    target_root = Path(args.target).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    copy_key_files(target_root)

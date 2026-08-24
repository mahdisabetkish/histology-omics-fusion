"""Report which raw files are present, complete and readable.

Useful after an interrupted download: a part-finished image is full-size on
disk with holes in the middle, so a directory listing looks fine while the file
is unusable.

    python scripts/check_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.samples import SECTIONS, donor_of, split_of  # noqa: E402

FILES = ["filtered_feature_bc_matrix.h5", "tissue_positions_list.txt",
         "scalefactors_json.json", "tissue_lowres_image.png", "full_image.tif"]


def tiff_ok(path):
    try:
        with open(path, "rb") as handle:
            return handle.read(2) in (b"II", b"MM")
    except OSError:
        return False


def main():
    raw = Path("data/raw")
    processed = Path("data/processed")
    print(f"{'section':>8}  {'donor':<7} {'split':<5} "
          f"{'counts':>7} {'pos':>4} {'lowres':>7} {'image':>10}  built")
    print("-" * 72)

    ready = 0
    for section in sorted(SECTIONS):
        folder = raw / section
        marks, image_note = [], ""
        for name in FILES:
            path = folder / name
            if not path.exists():
                marks.append("-")
            elif name == "full_image.tif":
                if (folder / "full_image.tif.parts").exists():
                    marks.append("partial")
                elif not tiff_ok(path):
                    marks.append("CORRUPT")
                else:
                    marks.append(f"{path.stat().st_size / 1e6:.0f}MB")
            else:
                marks.append("ok")
        built = "yes" if (processed / section / "patches.npy").exists() else "no"
        if all(m not in ("-", "partial", "CORRUPT") for m in marks):
            ready += 1
        print(f"{section:>8}  {donor_of(section):<7} {split_of(section):<5} "
              f"{marks[0]:>7} {marks[1]:>4} {marks[3]:>7} {marks[4]:>10}  {built}")

    print("-" * 72)
    print(f"{ready}/{len(SECTIONS)} sections have all raw files")
    if ready < len(SECTIONS):
        print("run `python -m src.data.download` to fetch or resume the rest")
    return 0 if ready == len(SECTIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())

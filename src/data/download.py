"""Fetch the raw spatialLIBD DLPFC files.

Four things are needed per section: the filtered UMI count matrix, the
full-resolution H&E image, the spot coordinates in full-resolution pixel space,
and the manual layer annotation. The counts and images live in the public
spatial-dlpfc S3 bucket; the coordinates and annotations are distributed with
the HumanPilot analysis repository.

The host throttles each connection to roughly 0.2 MB/s, which makes the twelve
533 MB histology images painful over a single stream. Large files are therefore
pulled as parallel byte ranges and reassembled. Completed ranges are recorded
next to the output file, so an interrupted download resumes where it stopped
instead of starting over.

    python -m src.data.download                      # everything, ~6.6 GB
    python -m src.data.download --skip-images        # counts and metadata only
    python -m src.data.download --sections 151673 151674
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .samples import SECTIONS

S3 = "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/"
GITHUB_RAW = "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/"

# The S3 bucket and the GitHub CDN both reject the default urllib agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/131.0"}

CHUNK = 4 << 20
WORKERS = 32
PARALLEL_ABOVE = 32 << 20


def remote_size(url):
    request = urllib.request.Request(url, headers={**HEADERS, "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
    return int(content_range.rsplit("/", 1)[1]) if "/" in content_range else None


def _read_range(url, start, end, retries=5):
    """Return bytes [start, end) of `url`, retrying on transient failures."""
    for attempt in range(retries):
        try:
            headers = {**HEADERS, "Range": f"bytes={start}-{end - 1}"}
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            if len(payload) == end - start:
                return payload
            raise OSError(f"short read {len(payload)} != {end - start}")
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise OSError("unreachable")


def _fetch_small(url, target):
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=120) as response:
                target.write_bytes(response.read())
            return
        except Exception as err:
            if attempt == 3:
                raise
            print(f"  retry {target.name} ({err})")
            time.sleep(5 * (attempt + 1))


def _fetch_parallel(url, target, total, workers):
    """Download `url` as parallel byte ranges into a sparse file."""
    ledger = target.with_suffix(target.suffix + ".parts")
    done = set()
    if target.exists() and ledger.exists():
        try:
            state = json.loads(ledger.read_text())
            if state.get("total") == total:
                done = set(state.get("done", []))
            else:
                target.unlink(missing_ok=True)
        except (ValueError, OSError):
            done = set()

    n_chunks = (total + CHUNK - 1) // CHUNK
    with open(target, "r+b" if target.exists() else "wb") as handle:
        handle.truncate(total)

    todo = [i for i in range(n_chunks) if i not in done]
    if not todo:
        print(f"  have  {target.name}")
        ledger.unlink(missing_ok=True)
        return

    # The file is now full-size but mostly holes. Record that before fetching
    # anything, so an interrupted run leaves an obvious marker rather than a
    # complete-looking file that turns out to be zeros when something reads it.
    ledger.write_text(json.dumps({"total": total, "done": sorted(done)}))

    lock = threading.Lock()
    handle = open(target, "r+b")
    started = time.time()
    counter = {"n": len(done)}

    def worker(index):
        start = index * CHUNK
        payload = _read_range(url, start, min(start + CHUNK, total))
        with lock:
            handle.seek(start)
            handle.write(payload)
            done.add(index)
            counter["n"] += 1
            fetched = (counter["n"] - (n_chunks - len(todo))) * CHUNK
            rate = fetched / max(time.time() - started, 1e-6) / 1e6
            pct = 100 * counter["n"] / n_chunks
            sys.stdout.write(
                f"\r  get   {target.name:<28s} {pct:5.1f}%  {rate:5.2f} MB/s"
            )
            sys.stdout.flush()
            if counter["n"] % 25 == 0:
                ledger.write_text(json.dumps({"total": total, "done": sorted(done)}))

    try:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(worker, todo))
    except BaseException:
        with lock:
            ledger.write_text(json.dumps({"total": total, "done": sorted(done)}))
        raise
    finally:
        handle.close()
    sys.stdout.write("\n")
    ledger.unlink(missing_ok=True)


def download(url, target, workers=WORKERS):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    total = remote_size(url)
    if total is None:
        if not target.exists():
            _fetch_small(url, target)
            print(f"  get   {target.name}")
        return target

    ledger = target.with_suffix(target.suffix + ".parts")
    if target.exists() and target.stat().st_size == total and not ledger.exists():
        print(f"  have  {target.name}")
        return target

    if total < PARALLEL_ABOVE:
        _fetch_small(url, target)
        print(f"  get   {target.name:<28s} {total / 1e6:.1f} MB")
    else:
        _fetch_parallel(url, target, total, workers)
    return target


def fetch_section(section_id, raw_dir, skip_images=False, workers=WORKERS):
    print(f"[{section_id}]")
    out = raw_dir / section_id
    download(f"{S3}h5/{section_id}_filtered_feature_bc_matrix.h5",
             out / "filtered_feature_bc_matrix.h5", workers)
    download(f"{GITHUB_RAW}10X/{section_id}/tissue_positions_list.txt",
             out / "tissue_positions_list.txt", workers)
    download(f"{GITHUB_RAW}10X/{section_id}/scalefactors_json.json",
             out / "scalefactors_json.json", workers)
    # Used only as a backdrop for the spatial maps in the dashboard.
    download(f"{S3}images/{section_id}_tissue_lowres_image.png",
             out / "tissue_lowres_image.png", workers)
    if not skip_images:
        download(f"{S3}images/{section_id}_full_image.tif",
                 out / "full_image.tif", workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="*", default=sorted(SECTIONS))
    parser.add_argument("--skip-images", action="store_true",
                        help="skip the full-resolution TIFFs (~6.4 GB total)")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help="parallel range requests per large file")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    raw_dir = Path(args.out)
    print("layer annotations")
    download(f"{GITHUB_RAW}10X/barcode_level_layer_map.tsv",
             raw_dir / "barcode_level_layer_map.tsv", args.workers)

    unknown = [s for s in args.sections if s not in SECTIONS]
    if unknown:
        raise SystemExit(f"unknown section(s): {', '.join(unknown)}")

    for section_id in args.sections:
        fetch_section(section_id, raw_dir, args.skip_images, args.workers)

    total = sum(f.stat().st_size for f in raw_dir.rglob("*") if f.is_file())
    print(f"\ndone. {total / 1e9:.2f} GB in {raw_dir}{os.sep}")


if __name__ == "__main__":
    main()

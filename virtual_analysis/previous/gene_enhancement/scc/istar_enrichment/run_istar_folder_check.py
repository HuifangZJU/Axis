import sys
from pathlib import Path
import subprocess

REQUIRED = [
    "he-raw.jpg",
    "cnts.tsv",
    "locs-raw.tsv",
    "pixel-size-raw.txt",
    "radius-raw.txt",
]

def main(list_file: str):
    with open(list_file) as f:
        for line in f:
            prefix = line.strip()
            if not prefix or prefix.startswith("#"):
                continue

            folder = Path(prefix)
            # normalize to folder/
            if not str(folder).endswith("/"):
                folder = Path(str(folder) + "/")

            if not folder.is_dir():
                print(f"[SKIP] Folder does not exist: {folder}")
                continue

            missing = [name for name in REQUIRED if not (folder / name).is_file()]
            if missing:
                print(f"[SKIP] {folder} is missing: {', '.join(missing)}")
                continue

            print(f"[RUN] All required files present in: {folder}")



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} folder_list.txt [hf_run.sh]", file=sys.stderr)
        sys.exit(1)

    list_file = sys.argv[1]
    hf_run = sys.argv[2] if len(sys.argv) >= 3 else "hf_run.sh"
    main(list_file)

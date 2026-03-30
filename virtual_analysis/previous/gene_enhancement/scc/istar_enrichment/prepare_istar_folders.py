#!/usr/bin/env python
import sys
from pathlib import Path

from PIL import Image  # pip install pillow if needed

# Allow large images; you're working with trusted SCC histology data.
Image.MAX_IMAGE_PIXELS = None  # or e.g. 500_000_000

def pick_first_match(folder: Path, patterns):
    """Return the first Path matching any of the given glob patterns, or None."""
    for pat in patterns:
        matches = sorted(folder.glob(pat))
        if matches:
            return matches[0]
    return None


def ensure_he_raw(folder: Path):
    """
    1) Find an image (.jpg or .png).
    2) If .png: convert to JPEG and save as he-raw.jpg.
       If .jpg: rename to he-raw.jpg.
    """
    target = folder / "he-raw.jpg"

    # If target already exists, assume it's fine
    if target.is_file():
        print(f"  [OK] he-raw.jpg already exists.")
        return

    img_path = pick_first_match(folder, ["*.jpg", "*.jpeg", "*.png"])
    if img_path is None:
        print(f"  [WARN] No .jpg/.jpeg/.png image found in {folder}")
        return

    suffix = img_path.suffix.lower()
    if suffix == ".png":
        # Convert PNG → JPEG
        print(f"  [IMG] Converting PNG -> he-raw.jpg from {img_path.name}")
        with Image.open(img_path) as img:
            # Handle alpha
            if img.mode in ("RGBA", "LA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            else:
                img = img.convert("RGB")
            img.save(target, quality=95)
        # If you want to delete the original PNG, uncomment:
        # img_path.unlink()
    else:
        # JPEG already; just rename if needed
        print(f"  [IMG] Renaming {img_path.name} -> he-raw.jpg")
        img_path.rename(target)


def ensure_cnts(folder: Path):
    """
    2) Find *cnts*.tsv and rename to cnts.tsv (if not already).
    """
    target = folder / "cnts.tsv"
    if target.is_file():
        print("  [OK] cnts.tsv already exists.")
        return

    src = pick_first_match(folder, ["*cnts*.tsv"])
    if src is None:
        print(f"  [WARN] No *cnts*.tsv found in {folder}")
        return

    print(f"  [CNT] Renaming {src.name} -> cnts.tsv")
    src.rename(target)


def ensure_locs_raw(folder: Path):
    """
    3) Find *locs*.tsv and rename to locs-raw.tsv (if not already).
    """
    target = folder / "locs-raw.tsv"
    if target.is_file():
        print("  [OK] locs-raw.tsv already exists.")
        return

    src = pick_first_match(folder, ["*locs*.tsv"])
    if src is None:
        print(f"  [WARN] No *locs*.tsv found in {folder}")
        return

    print(f"  [LOC] Renaming {src.name} -> locs-raw.tsv")
    src.rename(target)


def main(list_file: str):
    with open(list_file) as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue

            folder = Path(raw)
            # normalize to folder/
            if not str(folder).endswith("/"):
                folder = Path(str(folder) + "/")

            if not folder.is_dir():
                print(f"[SKIP] Not a directory: {folder}")
                continue

            print(f"\n[PROC] Folder: {folder}")

            # ---- folder-level "already prepared" check ----
            he_ok = (folder / "he-raw.jpg").is_file()
            cnts_ok = (folder / "cnts.tsv").is_file()
            locs_ok = (folder / "locs-raw.tsv").is_file()

            if he_ok and cnts_ok and locs_ok:
                print("  [SKIP] he-raw.jpg, cnts.tsv, and locs-raw.tsv all exist. Skipping.")
                continue

            # Otherwise, do per-file preparation
            ensure_he_raw(folder)
            ensure_cnts(folder)
            ensure_locs_raw(folder)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} folder_list.txt", file=sys.stderr)
        sys.exit(1)

    main(sys.argv[1])

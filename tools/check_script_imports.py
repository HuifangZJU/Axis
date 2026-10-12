#!/usr/bin/env python
"""Check that every import statement in the reproducibility scripts resolves.

Each script is analysed statically (nothing in the script is executed): all
``import`` / ``from ... import`` statements, including those nested in
functions, are resolved with the script's own directory first on
``sys.path`` (as when the script is run directly), and for
``from module import name`` the name must exist in the module.

Failures are split into

* **broken** – a module or name that should come from this repository
  (``axis`` or a sibling helper script) does not resolve: always an error;
* **missing third-party** – an external package is not installed in the
  current environment: reported, and an error only with ``--strict``.

Usage::

    python tools/check_script_imports.py reproducibility examples [--strict]
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def iter_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, None
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative imports do not occur in scripts
                yield node.lineno, "." * node.level + (node.module or ""), [a.name for a in node.names]
            else:
                yield node.lineno, node.module, [a.name for a in node.names]


#: Third-party distributions the scripts may use (core, extras and figure-only
#: dependencies). Any other unresolved top-level module is treated as a broken
#: reference to repository code.
THIRD_PARTY = {
    "alphashape",
    "anndata",
    "cv2",
    "einops",
    "h5py",
    "igraph",
    "imageio",
    "imgaug",
    "joblib",
    "leidenalg",
    "matplotlib",
    "mpl_toolkits",
    "networkx",
    "numpy",
    "ot",
    "pandas",
    "paste",
    "patsy",
    "PIL",
    "plotly",
    "pyreadr",
    "pyvista",
    "scanpy",
    "scipy",
    "scvi",
    "seaborn",
    "shapely",
    "skimage",
    "sklearn",
    "squidpy",
    "statsmodels",
    "tensorboardX",
    "tifffile",
    "torch",
    "torchvision",
    "tqdm",
    "umap",
    "xgboost",
    "yaml",
    "nibabel",
}


def is_local(module: str, script_dir: Path) -> bool:
    top = module.split(".")[0]
    if top == "axis" or (script_dir / f"{top}.py").exists() or (script_dir / top / "__init__.py").exists():
        return True
    if top in sys.stdlib_module_names or top in THIRD_PARTY:
        return False
    return True


def check_script(path: Path):
    script_dir = path.parent.resolve()
    tree = ast.parse(path.read_text(), filename=str(path))
    broken, missing = [], []
    sys.path.insert(0, str(script_dir))
    try:
        for lineno, module, names in iter_imports(tree):
            local = is_local(module, script_dir)
            try:
                mod = importlib.import_module(module)
            except ModuleNotFoundError as exc:
                missing_top = (exc.name or "").split(".")[0]
                third_party_missing = missing_top in THIRD_PARTY or (missing_top in sys.stdlib_module_names)
                bucket = missing if third_party_missing else broken
                bucket.append(f"{path}:{lineno}: cannot import {module!r} ({exc})")
                continue
            except Exception as exc:  # import-time error inside a dependency
                (broken if local else missing).append(f"{path}:{lineno}: importing {module!r} raised {type(exc).__name__}: {exc}")
                continue
            for name in names or []:
                if name == "*" or hasattr(mod, name):
                    continue
                try:
                    if importlib.util.find_spec(f"{module}.{name}") is not None:
                        continue
                except (ImportError, AttributeError, ValueError):
                    pass
                (broken if local or module.split(".")[0] == "axis" else missing).append(
                    f"{path}:{lineno}: {module!r} has no attribute {name!r}"
                )
    finally:
        sys.path.remove(str(script_dir))
        # forget sibling helper modules so that scripts in other folders get their own copies
        for name, mod in list(sys.modules.items()):
            file = getattr(mod, "__file__", None) or ""
            if file and Path(file).resolve().parent == script_dir:
                del sys.modules[name]
    return broken, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--strict", action="store_true", help="also fail on missing third-party packages")
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    scripts = sorted(p for root in args.paths for p in ([root] if root.is_file() else root.rglob("*.py")))
    all_broken, all_missing = [], []
    for script in scripts:
        broken, missing = check_script(script)
        all_broken += broken
        all_missing += missing
        status = "BROKEN" if broken else ("missing-deps" if missing else "ok")
        print(f"[{status:>12}] {script}")
    if all_missing:
        print("\nMissing third-party packages:")
        print("\n".join(sorted(set(all_missing))))
    if all_broken:
        print("\nBroken imports:")
        print("\n".join(all_broken))
    print(f"\n{len(scripts)} scripts checked: {len(all_broken)} broken imports, {len(all_missing)} missing third-party imports")
    return 1 if all_broken or (args.strict and all_missing) else 0


if __name__ == "__main__":
    sys.exit(main())

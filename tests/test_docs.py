"""The code shown in the README and the documentation must match the API.

Every Python block is parsed; each ``axis`` name it uses must exist and each
call must bind to the function's signature (argument count and keywords).
Attributes of objects returned by ``axis`` functions (e.g. ``fields.save``) are
checked against the returned class. Nothing is executed.
"""

import ast
import inspect
import re
import typing
from pathlib import Path

import pytest

import axis

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
FENCE = re.compile(r"^( *)```python[^\n]*\n(.*?)^\1```", re.S | re.M)


def _blocks():
    for f in FILES:
        text = f.read_text(encoding="utf-8")
        for m in FENCE.finditer(text):
            indent = len(m.group(1))
            code = "\n".join(line[indent:] for line in m.group(2).splitlines())
            line = text[: m.start()].count("\n") + 1
            yield pytest.param(code, id=f"{f.relative_to(ROOT).as_posix()}:{line}")


class _Instance:
    """Stands for an object of an ``axis`` class returned by a call."""

    def __init__(self, cls):
        self.cls = cls
        src = inspect.getsource(cls)
        self.attrs = set(re.findall(r"self\.(\w+)\s*[:=]", src)) | set(dir(cls))
        self.dynamic = "__getattr__" in vars(cls)  # e.g. DatasetConfig exposes its keys


def _chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return [node.id, *reversed(parts)]


def _lookup(parts, env):
    obj = env[parts[0]]
    for i, name in enumerate(parts[1:], start=1):
        where = ".".join(parts[:i])
        if isinstance(obj, _Instance):
            assert obj.dynamic or name in obj.attrs, f"{where} ({obj.cls.__name__}) has no attribute {name!r}"
            return None  # attribute values of instances are not tracked further
        assert hasattr(obj, name), f"{where} has no attribute {name!r}"
        obj = getattr(obj, name)
    return obj


def _returned_instance(func):
    try:
        ret = typing.get_type_hints(func).get("return")
    except Exception:
        return None
    if inspect.isclass(ret) and ret.__module__.startswith("axis"):
        return _Instance(ret)
    return None


@pytest.mark.parametrize("code", list(_blocks()))
def test_documented_code_matches_api(code):
    tree = ast.parse(code)
    env = {"axis": axis}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "axis":
            module = _lookup(node.module.split("."), env)
            for alias in node.names:
                assert hasattr(module, alias.name), f"{node.module} has no attribute {alias.name!r}"
                env[alias.asname or alias.name] = getattr(module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "axis":
                    env[alias.asname or alias.name.split(".")[0]] = _lookup(alias.name.split("."), env) if alias.asname else axis

    # statements in order, so that variables bound to axis objects are known when used
    for stmt in tree.body if isinstance(tree, ast.Module) else []:
        nested = list(ast.walk(stmt))
        inner = {id(n.value) for n in nested if isinstance(n, ast.Attribute)}
        for node in nested:
            if isinstance(node, ast.Attribute) and id(node) not in inner:
                parts = _chain(node)
                if parts and parts[0] in env:
                    _lookup(parts, env)
            if isinstance(node, ast.Call):
                parts = _chain(node.func)
                if not parts or parts[0] not in env:
                    continue
                func = _lookup(parts, env)
                if func is None or not callable(func):
                    continue
                if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
                    continue
                sig = inspect.signature(func)
                try:
                    sig.bind(*node.args, **{k.arg: k.value for k in node.keywords})
                except TypeError as err:
                    raise AssertionError(f"{'.'.join(parts)}{sig}: {err}") from None
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            parts = _chain(stmt.value.func)
            if parts and parts[0] in env:
                func = _lookup(parts, env)
                inst = _returned_instance(func) if callable(func) else None
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        if inst is not None:
                            env[target.id] = inst
                        else:
                            env.pop(target.id, None)

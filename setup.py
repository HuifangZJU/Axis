"""Build hook: ship the dataset configs in ``configs/`` inside the wheel.

``configs/`` stays the single source of truth in the repository; the wheel
gets a copy at ``axis/_configs`` so that ``axis.config.load_config("scc")``
also works after ``pip install``. All other metadata lives in pyproject.toml.
"""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class build_py_with_configs(build_py):
    def run(self):
        super().run()
        target = Path(self.build_lib) / "axis" / "_configs"
        self.mkpath(str(target))
        for config in sorted((Path(__file__).parent / "configs").glob("*.yaml")):
            self.copy_file(str(config), str(target / config.name))


setup(cmdclass={"build_py": build_py_with_configs})

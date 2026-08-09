"""Project-root resolution must stay inside the checkout.

This release is published as ``MammoDiffusion-v1.1`` but its historical working
copy sits inside a parent directory literally named ``MammoDiffusion``, which
now also holds a separate successor project. A resolver that matches a bare
directory name before a repository marker escapes the checkout as soon as
``data/`` is absent — the state a fresh clone starts from, and the state the
documented rebuild workflow begins in. Escaping is not a harmless miss: the
generator notebooks call ``EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)``
during setup, so a wrong root writes directories next to the successor project.

These tests execute the real resolver snippets shipped in the notebooks against
a synthetic collision layout, plus the two utility resolvers.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import ldm_project_paths  # noqa: E402
import shared_diffusers_assets as assets  # noqa: E402

BOOTSTRAP_PATTERN = re.compile(r"def _find_mammo_root\(\):.*?raise FileNotFoundError\(", re.S)
RESOLVER_PATTERN = re.compile(
    r"def find_project_root\(.*?\n(?=\S|\Z)", re.S
)
EXPECTED_BOOTSTRAPS = 9


def notebook_code_cells(path: Path) -> list[str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code"]


def build_collision_layout(base: Path, *, with_git: bool, with_data: bool) -> Path:
    """A checkout named like the release inside a parent carrying the bare name."""
    parent = base / "MammoDiffusion"
    checkout = parent / "MammoDiffusion-v1.1"
    for relative in ("notebooks/1_preprocessing", "notebooks/2_diffusers",
                     "notebooks/utility", "configs"):
        (checkout / relative).mkdir(parents=True)
    (checkout / "configs" / "classifier_protocol.json").write_text("{}", encoding="utf-8")
    (checkout / "README.md").write_text("x", encoding="utf-8")
    if with_git:
        (checkout / ".git").mkdir()
    if with_data:
        (checkout / "data").mkdir()
    # The separate successor project shares the parent; it must never be selected.
    (parent / "MammoDiffusion-v2").mkdir()
    return checkout


def resolve_from(checkout: Path, resolver, subdirectory: str) -> Path:
    previous = Path.cwd()
    os.chdir(checkout / subdirectory)
    try:
        return Path(resolver()).resolve()
    finally:
        os.chdir(previous)


class NotebookResolverTests(unittest.TestCase):
    """Run every resolver a notebook actually ships against the collision layout."""

    LAYOUTS = ((True, False), (False, True), (True, True))

    def _snippets(self, pattern: re.Pattern) -> list[tuple[str, str]]:
        found = []
        for path in sorted(ROOT.glob("notebooks/**/*.ipynb")):
            for source in notebook_code_cells(path):
                match = pattern.search(source)
                if match:
                    found.append((path.relative_to(ROOT).as_posix(), match.group(0)))
        return found

    def _assert_resolves_to_checkout(self, label: str, resolver) -> None:
        for with_git, with_data in self.LAYOUTS:
            with tempfile.TemporaryDirectory() as temporary:
                checkout = build_collision_layout(Path(temporary), with_git=with_git,
                                                  with_data=with_data)
                resolved = resolve_from(checkout, resolver, "notebooks/1_preprocessing")
            with self.subTest(source=label, git=with_git, data=with_data):
                self.assertEqual(resolved, checkout.resolve())
                self.assertNotEqual(resolved.name, "MammoDiffusion",
                                    "the resolver escaped to the identically named parent")

    def test_every_notebook_still_ships_the_shared_bootstrap(self):
        self.assertEqual(len(self._snippets(BOOTSTRAP_PATTERN)), EXPECTED_BOOTSTRAPS)

    def test_bootstrap_prefers_the_checkout_over_the_identically_named_parent(self):
        snippets = self._snippets(BOOTSTRAP_PATTERN)
        self.assertTrue(snippets)
        for label, snippet in snippets:
            body = snippet[: snippet.rindex("raise FileNotFoundError(")].rstrip()
            code = ("from pathlib import Path as _Path\n" + body
                    + "\n    raise FileNotFoundError('root not found')\n")
            namespace: dict = {}
            exec(compile(code, f"bootstrap:{label}", "exec"), namespace)  # noqa: S102
            self._assert_resolves_to_checkout(label, namespace["_find_mammo_root"])

    def test_notebook_resolvers_prefer_the_checkout_over_the_named_parent(self):
        snippets = self._snippets(RESOLVER_PATTERN)
        self.assertGreaterEqual(len(snippets), 9)
        for label, snippet in snippets:
            code = textwrap.dedent(snippet)
            preamble = ("import sys\nfrom pathlib import Path\n"
                        'PROJECT_NAME = "MammoDiffusion"\nPROJECT_ROOT_OVERRIDE = None\n')
            namespace: dict = {}
            exec(compile(preamble + code, f"resolver:{label}", "exec"), namespace)  # noqa: S102
            self._assert_resolves_to_checkout(label, namespace["find_project_root"])


class UtilityResolverTests(unittest.TestCase):
    def test_ldm_project_paths_resolver_stays_inside_the_checkout(self):
        for with_git, with_data in ((True, False), (False, True)):
            with tempfile.TemporaryDirectory() as temporary:
                checkout = build_collision_layout(Path(temporary), with_git=with_git,
                                                  with_data=with_data)
                resolved = resolve_from(checkout, ldm_project_paths.find_project_root,
                                        "notebooks/2_diffusers")
            with self.subTest(git=with_git, data=with_data):
                self.assertEqual(resolved, checkout.resolve())

    def test_shared_assets_resolver_stays_inside_the_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = build_collision_layout(Path(temporary), with_git=True, with_data=False)
            self.assertEqual(assets.project_root(checkout / "notebooks" / "utility"),
                             checkout.resolve())


if __name__ == "__main__":
    unittest.main()

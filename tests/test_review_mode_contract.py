"""The review-mode contract: opening a notebook and pressing Run All is safe.

Three separate guarantees are pinned here, because each has failed at least once
during this project's history:

1. the runtime guard actually blocks network access, package managers, and
   repository clones, while leaving loopback and local git reads alone;
2. no tracked notebook can reach a download, an install, or a clone before an
   explicit opt-in, checked statically so a new cell cannot reintroduce one;
3. no notebook ships a destructive or downloading flag switched on.

Together these are what makes the documented "review mode" a property of the
code rather than a promise in the README.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks/utility"))

import review_mode  # noqa: E402

def _project_notebooks() -> list[str]:
    """Every notebook the project ships.

    Git is preferred because it excludes the vendored Diffusers examples, but a
    source-only export has no git metadata, so fall back to a scan that skips
    the vendored checkout explicitly.
    """
    result = subprocess.run(["git", "ls-files", "*.ipynb"], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return sorted(result.stdout.split())
    return sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.glob("notebooks/**/*.ipynb")
        if "diffusers_repo" not in path.parts and ".ipynb_checkpoints" not in path.parts
    )


NOTEBOOKS = _project_notebooks()

FLAG_ASSIGNMENT = re.compile(r"^(\s*)([A-Z][A-Z0-9_]{2,})\s*=\s*(True|False)\s*(?:#.*)?$", re.M)

# Flags whose *enabled* state deletes data, rewrites a cohort, installs packages,
# or reaches the network. ALLOW_COMPLETE_PROCESSED_REUSE is excluded on purpose:
# it permits a read-only fallback and is safe precisely when it is on.
SIDE_EFFECTING_FLAG = re.compile(
    r"^(RESET_[A-Z0-9_]*|FORCE_[A-Z0-9_]*|OVERWRITE_[A-Z0-9_]*|INSTALL_[A-Z0-9_]*"
    r"|ALLOW_NETWORK[A-Z0-9_]*|ALLOW_[A-Z0-9_]*DOWNLOAD[A-Z0-9_]*|RUN_[A-Z0-9_]*_PHASE"
    r"|RUN_REAL_BENCHMARK|RUN_FINAL_EVALUATION)$"
)

NETWORK_CALLS = ("gdown.download", "urlopen", "urlretrieve", "hf_hub_download",
                 "snapshot_download", "requests.get", "requests.post")
INSTALL_MARKERS = ("pip install", "pip\", \"install", "'pip', 'install'")


def code_cells(relative: str) -> list[str]:
    notebook = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def shipped_flags(relative: str) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for source in code_cells(relative):
        for match in FLAG_ASSIGNMENT.finditer(source):
            if match.group(1) == "":
                flags[match.group(2)] = match.group(3) == "True"
    return flags


class RuntimeGuardTests(unittest.TestCase):
    """The guard has to block real escapes and permit real work."""

    def setUp(self):
        review_mode.deactivate()
        review_mode.activate()

    def tearDown(self):
        review_mode.deactivate()

    def test_outbound_connections_and_name_resolution_are_blocked(self):
        import socket
        with self.assertRaises(review_mode.ReviewModeViolation):
            socket.create_connection(("example.com", 80), timeout=2)
        with self.assertRaises(review_mode.ReviewModeViolation):
            socket.getaddrinfo("huggingface.co", 443)

    def test_loopback_and_local_git_still_work(self):
        import socket
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        try:
            client = socket.create_connection(listener.getsockname(), timeout=2)
            client.close()
        finally:
            listener.close()
        self.assertEqual(socket.getaddrinfo("localhost", 80)[0][0], socket.AF_INET)
        # A local git read must not be blocked. Whether it succeeds depends on
        # git metadata being present, which a source-only export lacks.
        try:
            subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, check=False)
        except review_mode.ReviewModeViolation:  # pragma: no cover
            self.fail("a local git read was blocked")

    def test_package_managers_and_clones_are_blocked(self):
        blocked = (
            [sys.executable, "-m", "pip", "install", "numpy"],
            ["pip", "install", "numpy"],
            ["conda", "install", "numpy"],
            ["git", "clone", "https://example.com/x.git"],
            ["git", "-C", "/tmp", "fetch", "origin"],
            ["wget", "https://example.com/x.zip"],
            ["curl", "-O", "https://example.com/x.zip"],
        )
        for command in blocked:
            with self.subTest(command=command[:3]):
                with self.assertRaises(review_mode.ReviewModeViolation):
                    subprocess.run(command, check=False)

    def test_harmless_subprocesses_are_not_blocked(self):
        for command in (["echo", "ok"], ["git", "ls-files", "VERSION"],
                        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]):
            with self.subTest(command=command[:3]):
                subprocess.run(command, cwd=ROOT, capture_output=True, check=False)

    def test_permission_helpers_name_the_flag_to_set(self):
        for helper, flag in ((review_mode.require_network, "ALLOW_NETWORK_ACCESS"),
                             (review_mode.require_dependency_install, "INSTALL_DEPENDENCIES"),
                             (review_mode.require_processed_download, "ALLOW_PROCESSED_DOWNLOAD")):
            with self.subTest(flag=flag):
                with self.assertRaises(review_mode.ReviewModeViolation) as caught:
                    helper("test action")
                self.assertIn(flag, str(caught.exception))

    def test_opt_in_lifts_the_restriction(self):
        review_mode.activate(allow_network=True)
        import socket
        try:
            socket.getaddrinfo("localhost", 80)
        except review_mode.ReviewModeViolation:  # pragma: no cover
            self.fail("opt-in did not lift the network restriction")
        self.assertFalse(review_mode.status()["review_mode"] and
                         not review_mode.allowances()["network"])

    def test_environment_variables_grant_permissions(self):
        import os
        review_mode.deactivate()
        os.environ["MAMMODIFFUSION_ALLOW_NETWORK"] = "1"
        try:
            review_mode.activate()
            self.assertTrue(review_mode.allowances()["network"])
        finally:
            os.environ.pop("MAMMODIFFUSION_ALLOW_NETWORK", None)


class NotebookContractTests(unittest.TestCase):
    """Static guarantees, so a new cell cannot quietly reintroduce an escape."""

    def test_every_notebook_installs_the_contract(self):
        for relative in NOTEBOOKS:
            with self.subTest(notebook=relative):
                code = "\n".join(code_cells(relative))
                self.assertIn("import review_mode", code)
                self.assertIn("review_mode.activate(", code)

    def test_no_notebook_ships_a_side_effecting_flag_enabled(self):
        for relative in NOTEBOOKS:
            for name, value in shipped_flags(relative).items():
                if SIDE_EFFECTING_FLAG.match(name):
                    with self.subTest(notebook=relative, flag=name):
                        self.assertFalse(value, f"{relative}: {name} ships enabled")

    def test_every_package_install_is_preceded_by_a_permission_check(self):
        for relative in NOTEBOOKS:
            for index, source in enumerate(code_cells(relative)):
                for match in re.finditer(r".*pip[\"'\s,\]]*install.*", source):
                    line = match.group(0).strip()
                    if line.startswith("#") or line.startswith("%") or line.startswith("!"):
                        continue  # magics live inside `if INSTALL_DEPENDENCIES:`
                    preceding = source[max(0, match.start() - 400):match.start()]
                    # Either gate is acceptable: an explicit permission call, or a
                    # surrounding `if INSTALL_DEPENDENCIES:` block that ships False.
                    gated = ("require_dependency_install" in preceding
                             or "if INSTALL_DEPENDENCIES:" in preceding)
                    with self.subTest(notebook=relative, cell=index):
                        self.assertTrue(gated,
                                        f"{relative} cell {index}: ungated {line[:70]}")

    def test_every_pip_magic_sits_behind_the_install_flag(self):
        for relative in NOTEBOOKS:
            for index, source in enumerate(code_cells(relative)):
                for match in re.finditer(r"^\s*%pip .*$", source, re.M):
                    preceding = source[:match.start()]
                    with self.subTest(notebook=relative, cell=index):
                        self.assertIn("if INSTALL_DEPENDENCIES:", preceding)

    def test_every_download_helper_starts_with_a_permission_check(self):
        checked = 0
        for relative in NOTEBOOKS:
            for index, source in enumerate(code_cells(relative)):
                cleaned = re.sub(r"^\s*[%!].*$", "pass", source, flags=re.M)
                try:
                    tree = ast.parse(cleaned)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.FunctionDef):
                        continue
                    body = ast.unparse(node)
                    if not any(call in body for call in NETWORK_CALLS):
                        continue
                    checked += 1
                    first = node.body[0]
                    rendered = ast.unparse(first) if not isinstance(first, ast.Expr) \
                        else ast.unparse(first.value)
                    with self.subTest(notebook=relative, cell=index, function=node.name):
                        self.assertIn("review_mode.require_", rendered,
                                      f"{relative}:{node.name} performs a download without "
                                      "an opening permission check")
        self.assertGreater(checked, 0, "no download helper was inspected")

    def test_shared_asset_helpers_refuse_to_clone_or_install_in_review_mode(self):
        source = (ROOT / "notebooks/utility/shared_diffusers_assets.py").read_text(encoding="utf-8")
        clone = source[source.index("def ensure_shared_diffusers_repo"):]
        clone = clone[:clone.index("\ndef ", 1)]
        self.assertIn("_require_network(", clone)
        self.assertLess(clone.index("_require_network("), clone.index('"git", "clone"'))

        install = source[source.index("def ensure_diffusers_editable_install"):]
        install = install[:install.index("\ndef ", 1)]
        self.assertIn("_require_dependency_install(", install)
        self.assertLess(install.index("_require_dependency_install("),
                        install.index('"pip", "install"'))


class DocumentedContractTests(unittest.TestCase):
    """What README and PROTOCOL promise must be what the code enforces."""

    def test_readme_documents_the_flags_the_code_reads(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for flag in ("ALLOW_NETWORK_ACCESS", "INSTALL_DEPENDENCIES", "ALLOW_PROCESSED_DOWNLOAD"):
            self.assertIn(flag, readme)
        for variable in review_mode._ENVIRONMENT_VARIABLE.values():
            self.assertIn(variable, readme)

    def test_no_notebook_fetches_the_rsna_source_archive(self):
        """README states this outright, so pin it."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("No notebook fetches the RSNA source archive", readme)
        preprocessing = "\n".join(
            code_cells("notebooks/1_preprocessing/01_Preprocessing_RSNA_512_gray_MLO.ipynb"))
        self.assertIn("never perform an opaque download", preprocessing)
        self.assertNotIn("gdown.download", preprocessing)


if __name__ == "__main__":
    unittest.main()

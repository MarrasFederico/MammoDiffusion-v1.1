"""The execution contract every notebook runs under by default.

MammoDiffusion v1.1 is a frozen study, so opening a notebook and pressing
*Run All* must be a safe, offline, read-only act. Two things previously made
that untrue: several notebooks reached a download or a ``pip install`` before
any scientific phase flag was consulted, and the shared Diffusers checkout was
cloned and installed at import time.

Rather than scatter another dozen ad-hoc conditionals, this module states the
contract once and enforces it at the primitives every escape route has to pass
through:

* outbound sockets — blocks anything that is not loopback, so ``gdown``,
  ``requests``, ``urllib``, and ``huggingface_hub`` all stop here;
* subprocesses — blocks package managers and repository clones by inspecting
  the argument vector, which is how ``pip`` and ``git`` would otherwise slip
  past the socket guard;
* the shared-asset helpers — refused explicitly so the failure names the flag
  to flip instead of surfacing as a connection error.

Review mode is the default. A real scientific run opts in explicitly, either in
the notebook::

    review_mode.activate(allow_network=True, allow_dependency_install=True)

or from the environment, which keeps the committed notebooks unmodified::

    MAMMODIFFUSION_ALLOW_NETWORK=1
    MAMMODIFFUSION_ALLOW_DEPENDENCY_INSTALL=1
    MAMMODIFFUSION_ALLOW_PROCESSED_DOWNLOAD=1

Loopback stays open because the Jupyter kernel talks to itself over ZMQ; a
guard that broke the kernel would simply be switched off, which is worse than
no guard at all.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from typing import Any, Iterable

__all__ = [
    "ReviewModeViolation",
    "activate",
    "deactivate",
    "is_active",
    "status",
    "require_network",
    "require_dependency_install",
    "require_processed_download",
    "allowances",
]


class ReviewModeViolation(RuntimeError):
    """Raised when review mode blocks a network, install, or download action."""


PERMISSIONS = ("network", "dependency_install", "processed_download")

_ENVIRONMENT_VARIABLE = {
    "network": "MAMMODIFFUSION_ALLOW_NETWORK",
    "dependency_install": "MAMMODIFFUSION_ALLOW_DEPENDENCY_INSTALL",
    "processed_download": "MAMMODIFFUSION_ALLOW_PROCESSED_DOWNLOAD",
}

_FLAG_HINT = {
    "network": "ALLOW_NETWORK_ACCESS",
    "dependency_install": "INSTALL_DEPENDENCIES",
    "processed_download": "ALLOW_PROCESSED_DOWNLOAD",
}

_state: dict[str, bool] = dict.fromkeys(PERMISSIONS, False)
_installed = False
_original: dict[str, Any] = {}

# Package managers and repository tools. A subprocess whose program or first
# arguments match one of these is an environment mutation, not computation.
_DEPENDENCY_TOOLS = frozenset({"pip", "pip3", "conda", "mamba", "micromamba", "poetry", "uv"})
# Only the subcommands that change the environment. Third-party libraries query
# `pip list` / `pip show` during import to detect optional backends, and blocking
# those would break imports that have nothing to do with installing anything.
_DEPENDENCY_MUTATING_SUBCOMMANDS = frozenset({
    "install", "uninstall", "download", "wheel", "add", "remove", "update", "upgrade",
    "sync", "create", "env", "clean", "build",
})
_NETWORK_TOOLS = frozenset({"wget", "curl", "rclone", "gdown", "aria2c", "scp", "rsync"})
# git is overwhelmingly used here for local reads (`ls-files`, `rev-parse`, `status`),
# so name the subcommands that actually reach a remote instead of trying to prove a
# command is local -- option forms like `git -C <path> rev-parse` defeat that.
_GIT_NETWORK_SUBCOMMANDS = frozenset({
    "clone", "fetch", "pull", "push", "remote", "submodule", "ls-remote", "request-pull",
})


def _environment_default(permission: str) -> bool:
    raw = os.environ.get(_ENVIRONMENT_VARIABLE[permission], "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def allowances() -> dict[str, bool]:
    """The permissions currently granted."""
    return dict(_state)


def status() -> dict[str, Any]:
    """A compact, printable description of the active contract."""
    return {
        "review_mode": is_active(),
        "mode": "REVIEW (offline, read-only)" if is_active() else "REAL RUN (explicit opt-in)",
        **{f"allow_{name}": _state[name] for name in PERMISSIONS},
    }


def is_active() -> bool:
    """True when at least one restriction is still in force."""
    return _installed and not all(_state.values())


# --- permission checks used by helpers that want a precise message -----------------------------

def _require(permission: str, action: str) -> None:
    if _state[permission]:
        return
    raise ReviewModeViolation(
        f"{action} is blocked in review mode. MammoDiffusion v1.1 opens offline and read-only. "
        f"Set {_FLAG_HINT[permission]} = True in the notebook (or export "
        f"{_ENVIRONMENT_VARIABLE[permission]}=1) to run this deliberately."
    )


def require_network(action: str = "network access") -> None:
    _require("network", action)


def require_dependency_install(action: str = "dependency installation") -> None:
    _require("dependency_install", action)


def require_processed_download(action: str = "downloading the processed cohort") -> None:
    _require("processed_download", action)


# --- guards -------------------------------------------------------------------------------------

def _is_loopback(address: Any) -> bool:
    """Allow the kernel's own ZMQ/IPC traffic; block everything outbound."""
    if isinstance(address, (str, bytes)):
        return True  # AF_UNIX socket path
    if isinstance(address, tuple) and address:
        host = address[0]
        if not isinstance(host, str):
            return False
        return host in {"127.0.0.1", "::1", "localhost", "", "0.0.0.0", "::"} \
            or host.startswith("127.")
    return False


_LOOPBACK_NAMES = frozenset({"127.0.0.1", "::1", "localhost", "localhost.localdomain",
                             "", "0.0.0.0", "::"})


def _guard_getaddrinfo(original):
    """Block name resolution too, so review mode emits no DNS traffic at all."""
    def getaddrinfo(host, *args, **kwargs):
        if not _state["network"]:
            name = host if isinstance(host, str) else ""
            if name not in _LOOPBACK_NAMES and not name.startswith("127."):
                raise ReviewModeViolation(
                    f"Resolving {host!r} is blocked in review mode. Set "
                    f"{_FLAG_HINT['network']} = True (or export "
                    f"{_ENVIRONMENT_VARIABLE['network']}=1) for a deliberate real run."
                )
        return original(host, *args, **kwargs)
    return getaddrinfo


def _guard_connect(original):
    def connect(self, address, *args, **kwargs):
        if not _state["network"] and not _is_loopback(address):
            raise ReviewModeViolation(
                f"Outbound network connection to {address!r} is blocked in review mode. "
                f"Set {_FLAG_HINT['network']} = True (or export "
                f"{_ENVIRONMENT_VARIABLE['network']}=1) for a deliberate real run."
            )
        return original(self, address, *args, **kwargs)
    return connect


def _classify_command(argv: Iterable[Any]) -> str | None:
    """Return the permission a subprocess needs, or None when it is harmless."""
    parts = []
    for item in argv:
        if isinstance(item, (list, tuple)):
            parts.extend(str(x) for x in item)
        else:
            parts.append(str(item))
    if not parts:
        return None
    tokens = [os.path.basename(p) for p in parts]
    lowered = [t.lower() for t in tokens]

    def _mutating(after: list[str]) -> bool:
        return any(token in _DEPENDENCY_MUTATING_SUBCOMMANDS for token in after)

    # `python -m pip ...`
    if any(t.startswith("python") for t in lowered) and "-m" in parts:
        index = parts.index("-m")
        if index + 1 < len(parts) and parts[index + 1].split(".")[0] in _DEPENDENCY_TOOLS:
            if _mutating(lowered[index + 2:]):
                return "dependency_install"
    for position, token in enumerate(lowered[:3]):
        if token in _DEPENDENCY_TOOLS and _mutating(lowered[position + 1:]):
            return "dependency_install"

    if lowered[0] == "git":
        return "network" if any(t in _GIT_NETWORK_SUBCOMMANDS for t in lowered[1:]) else None
    if lowered[0] in _NETWORK_TOOLS:
        return "network"
    return None


def _guard_subprocess(name, original):
    def wrapper(*args, **kwargs):
        argv = kwargs.get("args", args[0] if args else [])
        if isinstance(argv, str):
            argv = argv.split()
        permission = _classify_command(argv or [])
        if permission and not _state[permission]:
            shown = " ".join(str(x) for x in (argv or []))[:160]
            raise ReviewModeViolation(
                f"subprocess.{name} is blocked in review mode: {shown!r} would "
                f"{'install or change dependencies' if permission == 'dependency_install' else 'access the network'}. "
                f"Set {_FLAG_HINT[permission]} = True (or export "
                f"{_ENVIRONMENT_VARIABLE[permission]}=1) for a deliberate real run."
            )
        return original(*args, **kwargs)
    return wrapper


def activate(*, allow_network: bool | None = None,
             allow_dependency_install: bool | None = None,
             allow_processed_download: bool | None = None,
             announce: bool = False) -> dict[str, Any]:
    """Install the guards and record the granted permissions.

    Every argument left as ``None`` falls back to its environment variable and
    then to ``False``. Calling this again updates the permissions without
    stacking another layer of patches.
    """
    global _installed

    requested = {
        "network": allow_network,
        "dependency_install": allow_dependency_install,
        "processed_download": allow_processed_download,
    }
    for permission, value in requested.items():
        _state[permission] = _environment_default(permission) if value is None else bool(value)

    if not _installed:
        _original["socket_connect"] = socket.socket.connect
        _original["socket_connect_ex"] = socket.socket.connect_ex
        _original["getaddrinfo"] = socket.getaddrinfo
        socket.socket.connect = _guard_connect(_original["socket_connect"])
        socket.socket.connect_ex = _guard_connect(_original["socket_connect_ex"])
        socket.getaddrinfo = _guard_getaddrinfo(_original["getaddrinfo"])
        for name in ("run", "call", "check_call", "check_output", "Popen"):
            if hasattr(subprocess, name):
                _original[f"subprocess_{name}"] = getattr(subprocess, name)
                setattr(subprocess, name, _guard_subprocess(name, _original[f"subprocess_{name}"]))
        _installed = True

    if announce:
        current = status()
        print(f"Execution mode: {current['mode']}", file=sys.stderr)
    return status()


def deactivate() -> None:
    """Restore the unguarded primitives. Intended for tests."""
    global _installed
    if not _installed:
        return
    socket.socket.connect = _original["socket_connect"]
    socket.socket.connect_ex = _original["socket_connect_ex"]
    socket.getaddrinfo = _original["getaddrinfo"]
    for name in ("run", "call", "check_call", "check_output", "Popen"):
        key = f"subprocess_{name}"
        if key in _original:
            setattr(subprocess, name, _original[key])
    _original.clear()
    _installed = False
    for permission in PERMISSIONS:
        _state[permission] = False

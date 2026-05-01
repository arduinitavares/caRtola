"""caRtola file for ensuring the package is executable
as `cartola` and `python -m cartola`
"""
import importlib
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Protocol

from kedro.framework.cli.utils import KedroCliError, load_entry_points
from kedro.framework.project import configure_project


class _CommandGroup(Protocol):
    commands: Mapping[str, Callable[..., object]]


def _find_run_command(package_name: str) -> Callable[..., object]:
    try:
        project_cli = importlib.import_module(f"{package_name}.cli")
        # fail gracefully if cli.py does not exist
    except ModuleNotFoundError as exc:
        if f"{package_name}.cli" not in str(exc):
            raise
        plugins = load_entry_points("project")
        run = _find_run_command_in_plugins(plugins) if plugins else None
        if run:
            # use run command from installed plugin if it exists
            return run
        # use run command from `kedro.framework.cli.project`
        from kedro.framework.cli.project import run

        return run
    # fail badly if cli.py exists, but has no `cli` in it
    if not hasattr(project_cli, "cli"):
        raise KedroCliError(f"Cannot load commands from {package_name}.cli")
    return project_cli.run


def _find_run_command_in_plugins(plugins: Iterable[_CommandGroup]) -> Callable[..., object] | None:
    for group in plugins:
        if "run" in group.commands:
            return group.commands["run"]
    return None


def main(*args: object, **kwargs: object) -> None:
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    run = _find_run_command(package_name)
    run(*args, **kwargs)


if __name__ == "__main__":
    main()

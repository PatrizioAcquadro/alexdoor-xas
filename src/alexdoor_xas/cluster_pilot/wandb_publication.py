"""Fail-closed, symlink-free W&B durable publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PUBLICATION_SCHEMA = "alexdoor_xas.cluster_pilot_wandb_publication.v1"
PUBLICATION_REPORT_NAME = "publication_report.json"


class WandbPublicationError(ValueError):
    """Raised when a W&B source tree cannot be published safely."""


@dataclass(frozen=True)
class _Entry:
    relative: Path
    source: Path
    kind: Literal["directory", "file", "materialized_symlink"]
    target: Path | None = None


def publish_wandb_tree(
    source: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Validate and copy one W&B tree without preserving any symlinks.

    The destination must not exist. Validation completes before it is created, and any
    copy/report failure removes the destination created by this call.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    source_root = _validate_source(source_path)
    _validate_destination(source_root, destination_path)
    entries, omitted = _inventory_source(source_path, source_root)

    created = False
    try:
        destination_path.mkdir()
        created = True
        materialized: list[dict[str, str]] = []
        directories: list[_Entry] = []
        for entry in entries:
            published = destination_path / entry.relative
            if entry.kind == "directory":
                published.mkdir()
                directories.append(entry)
                continue
            published.parent.mkdir(parents=True, exist_ok=True)
            copy_source = entry.target if entry.target is not None else entry.source
            if copy_source is None:  # pragma: no cover - guarded by the entry contract.
                raise WandbPublicationError(f"missing copy source: {entry.relative.as_posix()}")
            shutil.copy2(copy_source, published, follow_symlinks=False)
            if entry.kind == "materialized_symlink":
                materialized.append(
                    {
                        "path": entry.relative.as_posix(),
                        "sha256": _sha256_file(published),
                    }
                )

        report: dict[str, Any] = {
            "schema": PUBLICATION_SCHEMA,
            "materialized_symlinks": sorted(
                materialized, key=lambda item: item["path"]
            ),
            "omitted_latest_run_symlinks": sorted(
                relative.as_posix() for relative in omitted
            ),
            "destination_contains_symlinks": False,
            "destination_symlink_count": 0,
        }
        _write_report(destination_path / PUBLICATION_REPORT_NAME, report)
        symlinks = [
            path.relative_to(destination_path).as_posix()
            for path in destination_path.rglob("*")
            if path.is_symlink()
        ]
        if symlinks:
            raise WandbPublicationError(
                f"destination contains forbidden symlinks: {sorted(symlinks)}"
            )

        for entry in sorted(
            directories, key=lambda item: len(item.relative.parts), reverse=True
        ):
            shutil.copystat(
                entry.source,
                destination_path / entry.relative,
                follow_symlinks=False,
            )
        shutil.copystat(source_path, destination_path, follow_symlinks=False)
        return report
    except Exception as error:
        if created:
            shutil.rmtree(destination_path)
        if isinstance(error, WandbPublicationError):
            raise
        raise WandbPublicationError(f"could not publish W&B tree: {error}") from error


def _validate_source(source: Path) -> Path:
    try:
        metadata = source.lstat()
    except OSError as error:
        raise WandbPublicationError(f"source W&B tree is missing: {source}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise WandbPublicationError("source W&B tree must be a real directory")
    try:
        return source.resolve(strict=True)
    except OSError as error:
        raise WandbPublicationError(f"cannot resolve source W&B tree: {source}") from error


def _validate_destination(source_root: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise WandbPublicationError(f"destination already exists: {destination}")
    try:
        parent_metadata = destination.parent.lstat()
        parent = destination.parent.resolve(strict=True)
    except OSError as error:
        raise WandbPublicationError(
            f"destination parent is missing or invalid: {destination.parent}"
        ) from error
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise WandbPublicationError("destination parent must be a real directory")
    if _is_within(parent, source_root):
        raise WandbPublicationError("destination must be outside the source W&B tree")


def _inventory_source(
    source_path: Path,
    source_root: Path,
) -> tuple[list[_Entry], list[Path]]:
    entries: list[_Entry] = []
    omitted: list[Path] = []

    def visit(directory: Path, relative_directory: Path) -> None:
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as error:
            raise WandbPublicationError(
                f"cannot read W&B directory: {relative_directory}"
            ) from error
        for child in children:
            relative = relative_directory / child.name
            if relative == Path(PUBLICATION_REPORT_NAME):
                raise WandbPublicationError(
                    f"source uses reserved publication report path: {relative.as_posix()}"
                )
            try:
                metadata = child.lstat()
            except OSError as error:
                raise WandbPublicationError(
                    f"cannot inspect W&B entry: {relative.as_posix()}"
                ) from error
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                target = _validated_symlink_target(child, relative, source_root)
                target_mode = target.stat().st_mode
                if stat.S_ISREG(target_mode):
                    entries.append(
                        _Entry(relative, child, "materialized_symlink", target=target)
                    )
                elif stat.S_ISDIR(target_mode):
                    if child.name != "latest-run":
                        raise WandbPublicationError(
                            "unexpected directory symlink: " f"{relative.as_posix()}"
                        )
                    omitted.append(relative)
                else:
                    raise WandbPublicationError(
                        f"symlink target is a special file: {relative.as_posix()}"
                    )
            elif stat.S_ISDIR(mode):
                entries.append(_Entry(relative, child, "directory"))
                visit(child, relative)
            elif stat.S_ISREG(mode):
                entries.append(_Entry(relative, child, "file"))
            else:
                raise WandbPublicationError(
                    f"special file is forbidden in W&B tree: {relative.as_posix()}"
                )

    visit(source_path, Path())
    return entries, omitted


def _validated_symlink_target(
    link: Path,
    relative: Path,
    source_root: Path,
) -> Path:
    try:
        os.readlink(link)
        target = link.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise WandbPublicationError(
            f"broken symlink in W&B tree: {relative.as_posix()}"
        ) from error
    if not _is_within(target, source_root):
        raise WandbPublicationError(
            f"symlink target escapes source W&B tree: {relative.as_posix()}"
        )
    return target


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = publish_wandb_tree(args.source, args.destination)
    except WandbPublicationError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "PASS: published symlink-free W&B tree "
        f"({len(report['materialized_symlinks'])} materialized, "
        f"{len(report['omitted_latest_run_symlinks'])} omitted)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "PUBLICATION_REPORT_NAME",
    "PUBLICATION_SCHEMA",
    "WandbPublicationError",
    "main",
    "publish_wandb_tree",
]

"""Regression checks for the canonical `knowledge/wiki` documentation surface."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WIKI_ROOT = REPO_ROOT / "knowledge" / "wiki"
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _wiki_pages() -> list[Path]:
    return sorted(WIKI_ROOT.rglob("*.md"))


def test_documentation_links_resolve_and_index_is_complete() -> None:
    pages = _wiki_pages()
    by_stem: dict[str, list[Path]] = {}
    for page in pages:
        by_stem.setdefault(page.stem, []).append(page)

    failures = []
    for page in pages:
        for raw_target in WIKILINK_RE.findall(page.read_text()):
            target = raw_target.split("|", 1)[0].split("#", 1)[0]
            relative = Path(target if target.endswith(".md") else f"{target}.md")
            if (WIKI_ROOT / relative).is_file():
                continue
            if len(by_stem.get(relative.stem, [])) != 1:
                failures.append(f"{page.relative_to(REPO_ROOT)}: {target}")
    for document in (REPO_ROOT / "README.md", REPO_ROOT / "datasets" / "README.md"):
        for raw_target in MARKDOWN_LINK_RE.findall(document.read_text()):
            target = raw_target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (document.parent / target).resolve().exists():
                failures.append(f"{document.relative_to(REPO_ROOT)}: {raw_target}")
    assert failures == []

    index = (WIKI_ROOT / "index.md").read_text()
    missing = []
    for page in pages:
        if page.name in {"index.md", "log.md"}:
            continue
        target = page.relative_to(WIKI_ROOT).with_suffix("").as_posix()
        if f"[[{target}|" not in index and f"[[{target}]]" not in index:
            missing.append(target)
    assert missing == []


def test_active_surfaces_do_not_reference_removed_legacy_documents() -> None:
    legacy_names = (
        "PROJECT_GUIDELINES.md",
        "architecture.md",
        "development.md",
        "status.md",
        "cluster.md",
        "code_quality_roadmap.md",
    )
    legacy_paths = tuple(f"{'docs'}/{name}" for name in legacy_names)
    active_roots = (
        REPO_ROOT / "configs",
        REPO_ROOT / "scripts",
        REPO_ROOT / "src",
        REPO_ROOT / "tests",
        WIKI_ROOT,
    )
    active_files = [
        REPO_ROOT / ".gitignore",
        REPO_ROOT / "README.md",
        REPO_ROOT / "datasets" / "README.md",
        REPO_ROOT / "knowledge" / "AGENTS.md",
        REPO_ROOT / "pyproject.toml",
    ]
    suffixes = {".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
    for root in active_roots:
        active_files.extend(path for path in root.rglob("*") if path.suffix in suffixes)

    failures = []
    for path in active_files:
        text = path.read_text(errors="replace")
        for legacy in legacy_paths:
            if legacy in text:
                failures.append(f"{path.relative_to(REPO_ROOT)}: {legacy}")
    assert failures == []


def test_active_config_and_output_surfaces_are_minimal() -> None:
    assert {path.name for path in (REPO_ROOT / "configs").iterdir()} == {
        "act.yaml",
        "alex_v2_door.json",
        "diffusion.yaml",
        "scripted_baseline.yaml",
    }
    outputs = REPO_ROOT / "outputs"
    entries = {path.name for path in outputs.iterdir()}
    assert {"README.md", "door_scene"} <= entries
    assert entries <= {"README.md", "door_scene", "door_push_alex_v2", "wandb"}
    assert {path.name for path in (outputs / "door_scene").iterdir()} == {
        "D0.usda",
        "D1.usda",
        "D2.usda",
        "D3.usda",
        "D4.usda",
    }
    assert not (outputs / "curated").exists()

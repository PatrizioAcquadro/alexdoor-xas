# AlexDoor-XAS Technical Wiki Rules

## Scope and Hierarchy

This file governs all work under `knowledge/`. 
The repository root `AGENTS.md` continues to govern repository-wide safety, implementation, testing, and Git behavior.

The allowed structure is:

```text
knowledge/
├── AGENTS.md
├── raw/
│   ├── papers/
│   ├── web/
│   ├── docs/
│   ├── notes/
│   ├── transcripts/
│   ├── data/
│   └── assets/
└── wiki/
    ├── implementation_phases/
    ├── topics/
    ├── decisions/
    ├── experiments/
    ├── sources/
    ├── index.md
    ├── status.md
    └── log.md
```

Do not add additional folders.

## Ownership and Safety

- `knowledge/raw/` is user-owned and immutable to agents. 
  Agents may inventory and read its contents, but must never edit, rename, move, overwrite, reformat, or delete anything in it.
  Raw storage is for research material, clipped web material, specifications,reports, notes, transcripts, small ingestable data, and local visual assets.
- `knowledge/wiki/` is primarily agent-maintained. 
  Preserve user editorial changes unless they conflict with verified evidence; resolve such conflicts with the smallest necessary edit and explain the evidence.
- Do not use destructive Git commands or rewrite history. 
  Repository Git rules and the user's explicit authorization govern branches, commits, pushes, and remotes.

## Evidence and Truth

Use evidence in this order:
1. Current code and deterministic tests for current executable behavior.
2. Current configurations, schemas, and verified small artifacts for declared contracts and recorded results.
3. Git history for phase attribution and meaningful change boundaries.
4. General documentation for explanation, after checking it against the stronger evidence above.

Do not infer implementation solely from a filename, comment, plan, or outdated document. 
Clearly label source claims, verified behavior, interpretation, inference, and unresolved uncertainty. 
Later behavior must not be attributed to an earlier phase merely because it exists in the current tree.

## Canonical Page Ownership

Every subject has one canonical documentation location:
- `wiki/status.md` — current completed work, evidence, limitations, boundaries, and next steps.
- `wiki/implementation_phases/` — what a phase actually introduced, including its stages, tests, deviations, replacements, and later meaningful updates.
- `wiki/topics/` — reusable concepts, technologies, systems, models, datasets, hardware, schemas, interfaces, or cross-cutting components.
- `wiki/decisions/` — choices with material architectural, interface, data, dependency, training, evaluation, performance, reproducibility, extensibility, or maintenance consequences.
- `wiki/experiments/` — durable training or evaluation runs requested by the user or clearly valuable to the project. 
  Ordinary correctness tests stay on the relevant phase page.
- `wiki/sources/` — provenance and synthesis for material actually ingested from `knowledge/raw/`.

Link to a canonical page instead of duplicating its explanation. 
Do not create a topic for every class or function, a decision for routine implementation details, an experiment for an ordinary test run, or a source page for material that was not ingested.

## Phase Pages

Maintain exactly one derived Markdown page for each top-level phase.
Keep stages and subphases inside that page. Use this structure:

```markdown
# Phase X — Phase Name

## Objective

## Subphase X.Y — Name
Follow this path for each subphase

#### Implementation
What should be / has been done.
Explain it in detail and in a well-organized way.

#### Key Decisions

#### Problems / Limitations
You need to say if they were fixed (and how), or not (so unsolved)

## Subphase Z.W — Name
Repeat the pattern from above for each subphase implemented.

## Artifacts
What has been generated (important metrics / plots / media / ecc)

## Files
Main files affected (repository-relative code references). Only the main ones.
```

## Writing and Linking

- Write clear, precise, direct US English. Report the key/most important points, not useless details.
- Prefer coherent technical explanations, NOT file-by-file summaries.
- Use meaningful Obsidian links with readable aliases where useful;
  for example `[[topics/system-architecture|System Architecture]]`.
- Give each subject one stable, lower-kebab-case filename.
- `wiki/index.md` is the navigation root. 
  It must list `status.md` prominently and every other wiki page (except itself and `log.md`), 
  grouped under Implementation Phases, Topics, Key Decisions, Experiments, and Sources, with a one-line description.
- `wiki/log.md` is append-only. Do not rewrite or reorder existing entries.
  In this way we keep track in a coincise way of all the changes done in the wiki.
  Log entries use `## YYYY-MM-DD — operation : Title`, followed by two-lines description.
  Allowed operations are: setup, ingest, update, query, experiment, and lint.
  Keep the log append-only and chronologically ordered. 
  This format must remain compatible with: grep "^## \[" knowledge/wiki/log.md | tail -5

## Ingest Workflow

1. Inventory the candidate material under `knowledge/raw/` without changing it.
2. Identify its type, origin, and date if known.
3. Read only the material necessary for the requested ingest.
4. Create or update one canonical `wiki/sources/` provenance page for material actually ingested. 
  Record path, claims used, uncertainty, and affected canonical pages.
5. Update according to this new information the affected phase, topic, decision, or experiment pages. 
  Keep provenance in the source page and avoid duplication.
6. Update `wiki/index.md` and append an ingest entry to `wiki/log.md`.

The conceptual setup document used to create this system is not project knowledge and must not receive a source page.

When ingesting Markdown that references local images, read the text first and then inspect the images that materially affect the source's meaning. 
Do not assume that image captions contain all relevant visual information.

## Query Workflow

1. Start at `wiki/index.md` and follow canonical wikilinks.
2. Check referenced repository files, tests, artifacts, and Git history when a current or high-consequence answer needs verification.
3. Answer with the distinction between verified state and uncertainty intact.
4. Update the wiki only when the user explicitly requests it or the query uncovers a meaningful, verified correction worth preserving.
5. If documentation is updated, update `wiki/index.md` and append an ingest entry to `wiki/log.md`.

## Synchronization Workflow

Update affected wiki pages when a code change materially changes behavior, architecture, interfaces, responsibilities, schemas, formats, serialization, data flow, control flow, algorithms, training procedures, evaluation metrics, or important configuration behavior. 
Also update them when the user explicitly requests documentation work.

Do not update the wiki for formatting, comments, local renaming, minor behavior-preserving refactoring, or temporary debugging. 
Code and tests remain the source of truth for executable behavior. 
Preserve phase attribution by using Git history when a current component has evolved across phases.

For a material implementation change:
1. Identify the canonical phase and cross-cutting pages affected.
2. Update them without retaining obsolete detail.
3. Add a short Version Notes entry only when the change is meaningful.
5. Update `wiki/index.md` for added, removed, or renamed pages, and append a documentation-update entry to `wiki/log.md`.

## Lint Workflow

Before finishing wiki work:

1. Confirm the allowed tree shape, and that `knowledge/raw/` and top-level `implementation_phases/` are unchanged.
2. Resolve every internal wikilink and ensure every page is listed in the index.
3. Check headings, repository-relative references, and concise Version Notes.
4. Search for duplicated canonical explanations, stale claims, unsupported completion language, copied plan text, and accidental source ingestion.
5. Verify cited files and symbols still exist and reconcile important claims against code, tests, configurations, artifacts, and Git history.
6. Run Markdown whitespace checks and inspect `git status` and `git diff`.
7. Append a concise lint entry to `wiki/log.md` after the checks pass.

# cuda-oxide roadmap campaign protocol

## Mission

Implement every capability in `cuda-oxide-book/appendix/roadmap.md` without
compatibility scaffolding or ceremonial layers. Each roadmap item gets an
explicit milestone plan before code starts, and temporary work artifacts are
removed once the corresponding roadmap item is complete.

## Board selection

1. Prefer the earliest roadmap item that is not done in `.memory/TODO.md`.
2. Work milestone-by-milestone inside that item.
3. If source inspection reveals missing prerequisite work, add it to the same
   item's milestone list before expanding implementation scope.

## Quality rules

- Follow `PRINCIPLES.md`.
- Apply the Rust guidance from `$tasteful-rust-minimal` and the Rust quality
  standard from the active agentic quality loop.
- Prefer direct Rust-shaped APIs over one-for-one CUDA surface cloning when the
  capability target still remains fully expressible.
- This is greenfield work: do not add fallbacks, deprecation layers, or
  backwards-compatibility branches.

## Validation ladder

For each substantial milestone:

1. `cargo fmt --check`
2. Targeted `cargo test` for touched workspace crates
3. Targeted example or compile validation when the change touches codegen,
   macros, or documented runtime flows
4. GPU-bound verification on a reusable B300 pod in `hou2-prod1` when the
   milestone's behavior depends on CUDA runtime or hardware semantics

Read the actual command output before closing the milestone.

## Reviewer gate

Before every commit:

1. Ask Claude CLI non-interactively for review, explaining the task and the
   relevant diff.
2. Treat blocking reviewer findings as work items, fix them, and rerun review.
3. Commit only after the review has no blocking issues.

Claude is advisory; source evidence and local validation decide the final shape.

## Commit policy

Pass-end commits are allowed after validation and the reviewer gate both pass.
Stage only files that belong to the completed milestone.

## Incident stops

- Do not mutate shared GPU infrastructure outside the requested B300 validation
  path.
- Reuse an existing B300 pod when possible.
- If the live validation path is blocked, preserve the exact blocker and keep
  progressing on source-backed local work that does not depend on the blocked
  boundary.

## Working artifacts

- `.memory/TODO.md` tracks roadmap items, milestone plans, and validation.
- `.memory/lessons.md` records only trial-and-error lessons worth reusing.
- Create a temporary debugging ledger only while actively debugging. Read it
  after compaction and delete it once the debugging goal is resolved.

## Stop condition

Continue until every roadmap item is implemented, validated at the required
boundary, reviewed, and reflected in source/docs/tests. Only then run the final
completion audit against the objective.

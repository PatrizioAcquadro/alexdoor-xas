# AlexDoor-XAS Project Guidelines

## Purpose

AlexDoor-XAS studies action representations for humanoid articulated-object
manipulation. The first benchmark is door pushing with the fixed-base IHMC Alex
V2 torso in Isaac Sim and Isaac Lab. The door is a controlled, contact-rich
research task, not the final scope of the project.

The central question is:

> Can object-centric, adapter-executable action chunks improve generalization,
> interpretability, safety, and transfer compared with robot-specific actions?

The intended result is a reusable benchmark and research package, not a
one-off door-opening demonstration.

## System boundaries

Keep these roles separate:

1. **Observation** — vision, robot state, object state, language, and the
   requested action-space tag.
2. **Policy** — predicts an action representation from observations.
3. **Action representation** — the main research variable.
4. **Adapter** — validates and converts a representation into executable Alex
   commands.
5. **Safety and logging** — enforces limits and records every trial.

Policies must not directly command hardware or bypass the adapter. Safety and
provenance are part of the experiment, not optional utilities.

## Action-space study

- **A1 — joint deltas:** robot-specific low-level baseline.
- **A2 — end-effector deltas:** practical Cartesian baseline.
- **A3 — object-relative end-effector deltas:** primary transfer-oriented
  baseline.
- **A4 — object-centric chunks:** interpretable interaction intent and the
  flagship representation for later VLA work.

Comparisons must use matched source episodes, splits, task conditions, and
evaluation protocols. The action-space tag is always explicit. Operational
formats and implemented execution paths are defined in
[`architecture.md`](architecture.md).

## Development roadmap

1. **Project and simulation readiness — complete.** Organize assets, define the
   minimal task and action taxonomy, and verify the simulator.
2. **Scripted baseline and deterministic data engine — complete.** Generate
   comparable episodes, action-space exports, metrics, and failure labels.
3. **Non-VLA learned baselines — locally complete.** Implement the dataset
   interface, adapter-v1, state-only ACT, state-only Diffusion Policy, and the
   local stabilization protocol.
4. **VLA and cross-action-space learning — not started.** Evaluate practical
   VLA fine-tuning and shared action-space-conditioned learning only after the
   cluster evidence is ready.
5. **Transfer and research packaging — future.** Progress from offline checks
   to guarded Alex execution and approved real-door trials.

WAM-lite and egocentric/Aria data remain optional later extensions. They may
inform the object-centric schema, but they are not initial dependencies.

## Evaluation principles

- Measure success, final object state, contact quality, safety, failure modes,
  and generalization; do not rely on videos alone.
- Compare representations under matched data and protocol.
- Bind datasets, splits, checkpoints, configurations, and evaluations with
  explicit provenance.
- Treat smoke runs as pipeline validation, not scientific performance
  evidence.
- Fail closed on non-finite state, incompatible provenance, invalid force
  evidence, or unsupported execution context.

## Hardware boundary

Hardware work is safety-first and requires explicit approval. Progression is:

1. logging-only checks;
2. air trajectories;
3. contact-only trials;
4. scripted fake-door pushes;
5. learned chunks through the adapter;
6. VLA chunks through the adapter;
7. approved real-door attempts.

Local simulation readiness is not hardware-readiness evidence. Do not add live
robot execution, calibration changes, or low-level actuator control implicitly.

## Sources of truth

- This file owns research intent, phase boundaries, and safety principles.
- [`architecture.md`](architecture.md) owns implemented technical contracts.
- [`development.md`](development.md) owns setup, workflows, and validation.
- [`status.md`](status.md) owns completed work, evidence, limitations, and next
  steps.
- [`cluster.md`](cluster.md) is the contract-bound Gilbreth pilot runbook.

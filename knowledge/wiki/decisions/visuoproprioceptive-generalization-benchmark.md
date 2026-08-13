# Visuoproprioceptive Generalization Benchmark

## Context

The completed state-only Phase 3 study covered A2/A3, ACT/Diffusion, one simulated
door family, and one training seed. All 576 closed-loop rollouts succeeded, so the
benchmark validated the pipeline but could not identify a better representation
or policy model.

## Scientific Question

Holding Alex V2, observations, matched demonstrations, dataset size, training
budget, and evaluation protocol fixed, how do A1-A4 action representations and
ACT/Diffusion policy models affect zero-shot success when opening push-door assets
that were geometrically unseen during training?

An unseen door is a complete held-out USD asset, not another pose of a training
door. The learned policy receives vision and proprioception and is evaluated
without retraining on the held-out doors.

## Benchmark Decision

The first study will:

- compare the complete A1-A4 x ACT/Diffusion matrix;
- use one fixed dataset size and multiple independent training seeds;
- separate door identities across training, development, and held-out test sets;
- derive all four representations from the same physical demonstrations and
  synchronized sensor observations;
- report success on seen doors, success on held-out doors, the generalization
  gap, and results per door instance.

The held-out test doors are not used for training, dataset generation, model
selection, or benchmark tuning.

## Simulation Data Strategy

Data generation is simulation-first and progressively hybrid:

- a privileged scripted expert is the primary source for the push-door study;
- Isaac Sim Replicator records and varies the visual domain;
- an Isaac Lab Mimic pilot determines whether transformed demonstrations add
  valid and useful spatial diversity before Mimic becomes part of production;
- targeted teleoperation covers recovery behavior or doors the scripted expert
  cannot handle;
- an RL teacher is deferred until handles or broader articulated-object tasks
  require coverage that scripted, Mimic, and teleoperated data cannot provide.

Every accepted physical episode is recorded once and exported to A1-A4. Its
teacher source remains explicit so representation comparisons use identical
underlying experience.

## Progression

1. Generalization across geometrically different push doors.
2. Handle operation and generalization across different handles.
3. Generalization across doors, cabinets, drawers, and other articulated objects.

Each stage first tests unseen instances of the same task family. Generalization
from pushing directly to handles or drawers is not claimed by the first study.

## Status

This decision defines the approved research direction. The visual observation
path, multi-door benchmark, learned A1/A4 support, new dataset, and multi-seed
evaluation have not yet been implemented.

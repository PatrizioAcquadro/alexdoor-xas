# AlexDoor-XAS Project Guidelines

## 1. Project Identity

**Project name:** AlexDoor-XAS  
**Full name:** Cross-Action-Space VLA/WAM Learning for Humanoid Articulated-Object Manipulation  
**Main robot platform:** IHMC Alex humanoid torso  
**Main simulator:** Isaac Sim / Isaac Lab  
**First benchmark task:** Door pushing/opening  
**Broader task family:** Humanoid articulated-object manipulation

AlexDoor-XAS studies how humanoid manipulation actions should be represented, learned, evaluated, and safely transferred from simulation to hardware.

The first task is door interaction because it is measurable, contact-rich, object-centric, and relevant to humanoid manipulation. However, the project is **not** simply about making Alex open one door. The door is the first controlled benchmark for studying action representations.

---

## 2. Core Research Thesis

The core thesis is:

> Object-centric, adapter-executable action chunks can improve generalization, interpretability, safety, and sim-to-real transfer compared with raw robot-specific action spaces.

The main scientific variable is the **action interface**.

The project compares multiple ways of representing the same manipulation behavior, from robot-specific actions to object-centric action chunks.

---

## 3. Main Research Question

The main research question is:

> Can object-centric, adapter-executable action chunks improve humanoid articulated-object manipulation compared with joint-level or end-effector-level action representations?

Secondary questions:

1. Can one model learn across multiple action spaces when the requested action representation is explicitly specified?
2. Do object-relative end-effector actions improve generalization across door poses, object geometry, viewpoints, and language instructions?
3. Can VLA-predicted actions remain useful when executed through a safe robot-specific adapter?
4. Can an action-conditioned WAM-lite component help rank, diagnose, or reject candidate manipulation chunks?

---

## 4. What the Project Is

AlexDoor-XAS is a research project for studying cross-action-space learning in humanoid articulated-object manipulation:

- a controlled humanoid manipulation benchmark;
- a study of action representations for VLA-style robotic manipulation;
- a simulation-to-hardware research pipeline;
- a system for comparing scripted, imitation-learning, diffusion-policy, VLA, and optional WAM-lite components under a shared task definition;
- a foundation for later work on pulling, handles, latches, drawers, cabinets, and bimanual stabilization.

It uses door interaction with the IHMC Alex torso as the first controlled benchmark. 
The project compares joint-level, end-effector, object-relative end-effector, and object-centric action representations across scripted, imitation-learning, diffusion-policy, VLA, and optional WAM-lite components.

The central objective is to understand whether object-centric, adapter-executable action chunks provide a better interface for learning, generalization, safety, and transfer than raw robot-specific commands.

The final artifact should be a reusable benchmark and research package, not a one-off door-opening demo.
The project should be developed as a research system, not as a one-off demo.

---

## 5. Core System Separation

The project must preserve a clear separation between five roles.

### Observation

What the model or controller receives, such as vision, robot state, object state, door state, language instruction, and action-space context.

### Policy

The component that predicts an action representation from observations.

They can be of different types (see below)

### Action Representation

The format of the predicted action.

This is the main research object of the project.

### Adapter

The robot-specific component that converts an action representation into executable Alex commands.

The adapter must keep the policy reusable and prevent the learned model from directly controlling unsafe low-level hardware actions.

### Safety and Logging

The component that enforces execution constraints and records every trial.

Safety and logging are part of the research system, not optional utilities.

---

## 6. Action Spaces

The project focuses on four core action spaces.

### A1: Joint Delta Chunks

Robot-specific sequences of joint changes.

Use mainly as a low-level baseline or debugging representation. It is not expected to transfer well.

### A2: End-Effector Delta Chunks

Cartesian hand or wrist motion chunks.

Use as a strong practical baseline. It is easier to learn than raw joint actions but still depends on robot kinematics.

### A3: Object-Relative End-Effector Deltas

End-effector motion expressed relative to the manipulated object, such as the door, hinge, panel, or handle frame.

Use as the main transfer-oriented baseline.

### A4: Object-Centric Action Chunks

Structured chunks describing the intended interaction with the object.

A4 is the flagship representation. It should be interpretable, adapter-executable, and suitable for safe humanoid manipulation.

The VLA should eventually predict this kind of chunk, and the robot adapter should convert it into executable Alex commands.

---

## 7. Model and Baseline Hierarchy

The project should introduce models progressively, only when the current phase needs them.

Core baselines:

1. **Scripted deterministic controller**  
   Used for data generation, debugging, repeatable trajectories, failure labels, and action-space conversion.

2. **ACT-style imitation baseline**  
   Used as the first learned sanity check for action-chunk imitation.

3. **Diffusion Policy baseline**  
   Used as a stronger multimodal continuous-action baseline when practical.

4. **OpenVLA-OFT baseline**  
   Used as the main practical VLA fine-tuning path.

5. **Mixed-action-space OpenVLA-OFT**  
   Used to test whether one shared VLA can learn across multiple action spaces through action-space conditioning.

Optional or later components:

6. **WAM-lite scorer**  
   Used only if it predicts action-conditioned futures and supports chunk ranking, diagnosis, or failure prediction.

7. **openpi/π0**  
   Used later if data conversion, inference, and small-to-medium Alex-compatible fine-tuning are practical.

8. **Custom VLA / EO-1-derived flow head**  
   Used only as a later learning exercise, appendix baseline, or research extension.

Do not make optional later components the first implementation dependency.

---

## 8. WAM-Lite Role

WAM-lite is optional/later and should remain scoped.

In this project, WAM-lite means an object-centric, low-dimensional, action-conditioned future predictor.

It may predict outcomes such as:

- future door angle;
- contact state;
- success or failure;
- future object keypoints or latent state.

It may be used as:

- a **critic** to rank candidate action chunks;
- a **diagnostic tool** to test whether object-centric chunks are more predictable;
- a **bridge** toward later human-egocentric data pretraining.

It should not be treated as a full large-scale world model in the initial version.

---

## 9. Relationship to Project Aria and Egocentric Human Data

I have a parallel project involving egocentric manipulation data, including Project Aria-style data.

This is relevant because human egocentric videos can provide object-centric interaction priors, such as contact phases, hand-object trajectories, manipulated object parts, motion axes, and subgoal sequences.

However, AlexDoor-XAS should not depend on Aria data for the initial implementation.

Aria data may inform the schema and object-centric action abstraction from the beginning, but it becomes a training or pretraining resource only in later phases.

---

## 10. Development Phases

The project should evolve through five high-level phases.

### Phase 1: Project Definition, Asset Organization, and Simulation Readiness

Create the clean project scaffold, organize local Alex and scene assets, verify Isaac Sim / Isaac Lab readiness, and define only the minimal schema and action-space taxonomy needed to proceed.

Phase 1 should stay minimal and should not overdesign future policy, VLA, WAM, or hardware infrastructure.

### Phase 2: Scripted Baseline and Deterministic Data Engine

Build a reliable door-interaction baseline, generate controlled simulation episodes, export comparable action representations, and produce first metrics, plots, videos, and failure labels.

### Phase 3: Imitation and Diffusion Baselines

Train non-VLA learned baselines, test whether the data is learnable, compare action spaces, and identify which representations should be carried into VLA fine-tuning.

### Phase 4: VLA Fine-Tuning and Cross-Action-Space Learning

Fine-tune or adapt a practical VLA baseline, compare one-model-per-action-space against shared action-space-conditioned learning, and evaluate whether VLA outputs are plausible and adapter-executable.

### Phase 5: Alex Transfer, Research Package, and Extension Plan

Test safe transfer to Alex through staged execution, collect real or fake-door evidence, package the work as a research artifact, and define the next research arc.

---

## 11. Evaluation Philosophy

The project should be evaluated through controlled metrics, not only videos.

Evaluation should measure whether action representations affect:

- task success;
- contact quality;
- final object state;
- failure modes;
- safety;
- generalization;
- sim-to-hardware or sim-to-fake-door transfer.

Exact metrics should be defined in the relevant phase, not overdesigned in advance.

The strongest results compare action spaces under matched data, matched task conditions, and matched evaluation protocols.

---

## 12. Hardware Transfer Principle

Hardware transfer must be safety-first.

The learned model should not directly command low-level raw hardware actions. Model outputs should pass through an adapter and safety supervisor.

Hardware evaluation should progress through staged levels:

1. logging-only checks;
2. air trajectories;
3. contact-only trials;
4. scripted fake-door pushes;
5. learned chunk replay through the adapter;
6. VLA-generated chunk replay through the adapter;
7. real-door attempts only with approval.

The hardware objective is not immediate full autonomy. The objective is safe evidence that the action representation, adapter, and logging pipeline remain meaningful on Alex.

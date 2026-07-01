# Action Spaces (taxonomy, Phase 1)

The **action interface** is the project's main research variable. Every episode
declares which of the four spaces its actions are in (the `action_space` tag in
[episode_schema.md](episode_schema.md)), so the same behavior can be re-expressed
and compared across spaces under matched data and evaluation.

This document defines the taxonomy operationally. The types/encoders/converters
themselves are Phase 2 (`src/alexdoor_xas/action/`); nothing is coded in Phase 1.

Common conventions:
- Actions are **chunks**: a policy call emits `chunk_len` steps (Phase 1 uses the
  tag only; `chunk_len=1` degrades to single-step).
- Deltas are per control tick (`control_dt`, ~0.02 s).
- Frames use the scene's Z-up, meters convention.
- Every space is executed through the **Adapter** (guidelines §5, §12) — the policy
  never commands raw hardware directly.

| ID | Tag | What it is | Shape (per step) | Frame | Role in the study |
|----|-----|-----------|------------------|-------|-------------------|
| **A1** | `A1_joint_delta` | Per-joint position deltas | `[n_act_joints]` (Alex: 23 nub / 29 full body) | joint space | Low-level baseline / debugging. Robot-specific; not expected to transfer. |
| **A2** | `A2_ee_delta` | Cartesian hand/wrist motion deltas | `[6]` per hand `(dx,dy,dz,drx,dry,drz)` (+ gripper) | robot base / world | Strong practical baseline; easier than raw joints but still kinematics-dependent. |
| **A3** | `A3_obj_rel_ee_delta` | End-effector deltas expressed **relative to the manipulated object** (door/hinge/handle/panel frame) | `[6]` per hand (+ gripper), in object frame | object frame | Main transfer-oriented baseline (pose/geometry/viewpoint robustness). |
| **A4** | `A4_obj_centric_chunk` | Structured, interpretable description of the intended object interaction (contact target, motion axis, subgoal, phase) | struct / variable | object-centric | **Flagship.** Interpretable + adapter-executable; the representation the VLA should eventually predict. |

## Notes per space

- **A1 — joint deltas.** Direct, unambiguous, but tied to Alex's exact DoF layout. Uses the actuator
  groups defined in the Alex config (legs / torso / arms). Good for verifying the adapter and for
  scripted data generation.
- **A2 — end-effector deltas.** Requires forward/inverse kinematics in the adapter. Decouples the
  policy from joint count but not from the robot's arm geometry / reach.
- **A3 — object-relative EE deltas.** Same as A2 but expressed in the door/handle frame, so a learned
  policy generalizes across door poses without relearning absolute geometry. Needs the object frame
  from `object_state`.
- **A4 — object-centric chunks.** Encodes *what to do to the object* (e.g. "contact handle → rotate
  about hinge axis by Δθ"), leaving the *how* to the adapter. This is where interpretability,
  safety, and cross-embodiment transfer are expected to pay off. The exact struct is designed in the
  phase that first needs it; egocentric/Aria priors (guidelines §9) may inform it later.

## Cross-action-space conditioning
When one model is trained across multiple spaces (guidelines §7 item 5), the `action_space` tag is
provided to the policy as an explicit input, and the adapter dispatches on the same tag to decode
and execute. Keeping the tag canonical here (the strings in the table above) is what makes that
dispatch unambiguous.

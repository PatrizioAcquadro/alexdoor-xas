# Action Representations and Adapters

AlexDoor-XAS changes the action representation while holding the robot, task, physical episode, and evaluation protocol fixed.

## Canonical Representations

| Tag | Meaning | Frame and form | Current use |
|---|---|---|---|
| `A1_joint_delta` | Joint-target delta | Robot joint coordinates | Matched export and interface verification only |
| `A2_ee_delta` | End-effector delta | World-frame 6D delta | Scripted and learned execution currency |
| `A3_obj_rel_ee_delta` | Object-relative end-effector delta | Static hinge-anchored door-frame 6D delta | Scripted and learned; transformed to A2 |
| `A4_obj_centric_chunk` | Object-centric contact-intent chunk | Contact targets in the moving panel frame | Matched export and guarded adapter execution |

Frames are Z-up, distances are meters, angles are radians, and quaternions use `(x, y, z, w)`. The A3 frame is fixed at the hinge with +Z along the hinge axis. A4 contact targets move with the panel.

## Adapter Boundary

`AdapterDecision` records whether a request was accepted, corrected, or rejected, along with the applied action and structured warnings. Corrections therefore remain visible to evaluation and replay.

- A2 validates shape and finiteness, clamps translation and rotation, enforces workspace and joint-related limits, and shapes contact entry.
- A3 validates the supplied door frame, rotates the request into world coordinates, and delegates to A2. Invalid or reflected frames fail closed.
- A4 executes guarded approach, contact, and push stages through A3 and A2. Invalid chunks, stalls, timeouts, and simulator stops terminate the stage sequence explicitly.

There is no learned A1 adapter and no learned A4 policy.

## Alex V2 Execution

The Alex V2 runtime consumes applied A2 translation through six-joint position-only differential IK at the collision-derived tool point. A2/A3 rotational values remain represented, validated, clamped, and recorded but are not commanded.

`src/alexdoor_xas/adapters/rollout.py` is the learned-policy execution boundary. It validates state, stops on simulator termination or truncation, preserves the last pre-reset terminal state, caches static door-pose terms after reset, and keeps each rollout's decisions isolated.

## Contact and Force Semantics

Task force comes from raw PhysX GPU contacts selected by the exact door actor ID. Pre-action contact belongs to the recorded step; terminal contact records the response to the final action. The runtime never silently substitutes aggregate gripper force or a geometric estimate.

## Matched Comparison

A2 and A3 products derived from one physical episode share episode identity, outcome, pose, split, and evaluation seeds while retaining different action arrays and normalization. A4 preserves the same physical identity as a staged object-centric representation.

This controls major task-distribution confounds but does not prove that representation is the only cause of every learning difference.

## Primary References

- `src/alexdoor_xas/action/spaces.py`
- `src/alexdoor_xas/action/frames.py`
- `src/alexdoor_xas/adapters/`
- `tests/test_action_spaces.py`
- `tests/test_adapters.py`

## Version Notes

- 2026-08-13 — Reduced the topic to the four active representation contracts and their maintained adapter/runtime behavior.

# Action Representations and Adapters

AlexDoor-XAS holds the task and physical trajectory identity constant while
changing how action is represented. Tags, dimensions, frames, units, and
adapter behavior are project invariants.

## Canonical Representations

| Tag | Meaning | Frame and form | Current use |
|---|---|---|---|
| `A1_joint_delta` | Joint-target delta | Robot joint coordinates | Exported for Alex episodes; no learned adapter |
| `A2_ee_delta` | End-effector delta | World frame, 6D `(dx, dy, dz, drx, dry, drz)` | Learned and scripted execution currency |
| `A3_obj_rel_ee_delta` | Object-relative end-effector delta | Static hinge-anchored door frame, 6D delta | Learned and scripted; transformed to A2 |
| `A4_obj_centric_chunk` | Object-centric contact intent chunk | Contact targets in moving panel frame | Scripted/exported guarded stages; no learned policy |

Frames are Z-up, distances are meters, angles are radians, and quaternions are
`(x, y, z, w)`. The A3 door frame is static: its origin is at the hinge and +Z
is the hinge axis. A4 contact targets move with the panel.

## Adapter Contract

`src/alexdoor_xas/adapters/base.py` defines `AdapterDecision` and the accepted,
corrected, and rejected statuses. Each decision retains the request, applied
action if any, and warnings. A correction is therefore visible in metrics and
replay rather than being hidden inside the controller.

`src/alexdoor_xas/adapters/a2.py` is the final Cartesian safety boundary. It
validates shape and finiteness, clamps per-tick translation to 0.02 m and
rotation to 0.05 rad, enforces workspace and joint-related constraints, and
shapes entry into contact.

`src/alexdoor_xas/adapters/a3.py` validates the supplied static door frame,
requires its rotation to be orthonormal with determinant +1, rotates the A3
delta into world coordinates, and delegates to A2. Reflected frames fail
closed before an action is transformed.

`src/alexdoor_xas/adapters/a4.py` validates an intent chunk and executes guarded
approach/contact/push stages through A3 and A2. Every A4 adapter requires an
explicit configuration; `alex_v2_a4_cfg` derives its stage standoffs and
clearances from the validated Alex V2 door calibration. Contact targets use the
collision-derived tool point against the physical panel thickness, without a
synthetic end-effector radius. Stalls and stage timeouts reject the chunk. The
stage executor stops immediately on simulator termination or truncation and
retains the last valid pre-reset state, matching the A2/A3 rollout boundary.

## Robot Execution

For the current [[alex-v2-benchmark|Alex V2 Benchmark]], the executor consumes
the applied A2 action. Only translation is actuated through six-joint
position-only differential IK at the collision-derived tool point. Rotational
components remain part of A2/A3 data, clamping, and decision records but are not
commanded to the robot.

`src/alexdoor_xas/adapters/rollout.py` is the learned-policy execution boundary.
It stops on simulator termination/truncation, records pre-reset terminal state,
and rejects non-finite simulator observations. Policies do not import or bypass
this layer.

## Contact and Force Semantics

Task force uses PhysX raw GPU contact buffers and selects only contacts whose
opposite actor ID belongs to the door panel body. It does not use the gripper's
unfiltered net force or the unsupported shape-filter API. Pre-action contact
belongs to each recorded step;
`terminal_contact` records the response to the final action. Two current input
validation gaps are documented: finite scalar contact values are coerced to
Boolean rather than restricted to exact Boolean/0/1, and geometric contact
checks omit the panel Z extent.

## Representation Comparison

Matched A2 and A3 datasets derive from the same physical episodes. They share
episode IDs, outcome, split, and pose distribution but contain representation-
specific action values and fingerprints. This isolates representation more
carefully than independently generating one dataset per space; see
[[decisions/door-relative-task-and-matched-representations|Door-Relative Task and Matched Representations]].

## Primary References

- `src/alexdoor_xas/action/spaces.py`
- `src/alexdoor_xas/action/frames.py`
- `src/alexdoor_xas/adapters/base.py`
- `src/alexdoor_xas/adapters/a2.py`
- `src/alexdoor_xas/adapters/a3.py`
- `src/alexdoor_xas/adapters/a4.py`
- `tests/test_action_spaces.py`
- `tests/test_adapters.py`

## Version Notes

- 2026-08-12 — Required proper A3 rotations and aligned A4 environment-end
  handling with the shared pre-reset rollout contract.
- 2026-08-12 — Bound A4 execution to validated Alex V2 calibration and replaced synthetic end-effector-radius contact geometry with the collision-derived tool point and physical panel thickness.
- 2026-07-03 — Canonical A1–A4 tags, matched exports, and scripted A3/A4 use
  were established.
- 2026-07-05 onward — Explicit adapters and learned rollouts made action
  acceptance, correction, rejection, and frame conversion auditable.

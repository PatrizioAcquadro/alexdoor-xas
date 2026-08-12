# Action Spaces
## Purpose
An **action space** defines the language in which a policy or controller expresses what the robot should do next.

The same physical behavior can be represented in different languages. For example, “move the hand toward the door” may be expressed as:
- changes to individual joint targets;
- a Cartesian end-effector displacement in the world;
- the same displacement relative to the door;
- a structured intent such as “contact this point on the panel and open the hinge.”

AlexDoor-XAS studies how this representational choice affects learning, generalization, interpretability, safety, and transfer while holding the physical task as constant as possible.

Frames, transformations, quaternions, Jacobians, and timing conventions are explained in [`05_MeasurementSetup.md`](05_MeasurementSetup.md).

## 1. Action Space Is Not State Space
The **observation** or state describes what the system currently knows:
- door angle and angular velocity;
- end-effector pose;
- joint state;
- contact and force.

The **action** describes the command or intent produced from that information.

The closed-loop relationship is:

    observation s_k
      -> policy or controller
      -> requested action a_k
      -> adapter and executor
      -> physical response
      -> next observation s_(k+1)

Changing the action space changes the interface between the policy and the execution system, even if the task, observations, robot, and scene remain unchanged.

## 2. Common Action-Space Design Choices
An action representation must answer several questions.

### 2.1 What Is Controlled?
Examples include:
- joint position;
- joint velocity;
- joint torque;
- end-effector position or pose;
- end-effector velocity;
- object motion;
- contact location;
- a high-level skill or phase.

### 2.2 Is the Command Absolute or Relative?
An absolute command specifies a target:

    q_target = desired joint configuration
    p_target = desired end-effector position

A relative command specifies a change:

    delta_q = change in joint target
    delta_p = change in end-effector position

Relative commands are often easier to bound per control tick, but they accumulate over time and depend on correct temporal alignment.

### 2.3 In Which Frame Is It Expressed?
A Cartesian command may be expressed in:
- world frame;
- robot-base frame;
- end-effector frame;
- object frame;
- another task-specific frame.

The numeric vector can change when the frame changes even though the intended physical motion is identical.

### 2.4 What Is the Temporal Unit?
An action may represent:
- one control tick;
- a velocity to hold for a duration;
- a sequence of future per-tick actions;
- one skill or phase with a termination condition.

The action representation is therefore both geometric and temporal.

## 3. AlexDoor-XAS Canonical Representations
AlexDoor-XAS defines four canonical action representations:

| Tag | Meaning | Coordinates and form | Current role |
|---|---|---|---|
| `A1_joint_delta` | Joint-position-target delta | Full robot joint order | Exported for Alex episodes; no current learned-policy adapter |
| `A2_ee_delta` | End-effector delta | World frame, 6D `(dx,dy,dz,drx,dry,drz)` | Final Cartesian execution currency; scripted and learned |
| `A3_obj_rel_ee_delta` | Object-relative end-effector delta | Static hinge-anchored door frame, 6D | Converted to A2; scripted and learned |
| `A4_obj_centric_chunk` | Object-centric contact and motion intent | Contact target in moving panel frame plus phase, hinge intent, and duration | Scripted/exported guarded execution; no current learned policy |

All four representations describe behavior related to the same physical door-pushing task. They differ in what information is made explicit and what work is delegated to the adapter or execution system.

## 4. A1 — Joint-Target Delta
### 4.1 Meaning
`A1_joint_delta` represents the change in the robot's joint-position targets:

    A1[k] = q_target[k+1] - q_target[k]

It is not:
- the measured joint position `q[k]`;
- measured joint velocity;
- torque;
- the difference between two realized physical configurations.

It specifically describes how the position targets sent to the actuators changed because of one control action.

### 4.2 Shape and Ordering
The current Alex V2 runtime exposes 29 movable joints in a fixed, validated order. A1 is therefore a 29-dimensional vector for this robot.

Only six right-arm joints are actively changed by the current IK executor:
- `RIGHT_SHOULDER_Y`;
- `RIGHT_SHOULDER_X`;
- `RIGHT_SHOULDER_Z`;
- `RIGHT_ELBOW_Y`;
- `RIGHT_WRIST_Z`;
- `RIGHT_WRIST_X`.

The remaining components normally remain zero because their targets are held.

The dimension alone is not sufficient to interpret A1. The joint-name order is part of the contract. Reordering values without reordering their names produces a different and incorrect command.

### 4.3 Why Target Delta Is Different From Real Motion
Suppose the target for one joint changes by `+0.05 rad`. The actual joint may initially move by less than `0.05 rad` because:
- the actuator takes time to track the target;
- the link has inertia;
- contact resists motion;
- the target is clamped at a joint limit;
- controller dynamics create lag.

Therefore:

    requested target change != guaranteed realized joint change

### 4.4 Timing Alignment
The dataset records `joint_pos_target[k]` before action `k` is applied. The target generated by action `k` becomes the next sample:

    A1[k] = joint_pos_target[k+1] - joint_pos_target[k]

The recorder stores one final post-loop target to close the last delta.

### 4.5 Strengths and Limitations
Strengths:
- direct relation to the robot's actuated coordinates;
- no online inverse-kinematics conversion is needed if A1 is executed directly;
- joint limits are easy to express.

Limitations:
- strongly tied to one robot and one joint ordering;
- difficult to interpret geometrically;
- the same object-level intent may require very different vectors on another embodiment;
- small joint errors can produce configuration-dependent Cartesian effects.

### 4.6 Current Project Boundary
A1 is derived and exported from episodes that contain joint targets. The current learned baselines and online adapter path do not execute learned A1 policies.

## 5. A2 — World-Frame End-Effector Delta
### 5.1 Meaning
`A2_ee_delta` is a six-dimensional per-control-tick end-effector delta:

    A2 = (dx_W, dy_W, dz_W, drx_W, dry_W, drz_W)

The first three values are a position displacement in meters. The final three values form a small axis-angle rotation vector in radians.

A2 is not:
- an absolute world pose;
- a quaternion;
- a joint command;
- a velocity unless a consumer explicitly divides the delta by `control_dt`.

### 5.2 World-Frame Semantics
The suffix `W` means that the component directions follow the fixed world axes.

For example:

    delta_p_W = (0.01, 0, 0) m

means “move 1 cm along `+X_W` during this control tick,” regardless of how the door is oriented.

This makes A2 easy for the simulator executor to consume, but it means that the same door-relative intent may require different numeric vectors when the complete door assembly is rotated.

### 5.3 Adapter and Safety Boundary
The A2 adapter is the final Cartesian safety boundary. It:
- validates shape and finite values;
- applies bounded corrections such as per-tick clamping;
- evaluates workspace and joint-related constraints;
- records whether the request was accepted, corrected, or rejected.

The environment also enforces per-component bounds:

    maximum translation component = 0.02 m per control tick
    maximum rotation component    = 0.05 rad per control tick

The scripted controller normally uses a smaller vector-norm step limit.

### 5.4 From A2 to Joint Targets
The execution path is:

    desired world-frame end-effector delta
      -> tool-point Jacobian
      -> damped-least-squares differential IK
      -> raw six-joint arm targets
      -> joint-limit clamp
      -> actuator targets
      -> PhysX response

The controlled point is the calibrated contact tool point offset from `RIGHT_GRIPPER_Z_LINK`, not merely the gripper-link origin.

### 5.5 Rotation Limitation
A2 stores, validates, clamps, and records all six components. However, the current Alex V2 executor uses position-only differential IK:
- translation components are actuated;
- rotational components remain represented but are not physically commanded.

This limitation must be remembered when interpreting 6D action errors or claiming pose control.

### 5.6 Strengths and Limitations
Strengths:
- geometrically interpretable in the scene;
- convenient execution interface for Cartesian IK;
- more independent of joint ordering than A1.

Limitations:
- tied to the world orientation;
- the same object-relative intent changes numerically across door poses;
- requires IK and embodiment-specific safety logic;
- Cartesian feasibility still depends on the robot's current configuration.

## 6. A3 — Door-Frame End-Effector Delta
### 6.1 Meaning
`A3_obj_rel_ee_delta` has the same six-dimensional layout as A2:

    A3 = (dx_D, dy_D, dz_D, drx_D, dry_D, drz_D)

The difference is the coordinate frame. A3 is expressed in the static, hinge-anchored door frame `D`.

### 6.2 Conversion to A2
Let `R_WD` map door-frame vectors into world coordinates. The adapter computes:

    delta_p_W   = R_WD delta_p_D
    delta_rot_W = R_WD delta_rot_D

or, as one 6D operation:

    A2 = rotate both 3D halves of A3 by R_WD

No translation is added because A3 contains deltas, not absolute points.

The resulting A2 command then follows the normal A2 safety and IK path.

### 6.3 Why the Door Frame Is Static
The A3 frame is fixed to the door fixture and hinge. It does not rotate with the panel.

This makes directions stable relative to the task:
- `+Z_D` remains the hinge axis;
- `+Y_D` remains the closed-panel direction from hinge to free edge;
- `+X_D` remains the push-face normal direction of the fixture.

The current hinge angle is still observed by the controller, but it is not multiplied into the A3-to-A2 frame conversion.

### 6.4 Generalization Across Door Poses
Suppose the same door-relative command is:

    delta_p_D = (-0.01, 0, 0) m

In D0, where the door frame is nominally aligned with the world, the world action may also be approximately:

    delta_p_W = (-0.01, 0, 0) m

If the complete door assembly is yawed, the A3 command remains numerically unchanged while `R_WD` produces the appropriate rotated A2 command.

A3 therefore factors out static door placement more directly than A2.

### 6.5 Strengths and Limitations
Strengths:
- command meaning remains stable relative to the door fixture;
- better geometric invariance across D0-D4 placement changes;
- same dimension and downstream executor as A2;
- easy to compare with A2 using matched physical episodes.

Limitations:
- still represents low-level per-tick motion rather than a complete object-level goal;
- requires a valid door-frame pose;
- remains coupled to the selected task frame;
- does not by itself guarantee embodiment transfer or physical feasibility.

### 6.6 Current Project Role
A3 is used by the scripted controller and by learned ACT/Diffusion baselines. The online A3 adapter validates the door frame, rotates A3 into A2, and delegates to the A2 adapter.

## 7. A4 — Object-Centric Intent Chunk
### 7.1 Meaning
`A4_obj_centric_chunk` represents a structured object-level intent for one controller phase rather than one raw per-tick motion vector.

Each chunk contains:

    phase
    contact_target_panel
    motion_hinge_delta_rad
    duration_ticks

The fields answer:
- **phase:** what stage of interaction is intended;
- **contact target:** where on the moving panel the interaction belongs;
- **hinge delta:** how much hinge motion is intended during a push phase;
- **duration:** how many control ticks the recorded phase lasted.

### 7.2 Phase Vocabulary
The canonical phases are:

    approach
    align
    pre_contact
    contact
    push
    hold
    release

`done` is a controller terminal state, not an emitted A4 chunk phase.

### 7.3 Contact Target in the Panel Frame
`contact_target_panel` is an absolute point in the moving panel frame:

    p_P = (x_P, y_P, z_P)

Because the panel frame rotates with the hinge angle, the same coordinates identify the same material region of the panel throughout opening.

At hinge angle `theta`:

    p_W(theta) = t_WD + R_WD Rz(theta) p_P

This is different from A3:
- A3 stores a delta in the static door frame;
- A4 stores a target point in the moving panel frame.

### 7.4 Current Door Geometry
The panel geometry is approximately:

    thickness = 0.036 m
    width     = 0.83 m
    height    = 2.0 m

In panel coordinates:

    x_P in [0, thickness]
    y_P in [0, width]
    z_P in [-height/2, +height/2]

The origin lies on the hinge at panel mid-height. The current nominal Alex push radius is:

    y_P = 0.35 * 0.83 m = 0.2905 m

and the nominal push height is:

    z_P = 0.15 m

### 7.5 Hinge Motion Is Intent, Not Guaranteed Outcome
`motion_hinge_delta_rad` describes intended hinge motion. It must not be interpreted as guaranteed achieved motion.

Execution may:
- reach the requested change;
- achieve only part of it;
- miss contact;
- stall;
- time out;
- reject the chunk before execution.

The A4 execution result therefore keeps requested and achieved hinge motion separate.

### 7.6 Duration Semantics
`duration_ticks` records the completed phase length in control ticks.

At 60 Hz:

    duration_seconds = duration_ticks / 60

The duration is not a universal physical law. Another embodiment or controller may need a different number of ticks to execute the same object-level intent.

### 7.7 A4 Execution
The guarded A4 adapter:
1. validates phase and numeric fields;
2. checks that the target lies on a valid panel region;
3. checks handle avoidance, hinge travel, and workspace reachability;
4. plans or synthesizes required interaction stages;
5. converts the live panel target into a door-frame waypoint using the current hinge angle;
6. emits per-tick A3 deltas;
7. passes A3 through the A2 safety and IK path;
8. monitors contact, progress, timeouts, termination, and stalls.

The execution chain is:

    A4 object intent
      -> panel-frame target at current hinge angle
      -> door-frame waypoint
      -> per-tick A3 delta
      -> world-frame A2 delta
      -> differential IK
      -> joint targets

### 7.8 Strengths and Limitations
Strengths:
- directly expresses contact location and object motion;
- separates object-level intent from embodiment-specific execution;
- target stays attached to the moving panel;
- supports explicit validation and guarded staged execution;
- is more human-readable than raw joint or Cartesian vectors.

Limitations:
- requires more adapter logic;
- relies on known object geometry and state;
- intent may not be physically achievable;
- duration and execution details are not automatically transferable;
- no learned A4 policy is currently established in the project.

## 8. One Physical Behavior in Four Representations
Consider the intent:

> Move the tool to a safe point on the panel, make contact, and push until the hinge opens.

The representations describe this at different levels:

### A1 view

    Change these 29 joint-position targets,
    with nonzero changes mainly in the six right-arm entries.

### A2 view

    Move the tool by this world-frame 6D delta at each control tick.

### A3 view

    Move the tool by this door-frame 6D delta at each control tick.

### A4 view

    Contact this panel-fixed point during this interaction phase
    and request this hinge-angle change.

The intended physical behavior may be related, but the numeric structure, invariance, interpretation, and execution burden differ substantially.

## 9. Adapter Responsibilities
An adapter is not merely a coordinate converter. It is the boundary that turns a requested representation into an executable and auditable command.

Each adapter decision is classified as:
- **accepted:** applied without correction;
- **corrected:** applied after a bounded, recorded correction;
- **rejected:** not executed.

The decision record retains:
- requested command;
- applied command, if any;
- validation checks;
- warnings;
- reason for correction or rejection.

This prevents safety changes from being hidden inside the controller.

### A2 adapter
Validates and bounds the final world-frame Cartesian command.

### A3 adapter
Validates `R_WD`, converts the door-frame delta to A2, and then uses the A2 adapter.

### A4 adapter
Validates and plans the object-centric chunk, produces live A3 commands, and relies on A3 and A2 for downstream execution.

### A1 boundary
A1 is currently an exported representation. There is no corresponding learned online A1 adapter in the active benchmark.

## 10. Relationship to the Jacobian
Only A2 and A3 directly use the Cartesian-to-joint conversion during the active execution path:

    A3 --rotation by R_WD--> A2
    A2 --differential IK using J--> joint targets

A1 already expresses joint-target changes, so a hypothetical direct A1 executor would not need Cartesian differential IK for that command. It would still need safety checks and actuator handling.

A4 is above both levels:

    A4 -> A3 -> A2 -> Jacobian/IK -> joint targets

## 11. Matched Representation Comparison
AlexDoor-XAS records one physical episode and derives representation-specific products from it.

For the matched A2/A3 products:
- episode identity is shared;
- physical outcome is shared;
- observations are shared;
- split membership is shared;
- pose and seed allocation are shared;
- action values differ because their frames differ.

This is more controlled than independently generating unrelated trajectories for each representation. It reduces the risk that a measured difference is caused by different physical data rather than by action representation.

A1 can be derived when joint targets are available. A4 is derived from the scripted controller's phase and object-intent log.

## 12. What the Current Project Does and Does Not Establish
The current project establishes:
- a fixed-base Alex V2 door-pushing benchmark;
- four explicit action-representation contracts;
- scripted execution and export for the representations;
- learned ACT and Diffusion baselines for A2 and A3;
- adapter-mediated safety and logging;
- matched representation-aware datasets and evaluation.

It does not yet establish:
- a learned A1 policy;
- a learned A4 policy;
- full 6D orientation actuation for A2/A3;
- a universal action space proven across multiple robot embodiments;
- automatic transfer without embodiment-specific execution logic.

Therefore, A1-A4 should be understood as controlled action-representation choices within the current Alex benchmark, not as proof that one interface is universally optimal.

## 13. Compact Comparison
| Question | A1 | A2 | A3 | A4 |
|---|---|---|---|---|
| What is commanded? | Joint-target change | EE delta | EE delta | Object/contact intent |
| Coordinate system | Robot joints | World frame | Static door frame | Moving panel frame |
| Per-tick vector? | Yes | Yes | Yes | No, structured phase chunk |
| Dimension/form | 29D for current Alex | 6D | 6D | Phase + point + hinge delta + duration |
| Needs IK in current path? | No direct path exists | Yes | Yes, after A3->A2 | Yes, through A3->A2 |
| Robot-specific? | Strongly | Executor-dependent | Executor-dependent | Adapter-dependent |
| Object-relative? | No | No | Relative to fixture | Relative to moving panel |
| Current learned baseline? | No | ACT and Diffusion | ACT and Diffusion | No |

## Final Mental Model

    A1 = how joint targets change
    A2 = how the tool moves in the world
    A3 = how the tool moves relative to the fixed door fixture
    A4 = what interaction is intended at a panel-fixed location

The full current execution ladder is:

    object intent
      -> task-relative Cartesian motion
      -> world-frame Cartesian motion
      -> joint targets
      -> actuator and physics response

The higher the representation sits in this ladder, the more interpretation and embodiment-specific execution work must be performed below it.

## Primary Project References
- [`src/alexdoor_xas/action/spaces.py`](../../../src/alexdoor_xas/action/spaces.py)
- [`src/alexdoor_xas/action/frames.py`](../../../src/alexdoor_xas/action/frames.py)
- [`src/alexdoor_xas/adapters/a2.py`](../../../src/alexdoor_xas/adapters/a2.py)
- [`src/alexdoor_xas/adapters/a3.py`](../../../src/alexdoor_xas/adapters/a3.py)
- [`src/alexdoor_xas/adapters/a4.py`](../../../src/alexdoor_xas/adapters/a4.py)
- [`src/alexdoor_xas/data_engine/export.py`](../../../src/alexdoor_xas/data_engine/export.py)
- [`src/alexdoor_xas/envs/door_task/door_push_robot_env.py`](../../../src/alexdoor_xas/envs/door_task/door_push_robot_env.py)

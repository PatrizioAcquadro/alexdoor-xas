# Measurement Setup
## Purpose
Frames, coordinates, conventions, and timing answer four questions that must be explicit in every robotics project:
- **What quantity is represented?** A point, direction, displacement, pose, velocity, force, or joint value.
- **Relative to which reference?** The world, robot base, link, end-effector, door frame, or moving panel.
- **Under which convention?** Axis directions, handedness, units, angle convention, and quaternion order.
- **At what time?** Before or after an action, at a physics step, at a control tick, or at an episode boundary.

A tuple such as `(0.30, 0.10, 0.80)` is not meaningful by itself. It becomes meaningful only when its quantity, frame, units, and timestamp are known.

This document first explains these ideas in a project-independent way and then defines how AlexDoor-XAS uses them. The A1-A4 action representations are documented separately in [`06_ActionSpaces.md`](06_ActionSpaces.md).

## Part I — General Frames and Coordinates
### 1. What Is a Frame?
A **frame**, or coordinate frame, is a three-dimensional measuring reference. It consists of:
- an origin;
- an X axis;
- a Y axis;
- a Z axis;
- a declared convention for the directions and orientation of those axes.

A useful mental model is a small rigid tripod of arrows attached to something. A project may attach frames to the world, a robot base, each robot link, a camera, an end-effector, or an object.

The frame is not the measured point. The frame is the ruler used to describe the point.

The same physical point can therefore have different coordinates in different frames. A cup may be:
- at `(2.0, 1.0, 0.8)` in a room frame;
- at `(0.6, -0.2, 0.1)` in a robot-base frame;
- at `(0.0, 0.0, 0.15)` in a table frame.

The cup has not moved. Only the reference used to describe it has changed.

### 2. Frame Versus Coordinates
The distinction is:
- the **frame** defines the origin and axes;
- the **coordinates** are the numeric components measured along those axes.

The notation

    p_W = (1.0, 2.0, 0.5)

means that the coordinates of point `p), expressed in world frame `W), are `(1.0, 2.0, 0.5)`.

Project names such as `ee_pos_w` follow the same idea:
- `ee_pos` means end-effector position;
- the suffix `_w` means that the value is expressed in the world frame.

### 3. Point, Vector, Delta, and Pose
Several quantities may all contain three numbers, but they do not transform in the same way.

#### 3.1 Point
A **point** represents a location:

    p = position of the end-effector

Its coordinates are measured from a frame origin. Changing the frame therefore requires both:
- a rotation, because the axes may differ;
- a translation, because the origins may differ.

#### 3.2 Vector
A **vector** represents a magnitude and direction:

    v = direction, linear velocity, force, or displacement

A free vector does not identify an absolute location. When it is re-expressed in another frame at the same physical reference point, only the axes change, so only a rotation is needed.

#### 3.3 Delta
A **delta** is a change:

    delta_p = p_after - p_before

The translation between frame origins cancels when two points are subtracted. A positional delta is therefore transformed like a vector, using rotation only.

#### 3.4 Pose
A **pose** combines:

    pose = position + orientation

The position locates a frame origin. The orientation specifies how that frame's axes are rotated relative to another frame.

### 4. Translation and Rotation
Assume two frames:
- `W`: world frame;
- `D`: a local frame.

The relationship between them contains two different pieces of information.

#### 4.1 Translation
`t_WD` is the vector from the origin of `W` to the origin of `D`, expressed in world coordinates.

It answers:

> Where is the origin of frame D in the world?

Translation changes location but does not, by itself, change axis directions.

#### 4.2 Rotation
`R_WD` is the rotation matrix that re-expresses coordinates from frame `D` into frame `W`.

It answers:

> How are the axes of frame D oriented relative to the axes of frame W?

The subscript can be read as:

    R_WD: world <- D

Each column of `R_WD` has a concrete geometric meaning:
- column 1 is the `+X_D` unit axis expressed in world coordinates;
- column 2 is the `+Y_D` unit axis expressed in world coordinates;
- column 3 is the `+Z_D` unit axis expressed in world coordinates.

A valid rigid-body rotation matrix satisfies:

    R_WD^T R_WD = I
    det(R_WD) = +1

Its inverse is its transpose:

    R_DW = R_WD^T

### 5. Where Does a Rotation Matrix Come From?
A rotation matrix is not guessed. It is obtained from the known or estimated pose of one frame relative to another.

Depending on the project, that pose may come from:
- a URDF, USD, CAD, or scene description;
- forward kinematics computed from measured joint positions;
- a calibrated static transform between sensors;
- visual tracking or object-pose estimation;
- SLAM or motion capture;
- an IMU orientation estimate;
- a manually measured and validated transform.

The producer may provide orientation as a rotation matrix, Euler angles, or a quaternion. The project converts that representation into `R_WD` when matrix operations are needed.

Therefore, asking “who gives us `R_WD`?” has two levels:
1. **The physical or software source gives the relative pose.**
2. **Project math converts its orientation into the rotation matrix.**

The matrix represents the same orientation as the source quaternion or Euler angles. It is not an additional independent measurement.

### 6. Transforming a Point
Let `p_D` be a point expressed in frame `D`. Its world coordinates are:

    p_W = t_WD + R_WD p_D

The operations have different responsibilities:
1. `R_WD p_D` re-expresses the point offset using world-axis directions.
2. `t_WD` places that rotated offset at the world location of frame `D`'s origin.

The inverse transform is:

    p_D = R_WD^T (p_W - t_WD)

The order matters. To express a world point in `D`, first subtract the location of `D`'s origin, then rotate into `D`'s axes.

### 7. Why a Vector or Delta Uses Only Rotation
Consider two points in frame `D`:

    p1_W = t_WD + R_WD p1_D
    p2_W = t_WD + R_WD p2_D

Subtract them:

    p2_W - p1_W
      = (t_WD + R_WD p2_D) - (t_WD + R_WD p1_D)
      = R_WD (p2_D - p1_D)

The same translation appears in both points and cancels. Therefore:

    delta_p_W = R_WD delta_p_D
    v_W       = R_WD v_D

This is why:
- an absolute contact point uses rotation and translation;
- a direction, force vector, or small positional delta uses rotation only.

This rule assumes that the vector is re-expressed at the same physical reference point. More advanced spatial quantities, such as a wrench or a velocity moved from one reference point to another, may require an additional lever-arm term. The simple point/vector rule remains correct for the free 3D vectors and end-effector deltas used by this project.

### 8. Homogeneous Transform
Rotation and translation are often combined into one `4 x 4` homogeneous transform:

    T_WD = [ R_WD  t_WD ]
           [ 0 0 0   1   ]

A point is extended with a final value of `1`:

    [p_W]         [p_D]
    [ 1 ] = T_WD [ 1 ]

A free vector is extended with a final value of `0`:

    [v_W]         [v_D]
    [ 0 ] = T_WD [ 0 ]

The final zero removes the translation term for a vector. This is another way to see why points translate but free vectors do not.

### 9. Orientation Representations
Orientation may be stored as:
- Euler angles;
- a `3 x 3` rotation matrix;
- an axis-angle or rotation vector;
- a quaternion.

These are different numeric representations of orientation, not different physical orientations.

#### 9.1 Euler Angles
Euler angles use three sequential rotations, such as roll, pitch, and yaw. They are intuitive but require a declared order and can encounter singularities such as gimbal lock.

#### 9.2 Rotation Matrix
A rotation matrix directly maps vector coordinates between frames. It is convenient for transformations but uses nine stored values to represent three rotational degrees of freedom.

#### 9.3 Axis-Angle
Axis-angle describes an orientation as:
- a unit rotation axis `u`;
- an angle `theta` about that axis.

A rotation vector stores:

    r = u theta

Its direction is the rotation axis and its norm is the angle in radians.

#### 9.4 Quaternion
A unit quaternion stores four values. AlexDoor-XAS uses:

    q = (x, y, z, w)

For a rotation by angle `theta` about unit axis `u = (u_x, u_y, u_z)`:

    x = u_x sin(theta / 2)
    y = u_y sin(theta / 2)
    z = u_z sin(theta / 2)
    w =       cos(theta / 2)

The first three values form the **vector part**. The final value `w` is the **scalar part**.

### 10. What Does w Mean in a Quaternion?
`w` is not:
- a fourth spatial axis;
- a position;
- a timestamp;
- an angular velocity.

It is the scalar component that, together with `(x, y, z)`, encodes the rotation angle. Specifically:

    w = cos(theta / 2)

The half-angle appears because quaternion multiplication is constructed to compose three-dimensional rotations smoothly.

Examples in `(x, y, z, w)` order:

#### Identity rotation

    theta = 0
    q = (0, 0, 0, 1)

Here `w = cos(0) = 1`.

#### 180 degrees about +Z

    theta = pi
    q = (0, 0, 1, 0)

Here `sin(pi/2) = 1` and `w = cos(pi/2) = 0`.

#### 90 degrees about +Z

    theta = pi/2
    q = (0, 0, sin(pi/4), cos(pi/4))
      approximately (0, 0, 0.7071, 0.7071)

Important quaternion facts:
- a rotation quaternion must have unit norm;
- `q` and `-q` represent the same physical orientation;
- quaternion components should not be interpreted as independent Euler angles;
- quaternion order must always be declared because both `(x,y,z,w)` and `(w,x,y,z)` are common.

### 11. Right-Handed Rotations
In a right-handed coordinate system, the positive rotation direction follows the right-hand rule:
- point the right thumb along the positive rotation axis;
- the curl of the fingers gives the positive rotation direction.

For a positive rotation about `+Z`:

    +X rotates toward +Y
    +Y rotates toward -X

The standard matrix is:

    Rz(theta) = [ cos(theta)  -sin(theta)  0 ]
                [ sin(theta)   cos(theta)  0 ]
                [     0            0       1 ]

### 12. Cartesian Space and Joint Space
**Cartesian space**, also called task space, describes what a robot point or frame does in physical space:
- end-effector position;
- end-effector orientation;
- Cartesian displacement;
- linear and angular velocity.

**Joint space** describes the robot's internal configuration:

    q = [q1, q2, ..., qN]

Each coordinate belongs to one named joint. A revolute-joint coordinate is usually measured in radians; a prismatic-joint coordinate is usually measured in meters.

Joint space is not a three-dimensional frame. It has one mathematical axis per joint, and the ordering of those joints is part of the data contract.

The two directions of conversion are:

    joint positions --forward kinematics--> Cartesian pose
    Cartesian target --inverse kinematics--> joint target

### 13. What Is the Jacobian?
The **Jacobian** is the local relationship between joint motion and Cartesian motion.

Suppose the robot has joint configuration `q` and an end-effector pose `x = f(q)`. The function `f` is forward kinematics. Around the current configuration, the Jacobian is:

    J(q) = partial f(q) / partial q

Operationally:

    x_dot = J(q) q_dot

or, for sufficiently small changes:

    delta_x approximately J(q) delta_q

Each Jacobian column answers:

> If this one joint moves a small positive amount while the others stay fixed, in which Cartesian direction and by how much will the end-effector move?

For a six-dimensional end-effector velocity, the rows normally represent:

    [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]

The columns correspond to the controlled joints in their declared order.

For example, a `6 x 6` Jacobian for a six-joint arm maps six joint velocities to:
- three components of end-effector linear velocity;
- three components of end-effector angular velocity.

The Jacobian depends on the current joint configuration. It changes as the arm moves.

### 14. How Inverse Kinematics Uses the Jacobian
Differential inverse kinematics starts with a desired small Cartesian motion `delta_x` and seeks a joint change `delta_q` satisfying:

    J delta_q approximately delta_x

If `J` were square and safely invertible:

    delta_q = J^-1 delta_x

Real robot Jacobians may be rectangular, redundant, singular, or poorly conditioned. A common robust solution is damped least squares:

    delta_q = J^T (J J^T + lambda^2 I)^-1 delta_x

The damping term `lambda` avoids very large joint commands near singular configurations.

The result is still only a local approximation. The controller must also consider:
- joint-position limits;
- joint-velocity limits;
- reachability;
- collisions;
- controller gains;
- the difference between commanded and physically realized motion.

## Part II — General Timing
### 15. Why Robotics Has More Than One Clock
A robotics system normally has several related time concepts. Treating them as one clock causes alignment and causality errors.

#### 15.1 Wall-Clock Time
Wall-clock time is calendar or host-machine time. It is useful for:
- creation timestamps;
- provenance;
- correlating logs from different machines.

It is not necessarily the time used to integrate simulation dynamics.

#### 15.2 Simulation Time
Simulation time is time inside the simulated world. Ten seconds of simulation may require less or more than ten seconds of wall-clock time.

Physical equations must advance according to simulation time, not according to how quickly the workstation renders or computes them.

#### 15.3 Physics Timestep
The physics timestep `physics_dt` is the amount of simulation time advanced by one numerical integration step.

Each physics step updates quantities such as:
- rigid-body poses and velocities;
- articulation states;
- actuator response;
- collisions and contacts.

#### 15.4 Control Timestep
The control timestep `control_dt` is the interval between two newly issued controller or policy actions.

Physics often runs faster than control:

    control_dt = decimation * physics_dt

The action or resulting low-level target is held while the physics engine performs the configured number of physics steps.

#### 15.5 Sensor Timestep
A sensor may update at another frequency. A complete system must preserve:
- acquisition time;
- availability time;
- the state or action interval to which the sample belongs.

#### 15.6 Episode Timing
Episode timing defines the experimental lifecycle:
- reset;
- first valid observation;
- maximum duration;
- success condition;
- termination or truncation;
- final state.

The physics engine does not intrinsically understand episodes. Project and environment code impose that structure.

### 16. Causal State-Action Alignment
A clear discrete-time convention is:

    t_k = k * control_dt

At tick `k`:

    observe state s_k
    compute action a_k
    record the pair (s_k, a_k)
    apply a_k during [t_k, t_(k+1))
    obtain state s_(k+1)

Therefore:

    s_k --a_k--> s_(k+1)

The observation stored with action `a_k` is a **pre-action observation**. It does not yet contain the response caused by `a_k`.

### 17. Latency
In a physical robot, these events may occur at different wall-clock times:
- sensor exposure or acquisition;
- delivery of the measurement;
- policy inference;
- command transmission;
- actuator response.

The relevant control delay is the age of the physical state by the time its resulting command affects the robot.

A synchronous simulator simplifies this chain, but the recorded pre-action/post-action convention must still be explicit.

### 18. Action Chunks and Receding Horizon
A policy may predict a sequence:

    [a_k, a_(k+1), ..., a_(k+H-1)]

`H` is the prediction horizon in control ticks. A project must state:
- how many future actions are predicted;
- how many are executed;
- when the policy observes the world again;
- whether overlapping predictions are combined.

At a 60 Hz control rate:

    40 ticks = 40 / 60 s = 0.667 s
    16 ticks = 16 / 60 s = 0.267 s
     8 ticks =  8 / 60 s = 0.133 s

## Part III — AlexDoor-XAS Coordinate Conventions
### 19. Project-Wide Conventions
AlexDoor-XAS uses:

| Property | Convention |
|---|---|
| Linear distance | meters |
| Angles | radians |
| Up axis | `+Z` |
| Rotation sign | right-handed |
| Quaternion order in project arrays | `(x, y, z, w)` |
| Cartesian delta order | `(dx, dy, dz, drx, dry, drz)` |
| Cartesian rotation delta | axis-angle vector in radians |

These contracts are implemented in [`src/alexdoor_xas/action/frames.py`](../../../src/alexdoor_xas/action/frames.py) and [`src/alexdoor_xas/action/spaces.py`](../../../src/alexdoor_xas/action/spaces.py).

One format exception must be kept explicit: the textual USD syntax for a value such as `quatf` places the real component first, effectively `(w,x,y,z)`. Project runtime arrays use `(x,y,z,w)`. [`src/alexdoor_xas/envs/door_task/door_runtime.py`](../../../src/alexdoor_xas/envs/door_task/door_runtime.py) performs this conversion when it reads the USD Stage.

### 20. World Frame W
The world frame is the global reference of the Isaac scene:
- the floor lies in the `XY` plane;
- `+Z_W` points upward;
- values ending in `_w` are expressed in world coordinates;
- world-frame Cartesian values are the final input currency of the environment executor.

The project does not define `+X_W` as universally “robot forward.” The robot itself may be rotated relative to the world.

### 21. Robot Base, Link, and Tool Point
Alex V2 is fixed-base during the benchmark. The base pose is part of scene setup, not a controlled floating-base degree of freedom.

The operational end-effector is a calibrated tool point rigidly offset from `RIGHT_GRIPPER_Z_LINK`. The current calibration defines the offset in that link's coordinates. The executor:
1. reads the gripper-link world pose;
2. rotates the link-local offset into the world;
3. adds it to the link position;
4. obtains the tool-point world pose.

The Jacobian is also shifted from the gripper-link origin to this tool point. This is necessary because rotating a link moves any point that is offset from the link origin. The point velocity obeys:

    v_tool = v_link + omega_link cross r_world

The implementation is in [`src/alexdoor_xas/envs/door_task/door_push_alex_v2_executor.py`](../../../src/alexdoor_xas/envs/door_task/door_push_alex_v2_executor.py) and [`src/alexdoor_xas/kinematics/offset_point.py`](../../../src/alexdoor_xas/kinematics/offset_point.py).

Consequently, `ee_pos_w` is the world position of the operational contact tool point, not merely the graphical or link-frame origin of the gripper.

### 22. Door Frame D
The door frame is the static frame of the door fixture:
- its origin is the `Doorframe` body origin;
- the hinge axis passes through the origin;
- `+Z_D` is the hinge axis;
- `+Y_D` extends from the hinge toward the free edge of the closed panel;
- `+X_D` is normal to the panel and points toward the push face;
- it remains fixed during one episode.

The door frame is static even though the door panel moves. “Static” describes the fixture frame, not the articulated panel.

### 23. How AlexDoor-XAS Obtains t_WD and R_WD
The door asset and the selected D0-D4 task layer define the composed pose of the `Doorframe` prim on the USD Stage.

At environment initialization:
1. USD composition combines the source door asset, the selected task-layer pose, and the environment namespace.
2. `UsdGeom.XformCache.GetLocalToWorldTransform` returns the composed `Doorframe` transform.
3. `ExtractTranslation()` gives `t_WD`.
4. `ExtractRotationQuat()` gives the door-frame orientation.
5. Project code reorders the quaternion into `(x,y,z,w)`.
6. `quat_to_rot_matrix()` converts that quaternion into `R_WD`.

The relevant readers are:
- [`src/alexdoor_xas/envs/door_task/door_runtime.py`](../../../src/alexdoor_xas/envs/door_task/door_runtime.py);
- [`src/alexdoor_xas/action/frames.py`](../../../src/alexdoor_xas/action/frames.py).

Therefore, USD/Isaac provides the composed door-frame pose, and AlexDoor-XAS converts its orientation into `R_WD`. The matrix is not learned, manually guessed at every tick, or derived from the current hinge opening.

The environment reads and stores this door-frame pose when it initializes. The fixed `Doorframe` body keeps it static through the episode.

### 24. Panel Frame P
The panel frame is attached to the moving door panel:
- it uses the hinge origin;
- `+Z_P` remains the hinge axis;
- `+Y_P` runs across the panel from hinge to free edge;
- `+X_P` is normal to the push face;
- it rotates with the current hinge angle `theta`.

Its orientation is:

    R_WP(theta) = R_WD Rz(theta)

Its origin is:

    t_WP = t_WD

A panel-fixed point `p_P` therefore maps to the world as:

    p_W(theta) = t_WD + R_WD Rz(theta) p_P

The point's panel coordinates stay constant while its world coordinates change as the panel opens.

The project implementation is `panel_frame()` in [`src/alexdoor_xas/action/frames.py`](../../../src/alexdoor_xas/action/frames.py).

### 25. Door Frame Versus Panel Frame
The essential distinction is:

    door frame D  = fixed to the hinge fixture
    panel frame P = rotates with the panel

When the hinge angle is zero:

    R_WP = R_WD

When the panel opens:

    R_WP = R_WD Rz(theta)

A direction expressed in the door frame keeps the same meaning relative to the stationary fixture. A contact point expressed in the panel frame stays attached to the same material location on the moving panel.

### 26. Door Pose Versus Hinge Angle
AlexDoor-XAS uses two different rotations that must not be conflated.

#### Door yaw
`door_yaw_rad` is the static placement of the complete door assembly in the world. It changes between canonical D0-D4 scenes and remains constant within an episode.

#### Hinge angle
`hinge_angle_rad` is the dynamic opening angle of the panel relative to the static door frame. It changes during the push.

For pure yaw motion, the panel's approximate world yaw is:

    panel world yaw = door yaw + hinge angle

The canonical task poses are:

| Pose | Static door yaw | World XY offset |
|---|---:|---:|
| `D0` | `0.00 rad` | `(0.00, 0.00) m` |
| `D1` | `+0.05 rad` | `(+0.02, 0.00) m` |
| `D2` | `-0.05 rad` | `(0.00, -0.02) m` |
| `D3` | `+0.10 rad` | `(+0.02, +0.02) m` |
| `D4` | `-0.10 rad` | `(+0.02, -0.02) m` |

These variations move the door assembly. Alex's fixed base does not move between these pose definitions.

### 27. Joint Space in AlexDoor-XAS
The current runtime asset exposes 29 movable robot joints in an exact declared order. The benchmark controller drives only the following six right-arm joints:
- `RIGHT_SHOULDER_Y`;
- `RIGHT_SHOULDER_X`;
- `RIGHT_SHOULDER_Z`;
- `RIGHT_ELBOW_Y`;
- `RIGHT_WRIST_Z`;
- `RIGHT_WRIST_X`.

The environment distinguishes:
- `joint_pos`: physically realized joint position;
- `joint_vel`: physically realized joint velocity;
- `joint_pos_target`: position target sent to the joint actuators.

The Cartesian execution path uses the world-frame tool-point Jacobian for the six controlled joints. Differential IK computes new arm targets, those targets are clamped to the joint limits, and the actuator/physics system attempts to realize them.

The requested target and realized joint position need not be identical at the same tick because of actuator dynamics, inertia, contacts, and limits.

## AlexDoor-XAS Timing
### 28. Who Controls Time?
AlexDoor-XAS uses three central temporal concepts: the physics timestep, the control timestep, and episode timing. They are related, but they have different responsibilities.

The responsibility chain is:
- **Project code** selects timing values and experimental rules.
- **Isaac Lab** schedules the environment lifecycle.
- **Isaac Sim** advances simulated time.
- **PhysX** numerically integrates physical evolution within each physics step.

### 29. Physics Timestep
AlexDoor-XAS uses:

    physics_dt = 1/120 s
    physics frequency = 120 Hz

The project declares the value through `SimulationCfg` in [`src/alexdoor_xas/envs/door_task/door_push_robot_env_cfg.py`](../../../src/alexdoor_xas/envs/door_task/door_push_robot_env_cfg.py).

At each physics step, PhysX updates bodies, articulations, drives, collisions, and contacts over a simulated interval of `1/120 s`.

### 30. Control Timestep
AlexDoor-XAS uses:

    decimation = 2
    control_dt = physics_dt * decimation
               = (1/120 s) * 2
               = 1/60 s
    control frequency = 60 Hz

For each control tick:

    one new action
      -> physics step 1 at 1/120 s
      -> physics step 2 at 1/120 s
      -> next control observation

The resulting joint-position targets remain active across those two physics steps.

### 31. One Recorded Control Tick
For dataset generation, tick `k` has timestamp:

    t_k = k * control_dt = k / 60 s

The logical sequence is:
1. read hinge angle and velocity;
2. read the tool pose;
3. read contact, force, and joint state when available;
4. compute the controller action;
5. record the pre-action state and requested action at `t_k`;
6. convert and apply the action;
7. execute two physics steps;
8. expose the resulting next state.

Therefore:

    recorded state[k] --action[k]--> resulting state[k+1]

The data-generation implementation is in [`src/alexdoor_xas/data_engine/generate.py`](../../../src/alexdoor_xas/data_engine/generate.py).

### 32. Contact Alignment
The contact value stored in ordinary step `k` is sampled before action `k` is executed. It therefore describes the pre-action state and generally reflects the response to earlier actions.

The response to the final action has no following ordinary step. The recorder therefore stores a separate `terminal_contact` sample at:

    t_terminal = n_steps * control_dt

This closes the final action-response interval without falsely assigning a post-action force to the pre-action observation.

### 33. Joint-Target Alignment
`joint_pos_target[k]` is also sampled before action `k`.

The new target generated by action `k` is visible in the next target sample:

    joint target delta[k]
      = joint_pos_target[k+1] - joint_pos_target[k]

The recorder captures one final post-loop target so that the last executed action also has a corresponding target delta.

### 34. Episode Timing
The nominal benchmark limits are:

    episode duration = 10 s
    maximum control ticks = 600
    corresponding physics steps = 1200

An episode reset is not ordinary physical evolution. The environment deliberately restores approved robot and door joint states, targets, counters, and buffers.

If a randomized start offset is requested, the environment may run a bounded IK settle before the recorded rollout. Consequently, recorded `t = 0` means:

> the first official episode observation after reset and required start-state preparation

It does not necessarily mean the first simulator step after the process was launched.

### 35. Success and Stopping
The benchmark success threshold is the first hinge-angle crossing of:

    pi / 4 rad = 45 degrees

The scripted controller may use a higher internal target so that the physical trajectory crosses the success threshold robustly. Success measurement and controller target are related but distinct concepts.

An episode may also stop because of:
- controller completion;
- controller timeout;
- rollout tick budget;
- environment termination;
- environment truncation;
- step failure.

### 36. Learned-Policy Timing
At the current 60 Hz control rate:
- ACT predicts 40 control ticks, approximately `0.667 s`; without temporal ensembling, the policy is queried again after that chunk is executed.
- Diffusion predicts 16 control ticks, approximately `0.267 s`, and normally executes the first 8 ticks, approximately `0.133 s`, before observing and replanning.
- With ACT temporal ensembling enabled, a fresh chunk is predicted each tick and overlapping predictions for the current action are combined.

These are policy execution schedules layered on top of the same 60 Hz control loop. They do not change the 120 Hz physics timestep.

## Final Interpretation Checklist
Before using any geometric or temporal value, ask:
1. What physical quantity is this?
2. Is it a point, free vector, delta, orientation, pose, or joint quantity?
3. In which frame is it expressed?
4. What are the axis, unit, handedness, and quaternion conventions?
5. Where did the frame pose come from?
6. Is this a requested target or a physically realized state?
7. Is the sample pre-action or post-action?
8. Which timestep and episode interval does it represent?

For AlexDoor-XAS, the shortest correct summary is:

    World frame  = where something is in the scene
    Door frame   = a static reference attached to the hinge fixture
    Panel frame  = a reference that rotates with the door panel
    Joint space  = the robot's internal articulated configuration
    Timing       = which state produced which command and which response

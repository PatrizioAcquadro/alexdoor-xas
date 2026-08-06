# Benchmark
## 1. What Is a Benchmark?
A benchmark is a **standardized experimental system** used to *measure and compare methods under clearly defined and repeatable conditions*. 
It specifies the problem to solve, the conditions in which it must be solved, the information and actions available to each method, and the evidence used to judge the result.

The **purpose of this standardization** is to make *results*:
- *comparable:* different methods are tested under equivalent conditions;
- *repeatable:* another researcher can execute the same experiment again;
- *interpretable:* a performance difference can be connected to a known experimental variable;
- *verifiable:* the result is supported by configurations, data, models, and recorded outputs rather than by a qualitative demonstration alone.

A task and a benchmark are therefore not the same thing. 
A **task** describes the desired behavior, such as pushing a door. 
A **benchmark** is the complete measurement system built around that task. It determines exactly what is tested, what is held constant, what is allowed to change, how many times the test is repeated, how performance is measured, and what records are required to reproduce the result.

### Main Types and Dimensions of Benchmarks
**There is no single universal classification.** The same benchmark can belong to *several categories at the same time* because each dimension answers a different question.

**Scope:** how much of the system the benchmark evaluates.
  - **Component microbenchmark:** isolates one component, operation, or subsystem.
  - **End-to-end benchmark:** evaluates the complete pipeline from input to task outcome.
  - **Task suite:** evaluates the same method across a collection of tasks.
**Interaction:** whether the evaluated method can affect what happens next.
  - **Offline dataset evaluation:** the method is tested on fixed recorded data and cannot change future observations.
  - **Online closed-loop evaluation:** every action changes the environment, the next observation depends on that change, and the policy acts again.
  - **Hybrid offline-to-online evaluation:** training or initial evaluation uses recorded data, followed by closed-loop execution in an environment.
**Environment:** where the experiment is executed.
  - **Simulation:** the robot and world are represented by software.
  - **Real hardware:** the experiment is performed with a physical robot.
  - **Sim-to-real:** a method developed in simulation is also evaluated on the physical system.
**Task coverage:** how many distinct tasks are included.
  - **Single-task:** all experiments address one defined task.
  - **Multi-task:** the same method is tested on several different tasks.
**Robot coverage:** how many robot embodiments are included.
  - **Single embodiment:** every experiment uses one robot design.
  - **Multi-robot:** multiple robots are evaluated, without necessarily requiring one shared representation or policy.
  - **Cross-embodiment:** knowledge, policies, or representations are explicitly compared or transferred across different robot bodies.
**Information:** which observations or instructions are available to the method.
  - **State-based:** the method receives structured quantities such as joint positions, poses, velocities, or forces.
  - **Vision-based:** the method receives camera images or video.
  - **Language-conditioned:** natural-language instructions specify or modify the desired behavior.
  - **Multimodal:** two or more information sources, such as state, vision, language, or audio, are combined.
**Objective:** which aspect of performance the benchmark is designed to measure.
  - **Success:** whether the task is completed.
  - **Efficiency:** how much time, energy, data, or computation is required.
  - **Robustness:** whether performance remains stable under noise or controlled disturbances.
  - **Generalization:** whether the method works in conditions that differ from those used for training.
  - **Safety:** whether execution respects force, collision, or other safety limits.
**Distribution:** how evaluation conditions relate to the training conditions.
  - **In-distribution:** evaluation conditions follow the same distribution as the training data.
  - **Controlled variations:** selected factors are changed systematically while the rest of the experiment remains controlled.
  - **Out-of-distribution:** evaluation introduces conditions outside the training distribution to test broader generalization.

### What a Complete Benchmark Includes
A *complete benchmark* includes at least:
1. **Task:** what must be done.
2. **Environment:** where the task takes place and what the method can observe and affect.
3. **Protocol:** initial conditions, duration, randomizations, repetitions, and execution rules.
4. **Metrics:** how performance is measured.
5. **Baselines:** reference methods used for comparison.
6. **Experimental control:** which variables remain fixed and which variables change.
7. **Provenance:** which data, configurations, models, software versions, and artifacts produced the result.

A concrete and well-known example is the [Arcade Learning Environment (ALE)](https://ale.farama.org/), commonly known as the Atari benchmark. 
It is used to compare agents that play Atari 2600 games. The same seven elements can be identified clearly:
1. **Task:** in each game, the agent observes the game state and selects joystick or fire actions to maximize the game score.
2. **Environment:** ALE provides the Atari 2600 emulator, the game ROMs, and a standard interface through which every agent observes and acts.
3. **Protocol:** the evaluation specifies elements such as the games, observation format, frame skip, sticky-action probability, episode termination, training budget, and evaluation procedure.
4. **Metrics:** agents are compared using episode scores for each game and aggregate normalized scores across the selected games.
5. **Baselines:** reference results such as random play, human play, and published agents show whether a new method performs poorly, reasonably, or better than established approaches.
6. **Experimental control:** competing agents use the same games, emulator settings, preprocessing, budgets, and evaluation rules. The learning method is the variable that changes.
7. **Provenance:** a reproducible result records the ALE and environment versions, game configuration, agent code and checkpoint, preprocessing and wrapper settings, and random seeds.

The distinction is important: playing *Breakout* is a task, whereas ALE, together with its protocol, metrics, controls, and reference results, forms the benchmark.

**Likewise, saying "the robot opened the door" is not enough.** 
We must know which door, which robot, which initial configuration, which policy, which success threshold, how many trials, which forces were measured, and which failures occurred.

## 2. Our Benchmark: AlexDoor-XAS
AlexDoor-XAS studies action representations for humanoid articulated-object manipulation. 
The current door-pushing benchmark is the **first concrete benchmark in that broader research project**, not the final limit of the project.

In this first benchmark, a simplified **fixed-base IHMC Alex V2 torso pushes a simulated hinged door**. 
*The robot, door, task, data, and evaluation conditions are kept as consistent as possible, while the main experimental variable is the language in which the policy describes the action to execute.*

The benchmark is therefore **not designed merely to prove that Alex can push a door once**. It is designed to compare whether different action representations produce different closed-loop behavior under matched conditions, including differences in task completion, efficiency, contact behavior, force, adapter corrections, failures, and robustness.

The long-term research objective is to determine whether representations that are less tied to one robot can provide a better foundation for generalization, cross-embodiment learning, and future vision-language-action policies. **The current benchmark creates a controlled first test of that idea**; it does *not yet demonstrate broad transfer or real-robot generalization.*

The table below classifies AlexDoor-XAS along the dimensions introduced above.

| Dimension | AlexDoor-XAS today |
|---|---|
| Scope | End-to-end benchmark on one task; it is not currently a task suite |
| Interaction | Offline training from recorded datasets followed by online closed-loop evaluation |
| Environment | Simulation only, using Isaac Sim and Isaac Lab |
| Task coverage | Single-task: door pushing |
| Robot coverage | Single embodiment: the fixed-base Alex V2 torso |
| Information | State-based |
| Objective | Controlled comparison of action representations, including task-success, force, and safety measurements |
| Distribution | Small controlled D0-D4 variations, not broad out-of-distribution generalization |

In one sentence:
> AlexDoor-XAS is currently a simulated, single-task, single-embodiment, state-only, closed-loop benchmark
> for comparing action representations under matched conditions.

The following table maps the seven components of a complete benchmark to the current AlexDoor-XAS implementation:

| Benchmark component  | AlexDoor-XAS today                                                                                                                                                              |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Task                 | Use Alex's right-hand contact point to push the hinged door panel to 45 degrees under the D0-D4 task variations                                                                 |
| Environment          | State-based closed-loop simulation in Isaac Sim and Isaac Lab, using a fixed-base Alex V2 asset, one articulated door, contact sensing, and a supporting floor                  |
| Protocol             | Versioned configurations define initial conditions, pose and seed schedules, simulator timing, episode limits, and first-crossing termination                                   |
| Metrics              | Task completion is considered together with door motion, timing, contact, force, adapter behavior, warnings, and failures                                                       |
| Baselines            | A scripted reference controller provides demonstrations, while state-only ACT and Diffusion policies provide learned comparison methods where defined                           |
| Experimental control | Compared methods use matched robot, task, data, poses, seeds, controller contracts, safety limits, and evaluation conditions; the action representation is the primary variable |
| Provenance           | Calibrations, datasets, splits, configurations, checkpoints, evaluations, and retained artifacts are versioned or fingerprint-bound                                             |

### Task
The task defines the **physical behavior** that every compared method must **produce**. 
It is intentionally narrower than "understand a door and open it in any possible way":

> Starting from a valid initial state, **Alex's right-hand contact point must approach the door panel, establish contact, and push it until the revolute hinge reaches 45° (`pi/4`)**.

This is a **door-pushing task**, not a complete door-opening task. 
*The robot does not identify or operate a handle, form a grasp, pull the panel, decide which side to use, or choose among multiple opening strategies.* 
Those capabilities would introduce additional perception, planning, and manipulation problems that are outside the controlled question of this first benchmark.

The task is **complete** at the first **crossing** of the **45-degree hinge-angle threshold.** 
This gives every method the *same unambiguous physical objective*: produce enough controlled contact motion to rotate the same articulated object through the same angle.

#### Why Use a Door?
A door is a useful first benchmark object because:
- it has been requested by **IHMC**;
- the task requires **physical contact**;
- it is articulated around a clearly defined **hinge**;
	- the contact point and its distance from the hinge affect the resulting torque;
- progress can be measured directly through an **angle**;
- **force** in the wrong direction can fail or produce an impact;
- it is complex enough to be **scientifically interesting**, but controlled enough to support **repeatable comparisons**.

The door is therefore an experimental instrument: it makes contact, object-relative geometry, force, and articulated motion observable within one repeatable task. 
**It is not the final objective or intended limit of the project.**

#### D0-D4 Task Variations
The benchmark uses five poses of the same door. 
These are poses of the **door relative to the robot**, not different starting poses of the robot. Alex's fixed base remains in the same calibrated position, while the entire door assembly, including the frame and hinge, is rotated and translated.

Each door pose is defined by three values: 
- `door_yaw_deg`, the rotation of the complete door assembly about the hinge's vertical axis; 
- `door_offset_x_m`, its translation along the simulation's world X axis; 
- `door_offset_y_m`, its translation along the simulation's world Y axis. 

**N.B.** The `door_yaw` value is not the door-opening angle. It changes the initial orientation of the complete door and frame around the hinge location; 
the panel's opening motion is measured separately through the revolute hinge joint during task execution.

**N.B.** The translations are expressed in the world frame, not in the rotated door frame.

| Pose | Door yaw | World X translation | World Y translation |
|---|---:|---:|---:|
| `D0` | `0 degrees` | `0 cm` | `0 cm` |
| `D1` | `+2.8648 degrees` | `+2 cm` | `0 cm` |
| `D2` | `-2.8648 degrees` | `0 cm` | `-2 cm` |
| `D3` | `+5.7296 degrees` | `+2 cm` | `+2 cm` |
| `D4` | `-5.7296 degrees` | `+2 cm` | `-2 cm` |

`D0` is the nominal pose: it uses the door's original transform from the task scene without any additional yaw or XY translation. 
Notice that D1 and D2 have equal and opposite rotations, but their translations are not mirror images: D1 shifts the door along positive world X, whereas D2 shifts it along negative world Y.

**The fixed-versus-randomized episode setting is a separate source of variation.** 
In randomized episodes, the requested initial *right-hand tool point can receive an offset of up to approximately `+/-2 cm` along each axis of the door frame*, and the selected *push point can also vary* within its calibrated bounds. This does not move the robot's fixed base and must not be confused with the D0-D4 door poses.

**These 5 poses test whether a method can handle small, controlled changes in the placement of the same articulated object instead of memorizing one exact door location.** 
*Because the robot, door geometry, mechanics, and task remain unchanged, D0-D4 are variations of one task, not five different tasks.* 
They do **not demonstrate generalization** to different doors, handles, opening directions, geometries, robots, images, or viewpoints.

### Environment
The environment is the world in which the task is executed and the interface through which a method observes that world and affects it. AlexDoor-XAS currently evaluates policies entirely in simulation; the physical Alex003 platform is the reference robot and future transfer target, but it is not part of the present benchmark evidence.

#### Simulated Physical Scene
The scene includes:
- fixed-base Alex;
- a door with one revolute hinge;
- a panel approximately `0.83 m` wide and `0.036 m` thick;
- a panel whose mass is set to `25 kg`;
- a passive hinge with `4 N m s/rad` damping;
- a contact sensor filtered exclusively to the door panel;
- a supporting floor surface.

The controller geometry is defined in [`src/alexdoor_xas/policies/scripted/door_push.py`](../../../src/alexdoor_xas/policies/scripted/door_push.py), while the articulated door and its damping are configured in [`src/alexdoor_xas/envs/door_task/door_push_robot_env_cfg.py`](../../../src/alexdoor_xas/envs/door_task/door_push_robot_env_cfg.py).

#### What Is Isaac Sim?
Isaac Sim is NVIDIA's robotics simulator.

In general, it can:
- import robots from URDF, MJCF, or USD;
- construct 3D scenes;
- simulate rigid bodies, joints, collisions, and contacts;
- use PhysX;
- simulate cameras, depth sensors, lidar, IMUs, and physical sensors;
- integrate with ROS 2;
- generate synthetic data;
- execute and verify robot controllers.

The official description is available in the [Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/index.html).

In simple terms:
> Isaac Sim is the virtual world and the physics engine.

#### What Is Isaac Lab?
Isaac Lab is a robot-learning framework built on top of Isaac Sim.

It provides structures for:
- robots and articulations;
- sensors;
- controllers;
- environments;
- observations and actions;
- resets and terminations;
- randomization;
- reinforcement learning;
- imitation learning;
- motion planning;
- vectorized simulation.

It supports two main development styles:
- **Manager-based:** each part of the task is separated into configurable managers;
- **Direct:** the environment class directly implements observations, actions, resets, and terminations.

The distinction is explained in the [task workflow documentation](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html). The general overview is available in the [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/index.html).

In simple terms:
> Isaac Lab organizes Isaac Sim as an experimental laboratory for robot learning and policies.

#### How We Use Them
In this project, Isaac Sim manages:
- the scene and USD files;
- the robot imported from URDF;
- the door's rigid body;
- the hinge;
- collisions;
- contact force;
- time advancement;
- joints and dynamics.

Isaac Lab manages:
- `DirectRLEnv`;
- scene configuration;
- the Alex articulation;
- the door articulation;
- `DifferentialIKController`;
- `ContactSensor`;
- resets;
- observations;
- action application;
- simulation stepping.

The flow is:
```text
Alex and door state
        |
        v
      Policy
        |
        v
Current A2 or A3 action
        |
        v
Adapter: validate, limit, and convert
        |
        v
Differential IK
        |
        v
Six right-arm joints
        |
        v
Door contact and dynamics
        |
        +--------------------> new Alex and door state
```

An important detail is that the environment class derives from `DirectRLEnv`, but the current benchmark does not use reinforcement learning. The environment reward is zero. ACT and Diffusion policies are trained offline through imitation learning and then evaluated closed-loop.

In addition:
- Isaac runs on the workstation;
- the official physics simulation uses the CPU;
- training and inference can use a GPU;
- training on the Gilbreth cluster uses PyTorch but does not execute Isaac;
- closed-loop evaluation returns to the workstation and Isaac.

#### Observation Interface
The current policies do not receive images. The primary `core_door_pose` observation preset contains 14 values:
- tool-point position: 3;
- tool-point orientation: 4;
- door angle and angular velocity: 2;
- relative door-frame position: 3;
- sine and cosine of the door yaw: 2.

The policy therefore receives a directly structured geometric state. It does not yet have to perceive or understand the door from a camera image. This is why the current benchmark is classified as **state-based**, not vision-based or multimodal.

#### Physical Reference Platform: Alex003
The [Alex003 Usage Guide](https://docs.google.com/document/d/17QtexPK_RqmfRammA7CsvEUuhFZFkicrJfONCdlWkEg/edit?tab=t.sfoy44dcf7kc) describes Alex003, or AX003, as the manipulator version of the Alex humanoid platform intended for research at Purdue University.

**It is a research prototype, not an industrial product.**

##### General Structure
Alex003 is:
- fixed-base;
- mounted to a table or pedestal;
- powered and connected through tethers;
- composed of two arms;
- equipped with a central torso;
- equipped with a pan-and-tilt head;
- equipped with two grippers;
- built around `16` primary degrees of freedom.

The 16 degrees of freedom are:
- 2 in the head;
- 7 in the left arm;
- 7 in the right arm.

The physical torso primarily contains computing and power electronics. It does not add a torso degree of freedom in the Alex003 configuration described by the manual.

##### Joints in Each Arm
Each arm has:
1. shoulder pitch;
2. shoulder roll;
3. shoulder yaw;
4. elbow pitch;
5. forearm yaw;
6. wrist roll;
7. wrist pitch.

The software names for the right-arm joints are:
- `RIGHT_SHOULDER_Y`;
- `RIGHT_SHOULDER_X`;
- `RIGHT_SHOULDER_Z`;
- `RIGHT_ELBOW_Y`;
- `RIGHT_WRIST_Z`;
- `RIGHT_WRIST_X`;
- `RIGHT_GRIPPER_Y`.

##### Main Ranges of Motion

| Joint | Right | Left |
|---|---:|---:|
| Shoulder pitch | -180 to +70 degrees | -180 to +70 degrees |
| Shoulder roll | -165 to +15 degrees | -15 to +165 degrees |
| Shoulder yaw | -110 to +70 degrees | -70 to +110 degrees |
| Elbow pitch | -135 to +10 degrees | -135 to +10 degrees |
| Forearm yaw | -150 to +150 degrees | -150 to +150 degrees |
| Wrist roll | -55 to +85 degrees | -85 to +55 degrees |
| Wrist pitch | -45 to +45 degrees | -45 to +45 degrees |

The head has:
- neck pitch: -28 to +28 degrees;
- neck yaw: -70 to +70 degrees.

##### Actuators
Alex uses 16 modular cycloidal actuators with:
- BLDC motors;
- a 19:1 cycloidal transmission;
- 17-bit input and output encoders;
- four actuator sizes.

| Actuator | Where it is used | Continuous torque | Peak torque |
|---|---|---:|---:|
| 85x26 | Shoulder pitch J1 | 55.09 N m | 178.60 N m |
| 76x26 | Shoulder roll J2 and elbow J4 | 30.4 N m | 107.54 N m |
| 68x26 | Shoulder yaw J3 | 22.61 N m | 78.14 N m |
| 60x08 | J5-J7 and the head | 6.64 N m | 23.18 N m |

These are hardware specifications, not the limits used by our simulated benchmark.

##### Modularity
The robot is designed to simplify maintenance and component replacement:
- the torso and shoulder can be separated;
- the shoulder and bicep can be separated;
- the elbow and forearm can be separated;
- the wrist and gripper can be replaced;
- power and communication harnesses can be disconnected;
- electronics are distributed near the joints.

##### Grippers
The Purdue platform is supplied with SAKE Robotics EZGrippers.

They are separate modules from the main chain of 16 cycloidal actuators. The manual warns that they can overheat if they are maintained at high torque for too long.

The current benchmark:
- does not perform a grasp;
- does not actuate the gripper;
- does not operate a door handle;
- uses contact geometry on the distal part of the arm to push the panel.

##### Computing and Electronics
Alex003 contains two main computers:
- **NUC:** EtherCAT master and real-time low-level controller;
- **NVIDIA Jetson AGX Orin 64 GB:** networking, logging, sensor integration, perception, and AI processes.

Motor control is distributed through:
- EtherCAT;
- Elmo Platinum Twitter servo drives;
- EtherSNACKS modules;
- local IMUs;
- power and communication harnesses.

The torso distributes:
- `48 V` for the motors;
- `24 V` for the logic electronics.

The platform is supplied with a `48 V / 32 A` power supply and a wired emergency stop.

##### Sensors
The robot provides:
- joint position, velocity, and torque;
- actuator temperatures;
- distributed IMUs;
- device state and fault information;
- bus current and voltage;
- provisions for cameras.

The manual indicates provisions for:
- a ZED X Mini in the head;
- a RealSense D457 or ZED X Mini in the torso.
However, these cameras and their mounting hardware were not included in the delivery described by the manual.

##### Physical-Robot Software
The real robot has two control levels:
1. **Low-level controller:** runs on the NUC and manages EtherCAT, actuators, faults, power-up, and power-down.
2. **High-level controller:** receives state and sends commands through ROS 2.

Communication uses:
- ROS 2;
- Fast DDS;
- Domain ID `42`;
- `rt/alex_state`;
- `rt/alex_command`;
- `rt/hardware_status`.

For each joint, the interface can specify:
- desired position;
- desired velocity;
- feed-forward torque;
- stiffness;
- damping;
- error and torque limits.

The low-level controller combines these values through a PD or impedance law and produces the torque sent to the actuator.

SCS2 is primarily used for:
- visualization;
- logging;
- plotting;
- diagnostics;
- troubleshooting;
- joint zeroing.

The manual specifies that general joint control must use the ROS 2 API rather than SCS2.

##### Safety
The manual is explicit: Alex can be dangerous.

Important risks include:
- high actuator torque;
- pinch points;
- arms falling when actuators are disabled;
- hot components;
- possible electrical or software faults;
- lack of waterproofing and dust protection;
- the prototype nature of the platform.

Recommended precautions include maintaining a safe operating distance, using physical guards, keeping an emergency stop available, using personal protective equipment, and following supervised procedures.

Isaac Sim results do not automatically authorize physical-robot operation.

#### The Robot Used in the Benchmark Is Not All of Alex003
This is the most important distinction.

##### Physical Alex003
The manual describes:
- Purdue's manipulator platform;
- a 16-DoF upper body;
- two 7-DoF arms;
- a 2-DoF head;
- physical grippers;
- computers, EtherCAT, and ROS 2;
- real hardware and safety procedures.

##### Repository Simulation Asset
The repository instead uses a local `alex_v2.urdf` file governed by a strict, fingerprinted contract.

This asset:
- is identified as a standard full-body asset;
- contains 29 movable joints in the model;
- includes leg, ankle, spine, and other joint names;
- is converted to fixed-base operation;
- does not use external hands;
- keeps self-collision enabled.

The contract is verified in [`src/alexdoor_xas/assets/alex_v2_manifest.py`](../../../src/alexdoor_xas/assets/alex_v2_manifest.py) and [`src/alexdoor_xas/assets/alex_v2_contract.py`](../../../src/alexdoor_xas/assets/alex_v2_contract.py).

The number 29 therefore describes the URDF model used in simulation. It does not mean that the physical Alex003 platform has 29 operational degrees of freedom.

##### Only Six Joints Are Commanded
The benchmark controls:
- `RIGHT_SHOULDER_Y`;
- `RIGHT_SHOULDER_X`;
- `RIGHT_SHOULDER_Z`;
- `RIGHT_ELBOW_Y`;
- `RIGHT_WRIST_Z`;
- `RIGHT_WRIST_X`.

The seventh distal arm degree of freedom and the gripper are not commanded.

Therefore:
> The benchmark studies fixed-base manipulation through a six-joint right-arm kinematic chain, not through the complete humanoid robot.

The configuration is defined in [`src/alexdoor_xas/envs/door_task/door_push_alex_v2_env_cfg.py`](../../../src/alexdoor_xas/envs/door_task/door_push_alex_v2_env_cfg.py).

##### Position-Only Differential IK
The policy does not directly produce motor targets.

It produces a Cartesian end-effector motion. The `DifferentialIKController` computes the changes in the six joints needed to produce that motion.

The controller is position-only:
- the three translation components are executed;
- the three rotation components are validated, limited, and recorded;
- the rotation components are not actuated.

This is an important scientific limitation: the benchmark cannot yet fully evaluate complete 6D action representations.

##### Tool Point
The controlled point is not simply the origin of the gripper link.

It is derived from the collision geometry of `RIGHT_GRIPPER_Z_LINK`, using the `right_thumb_collision` support shape.

This matters because the point that physically touches the door is offset from the mathematical origin of the link. The Jacobian is also transformed to this contact point.

This behavior is implemented in [`src/alexdoor_xas/envs/door_task/door_push_alex_v2_executor.py`](../../../src/alexdoor_xas/envs/door_task/door_push_alex_v2_executor.py).

##### Timing and Limits
- physics timestep: `1/120 s`;
- control timestep: `1/60 s`;
- maximum Cartesian displacement: `0.02 m` per tick;
- maximum recorded rotational command: `0.05 rad` per tick;
- calibrated workspace: shoulder-to-tool distance between `0.2` and `0.8 m`;
- joint targets are clamped to their limits;
- shoulder and elbow PD gains: `600/15`;
- wrist PD gains: `150/4`.

Force is accepted as task evidence only when it comes from contact between the tool and the exact door-panel collider. Collisions with the frame, floor, or other bodies are not counted as door-pushing force.

### Experimental Control
Experimental control is what makes AlexDoor-XAS a comparison rather than a collection of unrelated robot runs. Its purpose is to isolate the effect of the action representation as far as the current benchmark allows.

**What remains fixed.** Within a matched comparison, the benchmark keeps the following elements constant:
- the robot asset and validated calibration;
- the door geometry and physical properties;
- the door-pushing task and its 45-degree completion criterion;
- the source demonstration trajectories;
- matched episode identities and training, validation, and test splits;
- the D0-D4 pose and evaluation-seed schedule;
- the downstream robot controller;
- the versioned adapter and safety contracts applicable to each representation;
- the evaluation protocol.

**What changes.** The primary variable is **how the action is represented**: the same intended physical behavior can be expressed in a more robot-specific, Cartesian, object-relative, or object-centric form.

Policy family and dataset scale can also be studied as secondary experimental axes. When the benchmark compares action representations, those secondary conditions must be matched so that the result is not explained by a different model, different data, or different evaluation trials.

The hypothesis is that a representation that is less dependent on one robot may support better generalization, interpretability, safety, and transfer. However, this remains a hypothesis to test. The current benchmark does not yet demonstrate broad cross-embodiment or real-robot transfer.

### Protocol, Metrics, Baselines, and Provenance
This document defines the benchmark's purpose, task, environment, and experimental control. The remaining benchmark components are introduced here only at a high level because their exact contracts depend on later parts of the project:
- **Datasets** will explain episode generation, action-space exports, matched identities and splits, normalization, and dataset provenance.
- **Models** will explain the compared policy families, their inputs and outputs, and which methods serve as learned baselines.
- **Training** will explain optimization procedures, configurations, checkpoints, compute environments, and training reproducibility.
- **Evaluation** will define the complete rollout protocol, metrics, aggregation, failure analysis, baseline comparisons, and result provenance.

Together, those later sections complete the operational definition needed to run and interpret the benchmark without duplicating their details here.

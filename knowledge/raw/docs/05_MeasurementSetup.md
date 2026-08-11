# Measurement Setup
## Who Controls Time?
AlexDoor-XAS uses three different temporal concepts: the **physics timestep**, the **control timestep**, and the **episode timing**. They are related, but they describe different responsibilities and must not be treated as one clock.

The common responsibility chain is:
- **Our project code chooses the values and experimental rules.**
- **Isaac Lab schedules the environment lifecycle according to those values.**
- **Isaac Sim executes each requested simulation step and maintains simulated time.**
- **PhysX calculates the physical evolution within each physics step.**

### 1. Physics Timestep
The **physics timestep** is the amount of simulated time advanced by one physics integration step.

For AlexDoor-XAS:
```text
physics_dt = 1/120 s
physics frequency = 120 Hz
```

The project configuration declares this value through `SimulationCfg` in [`src/alexdoor_xas/envs/door_task/door_push_robot_env_cfg.py`](../../../src/alexdoor_xas/envs/door_task/door_push_robot_env_cfg.py). Isaac Lab passes and uses that configuration when scheduling the environment. Isaac Sim performs the requested simulation step, while PhysX numerically integrates the bodies, articulations, joints, drives, collisions, and contacts over that `1/120 s` interval.

Therefore:
- project code **selects** the physics timestep;
- Isaac Lab **schedules** calls using that timestep;
- Isaac Sim **advances** simulated time;
- PhysX **calculates** what physically happens during the step.

### 2. Control Timestep
The **control timestep** is the interval between two new actions from the controller or policy. It does not have to equal the physics timestep because stable contact simulation may require physics to run faster than control.

AlexDoor-XAS uses:
```text
decimation = 2
control_dt = physics_dt * decimation
           = (1/120 s) * 2
           = 1/60 s
control frequency = 60 Hz
```

For every control tick, the controller or policy produces one new action. Isaac Lab processes that action and requests two physics steps before asking for the next action. During those two steps, the environment continues applying the resulting joint targets while Isaac Sim and PhysX advance the physical state.

The sequence is:
```text
one new action
    -> physics step 1 at 1/120 s
    -> physics step 2 at 1/120 s
    -> new observation and next action
```

This behavior is configured by `decimation = 2` and implemented through the Isaac Lab environment lifecycle used by [`src/alexdoor_xas/envs/door_task/door_push_robot_env.py`](../../../src/alexdoor_xas/envs/door_task/door_push_robot_env.py).

Therefore:
- project code **selects** the control frequency and produces the action;
- Isaac Lab **holds and schedules** that action across the configured number of physics steps;
- Isaac Sim **executes** those steps;
- PhysX **computes** the physical response during each step.

### 3. Episode Timing
**Episode timing** describes the logical duration and lifecycle of one benchmark trial. An episode has a beginning, a sequence of control ticks, and a termination reason. This is an experimental concept, not a concept understood by PhysX.

For AlexDoor-XAS, project code defines rules such as:
- the maximum episode duration of `10 s`;
- the reset state of the robot and door;
- success at the first door-hinge crossing of `pi/4`;
- timeout, invalid-state, controller, and other fail-closed termination conditions.

Isaac Lab maintains environment step counters, invokes reset and termination hooks, and writes the requested reset state into the simulator. Project rollout and evaluation code applies the benchmark-specific first-crossing and failure rules. Isaac Sim updates the runtime state when asked, but it does not decide what counts as an episode, success, or failure. PhysX has no concept of an episode at all; it only receives the current physical state and advances it by another physics timestep.

A reset is therefore not ordinary physical evolution. The environment deliberately writes the approved initial joint positions, velocities, targets, and buffers back into the simulation before the next trial begins.

Therefore:
- project code **defines** when a trial starts, succeeds, fails, or times out;
- Isaac Lab **tracks and executes** the environment lifecycle and reset hooks;
- Isaac Sim **applies and exposes** the resulting simulation state;
- PhysX **resumes physical integration** from that state without knowing that a new episode began.

### Complete Timing Relationship
```text
Episode
    contains many control ticks at 60 Hz
        each control tick contains two physics steps at 120 Hz
            each physics step is executed by Isaac Sim and integrated by PhysX
```

In one sentence:
> Project code defines the timing contract, Isaac Lab schedules the experiment, Isaac Sim advances simulated time, and PhysX calculates the physical evolution of each step.

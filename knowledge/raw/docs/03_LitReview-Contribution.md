XAS is not a single scientific field
SO the problem we're tackling is distributed/at the intersection among 3 areas:
- **Action Space Design:** what rep makes easier the "learning"? A single policy could use at the same time more representations?
- **Object-centric + Latent Action learning:** can we replace robots command w/ intention/effect on the obj / latent rep shared?
- **Cross-embodiment learning:** how to share data-knowledge across robots w/ diff commands

**Novelty briefly:** constant physical interaction, but diff language to describe it
- More controlled scientific question 
# Action Space Design
These works directly study how joint space, Cartesian space, delta actions, and hybrid representations affect learning and execution.
## 3.1 [Implicit Kinematic Policies: Unifying Joint and Cartesian Action Spaces in End-to-End Robot Learning](https://arxiv.org/abs/2203.01983)
#### **Problem addressed**
In robot learning, it is usually necessary to choose a single representation:
- joint actions, which directly control the robot's configuration;
- Cartesian actions, which describe end-effector motion.

This choice can drastically change the outcome:
- Cartesian space simplifies geometric tasks such as reaching and sweeping;
- joint space better represents behaviors that depend on the arm's entire posture;
- neither is always better.

The paper therefore asks:
> Can a policy use joint and Cartesian action representations simultaneously,
> instead of having to choose only one?

#### **Central idea and methodology**
The authors propose `Implicit Kinematic Policies`, or IKP.

The same action is presented to the policy in two kinematically consistent representations: a_{Cartesian} = FK(a_{joint})
- where `FK` is the robot's forward kinematics.

The policy therefore receives all of the following at the same time:
- joint configuration;
- Cartesian end-effector pose;
- Cartesian poses of intermediate links;
- an image of the scene.

The model is an implicit policy based on an **energy-based model.** 
Instead of directly producing an action, it assigns a cost to possible actions and **searches for the one with minimum energy.**

This formulation allows the model to exploit:
- the geometric simplicity of Cartesian space;
- the complete configuration information available in joint space;
- the exact kinematic relationship between the two representations.

The authors test the system on **tasks** such as:
- bimanual sweeping;
- whole-arm box flipping;
- block insertion;
- manipulation with small errors in the joint encoders.

#### **Results**
The results show that the **best representation depends on the task.**

For bimanual sweeping:
- Cartesian-only: 79.4%;
- joint-only: 44.3%;
- joint + Cartesian IKP: 85.9%.

For box flipping:
- Cartesian-only: 38.6%;
- joint-only: 98.4%;
- joint + Cartesian IKP: 97.5%.

*IKP therefore achieves performance close to or better than the best action space manually selected for each task.*

The model can also **compensate for small systematic errors** in joint encoders **through residual layers** inserted into the kinematic chain.

#### **Connection to AlexDoor-XAS**
**Similarities:**
- *joint space conceptually corresponds to A1;*
- *Cartesian end-effector space corresponds to A2;*
- *the same action is represented in multiple spaces;*
- the representations are connected through kinematics;
- *the goal is to avoid choosing a single action space in advance.*

**Differences:**
- *IKP mainly considers joint and Cartesian space;*
	- it does not introduce an object-relative frame like A3;
	- it does not introduce object-centric intent chunks like A4;
- it does not use separate, auditable adapters;
- it does not explicitly study force, contact safety, or provenance;
- it does not necessarily allow a specific output to be requested through an action-space tag.

**Lessons for the project:**
1. A1 and A2 may contain complementary information.
2. A shared policy could learn better than four completely separate policies.
3. The representations must remain kinematically consistent.
4. We should not expect one action space to be universally better.
5. IKP is an essential baseline for a future Phase 4 multi-action-space study.

---

## 3.2 [Demystifying Action Space Design for Robotic Manipulation Policies](https://arxiv.org/abs/2602.23408)
Feng et al., 2026.
#### **Problem addressed**
Many robot-learning systems **choose their action space** based on:
- conventions inherited from the framework;
- intuition;
- implementations that are already available;
- isolated results from prior work.

There is no systematic evaluation of **how performance changes when varying:**
- joint versus task space;
- absolute versus delta actions;
- single actions versus action chunks.

The paper therefore seeks to understand:
> Which properties of an action representation determine
> learnability, control stability, and generalization?

#### **Central idea and methodology**
The authors organize the **action space along two axes.**

*Spatial abstraction:*
- joint space;
- task space, meaning end-effector Cartesian space.

*Temporal representation:*
- absolute action;
- step-wise delta;
- chunk-wise delta;
- action chunking.

They conduct a **large-scale study** with:
- more than 500 models;
- more than 13,000 real-world rollouts;
- more than 2,000 demonstrations;
- precision, coordination, and bimanual manipulation tasks;
- variations in data quantity, training duration, and model capacity.

The **goal** is not simply to propose a new policy, but to **isolate the effect of the action representation.**

#### **Results**
The main results are:
- delta actions generally outperform absolute actions when implemented correctly;
- joint space and task space have complementary advantages;
- joint space favors control stability and precision under standard conditions;
- task space favors generalization and cross-embodiment transfer;
- action chunking interacts with how delta actions are defined;
	- it is not enough to say "we use action chunks": **it is necessary to specify the state relative to which each delta is computed.**

The paper therefore confirms that **the representation changes the optimization landscape**, not merely the output format.

#### **Connection to AlexDoor-XAS**
**Similarities:**
- *A1 is joint space;*
- *A2 and A3 are task-space actions;*
- *A1-A3 are delta representations;*
- ACT and Diffusion Policy produce action chunks;
- the project compares representations under controlled conditions.

**Differences:**
- the paper mainly compares joint versus task space and absolute versus delta;
- *it does not separate world-frame and object-relative task space;*
- *it does not include a symbolic representation like A4;*
- it does not study an A4→A3→A2 hierarchy;
- it does not treat articulated-object contact as the central variable.

**Lessons for the project:**
1. The decision to use delta actions is well motivated.
2. We must define unambiguously the reference from which every delta is computed.
3. A1 may favor stability, while A2/A3 may favor generalization.
4. The comparison must separate spatial abstraction from temporal abstraction.
5. If A4 uses longer chunks than A1-A3, we must avoid confusing the benefit of the object-centric representation 
	with the benefit of greater temporal abstraction.

---

## 3.3 [On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer](https://arxiv.org/abs/2312.03673)
Aljalbout et al., 2023/2024.
#### **Problem addressed**
A policy can have a high success rate in simulation and fail on the physical robot.

Part of the **sim-to-real gap** **depends** not only on **images** or **dynamics**, but **also** **on** the **control interface** used by the policy.

The paper asks:
> How do exploration, tracking error, constraint violations, and sim-to-real transfer change 
> when different action spaces are used?

#### **Central idea and methodology**
The authors train more than **250 RL agents on reaching and pushing.**

They compare **13 control spaces** constructed by combining:
- joint and Cartesian coordinates;
- position, velocity, and torque;
- absolute actions;
- one-step delta actions;
- multi-step integrated delta actions.

For every configuration, they study:
- sample efficiency;
- reward and success rate;
- tracking error;
- velocity, acceleration, and jerk violations;
- differences between simulated and real trajectories;
- success rate on the physical robot.

For pushing, they also randomize object mass and friction.

#### **Results**
The main results are:
- Cartesian spaces facilitate exploration in geometric tasks (such as pushing);
- velocity-based spaces generally transfer better than position-based spaces;
- delta spaces often transfer better than their corresponding base spaces;
- torque control is highly sensitive to the dynamics gap;
- multi-step integration can accumulate tracking error;
- one-step deltas are generally more robust to hyperparameters;
- *simulation success rate does not directly predict real-world success rate;*
- joint velocity is, overall, one of the most robust spaces among those studied.

The most important finding is that the **action space implicitly includes:**
- the feedback loop;
- the low-level controller;
- tracking dynamics;
- safety limiters.
*It is not merely a numerical vector.*

#### **Connection to AlexDoor-XAS**
**Similarities:**
- *A1-A3 are delta representations;*
- *A2/A3 are converted into joint targets through differential IK;*
- the project measures force, warnings, and adapter corrections;
- *the task includes pushing and contact;*
- the adapter and controller are part of the experiment.

**Differences:**
- the paper *uses RL*, while Phase 3 uses behavior cloning with ACT and Diffusion Policy;
- its *tasks are simpler* than articulated-door manipulation;
- *it does not compare world-frame and object-relative coordinates;*
- *it does not include object-centric intent chunks;*
- it does not use a fixed-base humanoid torso.

**Lessons for the project:**
1. We cannot select an action space using only simulation success rate.
	- We must measure tracking error, force, smoothness, corrections, and constraint violations.
2. *The adapter must be included in the scientific definition of the action space.*
3. *Two geometrically equivalent representations can produce different behaviors because of the controller.*
4. *Any future sim-to-real conclusion will have to be verified on the robot: simulation is not sufficient.*

---

## 3.4 [Redundancy-aware Action Spaces for Robot Learning](https://arxiv.org/abs/2406.04144)
Mazzaglia et al., 2024.
#### **Problem addressed**
For a redundant arm, **the same end-effector pose can be reached with many joint configurations.**

**Task-space control specifies** where the **hand** should go, but it does **not** necessarily **control**:
- elbow position;
- the arm's internal configuration;
- collision avoidance;
- the use of intermediate links;
- posture in confined spaces.

**Joint space controls the entire arm, but makes learning more difficult.**

The problem is therefore:
> How can we preserve the simplicity of task space 
> without losing control over joint redundancy?

**Joint Redundancy:** when a robot has more joints DOF than the ones that it strictly needs to execute a task.

#### **Central idea and methodology**
The authors introduce `End-effector Redundancy`, or ER.

The action space combines:
- an end-effector target;
- additional information about the redundant configuration.

They propose two implementations:
- `ERAngle`;
- `ERJoint`.

The **goal** is to *maintain a geometrically meaningful interface* 
while allowing the policy to *control parts of the configuration* that the end-effector pose does not determine.

The system is evaluated with:
- reinforcement learning;
- imitation learning;
- simulated tasks;
- real-world tasks;
- manipulation in confined spaces;
- behaviors that require using the elbow or other links.

#### **Results**
`ERJoint` achieves the **best results in tasks where the complete configuration is important.**

**Traditional task space remains effective for simple manipulations** focused on the end effector, but it fails when:
- the elbow must avoid an obstacle;
- the robot must enter a confined space;
- an intermediate link must contribute to the interaction;
- IK chooses an unsuitable configuration.

#### **Connection to AlexDoor-XAS**
**Similarities:**
- *A1 preserves complete joint information;*
- *A2/A3 simplify the task using end-effector motion;*
- Alex is redundant at the whole-body level;
- Cartesian execution depends on the IK solution.

**Differences:**
- the current benchmark moves only six joints in the right arm;
- Alex is fixed-base;
- whole-body redundancy is not yet used;
- A2/A3 rotation is recorded but not commanded;
- *ER does not introduce object-relative or object-centric semantics.*

**Lessons for the project:**
1. *The benefit of A2/A3 may change when the project moves to whole-body manipulation.*
	- A3 does not solve the elbow-posture problem.
	- A hybrid representation may eventually be needed: A3 plus a posture or null-space target.
2. The current benchmark must not be presented as evidence that applies to every form of humanoid manipulation.
3. A1 remains an important baseline because it contains information that A2/A3 may lose.

---

# 4. Task-Relative, Object-Centric, and Temporally Abstract Actions
These works are particularly relevant to A3 and A4.

## 4.1 [Affordance-Centric Policy Learning: Sample Efficient and Generalisable Robot Policy Learning using Affordance-Centric Task Frames](https://arxiv.org/abs/2410.12124)
Rana et al., 2024.
THIS PAPER IS VERY IMPORTANT
#### **Problem addressed**
**A policy expressed in global coordinates must relearn very similar behaviors when:**
- the object is moved;
- the object's orientation changes;
- the object instance changes;
- the background changes.

**Much of the visual information is irrelevant to the interaction.**

The problem is:
> Can observations and actions be expressed relative to 
> the part of the object that makes the task possible?

#### **Central idea and methodology**
The authors construct an `affordance-centric task frame`.

The frame is:
- **centered on the object region relevant to the interaction;**
- oriented consistently with the affordance;
- *extracted and tracked using large vision models.*

**The policy therefore operates in coordinates relative to the affordance**, instead of in the world frame.

The hypothesis is that this transformation produces:
- spatial invariance;
- intra-category invariance;
- greater sample efficiency.

#### **Results**
The authors show that an affordance-centric policy trained with 10 demonstrations
can achieve **generalization** comparable to an image-based policy trained with 305 demonstrations.

The representation **improves robustness to:**
- different object positions;
- different instances of the same category;
- irrelevant visual variations.

#### **Connection to AlexDoor-XAS**
**Similarities:**
- *A3 uses a task frame tied to the door;*
- the frame is *anchored to the hinge;*
- the goal is to remove irrelevant global variation;
- *A4 uses a contact target relative to the panel.*

**Differences:**
- the paper detects the *affordance through visual models;*
- AlexDoor uses *door geometry that is already known;*
	- A3 uses a static frame at the hinge;
	- A4 uses a moving frame attached to the panel;
- *the paper does not compare matched joint, world-frame, and object-centric actions.*

**Lessons for the project:**
1. A3 must be evaluated under genuine variations in door position and orientation.
2. Generalization must include new geometries, not only new world poses.
3. The frame must be estimated realistically, with noise.
4. A result obtained using perfect simulator state does not demonstrate visual object-relative generalization.
5. The hinge frame is a well-motivated inductive bias, but it must be subjected to ablation.

---

## 4.2 [Automatic Derivation of an Optimal Task Frame for Learning and Controlling Contact-Rich Tasks](https://arxiv.org/abs/2404.01900)
Mousavi Mohammadi et al., 2024.
#### **Problem addressed**
In contact-rich tasks, it is **necessary to choose a frame in which to express:**
- motion;
- force;
- wrench;
- task constraints.
This choice is **often made manually by an expert.**

But, an incorrect frame can mix components that should be independent and make control more difficult.

#### **Central idea and methodology**
The authors seek to **derive the task frame automatically** from demonstrations that include motion and wrench.

They use **screw theory** to generate candidates for:
- the frame origin;
- the frame orientation.

They then **select the frame** that **maximizes the decoupling of motion and force components.**
SO some directions describe only the motion, while others only the force; Correlations are minimal.

The method can determine whether the origin and orientation should be fixed relative to:
- the world;
- the robot tool;
- combinations of the two.

The resulting frame is used by a constraint-based controller.

#### **Results**
The method is validated on several contact-rich tasks, including articulated object manipulation.

*The automatically derived task frames are consistent with those selected by experts*
and enable execution of the studied behaviors.

#### **Connection to AlexDoor-XAS**
**Similarities:**
- A3 depends entirely on the choice of the door task frame;
- the task is contact-rich;
- the door is an articulated object;
- motion and force should be interpreted relative to the hinge and the panel.

**Differences:**
- *AlexDoor selects the frame manually;*
- A3 represents the action, but not the wrench;
- the A3 frame is static at the hinge;

**Lessons for the project:**
1. The door frame must not be treated as an obvious or neutral choice.
2. We need an ablation among world-, hinge-, and panel-frame representations.
3. Force and motion may require different frames.
4. *A future extension could learn or automatically validate the task frame.*
5. Frame errors must be included in robustness tests.

---

## 4.3 [Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots](https://arxiv.org/abs/2402.10329)
Chi et al., 2024.
#### **Problem addressed**
**Traditional robot-demonstration collection is expensive and hardware-specific.**

**Human videos** are easier to collect, but they introduce a large **embodiment gap**:
- hands differ from grippers;
- cameras differ;
- global coordinates are not shared;
- timing and latency differ;
- robot joint actions are absent.

#### **Central idea and methodology**
**UMI uses a portable gripper operated directly by a person.**

The gripper records:
- egocentric images;
- poses;
- gripper opening;
- relative trajectories.

**The policy uses a `relative-trajectory action representation` instead of absolute global targets.**

The system also introduces latency matching among:
- observations;
- policy inference;
- execution.

The policy is implemented using **Diffusion Policy** to model multimodal action distributions.

#### **Results**
UMI makes it possible to learn real-world tasks that are:
- dynamic;
- bimanual;
- precise;
- long-horizon.

The policies generalize to new objects and new environments when the training data contains sufficient diversity.

The authors also show that the relative action representation and latency matching are important components for stability.

#### **Connection to AlexDoor-XAS**
**Similarities:**
- A3 removes part of the dependence on global coordinates;
- A4 describes behavior at a less hardware-specific level;
- the goal is to produce transferable action representations;
- *Diffusion Policy* is present in both systems.

**Differences:**
- *UMI is primarily a data-collection system;*
- it uses a handheld gripper, not matched simulator exports;
- it does not compare A1-A4;
- it does not use an equivalent safety-adapter boundary;
- it does not specifically study articulated-object manipulation.

**Lessons for the project:**
1. Relative trajectories are a good interface for transfer.
2. Control frequency and latency are part of the operational representation.
3. Demonstrations must contain enough information to reconstruct task geometry.
4. *Generalization also comes from data diversity, not only from the action space.*
5. A3/A4 may be better interfaces for incorporating human or Aria data in the future.

---

## 4.4 [UMI-on-Legs: Making Manipulation Policies Mobile with Manipulation-Centric Whole-body Controllers](https://proceedings.mlr.press/v270/ha25a.html)
Ha et al., 2025.
#### **Problem addressed**
A manipulation policy trained for a fixed-base arm cannot directly command a mobile quadruped.
The quadruped must coordinate:
- locomotion;
- balance;
- body posture;
- the end-effector trajectory.

The problem is to **separate the meaning of the task from the robot-specific mechanics.**

#### **Central idea and methodology**
The system uses two policies.

The manipulation policy:
- is trained with UMI data;
- predicts end-effector trajectories in the task frame.

The whole-body controller:
- is trained in simulation;
- receives the task-frame trajectories;
- converts them into coordinated quadruped motion.

The end-effector trajectory therefore becomes the common interface between:
- task-level manipulation;
- robot-specific execution.

#### **Results**
The system achieves more than 70% success on the studied tasks, which include:
- prehensile manipulation;
- non-prehensile manipulation;
- dynamic manipulation.

The authors also demonstrate zero-shot cross-embodiment deployment on the quadruped
using a manipulation-policy checkpoint originally intended for a fixed-base arm.

#### **Connection to AlexDoor-XAS**
**Similarities:**
- policy and execution are separated;
- *A3 can be interpreted as a task-frame trajectory;*
- *the adapter converts the abstract command into robot-specific motion;*
- *the goal is to make the policy less dependent on the embodiment.*

**Differences:**
- UMI-on-Legs separates the manipulation policy and whole-body controller;
- AlexDoor uses an A4→A3→A2→differential-IK chain;
- our benchmark is fixed-base;
- we have not yet demonstrated transfer to a second embodiment.

**Lessons for the project:**
1. A3 is a candidate interface between a task policy and a whole-body controller.
2. True transfer requires at least a second embodiment or a substantially different robot configuration.
3. Adapter performance and policy performance must be evaluated separately.
4. The whole-body controller may change without necessarily retraining the task policy.
5. A3/A4 have greater value if they become genuinely robot-independent.

---

## 4.5 [HACMan: Learning Hybrid Actor-Critic Maps for 6D Non-Prehensile Manipulation](https://proceedings.mlr.press/v229/zhou23a.html)
Zhou et al., 2023.
#### **Problem addressed**
In non-prehensile manipulation, choosing an end-effector motion is not enough.

The system must decide:
- where to make contact;
- in which direction to push;
- how to move after contact;
- how to adapt the interaction to the object's geometry.

A simple Cartesian delta does not explicitly represent these decisions.
#### **Central idea and methodology**
HACMan introduces an action representation that is:
- object-centric;
- temporally abstract;
- spatially grounded;
- a discrete-continuous hybrid.

The action contains:
1. a contact point selected on the object point cloud;
2. motion parameters describing the motion after contact.

The policy uses off-policy reinforcement learning and produces a map of possible actions associated with points on the object.

The task is to move objects to target 6D poses through non-prehensile interactions.

#### **Results**
In the most difficult variant:
- 89% success on unseen objects in simulation;
- 50% success in zero-shot real-world transfer;
- more than three times the success rate of the best alternative action representation.

The system generalizes to new object categories without a significant reduction in simulation performance.

#### **Connection to AlexDoor-XAS**
**Similarities:**
- A4 contains a panel contact target;
- A4 describes the desired change in the object;
- both representations combine spatial and temporal abstraction;
- both are designed for contact-rich, non-prehensile manipulation.

**Differences:**
- HACMan already learns an object-centric policy;
- A4 does not yet have a learned policy;
- HACMan selects points from a point cloud;
- A4 operates on a door with known geometry and hinge;
- HACMan uses RL;
- A4 is compiled through guarded approach/contact/push stages.

**Lessons for the project:**
1. HACMan is the most important conceptual baseline for A4.
2. A4 must be compared with a contact-point-plus-motion representation.
3. The object-centric action must be evaluated on unseen geometries.
4. We must separate the advantage of the contact target from that of temporal abstraction.
5. A learned A4 policy should produce outputs that the adapter can validate directly.

---

## 4.6 [HYDRA: Hybrid Robot Actions for Imitation Learning](https://proceedings.mlr.press/v229/belkhale23a.html)
Belkhale et al., 2023.
#### **Problem addressed**
Low-level actions allow precise control, but accumulate small errors over time.

High-level actions, such as waypoints or skill primitives:
- reduce the effective length of the problem;
- may be insufficient when precision is required.

The problem is:
> How can temporal abstraction and fine control be combined
> within the same policy?

#### **Central idea and methodology**
HYDRA uses a hybrid action space with two levels:
- sparse high-level waypoints;
- dense low-level actions.
The policy also learns when to switch from one level to the other.

Action relabeling is also applied to make demonstrations more consistent and reduce distribution shift.

The system is evaluated on long-horizon tasks such as:
- making coffee;
- toasting bread;
- organizing dishes.

#### **Results**
HYDRA outperforms prior imitation-learning methods by 30-40% across seven simulated and real-world environments.

The benefit comes from the ability to:
- progress quickly through simple phases;
- use fine control during delicate phases;
- reduce error accumulation without sacrificing dexterity.

#### **Connection to AlexDoor-XAS**
**Similarities:**
- *A4 is a high-level action chunk;*
- *A2/A3 are low-level Cartesian actions;*
- the A4 adapter executes approach/contact/push stages;
- the task requires particular precision during the transition to contact.

**Differences:**
- *HYDRA dynamically switches between high and low levels;*
- AlexDoor primarily treats A2/A3/A4 as representations to compare;
- A4 is not yet produced by a learned policy;
- *our adapter compiles A4 into A3/A2, instead of leaving the selection entirely  to the policy.*

**Lessons for the project:**
1. We should not assume that A4 must completely replace A2/A3.
2. A hybrid policy could use A4 for planning and A3 for contact control.
3. We need a comparison among A4-only, A3-only, and A4+A3 hybrid systems.
4. The transition between levels must be observable and measurable.
5. Action relabeling could improve the consistency of A4 labels.

---

## 4.7 [EDAR: Learning Environment-Dependent Action Representations for Robotic Manipulation](https://arxiv.org/abs/2607.11427)
Xu et al., July 2026. Recent preprint.
#### **Problem addressed**
Many action tokenizers represent only trajectory structure:
- geometry;
- frequencies;
- smoothness;
- temporal patterns.

The same motor command, however, can have different meanings depending on the scene.

For example, closing the gripper while moving to the right can:
- have no effect in free space;
- grasp an object;
- close a drawer if the gripper is on the handle.

The problem is:
> How can an action be represented 
> according to the effect it should produce on the environment?

#### **Central idea and methodology**
EDAR constructs an `Environment-Dependent Action Representation`.

The action representation depends on:
- the executable command;
- the current observation;
- the expected visual consequence.

The model is trained to jointly represent:
- the structure of the control trajectory;
- the scene's future change.
The latent action should therefore distinguish two numerically similar commands when they produce different effects.

The system is evaluated on:
- LIBERO;
- CALVIN;
- Meta-World;
- long-horizon real-world manipulation.

#### **Results**
EDAR improves downstream policy learning compared with trajectory-only action representations.

The improvements are particularly evident in long-horizon tasks, where it is important to distinguish:
- motion without an effect;
- contact;
- grasping;
- object-state changes.

The ablations reported by the authors indicate contributions from:
- environment-conditioned encoding;
- future-consequence prediction;
- dual-target decoding.

#### **Connection to AlexDoor-XAS**
**Similarities:**
- A4 describes the intended effect on the door;
- the meaning of the push depends on contact state;
- the same A2 trajectory can produce different effects depending on its
  position relative to the panel;
- the project records hinge motion, contact, and force.

**Differences:**
- EDAR uses a learned latent representation;
- A4 uses symbolic, interpretable fields;
- EDAR predicts visual consequences;
- AlexDoor uses object state and simulator evidence;
- EDAR does not necessarily use a safety-aware adapter.

**Lessons for the project:**
1. A4 should be defined in terms of the desired effect, not merely as trajectory compression.
2. The model should distinguish free-space motion from contact-effective motion.
3. `intended_hinge_delta` is scientifically important because it makes the effect explicit.
4. It may be useful to add an auxiliary objective that predicts the achieved object change.
5. A4's symbolic representation is more interpretable, but may be less expressive than a latent action space.

---

# 5. Cross-Embodiment Learning and Unification of Heterogeneous Action Spaces
These works seek to train shared backbones / policies using robots with different output interfaces.

## 5.1 [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)
Open X-Embodiment Collaboration, 2023/2024.
#### **Problem addressed**
Every laboratory collects data with:
- different robots;
- different cameras;
- different tasks;
- different action spaces;
- different frequencies and controllers.

Individual datasets are too small to train generalist robot policies.

The problem is:
> Is it possible to aggregate data from many robots 
> and obtain positive transfer?

#### **Central idea and methodology**
The Open X-Embodiment project aggregates:
- more than one million trajectories;
- 22 robot embodiments;
- 60 datasets;
- 21 institutions.

The data is converted into a common RLDS format.

To train RT-X, actions are approximately normalized into a seven-dimensional end-effector vector:
- XYZ position;
- roll, pitch, yaw;
- gripper.

Two model families are trained:
- `RT-1-X`;
- `RT-2-X`.

#### **Results**
The models trained on the multi-robot mixture show positive transfer and, in
several settings, outperform policies trained only on the target robot's data.

The work demonstrates that data from other robots can improve:
- task performance;
- generalization;
- acquisition of new capabilities.

The alignment of the action spaces remains coarse, however:

- some values are absolute, while others are relative;
- some represent position, while others represent velocity;
- coordinate frames are not fully standardized;
- the same vector can produce different motions on different robots.

#### **Connection to AlexDoor-XAS**
**Similarities:**
- both aim to share knowledge across different representations;
- both require explicit action-space metadata;
- both depend on standardized datasets and provenance.

**Differences:**
- RT-X aggregates different episodes collected by different robots; AlexDoor starts from the same source episodes;
- RT-X forces many actions into a common Cartesian format; AlexDoor preserves A1-A4 as semantically distinct representations;
- AlexDoor can isolate the effect of the action representation more cleanly.

**Lessons for the project:**
1. Data standardization is necessary, but is not sufficient to standardize semantics.
2. The action-space tag must specify frame, units, temporal meaning, and controller.
3. Matched episodes are an important experimental advantage.
4. The project must distinguish data-scale transfer from representation transfer.
5. We must not call two vectors a "shared action space" merely because they have the same dimension when their semantics differ.

---

## 5.2 [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)
Octo Model Team, 2024.

**Problem addressed**
A generalist policy must be able to adapt to:

- new robots;
- new camera setups;
- new observations;
- new action spaces;
- new task specifications.

Many previous foundation policies work only with the interface used during
pretraining.

**Central idea and methodology**

Octo uses a modular architecture:

- observation and task tokenizers;
- a shared Transformer backbone;
- readout tokens;
- lightweight diffusion action heads.

New inputs or outputs can be added without reinitializing the backbone.

Octo is pretrained on approximately 800,000 trajectories selected from Open
X-Embodiment.

It produces action chunks using a diffusion head.

For a new action space, it is possible to:

1. preserve the backbone;
2. add a new action head;
3. fine-tune on a small target dataset.

**Results**

Octo is evaluated on nine robot platforms.

In the studied zero-shot tests, it outperforms RT-1-X on average.

For new inputs and new action spaces, the paper demonstrates data-efficient
fine-tuning with approximately:

- 100 target demonstrations;
- less than five hours on an NVIDIA A5000.

It is important, however, that the zero-shot tests primarily use delta
end-effector actions. Moving to joint-position control requires a new head and
fine-tuning.

**Connection to AlexDoor-XAS**

Similarities:

- a shared backbone;
- replaceable output heads;
- action-chunk prediction;
- Diffusion Policy;
- the possibility of conditioning the model on the task and action interface.

Differences:

- Octo cannot switch to arbitrary action spaces zero-shot;
- new action spaces require fine-tuning;
- it does not use matched A1-A4 labels for the same episode;
- it does not separate the policy and safety adapter as AlexDoor does;
- it does not study object-centric A4.

Lessons for the project:

1. A shared backbone with A1-A4 heads is a natural baseline.
2. We must distinguish pretraining transfer from zero-shot action-space
   switching.
3. We should compare full fine-tuning, head-only adaptation, and zero-shot
   conditioning.
4. The action-space tag could be integrated into the task/readout tokens.
5. A shared policy is not sufficient: it must demonstrate positive transfer
   between representations.

---

## 5.3 [Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation](https://arxiv.org/abs/2408.11812)

Doshi et al., 2024. System: `CrossFormer`.

**Problem addressed**

Robots can have radically different observation and action spaces:

- single-arm and dual-arm manipulators;
- quadrupeds;
- mobile robots;
- quadrotors;
- Cartesian outputs;
- joint outputs;
- waypoints;
- different action dimensions and control frequencies.

Manually aligning all these spaces is difficult and often discards information.

**Central idea and methodology**

CrossFormer uses a Transformer sequence model.

Inputs are tokenized into sequences:

- images;
- proprioception;
- language instructions or goal images.

The output uses action readout tokens and separate action heads for embodiment
classes.

The backbone is shared, but every head may have:

- a different dimension;
- different semantics;
- a different frequency.

The model is trained on approximately:

- 900,000 trajectories;
- 20 embodiments.

It does not require complete manual alignment of the action spaces.

**Results**

In the reported tests, CrossFormer achieves:

- approximately 73% average performance;
- approximately 67% for versions trained only on the target robot's dataset;
- approximately 51% for the prior specialist methods considered.

It controls very different systems using the same backbone weights.

The authors emphasize an important limitation, however: the paper does not yet
demonstrate strong positive transfer across embodiments. It primarily shows that
a shared model can absorb heterogeneous data without clear degradation.

**Connection to AlexDoor-XAS**

Similarities:

- a shared backbone;
- different output heads;
- action spaces with different dimensions and semantics;
- the possibility of using a tag or embodiment/action-space class.

Differences:

- CrossFormer uses unmatched datasets;
- its action heads mainly correspond to different embodiments;
- AlexDoor aims to study different representation levels of the same behavior;
- AlexDoor has known A3→A2 and A4→A3→A2 transformations;
- our task is more controlled but much less diverse.

Lessons for the project:

1. CrossFormer is the most natural architectural baseline for a shared A1-A4
   policy.
2. We must measure positive transfer, not merely the absence of negative
   transfer.
3. The correct comparison is a shared model versus four specialist models with
   controlled capacity.
4. We need ablations on the shared backbone, separate heads, and action-space
   token.
5. Matched episodes enable a cleaner analysis than the one available in
   CrossFormer.

---

## 5.4 [Latent Action Diffusion for Cross-Embodiment Manipulation](https://arxiv.org/abs/2506.14608)

Bauer et al., 2025.

**Problem addressed**

Robot hands, human hands, and parallel-jaw grippers have action spaces:

- with different dimensions;
- with different kinematics;
- with different actuators;
- that are difficult to combine in a single policy.

Forcing everything into a seven-dimensional Cartesian vector discards
hand-specific dexterity.

**Central idea and methodology**

The authors learn a shared latent action space.

Each embodiment has:

- an action encoder;
- an action decoder.

To align the spaces semantically, they generate corresponding poses through
retargeting from a human hand to robot end effectors.

The encoders are trained with contrastive learning so that equivalent actions
are placed close together in the latent space:

\[
E_i(a_i) \approx E_j(a_j)
\]

The policy is a latent diffusion policy:

\[
observation \rightarrow z
\]

The embodiment-specific decoder then converts:

\[
z \rightarrow a_i
\]

**Results**

The system enables:

- control of different end effectors with a single latent policy;
- co-training with data from different embodiments;
- positive skill transfer;
- improvement over diffusion policies trained on only one embodiment.

The paper reports average improvements on the order of approximately 13
percentage points, with larger gains in some comparisons.

**Connection to AlexDoor-XAS**

Similarities:

- different representations are connected through a common space;
- a shared policy can use data from multiple action spaces;
- each space retains a dedicated decoder or adapter;
- the project has matched representations that are useful for alignment.

Differences:

- the paper connects different end effectors;
- AlexDoor connects different levels of abstraction;
- A1-A3 have known geometric transformations;
- A4 is not directly invertible into a single low-level trajectory;
- the latent space is not very interpretable;
- AlexDoor prioritizes explicit and auditable outputs.

Lessons for the project:

1. A shared latent-action model is an important baseline.
2. Matched episodes may provide better positive pairs than retargeting.
3. We must compare latent alignment with explicit action-space conditioning.
4. Interpretability and decoder error must be explicit metrics.
5. A4 may require a hierarchical latent space, rather than simply the same
   encoding used for A1-A3.

---

## 5.5 [SPACE: Enabling Learning from Cross-Robot Data Toward Generalist Policies](https://arxiv.org/abs/2606.24049)

Lee et al., June 2026. Recent preprint.

**Problem addressed**

Two robots may need to produce different commands to achieve the same geometric
displacement.

Even two units of the same model may differ in:

- controller gains;
- friction;
- calibration;
- control frequency;
- payload;
- operating state.

Directly predicting native commands ties the policy to the dynamics of the
robot that collected the data.

**Central idea and methodology**

SPACE introduces a separation between:

1. `Cartesian State Delta Policy`;
2. `Adaptive Command Execution`.

The policy predicts the desired geometric change of the end effector.

The `Action Adapter` converts that change into commands appropriate for the
current robot.

The system seeks to adapt to variations:

- across embodiments;
- across units of the same embodiment;
- online, during operation.

The universal action therefore describes the desired kinematic result, while
the adapter handles its dynamic realization.

**Results**

SPACE outperforms policies that directly predict robot control commands in the
studied cross-robot settings.

It also shows robustness to changes in:

- control frequency;
- object weight;
- controller gains;
- system dynamics.

Because this is a very recent work, these results should still be considered
emerging evidence.

**Connection to AlexDoor-XAS**

Similarities:

- separation between policy and robot execution;
- Cartesian delta as a common interface;
- an Action Adapter;
- the ability to share data across robots;
- robustness to operational differences.

Differences:

- SPACE uses a universal Cartesian state delta;
- AlexDoor distinguishes world-frame A2 from object-relative A3;
- AlexDoor adds object-centric intent through A4;
- the AlexDoor adapter records accepted, corrected, and rejected decisions;
- AlexDoor includes force and articulated-object semantics;
- the project has not yet demonstrated cross-robot deployment.

Lessons for the project:
1. "Policy + adapter" is not, by itself, a novel contribution.
2. Novelty must focus on A1-A4, matched episodes, object-centricity, and safety
   evidence.
3. SPACE is a direct baseline for A2/A3 + adapter.
4. We must test variations in control frequency, payload, and controller gains.
5. An object-relative A3 adapter may provide additional invariance beyond the
   Cartesian world delta used by SPACE.

---

# Summary of Lessons for AlexDoor-XAS

Five conclusions emerge from the literature as a whole.

1. **There is no universally best action space.**
   Joint, Cartesian, object-relative, and object-centric spaces introduce different inductive biases.

2. **Cross-action-space learning can be implemented in different ways.**
   We can use:
	   - concatenated representations, as in IKP;
	   - a shared backbone and different heads, as in CrossFormer;
	   - a latent action space, as in Latent Action Diffusion;
	   - a canonical action plus an adapter, as in SPACE;
	   - a hybrid high-/low-level system, as in HYDRA.

3. **A3 is motivated by the task-frame literature.**
   It must, however, be evaluated with:
	   - new poses;
	   - new geometries;
	   - frame noise;
	   - perceptual estimation;
	   - real differences in dynamics.

4. **A4 is motivated by HACMan, HYDRA, and EDAR.**
   Its possible benefit comes from three properties that will have to be isolated:
   - object-centricity;
   - explicit contact intent;
   - temporal abstraction.

5. **Matched source episodes are AlexDoor-XAS's main experimental advantage.**
   They make it possible to compare A1-A4 while limiting differences in task,
   trajectory, outcome, and data distribution. The future Phase 4 must use this
   advantage to demonstrate genuine positive transfer across action spaces, not
   merely train a model with four output heads.

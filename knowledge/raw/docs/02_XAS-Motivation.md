### XAS - Cross Action Space Learning

Def: study of how a robot behavior changes when the physical action is described by using diff representations
SO: the behavior learnt by a robot does not dep only on the dataset (what it observes) and what model we use, BUT also from the language we want it to express its own actions
  
Our project: id what language of the action should be learned/reasoned by the robot
- The door is just the first benchmark (so it’s not simply about opening doors)

Goal: id which rep lv give the best compromise in terms of efficiency / generalization / transfer / interpretability / safety / adapter’s complexity / data needed
#### Conceptual architecture
1. Observation: current situation ⟹ 
2. Policy: decides what to do ⟹ 
3. Action Rep: format of the policy decision ⟹ 
4. Adapter: check - correct - convert the decision in what Alex can do ⟹ 
5. Environment: apply the command in the simulator / real robot

Logging - safety observing the whole process: record what have been req / corrected / applied / observed
#### Action Space
Def: way of representing an action/command/behavior (interface); It defines:
- number/structured produced by the policy + their meaning
- reference system in which they are expressed 
- units - limits used
- how the command is transformed into the robot action

Diff ways can lead to the same physical results: same physical episode rep in diff ways
- So we compare diff rep from the same physical origin
	- Same episode id, physical outcome, pose distrib, dataset split, evaluation
	- Diff action values / ref fram
- In this way we avoid confounding: the problem of having a dataset for each action space (w/ maybe some simpler than others), that could affect the results
	- So we are sure that we judge the policy’s effect wrt rep (not other factors)
- Compare results + errors + forces + interventions 

Our Project: progression from command closer to robot ⟹ to the interaction’s meaning, SO:
- decreases the dep on Alex only
- increases the semantic content of the action
- increases transferability
- increases interpretability
- BUT increases the work req from the adapter to transform the intention in execution
	- SO: that’s why we cannot assume that A4 is better from the beginning

A1: say how each arm joint must move = robot-spec low-lv baseline
- Action: rep as set of deltas of joints target
- The closest to the robot mechanics
- Pros: concrete, close to robot control, makes explicit the joints movement
- Cons: dep on robots structure (diff robot = diff interpretation), difficult to interpret, the intention is hidden behind multiple small joints movements
    
A2: say how the end-effector (hand/gripper) should move wrt world/room (global ref S) = practical cartesian baseline
- 6D vector (dx, dy, dz, drx, dry, drz) = 3 x translation, 3 x rotation
- Inverted controller transform the end-eff movement, into the joints movement; SO A2 is less related to the single joints
- Pros: more practical than A1
- Cons: less related to world frame (if door changes pos, the same numbers of the commands rep a diff intention wrt door)

A3: say how the hand should move wrt door = primary transfer-oriented baseline
- Still 6D vector form, BUT diff ref S = wrt door geometry (not world/room)
- Pros: if door pos changes, the same movement meaning (eg. get closer to the door’s panel) it’s still valid
- Cons: lot of delegation to the adapter (A3 ⟹ A2)
	1. checks doors frame
	2. converts A3 delta into the world frame
	3. pass the result to A2, to apply the same fine-grained controls

A4: describe intended action chunk (interaction phase, target point of contact on the door, desired door angular movement, chunk duration), not a single movement
- Pros: more interpretable + closer to VLA (w/ img + language)
- Cons: even more delegation to the adapter 
	1. converts this rep into a seq o phases (approach ⟹ contact ⟹ push)
	2. converts into A3
	3. converts into A2 
	4. IK: get alex execution

SO: 
- Action Space: available “vocabulary” 
	- e.g. A2 = 6D cartesian movement of the hand;
- Action Rep: concrete way in which an action is encoded, using that vocabulary 
	- e.g. move the hand slightly straight, w/o rotating it
#### Motivation
1. Actions format is important as the model - dataset size
	- Since it introduces inductive bias: suggest to the policy which regularities = important
2. Transfer (same command valid for diff robots, though diff adapters)
#### Advantages
Learning: rep w/ simple regularities = easy to approximate (target f is more uniform, w/ less variance b/w equivalent examples, easier to interpolate, less dep on training orientations)

e.g. Move the hand of 2cm towards the door opening

A2: door changes orientation in world ⟹ numerical target changes

- BUT the semantic behavior is still the same (push the door)
    
- SO policy needs to learn implicitly from the data diff things: door orientation, local dir x opening, how to rotate that dir in world frame, what x-y-z emit
    

A3: same numerical target x diff door orientation in world

- BUT is the adapter which does the transformation in the world frame
    

  

Sample efficiency: rep incorporates a useful symmetry ⟹ less examples to learn behavior

TO ASSESS: we compare learning curves of adapters wrt dataset dim

BUT: not guaranteed; it depends on

- door frame correctness
    
- observations quality
    
- data noise
    
- policy capability 
    
- presence of pose variations in dataset
    
- adapter accuracy 
    

  

e.g. Same of before

A2: policy needs to learn the rule “when the door rotates ⟹ action vector rotates”

- BUT to do this it needs tons of example w/ diff orientations of SAME behavior
    

A3: policy doesn't need to learn it (since rotations is dealt by the adapter)

- WHY: symmetry already incorporated in rep = lot of examples x same target action
    

- Physically equivalent situations rep in similar way ⟹ easier x policy to recognize that they belong to the same behavior = not separate learning
    

- SO: avoid requiring the model to id the same strategy in diff coordinates; dealt by S
    

  

Generalization: policy capability to work well in unseen situations(not in training set) at test t

- SO: policy does not remember the training examples ⟹ it has learned a general rule related to the obj (not wrt its pos in the scene) = same rel end-effector - door
    
- SO semantically stable: maintains the same meaning, even it the coords changes 
    

TO ASSESS: policy success on unseen/held-out poses

BUT: not guaranteed; it depends on

- correctness of door frame estimation (by cameras/sensors / obj detection S )
    
- new obj pose (pos+ orientation) makes reachable/not the door
    
- change of contact dyn (b/w hand-door)
    
- end-effector/arm starts from a very diff config (pos/orientation)
    
- obs do not rep well the hand-door rel (hand pos wrt door)
    
- robot reaches articulation lim (since each joint w/ its interval of movement)
    
- adapter transforms the command correctly, but the physical movement is not exe
    

  

Interpretability: how easily H understands what the policy is doing, by looking at is output

- By looking at the action/command gen by the model, can I understand its intentions?
    

USEFUL: for debugging (error analysis) + results communication

  

A1: joints delta vector (small articulations var) = 29 pos/neg n ⟹ difficult interpretation

- N describe how to move the body, not the why
    
- To know intention we need to know: which joint x that value; robot actual config ⟹ combine mentally all joints movement ⟹ understand hand’s final pos ⟹ effect
    

A4: obj-centric chunks (phase, contact point, desired movement, duration) = describe action goal (not each single articulation movement)

  

Safety: adapter as a security filter b/w policy - robot, since:

1. Verify if the command is geometrically valid, reachable in workspace, coherent w/ task, duration right in t, compatible w/ what the robot can do
    

- Duration: pos (not neg/zero) + w/ max tick budget (no endless actions)
    
- Door hinge delta direction: now just pos (since it pushes, not pulls) 
    
- Obj target: verify pos of P of contact on the door (coords / lim / reachable)
    

2. Determine the command type: valid, corrected, rejected (not secure/impossible)
    

  

A4 Adapter: check to reach these points at a certain t max (otherwise missed_contact):

- approach waypoint (intermediate P, in front of the door at a certain d from initial P),
    
- pre-contact waypoint (another intermediate P, really close to door),
    

- contact P, 
    
- final P (after movement = final angle)
    

  

Transfer: capability to use a policy diff from the one of the training 

- An Alex policy can be used also on another robot w/o learning everything again?
    
- This is possible bc of adapters (each robot has a spec one)
    

BUT: not guaranteed; it depends on

- correctness of adapter 
    
- robot morphology to make the action feasible
    

SO: transfer the semantic part + adapt the physical part

  

A1: w/ joint-spec rep we tell exactly which articulations of a spec robot we have to move

- We need to know:
    

- Alex articulations + arms segments lenght
    
- Their order in the vector
    
- Their direction - lim 
- BUT: another robot = diff values: same value ⟹ complete diff movement
- SO: related to robot’s morphology
    
A4: w/ obj-spec rep describes what should happen to the obj (door)
- We have phase, target P on the panel, desired delta hinge (movement), duration
- SO: comprehensible independently from the robot
- HOW: each robot has its own adapter (w/ IK / end-effector / workspace/ joint lim)
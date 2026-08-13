**Goals (TO-DO):** 
- Create a new dataset: more realistic, representative, harder => I want images, text, ecc. It should be a proper dataset
	- Let's understand from which paper we could take inspiration
	- At the moment these data should be just sim?
	- We need to understand what to do to enable generalization
	- Release it as a new version - fingerprint - split - normalization
- Conduct the training w/ more seeds (before just 0)
	- See what a seed is, what it impacts, why do we need to use different ones
- Multi-seed evaluation:
	- Understand what has been evaluated, and why
	- What is the most indicative + how to interpret the results
- Resolve the usage of A1-A4
	- A1: 
		- problem w/ training: only 6/29 DOF used (other 23 = 0) => we need to think proper loss (w/ masking)
		- problem w/ IK: now we always think cartesian space, instead of joint space = modify this => need direct-joint delta exe
	- A4 measures adapter not policy 

## **Unsafe Case: 219.95 N**
**Def:** it's a successful episode, but w/ this single anomalous peak for a single tick (55 - the first contact)
- Generated from ACT, with A3, dataset N50, seed 0

**Minimal investigation:** 
1) Exact replay of the case
2) Depending of the output:
	- If it is observed again: 1/2 tests w/ minimal perturbations = to see if it happens again
	- If not: 2/3 repetitions, to understand if it's unstable

**N.B.** It could disappear with the new dataset that we will create


**DONE:**
- Riexaminate the case w/ 219.95 N 
	- Understand what it means, why it's been generated
	- What to do to fix it
	- Ensure that this situation won't happen again for these reasons (gate before training needed?)
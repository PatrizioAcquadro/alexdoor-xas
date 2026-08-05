### **Environment**
**Def:** contains the configuration of the software environment of the project for the cluster, to train the policies 
- *Name:* logical name of the conda env
- *Channels:* repo from which conda can download the packages 
- *Dependecies:* programs/lubs to install (python + pip (for packages))
- **N.B.** pyTorch is not present bc it dep on the NVIDIA driver of the cluster
- **N.B.** Isaac Sim & Lab excluded since the cluster is only for training

**yml = YAML:** textual format to represent in a readable way structured data / config
- We use identation to indicate which elements belong to that section
	- *Identation:* space inserted at the beginning of a textual/coding row = to organize the structure 
- It's a doc read-interpreted by another program
	- Here by th script *bootstrap_gilbreth_pilot.sh*: does update (if env exists), or creates it from scratch (if not present)
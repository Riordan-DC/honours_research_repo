# Honours Research Notebooks
A repo containing all the code for my honours research project to be completed as part of my Bachelor of Information Technology (Computer Science) Honours (2020-2021).


Baseline_Instructor.ipynb
-------------------------
The primary baseline notebook. Contains a dataloader for test (ALFRED dataset) data, the model definition, a test loop, and visualisations for test data.
The baseline uses word vectors and a scene graph to infer object/action pairs from instructions.
Requires:
- ALFRED repo with json dataset
- AI2THOR 2.1.0 (An old version to support outdated ALFRED env initialisation code)

The AFLRED dataset provides the instructions being used to test the baseline. This data is not accurate and do not take its performance seriously. This is why the Interactive_Baseline was developed. In the future ALL the ALFRED code will be isolated because it decays my codebase quicker by forcing me to support older versions of AI2THOR.

Ideally, besides all the todos in the notebook, I'd like to extract the components of this notebook into several scripts such as:
models.py (Baseline, RL, etc)
utils.py (ALFRED specific dataloader)
While maintaining the testing and visualisations inside the notebook where iteration is quick and iteraction is required.


Interactive_Baseline.ipynb
-------------------------
The primary baseline with an interactive prompt.
It is a copy of the Baseline_Instructor notebook but with a prompt loop that allows users to write their instructions. If an instruction fails then 

Instructed_Agent_GridWorld.ipynb
-------------------------
An early experiment where I simplified the scene graph concept to a grid world. This meant we performed a local node search instead of a global one. Only considering nodes 1-hop from the robot. Overall, this taught me that a local search is inferior to a global one (up to a limit) because in reality we do not have the luxury of adjacent nodes. The paths through many goal-irrelevant nodes removes the benefits of abstraction that a scene graph brings in the first place. This notebook is no longer used. Kept simply as a record of previous investigations.

# New Investigations

ML_GOAP.ipynb
-------------------------
A implimentation of GOAP rewritten from the GOAP C implimentation. GOAP is based upon a robotics planner called STRIPS (Standford Research Institude Problem Solver) that uses A* search to find a path from a goal state back to an initial state by satisfying state constaints with actions with deterministic outcomes on the state. My vision was two fold:
- What if we could build a GOAP style planner that searches the state space of stochastic actions? I.e. Actions where outcome state is stochastic. Therefore we also need a stochastic A* heuristic.
- With a discrete action space and few atomic actions; the scene graph robot operates on a level of abstraction where GOAP can be used to make plans to goal states.

Instruction_Parse.ipynb
-------------------------
An investigation into methods for parsing natural language. This project, using natural langauge in robotics, is as much a scene understanding problem as it is a natural langauge understanding problem. For this reason this notebook tracks my exploration into NLU. My laptop also has the weekly course code for the standford NLP unit and my ML recipies GoogleDoc contains all my nodes throughout that unit.

# Custom Libraries

scene_graph.ipynb
-------------------------
Niko's scene graph implimentation.

thorEnvironment.ipynb
-------------------------
Vib's thor environment OpenAI-GYM API for easy use of the thor environment especially in RL experiments. 

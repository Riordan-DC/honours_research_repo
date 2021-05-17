# Honours Research Notebooks
A repo containing all the code for my honours research project to be completed as part of my Bachelor of Information Technology (Computer Science) Honours (2020-2021).


Baseline_Instructor.ipynb
-------------------------
The primary baseline notebook. Contains a test loop, and visualisations for test data.
The baseline uses word vectors and a scene graph to infer object/action pairs from instructions.
Requires:
- ALFRED repo with json dataset
- AI2THOR 2.1.0 (An old version to support outdated ALFRED env initialisation code)

The AFLRED dataset provides the instructions being used to test the baseline. This data is not accurate and do not take its performance seriously. This is why the Interactive_Baseline was developed. In the future ALL the ALFRED code will be isolated because it decays my codebase quicker by forcing me to support older versions of AI2THOR.

Ideally we want to move away from ALFRED and old AI2THOR. Instead, using Vib's new code and NEW graph representation without hundreds of "goto" pose nodes. In future I'd like to eradicate the "go" node entirely (maybe just one attached to each object).

Interactive_Baseline.ipynb
-------------------------
WARNING: Old and outdated baseline in use.
The primary baseline with an interactive prompt.
It is a copy of the Baseline_Instructor notebook but with a prompt loop that allows users to write their instructions. If an instruction fails then the user is prompted again, until the test passes. 

Instructed_Agent_GridWorld.ipynb
-------------------------
WARNING: Old and unused.
An early experiment where I simplified the scene graph concept to a grid world. This meant we performed a local node search instead of a global one. Only considering nodes 1-hop from the robot. Overall, this taught me that a local search is inferior to a global one (up to a limit) because in reality we do not have the luxury of adjacent nodes. The paths through many goal-irrelevant nodes removes the benefits of abstraction that a scene graph brings in the first place. This notebook is no longer used. Kept simply as a record of previous investigations.

Sentence_Embedding.ipynb
-------------------------
My third baseline incorporates modern NLP, neural NLP. This meant I needed to train a model to interpret the instructions for the cosine ranking model. The first model contained in this notebook is a RNN encoder that reads in the sentence and produces recurrent hidden states and an output vector and two decoders, on for actions and the other for arguments. The encoder is trained on the loss of the action and argument decoders. In the "Sentence_Embedding_Predictor" version of this notebook I describe below that I replaced the action decoder with an MLP action classifier.

Sentence_Embedding_Predictor.ipynb
-------------------------
A version of the sentence embedding notebook where I built a simple MLP action predictor instead of using an RNN. I can get better results with this simple MLP instead of the more complicated RNN. 

# New Investigations

ML_GOAP.ipynb
-------------------------
A implimentation of GOAP rewritten from the GOAP C implimentation. GOAP is based upon a robotics planner called STRIPS (Standford Research Institude Problem Solver) that uses A* search to find a path from a goal state back to an initial state by satisfying state constaints with actions with deterministic outcomes on the state. My vision was two fold:
- What if we could build a GOAP style planner that searches the state space of stochastic actions? I.e. Actions where outcome state is stochastic. Therefore we also need a stochastic A* heuristic.
- With a discrete action space and few atomic actions; the scene graph robot operates on a level of abstraction where GOAP can be used to make plans to goal states.

Instruction_Parse.ipynb
-------------------------
An investigation into methods for parsing natural language. This project, using natural langauge in robotics, is as much a scene understanding problem as it is a natural langauge understanding problem. For this reason this notebook tracks my exploration into NLU. My laptop also has the weekly course code for the standford NLP unit and my ML recipies GoogleDoc contains all my nodes throughout that unit.

GAE_Parser.ipynb
-------------------------
Graphs are used in linguistics to represent sentences and sentiment. I thought It would be interesting to study how a graph autoencoder could be used for one of these tasks. Of course this didnt make it into my thesis but I thought it is still an interesting idea, worth another look.

# Custom Libraries

scene_graph.ipynb
-------------------------
Niko's scene graph implimentation.

thorEnvironment.ipynb
-------------------------
Vib's thor environment OpenAI-GYM API for easy use of the thor environment especially in RL experiments. 

thor_environment.py
-------------------
Vib's thor environment OpenAI-GYM API for easy use of the thor environment especially in RL experiments. 
Extended with a new explore function that produces smaller graphs quicker but cannot simulate occlusion.
A new update_object function which builds graphs in accordance with the ALFRED action format.
A new step function which performs ALFRED defined action procedures.

utils.py
--------
Includes the ALFRED dataloader with instruction tokenisation and punctuation sanitization. All functions for debuging graphs and performing slow neighbourhood averaging functions. This file is included in Baseline_Instructor.ipynb and any script for running a baseline with ALFRED data.

models.py
---------
Includes the models I used as baselines in the thesis. All are written as PyTorch modules even the cosine model which contains no learnable parameters. Currently there are the Cosine Model, the sentence Encoder, the argument attention decoder, the action classifier / predictor and a few torch utility functions used to format data for the langauge models.

# Experiment 7
Model: Cosine Model
Graph Model: Minimised Graph Exploration, Seperate scoring/searching for instruction nouns (objects) then instruction verbs (affordances on the previously found objects). Model is ran twice, first on instruction nouns to create an object mask. Second on instruction verbs, the chosen affordance is the highest scored after the object mask is applied. 
Dataset: ALFRED, valid_unseen, All Tasks, Any Votes, All Graphs (Includes Partially Observed Graphs, i.e. found during exploration), All actions.
Preprocess: No punctuation, lemmatized, Nouns and Verbs are SpaCy detailed POS tags that begin with 'N' and 'V' respectively.

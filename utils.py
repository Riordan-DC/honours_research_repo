import tqdm
import glob
import json
import sys
import string
import torchtext
from torchtext.data import get_tokenizer
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # We will use lemmatization instead of stemming because stemming is radical and can often destroy the original meaning of a stentence.
    pass

# Loading and Transforming the ALFRED dataset utilites

def load_next_alfred_data(ALFRED_JSON_PATTERN):
    """
     Get list of all instructions and their trajectories
     glob.glob gets all files and stores them. iglob makes an iterator.
     
    Paramaters:
        @alfred_json_pattern
        A string which represents the location of the data with glob style
        regex expressions to denote multiple subdirectories.
    """ 
    train_json_files = glob.glob(ALFRED_JSON_PATTERN)
    tokenizer = get_tokenizer("basic_english")
    wnl = WordNetLemmatizer()
    dataset = []
    
    stop_words = ["the", "and", "of", "a", "is", "over"]
    
    def preprocess_sentence(sentence):
        sentence = tokenizer(sentence)
        sentence = filter(lambda x: not x in string.punctuation, sentence)
        sentence = [wnl.lemmatize(word) for word in sentence]
        sentence = list(filter(lambda x: not x in stop_words, sentence))
        return sentence
    
    # Yeild an alfred json
    for json_file_idx in tqdm.tqdm(range(len(train_json_files))):
        data = json.load(open(train_json_files[json_file_idx]))
        annotations = data['turk_annotations']['anns']
        actions = data['plan']['high_pddl']
        scene = data['scene']
        scene['task_id'] = data['task_id']
        
        instruction_actions = []
        for d in annotations:
            votes = d['votes']
            if any(votes): # WARNING: Limiting dataset based on votes
                trajectory = {'task_desc': [], 'instructions': []}
                trajectory['task_desc'] = preprocess_sentence(d['task_desc'])
                for i in range(len(d['high_descs'])):
                    sanitized_instruction = preprocess_sentence(d['high_descs'][i])
                    instruction = {'instruction': sanitized_instruction, 
                                   'action': actions[i]['discrete_action']['action'],
                                   'argument_1': actions[i]['discrete_action']['args'][0] if 0 < len(actions[i]['discrete_action']['args']) else '<unk>', 
                                   'argument_2': actions[i]['discrete_action']['args'][1] if 1 < len(actions[i]['discrete_action']['args']) else '<unk>'}
                    trajectory['instructions'].append(instruction)
                instruction_actions.append(trajectory)

        if len(instruction_actions) > 0:
            dataset.append((instruction_actions, scene))
    return dataset

def filtered_dataset_copy(alfred_data, scene=None):
    data_copy = alfred_data.copy()
    
    if scene:
        def filter_scene(x):
            return x[1]['floor_plan'] == scene

        data_copy = list(filter(filter_scene, data_copy)) 
    
    return data_copy

# Test / Training Utility Functions

def draw_graph(graph):
    plt.figure()
    x = []
    y = []
    for node in graph.nodes(data=True):
        index = node[0]
        data = node[1]
        if data['node_type'] == 'object':
            name = data['data']['name']
            pos = data['data']['position']
            x.append(pos['x'])
            y.append(pos['z'])
    ax = plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Environment Graph Map')

def add_object_features(graph):
    """
    Graph preprocessing step to add object features to their affordance features.
    Without this step an affordance selection task will be unaware of objects.
    """
    
    class graph_t:
        x = graph._torch_graph.x.clone()
        edge_index = None
    
    g = graph_t()
    # For each object in a graph, add its features to its affordance
    for idx in graph.nodes:
        affordances = graph.get_affordances(idx)
        affordance_count = len(affordances)
        if affordance_count > 0:
            for affordance in affordances:
                g.x[affordance] += g.x[idx]
                g.x[affordance] /= 2.0
    return g

def thor_restore(controller, init_action, object_poses, object_toggles, dirty_and_empty):
    """
    Restore the Thor simulator to an ALFRED defined state
    """    
    
    if len(object_toggles) > 0:
        controller.step((dict(action='SetObjectToggles', objectToggles=object_toggles)))
    
    if dirty_and_empty:
        controller.step(dict(action='SetStateOfAllObjects',
                           StateChange="CanBeDirty",
                           forceAction=True))
        controller.step(dict(action='SetStateOfAllObjects',
                           StateChange="CanBeFilled",
                           forceAction=False))
    
    controller.step((dict(action='SetObjectPoses', objectPoses=object_poses)))
    controller.step(init_action)

def valid_target(p_env, target):
    """
    Sanity Check. Skip impossible graphs, requiring multiple steps, or faulty exploration.
    It appears that the SceneGraph.find() method cannot replace this functionality.
    """
    in_sim = False
    objects = p_env.controller.step(dict(action='Pass')).metadata['objects']
    for obj in objects:
        if target == obj['objectType'].lower():
            in_sim = True
    
    in_graph = False
    for n in p_env.graph.nodes:
        if p_env.graph.nodes[n]['node_type'] == 'object':
            if p_env.graph.nodes[n]['obj'] == target:
                in_graph = True
    
    if not in_sim and not in_graph:
        print('[CHECK][MAJOR ERROR] - Target not found in simulation or graphmap')
        return False
    elif not in_sim and in_graph:
        print('[CHECK][MAJOR ERROR] - Target [%s] found in graphmap but not simulation. Perhaps a trajectory/sim mismatch?' % target)
        return False
    elif in_sim and not in_graph:
        print('[CHECK][NOTE] - Target found in sim but not added to graphmap due to not being found in exploration.')
        return True
    elif in_sim and in_graph:
        return True

def valid_action(p_env, action):
    """
    In future I'd like all actions/affordances to be supported. I am currently working on this.
    This function is used to invalidate trajectories, this reduces available data and is bad. 
    """
    try:
        normalize_action_name(action)
        return True
    except NotImplementedError:
        return False

def valid_trajectory(p_env, trajectory):
    """
    Sanity Check. Goes through each instuction and checks that the targets exist
    and its actions can be performed. 
    WARNING: An object meant to be discovered inside an object
    will not be detected, 
    TODO: Do Check thor Environment for things contains inside things.
    """
    valid = True
    for inst_idx, instruction in enumerate(trajectory['instructions']):
        target_object = instruction['argument_1']
        target_action = instruction['action']
        if not valid_target(p_env, target_object):
            print('\t[CHECK] Invalid Instruction \"%s\" Target Not Found: [%s]' % (' '.join(instruction['instruction']), target_object))
            valid = False
            break
        if not valid_action(p_env, target_action):
            print('\t[CHECK] Invalid action/affordance \" %s \" Is not currently supported!' % target_action)
            valid = False
            break
        
    return valid

def print_scores(instruction, scores):
    """
    A helpful debug function for viewing the max score node for each node in the graph.
    """
    print(instruction)
    print(scores)

def normalize_action_name(name):
    if name == 'GotoLocation':
        return 'go'
    elif name == 'PickupObject':
        return 'pick'
    elif name == 'PutObject':
        return 'put'
    elif name == 'OpenObject':
        return 'open'
    elif name == 'CloseObject':
        return 'close'
    elif name == 'SliceObject':
        return 'slice'
    elif name == 'CleanObject':
        return 'clean'
    elif name == 'HeatObject': # Custom ALFRED Action: Test
        return 'heat'
    elif name == 'CoolObject': # Custom ALFRED Action: Test
        return 'cool'
    elif name == 'ToggleObject': # Custom ALFRED Action: Test
        return 'toggle'
    else:
        raise NotImplementedError("Action %s not implimented yet." % name)


import json
import os

class Language:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0
    
    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2index: # Add word to langauge if unseen 
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def word(self, index):
        return self.index2word.get(str(index), f"<unk:{index}>")
    
    def index(self, word):
        return self.word2index.get(word, f"<unk:{word}>")
    
    def dump(self, filename):
        obj = {
            'word2index' : self.word2index,
            'word2count' : self.word2count,
            'index2word' : self.index2word,
            'n_words' : self.n_words
        }
        with open(filename, 'w') as fp:
            json.dump(obj, fp)
    
    def load(self, filename):
        with open(filename, 'r') as fp:
            obj = json.load(fp)
            self.word2index = obj['word2index']
            self.word2count = obj['word2count']
            self.index2word = obj['index2word']
            self.n_words = obj['n_words']
    
    def reset_counts(self):
        self.word2count = dict.fromkeys(self.word2count, 0)

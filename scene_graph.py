import numpy as np
import networkx as nx
import torch
import torch_geometric
import torchtext
import pickle
import matplotlib.pyplot as plt

try:
    __word_vec__
except NameError:
   # __word_vec__ = torchtext.vocab.FastText()
# Riordan's vector cache aha
   __word_vec__ = torchtext.vocab.FastText(cache='../.vector_cache')


class SceneGraph(nx.DiGraph):
    def __init__(self, word_vec=None, one_hot=False):
        super(SceneGraph, self).__init__()
        
        self.node_counter = 0    
        
        self.one_hot = one_hot
        
        self.vocab_size = 0
        self.vocab = {}
        
        self._torch_graph = None
        
#         self._word_vec = torchtext.vocab.FastText() if word_vec is None else word_vec
#         self._word_vec = __word_vec__.get_vecs_by_token if word_vec is None else word_vec
        self.init_word_vec()
    
        self._translate_unknown_words = {'stoveburner': 'stove',
                                         'stoveknob': 'stove',
                                        'sinkbasin': 'sink',
                                        'garbagecan': 'garbage',
                                        'soapbottle': 'soap',
                                        'coffeemachine': 'coffee',
                                        'dishsponge': 'sponge',
                                        'peppershaker': 'pepper',
                                        'spraybottle': 'spray',
                                        'glassbottle' : 'glass',
                                        'butterknife' : 'butter',
                                        'papertowelroll': 'towel',
                                        'winebottle': 'wine',
                                        'floorlamp': 'lamp',
                                        'diningtable': 'table',
                                        'showercurtain' : 'shower',
                                        'bathtubbasin' : 'bathtub',
                                        'toiletpaperhanger' : 'hanger',
                                        'towelholder' : 'holder',
                                        'breadsliced' : 'bread',
                                        'wateringcan' : 'water',
                                        'tvstand' : 'stand',
                                        'tissuebox' : 'tissue',
                                        'desklamp' : 'lamp'}
        
        # Using these reduces performance. I believe because they are averaged you lose a lot of useful information.
        # Learning word vectors would help 
        """ 
                                        {'stoveburner': ['stove', 'burner'],
                                         'stoveknob': ['stove', 'knob'],
                                        'sinkbasin': ['sink', 'basin'],
                                        'garbagecan': ['garbage', 'can'],
                                        'soapbottle': ['soap', 'bottle'],
                                        'coffeemachine': ['coffee', 'machine'],
                                        'dishsponge': ['dish', 'sponge'],
                                        'peppershaker': ['pepper', 'shaker'],
                                        'spraybottle': ['spray', 'bottle'],
                                        'glassbottle' : ['glass', 'bottle'],
                                        'butterknife' : ['butter', 'knife'],
                                        'papertowelroll': ['paper', 'towel', 'roll'],
                                        'winebottle': ['wine', 'bottle'],
                                        'floorlamp': ['floor', 'lamp'],
                                        'diningtable': ['dining', 'table'],
                                        'showercurtain' : ['shower', 'curtain'],
                                        'bathtubbasin' : ['bathtub', 'basin'],
                                        'toiletpaperhanger' : ['toilet', 'paper', 'hanger'],
                                        'towelholder' : ['towel', 'holder'],
                                        'breadsliced' : ['bread', 'slice'],
                                        'wateringcan' : ['water', 'can'],
                                        'tvstand' : ['tv', 'stand'],
                                        'tissuebox' : ['tissue', 'box'],
                                        'desklamp' : ['desk', 'lamp']}
        """
        self.setup()
        
    # =================================================================================== 
    def setup(self):
        # set up the robot with basic affordances
        self.robot_node = self.add_robot()
        
        # Rio Note: Not adding robot affordances because we abstract movement to a simple 'go'
        #for affordance in ['left', 'right', 'ahead', 'back', 'up', 'down', 'crouch', 'stand']:
        #    self.add_affordance(self.robot_node, affordance)
    
    # =================================================================================== 
    def to_pickle(self, filename):
        del self.get_word_vec
        nx.write_gpickle(self, environment_file)
        self.init_word_vec()
    
    # =================================================================================== 
    def from_pickle(self, filename):
        self = nx.read_gpickle(filename)  
        self.init_word_vec()
        return self
    
    # =================================================================================== 
    def init_word_vec(self):
        self.get_word_vec = __word_vec__.get_vecs_by_tokens
    
    # =================================================================================== 
    def add_robot(self, **kwargs):
        x = self.word2vec('robot') 
                
        self.add_node(self.node_counter, x=x, node_type='robot', **kwargs)                                        
        self.node_counter += 1                                
        return self.node_counter-1
    # =================================================================================== 
    def word2vec(self, word):
        if self.one_hot:
            if not word in self.vocab:
                self.vocab_size += 1
                self.vocab[word] = self.vocab_size
            return np.array([self.vocab[word]])
        else:
            if word in self._translate_unknown_words:
                word_set = self._translate_unknown_words[word]
                representation = self.get_word_vec(word_set, True)
                if len(representation.shape) > 1:
                    representation = torch.mean(representation, dim=0)
            else:
                representation = self.get_word_vec(word, True)
            
            if not torch.sum(torch.abs(representation)) > 0:
                print('zero word: %s' % word)

            return representation.numpy()
        
    # ===================================================================================
    def add_object(self, x, obj=None, **kwargs):
        
        if type(x)==str:
            if obj is None:
                obj = x
            x = self.word2vec(x)
        
        self.add_node(self.node_counter, x=x, node_type='object', obj=obj, **kwargs)               
        self.node_counter += 1        
        return self.node_counter-1
    
    # ===================================================================================
    def add_relation_edge(self, n1, n2, x, relation=None, **kwargs):
        if type(x)==str:
            if relation is None:
                relation = x
            x = self.word2vec(relation)  
            
        self.add_edge(n1, n2, x=x, edge_type='relation', relation=relation, **kwargs)
        
    # ===================================================================================        
    def add_affordance(self, obj, x, affordance=None, **kwargs):
        if type(x)==str:
            if affordance is None:
                affordance = x
            x = self.word2vec(affordance)               

        self.add_node(self.node_counter, x=x, node_type='affordance', affordance=affordance, **kwargs)        
        self.add_edge(obj, self.node_counter, x=self.word2vec('affordance'), edge_type='affordance')
                             
        self.node_counter += 1        
        return self.node_counter-1
    
    # ===================================================================================        
    def remove_affordance(self, obj, affordance):
        raise NotImplementedError()
    
    
    # ===================================================================================
    def to_torch_graph(self):        
        data = {}
        
        # make sure node indices are consecutive, this is important as we constantly remove and add nodes
        nlist = sorted(self.nodes())
        mapping = dict(zip(nlist, range(0, self.number_of_nodes())))
        # The relabel of nodes must also happen to indices kept outside of the networkx class i.e.
        # Rio Note: No robot node, the agent is disembodied. Movement is abstracted to 'go' affordances. 
        # Update: I have decided I cannot make the robot disembodied because it now needs to hold
        # objects.
        self.robot_node = mapping[self.robot_node]
        
        try:
            nx.relabel_nodes(self, mapping, copy=False)
        except Exception:
            print(mapping)
            raise(Exception)
#             self = nx.relabel_nodes(self, mapping, copy=True)                            
        
        edge_index = torch.tensor(list(self.edges)).t().contiguous()
        data['edge_index'] = edge_index.view(2, -1)
        data['x'] = torch.tensor([self.nodes[n]['x'] for n in self.nodes])        
#         data['edge_attr'] = torch.tensor([self.edges[n]['x'] for n in self.edges])
        
        
        graph = torch_geometric.data.Data.from_dict(data)
        graph.num_nodes = self.number_of_nodes()
        
        # a mask for all the affordance nodes
        mask = []
        for n in sorted(self.nodes):
            if self.nodes[n]['node_type']=='affordance':
                mask.append(True) 
            else:
                mask.append(False)
        
        self._torch_affordance_mask = torch.tensor(mask, dtype=torch.bool)
        
        # a mask for all the object nodes
        mask = []
        for n in sorted(self.nodes):
            if self.nodes[n]['node_type']=='object':
                mask.append(True) 
            else:
                mask.append(False)
        
        self._torch_object_mask = torch.tensor(mask, dtype=torch.bool)
        
        self._torch_graph = graph
        return self._torch_graph
        
    # ===================================================================================
    def from_torch_id(self, idx):
        
        return idx
    
    
        # is there a more efficient way, without converting this into a full list?
        return list(self.nodes.keys())[idx]
    # ===================================================================================
    def clear(self):
        super().clear()
        self.node_counter = 0
        self._torch_graph = None
        
        self.setup()
        
            
    # ===================================================================================
    def find(self, attribute=None, value=None):        
        return [x for x in self.nodes if self.nodes[x].get('data', {}).get(attribute, {}) == value]
            
    # ===================================================================================        
    def has_affordance(self, idx, affordance):
        a = [n for n in self.successors(idx) if (self.nodes[n]['node_type']=='affordance') and (self.nodes[n].get('affordance','')==affordance)]
        return len(a)>0
        
    # ===================================================================================        
    def get_affordances(self, idx, affordance=None):                
        if affordance is None:
            return [n for n in self.successors(idx) if (self.nodes[n]['node_type']=='affordance')]
        else:
            return [n for n in self.successors(idx) if (self.nodes[n]['node_type']=='affordance' and self.nodes[n].get('affordance','')==affordance) ]                
                    
    # ===================================================================================        
    def get_related_objects(self, idx):                
        succ = [n for n in self.successors(idx) if (self.nodes[n]['node_type']=='object')]
        pred = [n for n in self.predecessors(idx) if (self.nodes[n]['node_type']=='object')]        
        return succ, pred
                
    # ===================================================================================  
    def get_relations(self, idx, relation=None):        
        if relation is None:
            return [n for n in self.adj[idx] if self.adj[idx][n].get('edge_type','')=='relation']
        else:
            relations = []
            for n in self.adj[idx]:
                if self.adj[idx][n].get('edge_type','') == 'relation':
                    if self.adj[idx][n].get('relation') == relation:
                        relations.append(n)
            return relations

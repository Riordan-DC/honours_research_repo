import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear

# ==================================================
class CosineModel(torch.nn.Module):
    def __init__(self):
        super(CosineModel, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def forward(self, data, target):
        '''
        Consine model scores graph nodes on the cosine similarity between the node embedding and the word embeddings.
        '''
        x, edge_index = data.x, data.edge_index
        max_nodes = []
        
        scores = torch.zeros(x.shape[0], dtype=torch.float32)
        
        for word in target:
            word = word.expand_as(x)
            node_scores = self.cos(x, word)
            scores += node_scores
            # Save the highest scoring node for each word
            max_node = torch.argmax(node_scores)
            max_nodes.append(max_node)
        
        x = scores.view(-1,1)

        return x, max_nodes 
    
    def one_hot(self, data, target):
        '''
        Cosine model scores node based on the cosine similarity between the node embedding and the word embeddings.
        '''
        x, edge_index = data.x, data.edge_index
        max_nodes = []
        
        # Convert indices to one-hot encoding
        max_vector = torch.argmax(torch.cat([x,target]))
        num_vectors = x.shape[0]
        x_one_hot = torch.zeros(num_vectors, max_vector)
        x_one_hot[torch.arange(num_vectors), x.flatten()] = 1.0

        target = target.long()
        num_vectors = target.shape[0]
        target_one_hot = torch.zeros(num_vectors, max_vector)
        target_one_hot[torch.arange(num_vectors), target.flatten()] = 1.0
        
        scores = torch.zeros(x.shape[0], dtype=torch.float32)
        
        for word in target:
            word = word.expand_as(x)
            node_scores = self.cos(x, word)
            scores += node_scores
            # Save the highest scoring node for each word
            max_node = torch.argmax(node_scores)
            max_nodes.append(max_node)
        
        x = scores.view(-1,1)

        return x, max_nodes 

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
        
        return scores, max_nodes 
    
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
    

# Prepare data for tensor representation
def indices_from_sentence(lang, sentence):
    return [lang.index(word) for word in sentence]

def tensor_from_sentence(lang, sentence):
    indices = indices_from_sentence(lang, sentence)
    indices.append(lang.index('<EOS>'))
    return torch.tensor(indices, dtype=torch.long).view(-1,1)

def tensors_from_pair(input_lang, output_lang, pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

MAX_LENGTH = 10

# Sourced from Pytorch Tutorial: Seq2Seq Translation Tutorial
# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class ActionPredictor(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(ActionPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        output = F.relu(self.linear1(input))
        output = self.dropout(output)
        output = F.log_softmax(self.linear2(output), dim=0)
        return output

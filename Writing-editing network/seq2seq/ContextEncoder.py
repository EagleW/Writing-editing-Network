import torch.nn as nn
import torch

class ContextEncoder(nn.Module):
    def __init__(self, contextual_dim, number_of_contexts, word_embedding_dim):
        super(ContextEncoder, self).__init__()
        self.embedding = nn.Embedding(number_of_contexts, contextual_dim)
        self.transform = nn.Linear(contextual_dim, word_embedding_dim)
        self.tanh = nn.Tanh()

    def forward(self, topics):
        embedded = self.embedding(topics)
        transformed = self.transform(embedded)
        add = torch.sum(transformed, dim=1)
        non_linearity = self.tanh(add)
        return non_linearity
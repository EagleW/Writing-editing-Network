import torch.nn as nn
import torch

class ContextEncoder(nn.Module):
    def __init(self, contextual_dim, number_of_contexts, word_embedding_dim):
        self.embedding = nn.Embedding(number_of_contexts, contextual_dim)
        self.transform = nn.Linear(contextual_dim, word_embedding_dim)
        self.tanh = nn.tanh()

    def forward(self, topics):
        embedded = self.embedding(topics)
        transformed = self.transform(embedded)
        add = torch.sum(transformed, dim=0)
        non_linearity = self.tanh(add)
        return non_linearity
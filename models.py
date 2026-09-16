import torch
import torch.nn as nn

class ASNN(nn.Module):
    def __init__(self, input_dim, dropout, is_classification, tau=0.1):
        super(ASNN, self).__init__()
        self.is_classification = is_classification
        self.tau = tau
        self.embedding_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.head = nn.Linear(64, 1)
        self.memory_embeddings = None
        self.memory_labels = None
        self.eval_mode_memory_active = False
        
    def forward(self, x):
        emb = self.embedding_net(x)
        out = self.head(emb)
        
        if self.is_classification:
            # return raw logits here, nn.BCEWithLogitsLoss will handle it during training
            base_pred = out
        else:
            base_pred = out
            
        if self.eval_mode_memory_active and self.memory_embeddings is not None and not self.training:
            emb_norm = torch.nn.functional.normalize(emb, p=2, dim=1)
            mem_norm = torch.nn.functional.normalize(self.memory_embeddings, p=2, dim=1)
            sim = torch.mm(emb_norm, mem_norm.t()) # shape: (batch, num_memory)
            weights = torch.nn.functional.softmax(sim / self.tau, dim=1)
            mem_pred = torch.mm(weights, self.memory_labels)
            
            # Blend
            if self.is_classification:
                base_prob = torch.sigmoid(base_pred)
                blended = (base_prob + mem_pred) / 2.0
                return blended # returning blended probabilities
            else:
                blended = (base_pred + mem_pred) / 2.0
                return blended
                
        return base_pred

    def populate_memory(self, train_x, train_y):
        self.eval()
        with torch.no_grad():
            self.memory_embeddings = self.embedding_net(train_x)
            self.memory_labels = train_y
        self.eval_mode_memory_active = True

import torch
import torch.nn as nn
import torchvision.models as models

class TriNetSiamese(nn.Module):
    def __init__(self, embedding_dim=128):
        super(TriNetSiamese, self).__init__()
        
        # Load ResNet18 pretrained model
        resnet = models.resnet18(pretrained=True)
        
        # Remove the fully connected layer (we keep layers until AvgPool)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Removes last FC layer
        
        # Fully connected layer to get embeddings
        self.fc = nn.Sequential(
            nn.Linear(512, embedding_dim),  # ResNet18 has 512-dim output
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)  # Another layer for projection
        )
        
        # Normalization layer
        self.norm = nn.functional.normalize

    def forward_one(self, x):
        x = self.backbone(x)  # Extract ResNet features
        x = torch.flatten(x, start_dim=1)  # Flatten (batch_size, 512)
        x = self.fc(x)  # Fully connected projection
        x = self.norm(x, p=2, dim=1)  # L2-normalization
        return x
    
    def forward(self, anchor, positive, negative):
        """ Forwarding a triplet """
        anchor_emb = self.forward_one(anchor)
        positive_emb = self.forward_one(positive)
        negative_emb = self.forward_one(negative)
        
        return anchor_emb, positive_emb, negative_emb


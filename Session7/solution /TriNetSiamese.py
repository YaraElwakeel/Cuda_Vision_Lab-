import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models

class TriNetSiamese(nn.Module):
    def __init__(self, embedding_dim=128,input_size=(224, 224)):
        super(TriNetSiamese, self).__init__()
        
        # Load ResNet18 pretrained model
        resnet = models.resnet18(pretrained=True)
        
        # Remove the fully connected layer (we keep layers until AvgPool)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  
        
        # Compute feature map size Resnet_18 
        self.feature_map_size = self._compute_feature_map_size(input_size)
        
        # Fully connected layer to get embeddings
        self.fc = nn.Linear(self.feature_map_size, embedding_dim)  
 
        # Normalization layer
        self.norm = nn.functional.normalize
    
    def _compute_feature_map_size(self, input_size):
        """ Computes the feature map size after passing through the backbone """
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, *input_size)  
            x = self.backbone(dummy_input)  
            return x.numel()  

    def forward_one(self, x):
        x = self.backbone(x)  
        x = torch.flatten(x, start_dim=1)  
        x = self.fc(x) 
        x = self.norm(x, p=2, dim=1) 
        return x
    
    def forward(self, anchor, positive, negative):
        """ Forwarding a triplet """
        anchor_emb = self.forward_one(anchor)
        positive_emb = self.forward_one(positive)
        negative_emb = self.forward_one(negative)
        
        return anchor_emb, positive_emb, negative_emb

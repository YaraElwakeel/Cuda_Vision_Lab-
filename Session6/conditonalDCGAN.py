import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable





class ConvBlock(nn.Module):
    """
    Simple convolutional block: Conv + Norm + Act + Dropout
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, add_norm=True, activation="ReLU", dropout=None):
        """ Module Initializer """
        super().__init__()
        assert activation in ["ReLU", "LeakyReLU", "Sigmoid", "Tanh", None]
        padding = kernel_size // 2
        
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride))
        if add_norm:
            block.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            nonlinearity = getattr(nn, activation, nn.ReLU)()
            if isinstance(nonlinearity, nn.LeakyReLU):
                nonlinearity.negative_slope = 0.2
            block.append(nonlinearity)
            
        if dropout is not None:
            block.append(nn.Dropout(dropout))
            
        self.block =  nn.Sequential(*block)

    def forward(self, x):
        """ Forward pass """
        y = self.block(x)
        return y


class ConvTransposeBlock(nn.Module):
    """
    Simple convolutional block: ConvTranspose + Norm + Act + Dropout
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, add_norm=True, activation="ReLU", dropout=None):
        """ Module Initializer """
        super().__init__()
        assert activation in ["ReLU", "LeakyReLU", "Tanh", None]
        padding = kernel_size // 2
        
        block = []
        block.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=stride))
        if add_norm:
            block.append(nn.BatchNorm2d(out_channels))
        if activation is not None:
            nonlinearity = getattr(nn, activation, nn.ReLU)()
            if isinstance(nonlinearity, nn.LeakyReLU):
                nonlinearity.negative_slope = 0.2
            block.append(nonlinearity)
        if dropout is not None:
            block.append(nn.Dropout(dropout))
            
        self.block =  nn.Sequential(*block)

    def forward(self, x):
        """ Forward pass """
        y = self.block(x)
        return y


class Reshape(nn.Module):
    """ Reshaping a vector in a given shape """
    
    def __init__(self, shape):
        """ """
        super().__init__()
        self.shape = shape
        
    def forward(self, x):
        """ """
        B, N = x.shape
        x = x.view(B, N, 1, 1)
        y = x.repeat(1, 1, *self.shape)
        return y




class Generator(nn.Module):
    """
    A fully convolutional generator using ReLU activations. 
    Takes as input a latent vector and a class label, and outputs a fake sample.
       (B, latent_dim + num_classes, 1, 1)  --> (B, num_channels, 32, 32)
    """
    def __init__(self, latent_dim=128, num_channels=1, base_channels=32, num_classes=3):
        """ Model initializer """
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        layers = []
        for i in range(4):
            layers.append(
                ConvTransposeBlock(
                        in_channels=(latent_dim + num_classes) if i == 0 else base_channels * 2 ** (3-i+1),
                        out_channels=base_channels * 2 ** (3-i),
                        kernel_size=4,
                        stride=1 if i == 0 else 2,
                        add_norm=True,
                        activation="ReLU"
                    )
                )
        layers.append(
            ConvTransposeBlock(
                    in_channels=base_channels,
                    out_channels=num_channels,
                    kernel_size=4,
                    stride=2,
                    add_norm=False,
                    activation="Tanh"
                )
            )
        
        self.model = nn.Sequential(*layers)

    def forward(self, x, labels):
        """ Forward pass through generator 
        Args:
            x: Latent vector of shape (B, latent_dim, 1, 1)
            labels: Class labels of shape (B,)
        Returns:
            Generated samples of shape (B, num_channels, 32, 32)
        """
        # Embed class labels and concatenate with latent vector
        label_embeddings = self.label_embedding(labels).view(labels.size(0), self.num_classes, 1, 1)
        x = torch.cat([x, label_embeddings], dim=1)
        y = self.model(x)
        return y



class Discriminator(nn.Module):
    """ A fully convolutional discriminator using LeakyReLU activations. 
    Takes as input either a real or fake sample and predicts its authenticity and class.
       (B, num_channels + num_classes, 32, 32)  -->  (B, 1, 1, 1)
    """
    def __init__(self, in_channels=1, out_dim=1, base_channels=32, dropout=0.3, num_classes=3):
        """ Module initializer """
        super().__init__()

        self.num_classes = num_classes

        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, 32 * 32)

        layers = []
        for i in range(4):
            layers.append(
                ConvBlock(
                        in_channels=(in_channels + 1) if i == 0 else base_channels * 2 ** i,
                        out_channels=base_channels * 2 ** (i + 1),
                        kernel_size=4,
                        add_norm=True,
                        activation="LeakyReLU",
                        dropout=dropout,
                        stride=2
                    )
                )
        layers.append(
                ConvBlock(
                        in_channels=base_channels * 16,
                        out_channels=out_dim,
                        kernel_size=4,
                        stride=4,
                        add_norm=False,
                        activation="Sigmoid"
                    )
                )

        self.model = nn.Sequential(*layers)

    def forward(self, x, labels):
        """ Forward pass 
        Args:
            x: Input image of shape (B, num_channels, 32, 32)
            labels: Class labels of shape (B,)
        Returns:
            Discriminator output of shape (B, 1, 1, 1)
        """
        # Embed class labels and reshape to match input dimensions
        label_embeddings = self.label_embedding(labels.unsqueeze(1)).view(labels.size(0), 1, 32, 32)
        # Concatenate labels with the input image
        x = torch.cat([x, label_embeddings], dim=1)
        y = self.model(x)
        return y

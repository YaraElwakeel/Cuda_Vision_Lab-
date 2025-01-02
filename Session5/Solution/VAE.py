import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import numpy as np

def get_act(act_name):
    """ Gettign activation given name """
    assert act_name in ["ReLU", "Sigmoid", "Tanh"]
    activation = getattr(nn, act_name)
    return activation()

def reparameterize( mu, log_var):
    """Reparameterization trick to sample z ~ N(mu, sigma^2)"""
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)  # random sampling happens here
    z = mu + std * eps
    return z


class VanillaVAE(nn.Module):
    def __init__(self, in_size=(1, 32, 32), sizes=[1024, 128, 10], act="ReLU"):
        """ Model initlaizer """
        assert np.prod(in_size) == sizes[0]
        super().__init__()
        
        self.in_size = in_size
        self.sizes = sizes
        self.activation = get_act(act) 
        
        self.encoder = self._make_encoder()
        self.decoder = self._make_decoder()
        # learned mean of the distribution of the latent space
        self.fc_mu = nn.Linear(sizes[-2], sizes[-1]) 
        self.fc_sigma = nn.Linear(sizes[-2], sizes[-1]) 
        return
        
    def _make_encoder(self):
        """ Defining encoder """
        layers = [nn.Flatten()]
        
        # adding fc+act+drop for each layer
        for i in range(len(self.sizes)-2):
            layers.append( nn.Linear(in_features=self.sizes[i], out_features=self.sizes[i+1]) )
            layers.append( self.activation )
                
        # replacing last act and dropout with sigmoid
        encoder = nn.Sequential(*layers)
        return encoder
    
    def _make_decoder(self):
        """ Defining decoder """
        layers = []
        
        # adding fc+act+drop for each layer
        for i in range(1, len(self.sizes)):
            layers.append( nn.Linear(in_features=self.sizes[-i], out_features=self.sizes[-i-1]) )
            layers.append( self.activation )
                
        # replacing last act and dropout with sigmoid
        layers = layers[:-1] + [nn.Sigmoid()]
        decoder = nn.Sequential(*layers)
        return decoder
    
    def forward(self, x):
        """ Forward pass """
        # encoding and computng statistics
        x_enc = self.encoder(x)

        mu = self.fc_mu(x_enc)
        log_var = self.fc_sigma(x_enc)
        
        # reparametrization trick
        z = reparameterize(mu, log_var)
        
        # decoding
        x_hat_flat = self.decoder(z)
        x_hat = x_hat_flat.view(-1, *self.in_size)
        
        return x_hat, (z, mu, log_var)
    
class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=10, hidden_layers=[ 64, 128, 256,512]):
        """
        A Convolutional Variational Autoencoder (VAE)  configuration.

        Args:
            in_channels (int): Number of input channels 
            latent_dim (int): Dimensionality of the latent space.
            hidden_layers (list):  number of filters per layer.
        """
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers

        # Encoder: Convolutional Layers 
        encoder_layers = []
        input_channels = in_channels
        for filters in hidden_layers:
            encoder_layers.append(nn.Conv2d(input_channels, filters, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            
            input_channels = filters
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate the spatial dimensions after encoding
        self.final_feature_dim = hidden_layers[-1] * 4 * 4  

        # Fully Connected Layers for Latent Space
        self.fc_mu = nn.Linear(self.final_feature_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.final_feature_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.final_feature_dim)

        # Decoder:  Transposed Convolutional Layers 
        decoder_layers = []
        reversed_layers = hidden_layers[::-1]
        for i in range(len(reversed_layers) - 1):
            decoder_layers.append(nn.ConvTranspose2d(reversed_layers[i], reversed_layers[i + 1], kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU())

        decoder_layers.append(nn.ConvTranspose2d(reversed_layers[-1], in_channels, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.Sigmoid()) 
        self.decoder_net = nn.Sequential(*decoder_layers)

    def decode(self, z):
        """Decode from latent space."""
        batch_size = z.size(0)
        dec_input = self.fc_decode(z).view(batch_size, self.hidden_layers[-1], 4, 4)
        return self.decoder_net(dec_input)

    def forward(self, x):
        # Encode
        batch_size = x.size(0)
        enc_out = self.encoder(x)  
        enc_out_flat = enc_out.view(batch_size, -1)  # Flatten

        # Compute latent space statistics
        mu = self.fc_mu(enc_out_flat)
        log_var = self.fc_log_var(enc_out_flat)

        # Reparameterize 
        z = reparameterize(mu, log_var)

        # Decode
        x_hat = self.decode(z)

        return x_hat, (z, mu, log_var)
    
class ConditionalConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=10, hidden_layers=[64, 128, 256, 512], num_conditions=3):
        """
        A Conditional Convolutional Variational Autoencoder (VAE).

        Args:
            in_channels (int): Number of input channels 
            latent_dim (int): Dimensionality of the latent space.
            hidden_layers (list):  number of filters per layer.
            num_conditions (int): number of classes.
        """
        super(ConditionalConvVAE, self).__init__()

        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.num_conditions = num_conditions

        # Encoder: Convolutional Layers
        encoder_layers = []
        input_channels = in_channels + num_conditions
        for filters in hidden_layers:
            encoder_layers.append(nn.Conv2d(input_channels, filters, kernel_size=4, stride=2, padding=1))
            encoder_layers.append(nn.ReLU())
            input_channels = filters
        self.encoder = nn.Sequential(*encoder_layers)


        self.final_feature_dim = hidden_layers[-1] * 4 * 4

        # Fully Connected Layers for Latent Space
        self.fc_mu = nn.Linear(self.final_feature_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.final_feature_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + num_conditions, self.final_feature_dim)

        # Decoder: Transposed Convolutional Layers
        decoder_layers = []
        reversed_layers = hidden_layers[::-1]
        for i in range(len(reversed_layers) - 1):
            decoder_layers.append(nn.ConvTranspose2d(reversed_layers[i], reversed_layers[i + 1], kernel_size=4, stride=2, padding=1))
            decoder_layers.append(nn.ReLU())

        decoder_layers.append(nn.ConvTranspose2d(reversed_layers[-1], in_channels, kernel_size=4, stride=2, padding=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder_net = nn.Sequential(*decoder_layers)

    def add_condition(self, x, y):
        """Concatenate condition to the input."""
        # Reshape y to [batch_size, num_conditions, 1, 1]
        condition = y.view(y.size(0), y.size(1), 1, 1).repeat(1, 1, x.size(2), x.size(3))
        return torch.cat([x, condition], dim=1)


    def decode(self, z, y):
        """Decode from latent space with condition."""
        batch_size = z.size(0)
        z_cond = torch.cat([z, y], dim=1)  
        dec_input = self.fc_decode(z_cond).view(batch_size, self.hidden_layers[-1], 4, 4)
        return self.decoder_net(dec_input)

    def forward(self, x, y):
        # Add condition to input
        x_cond = self.add_condition(x, y)

        # Encode
        batch_size = x_cond.size(0)
        enc_out = self.encoder(x_cond)
        enc_out_flat = enc_out.view(batch_size, -1)  

        # Compute latent space statistics
        mu = self.fc_mu(enc_out_flat)
        log_var = self.fc_log_var(enc_out_flat)

        # Reparameterize
        z = reparameterize(mu, log_var)

        # Decode with condition
        x_hat = self.decode(z, y)

        return x_hat, (z, mu, log_var)
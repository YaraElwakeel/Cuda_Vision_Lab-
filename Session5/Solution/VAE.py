import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_act(act_name):
    """ Gettign activation given name """
    assert act_name in ["ReLU", "Sigmoid", "Tanh"]
    activation = getattr(nn, act_name)
    return activation()

def reparameterize( mu, log_var):
    """ Reparametrization trick"""
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
        self.fc_mu = nn.Linear(sizes[-2], sizes[-1]) #--> as 128 is the output of the encoder and 10 i sthe size of the latent space ,
        self.fc_sigma = nn.Linear(sizes[-2], sizes[-1]) #--> same thing but this represents the log_2(variance^2) of the distribution 
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
    def __init__(self, in_channels=1, latent_dim=10, encoder_channels=[32, 64, 128]):
        super(ConvVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.encoder_channels = encoder_channels

        # Encoder: Convolutional Layers
        encoder_layers = []
        input_channels = in_channels

        for output_channels in encoder_channels:
            encoder_layers.append(
                nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
            )
            encoder_layers.append(nn.ReLU())
            input_channels = output_channels
        self.encoder = nn.Sequential(*encoder_layers)

        # Fully Connected Layers for Mean and Log Variance
        self.fc_mu = nn.Linear(encoder_channels[-1] * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(encoder_channels[-1] * 4 * 4, latent_dim)

        # Fully Connected Layer for Latent to Decoder Input
        self.fc_decode = nn.Linear(latent_dim, encoder_channels[-1] * 4 * 4)

        # Decoder: Transposed Convolutional Layers
        decoder_layers = []
        decoder_channels = list(reversed(encoder_channels))
        input_channels = decoder_channels[0]
        for output_channels in decoder_channels[1:]:
            decoder_layers.append(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)
            )
            decoder_layers.append(nn.ReLU())
            input_channels = output_channels
        decoder_layers.append(
            nn.ConvTranspose2d(input_channels, in_channels, kernel_size=4, stride=2, padding=1)
        )
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Forward pass"""
        # Encode
        batch_size = x.size(0)
        enc_out = self.encoder(x)  # Shape: [B, C, H, W] based on encoder output
        enc_out_flat = enc_out.view(batch_size, -1)  # Flatten

        # Compute latent space statistics
        mu = self.fc_mu(enc_out_flat)
        log_var = self.fc_log_var(enc_out_flat)

        # Reparameterize to sample z
        z = reparameterize(mu, log_var)

        # Decode
        dec_input = self.fc_decode(z).view(batch_size, self.encoder_channels[-1], 4, 4)  # Reshape to match decoder input
        x_hat = self.decoder(dec_input)

        return x_hat, (z, mu, log_var)

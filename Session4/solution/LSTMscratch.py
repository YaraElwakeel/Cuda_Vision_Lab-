import torch
import torch.nn as nn
import numpy as np


class LSTMCell_scratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell_scratch, self).__init__()
        """
        Args:
            input_size: The number of input features.
            hidden_size: The number of features in the hidden state h_t and cell state c_t.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

         # Input gates
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x, hidden):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).
            states: Tuple containing the previous hidden state (h_t-1) and cell state (c_t-1).
                    Each of shape (batch_size, hidden_size).

        Returns:
            h_t: Updated hidden state of shape (batch_size, hidden_size).
            c_t: Updated cell state of shape (batch_size, hidden_size).
        """
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=1)

        f_t = torch.sigmoid(self.forget_gate(combined))  # Forget gate
        i_t = torch.sigmoid(self.input_gate(combined))   # Input gate
        g_t = torch.tanh(self.cell_gate(combined))       # Candidate cell state
        o_t = torch.sigmoid(self.output_gate(combined))  # Output gate


        c_next = f_t * c_prev + i_t * g_t
        h_next = o_t * torch.tanh(c_next)

        return h_next, c_next

class LSTMscratch(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_of_layers, device):
        super(LSTMscratch, self).__init__()
        self.lstms = []
        self.mode = "zeros"
        self.num_layers = number_of_layers
        self.hidden_dim = hidden_dim

        # for embedding rows into vector representations
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, input_dim, 3, 1, 1),
                nn.AdaptiveAvgPool2d((1, 1)))
        

        for layers in range(number_of_layers):
            self.lstms.append(LSTMCell_scratch(input_size=input_dim if layers == 0 else hidden_dim, hidden_size=hidden_dim).to(device))

        # FC-classifier
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=6)
        return
    

    def forward(self, x):
        b_size, num_frames, n_channels, n_rows, n_cols = x.shape

        h, c = self.init_state(b_size=b_size, device=x.device)

        # embedding rows
        x = x.view(b_size * num_frames, n_channels, n_rows, n_cols)
        embeddings = self.encoder(x)
        embeddings = embeddings.reshape(b_size, num_frames, -1)
        
        # iterating over sequence length
        lstm_out = []
        for i in range(embeddings.shape[1]):
            lstm_input = embeddings[:, i, :]
            # iterating over LSTM Cells
            for j, lstm_cell in enumerate(self.lstms):
                h[j], c[j] = lstm_cell(lstm_input, (h[j], c[j]))
                lstm_input = h[j]
            lstm_out.append(lstm_input)
        lstm_out = torch.stack(lstm_out, dim=1)
            
        # classifying
        y = self.classifier(lstm_out[:, -1, :])  # feeding only output at last layer
        
        return y

    def init_state(self, b_size, device):
        """ Initializing hidden and cell state """
        if(self.mode == "zeros"):
            h = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
            c = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
        elif(self.mode == "random"):
            h = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
            c = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
        return h, c

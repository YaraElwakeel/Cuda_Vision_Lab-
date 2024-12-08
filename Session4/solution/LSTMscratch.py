import torch
import torch.nn as nn

class LSTMCell_scratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell_scratch, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gates
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.cell_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden):
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
    def __init__(self, input_dim, hidden_dim, number_of_layers, device, use_custom=True, dropout=0.2):
        """
        Args:
            input_dim: The number of input features.
            hidden_dim: The number of hidden units.
            number_of_layers: The number of stacked LSTM layers.
            device: Device (CPU/GPU) to run the model.
            use_custom: If True, use the custom LSTM cell; otherwise, use PyTorch's nn.LSTMCell.
            dropout: Dropout probability for regularization.
        """
        super(LSTMscratch, self).__init__()
        self.mode = "zeros"
        self.num_layers = number_of_layers
        self.hidden_dim = hidden_dim
        self.use_custom = use_custom
        self.dropout_prob = dropout

        # Encoder for embedding rows into vector representations
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, input_dim, 3, 1, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Choose between custom or PyTorch LSTM cells
        self.lstms = []
        for layer in range(number_of_layers):
            if use_custom:
                self.lstms.append(
                    LSTMCell_scratch(input_size=input_dim if layer == 0 else hidden_dim, hidden_size=hidden_dim).to(device)
                )
            else:
                self.lstms.append(
                    nn.LSTMCell(input_size=input_dim if layer == 0 else hidden_dim, hidden_size=hidden_dim).to(device)
                )

        # Add dropout layers
        self.dropout = nn.Dropout(p=dropout)

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout),  # Dropout before the next linear layer
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=6)
        )

    def forward(self, x):
        b_size, num_frames, n_channels, n_rows, n_cols = x.shape

        # Initialize hidden and cell states
        h, c = self.init_state(b_size=b_size, device=x.device)

        # Embed rows
        x = x.view(b_size * num_frames, n_channels, n_rows, n_cols)
        embeddings = self.encoder(x)
        embeddings = embeddings.reshape(b_size, num_frames, -1)

        # Apply dropout to embeddings
        embeddings = self.dropout(embeddings)

        # Iterate over sequence length
        lstm_out = []
        for i in range(embeddings.shape[1]):
            lstm_input = embeddings[:, i, :]
            for j, lstm_cell in enumerate(self.lstms):
                h[j], c[j] = lstm_cell(lstm_input, (h[j], c[j]))
                lstm_input = h[j]
            lstm_out.append(lstm_input)
        lstm_out = torch.stack(lstm_out, dim=1)

        # Apply dropout to LSTM output
        lstm_out = self.dropout(lstm_out)

        # Classify based on the last time step of the final layer
        y = self.classifier(lstm_out[:, -1, :])
        return y

    def init_state(self, b_size, device):
        """ Initialize hidden and cell states """
        h = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
        c = [torch.zeros(b_size, self.hidden_dim).to(device) for _ in range(self.num_layers)]
        return h, c

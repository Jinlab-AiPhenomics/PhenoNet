import torch.nn as nn

class lstm_network(nn.Module):
    def __init__(self):
        super(lstm_network, self).__init__()
        self.recurrent_layer = nn.LSTM(168, 64, 2, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(64, 5)

    def forward(self, input1, h_t_1=None, c_t_1=None):
        rnn_outputs, (hn, cn) = self.recurrent_layer(input1)
        x = self.fc(rnn_outputs[:, -1, :])
        return x

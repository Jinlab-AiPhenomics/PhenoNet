from arch.lstm import lstm_network
from arch.phenovit import phenovit_network

class PhenoNet():
    def __init__(self):
        super(PhenoNet, self).__init__()
        self.phenovit = phenovit_network()
        self.lstm_network = lstm_network()

    def print_architecture(self):
        print(self.phenovit)
        print(self.lstm_network)
    
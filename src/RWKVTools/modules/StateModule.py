# create statefull torch.nn module
import torch
import torch.nn as nn
class StateModule(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.initState(*args)
    def initState(self, *args):
        self.state = torch.zeros(*args)

    def setState(self, state):
        self.state = state.detach().clone()

    def getState(self):
        return self.state.clone()
    
    def resetState(self):
        self.initState(*self.state.shape)
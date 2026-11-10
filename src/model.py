import torch 
import torch.nn as nn 

class FraudDetectionModel(nn.Module): 

    def __init__(self):
        super(FraudDetectionModel,self).__init__()

        self.network = nn.Sequential(
            nn.Linear(30,64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,1)
        )
    def forward(self,x):
        return self.network(x)
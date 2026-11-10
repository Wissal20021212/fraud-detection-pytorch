import torch
from model import FraudDetectionModel

model = FraudDetectionModel()

# Simulate a batch of 5 transactions
x = torch.randn(5, 30)

output = model(x)

print("Output shape:", output.shape)
print(output)
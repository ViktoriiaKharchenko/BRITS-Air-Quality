import torch
import torch.nn as nn
import torch.optim as optim


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# Ensure CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available")
else:
    device = torch.device('cpu')
    print("CUDA is not available")

# Move model to GPU
model = SimpleModel().to(device)

# Define a simple optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Create dummy data and move it to GPU
inputs = torch.randn(64, 10).to(device)
labels = torch.randn(64, 1).to(device)

# Before running the model
print("Memory allocated before model runs:", torch.cuda.memory_allocated())
print("Memory reserved before model runs:", torch.cuda.memory_reserved())

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, labels)

# Backward pass
loss.backward()

# Optimize
optimizer.step()

# After running the model
print("Memory allocated after model runs:", torch.cuda.memory_allocated())
print("Memory reserved after model runs:", torch.cuda.memory_reserved())

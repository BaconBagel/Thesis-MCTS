# Importing the necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Defining some hyperparameters
batch_size = 10  # The number of samples in each batch
seq_len = 10  # The number of timesteps in each sequence
input_size = 100  # The size of the grid (assuming it is square)
output_size = 100  # The size of the output grid (assuming it is square)
hidden_size = 256  # The size of the hidden state in the recurrent layer
num_layers = 2  # The number of layers in the recurrent layer
num_epochs = 5 # The number of epochs to train the model
learning_rate = 0.001  # The learning rate for the optimizer

# Creating some dummy data for demonstration purposes
# You should replace this with your actual data
# Assuming scores is a numpy array of shape (5, 100, 100, 100)
num_samples = 100
reward_array = np.load("reward_array.npy")[:num_samples]
reward_array = np.where(reward_array > 1000, 1000, reward_array)
reward_array = (reward_array - np.min(reward_array)) / (np.max(reward_array) - np.min(reward_array))

num_sets = 1 # Since you have only one type of data
scores = reward_array.reshape(num_samples, num_sets, 100, 100, 100) # Reshaping the array to add the num_sets dimension
print(np.mean(scores))
data = torch.from_numpy(scores).float() # Converting the numpy array to a torch tensor

# Assuming data is a tensor of shape (num_samples, num_sets, 100, 100, 100)
inputs = data[:, :, :-1, :, :] # A tensor of shape (num_samples, num_sets, 99, 100, 100) containing the input sequences
targets = data[:, :, (torch.arange(99) + 1) % 100, :, :] # A tensor of shape (num_samples, num_sets, 99, 100, 100) containing the target grids

print(np.count_nonzero(inputs), np.count_nonzero(targets))
# Defining the model class
class GridPredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(GridPredictor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Defining the recurrent layer
        self.rnn = nn.LSTM(input_size * input_size, hidden_size, num_layers, batch_first=True)

        # Defining the output layer
        self.fc = nn.Linear(hidden_size, output_size * output_size)

    def forward(self, x):
        # x is a tensor of shape (batch_size, num_sets, seq_len, input_size, input_size)

        # Reshaping x to (batch_size * num_sets, seq_len, input_size * input_size)
        x = x.view(-1, x.size(2), x.size(3) * x.size(4))

        # Passing x through the recurrent layer
        # output is a tensor of shape (batch_size * num_sets, seq_len, hidden_size)
        # hidden is a tuple of two tensors of shape (num_layers, batch_size * num_sets, hidden_size) representing the hidden state and the cell state at the last timestep
        output, hidden = self.rnn(x)

        # Taking the output at the last timestep
        # output is a tensor of shape (batch_size * num_sets, hidden_size)
        output = output[:, -1, :]

        # Passing output through the output layer
        # output is a tensor of shape (batch_size * num_sets, output_size * output_size)
        output = self.fc(output)

        # Reshaping output to (batch_size, num_sets, output_size, output_size)
        output = output.view(-1, x.size(0) // batch_size, self.output_size, self.output_size)

        return output


# Creating an instance of the model
model = GridPredictor(input_size, output_size, hidden_size, num_layers)

# Defining the loss function and the optimizer
criterion = nn.MSELoss()  # Mean squared error loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer

model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)

for epoch in range(num_epochs):
    # Shuffling the data
    perm = torch.randperm(num_samples)
    inputs = inputs[perm]
    targets = targets[perm]

    # Looping over batches
    for i in range(0, num_samples, batch_size):
        # Getting a batch of data
        input_batch = inputs[i:i + batch_size].to(device)
        target_batch = targets[i:i + batch_size].to(device)

        # Zeroing the gradients
        optimizer.zero_grad()

        # Passing the input batch through the model
        output_batch = model(input_batch)  # A tensor of shape (batch_size , num_sets , output_size , output_size )

        # Computing the loss
        loss = criterion(output_batch, target_batch)  # A scalar

        # Backpropagating the loss
        loss.backward()

        # Updating the parameters
        optimizer.step()

    # Printing the epoch and the loss
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

reward_array = np.load("reward_array_means.npy")[:num_samples]
num_samples = len(reward_array)
reward_array = np.where(reward_array > 1000, 1000, reward_array)
reward_array = (reward_array - np.min(reward_array)) / (np.max(reward_array) - np.min(reward_array))

num_sets = 1 # Since you have only one type of data
scores = reward_array.reshape(num_samples, num_sets, 100, 100, 100) # Reshaping the array to add the num_sets dimension
print(np.mean(scores))
data = torch.from_numpy(scores).float() # Converting the numpy array to a torch tensor


# Assuming data is a tensor of shape (num_samples, num_sets, 100, 100, 100)
inputs = data[:, :, :-1, :, :] # A tensor of shape (num_samples, num_sets, 99, 100, 100) containing the input sequences
targets = data[:, :, (torch.arange(99) + 1) % 100, :, :] # A tensor of shape (num_samples, num_sets, 99, 100, 100) containing the target grids
for i in range(0, num_samples, batch_size):
    # Getting a batch of data
    input_batch = inputs[i:i + batch_size].to(device)
    target_batch = targets[i:i + batch_size].to(device)

    # Passing the input batch through the model
    output_batch = model(input_batch)  # A tensor of shape (batch_size , num_sets , output_size , output_size )
    print(torch.mean(output_batch),torch.mean(target_batch))
    print(torch.nonzero(target_batch),torch.nonzero(output_batch))
    # Computing the loss
    loss = criterion(output_batch, target_batch)  # A scalar
    print(loss.item())
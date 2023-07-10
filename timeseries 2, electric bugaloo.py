# Import pytorch and numpy
import torch
import torch.nn as nn
import numpy as np


# Define the network class
class GridPredictor(nn.Module):
    def __init__(self, cnn_filters, rnn_hidden, grid_size):
        super(GridPredictor, self).__init__()
        # Define the CNN layer to process the current state of the grid
        # It has cnn_filters number of 3x3 filters with padding 1 and ReLU activation
        self.cnn = nn.Conv2d(1, cnn_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Define the RNN layer to process the temporal sequence of grid states
        # It has rnn_hidden number of hidden units and uses LSTM cells
        self.rnn = nn.LSTM(cnn_filters * grid_size * grid_size, rnn_hidden, batch_first=True)
        # Define the output layer to predict the score for the next timestep at a given grid position
        # It has a linear transformation followed by a sigmoid activation
        self.linear = nn.Linear(rnn_hidden, grid_size * grid_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is a tensor of shape (batch_size, seq_len + 1, grid_size, grid_size)
        # batch_size is the number of samples in a batch
        # seq_len is the number of timesteps in a sequence
        # grid_size is the size of the grid in both x and y coordinates
        batch_size, seq_len_plus_one, grid_size, _ = x.shape
        # Take only the first seq_len timesteps as input for the CNN and RNN layers
        # The input shape is (batch_size, seq_len, grid_size, grid_size)
        x_input = x[:, :-1, :, :]
        # Apply the CNN layer to each timestep of the sequence
        # The output shape is (batch_size, seq_len, cnn_filters, grid_size, grid_size)
        x = self.cnn(x_input.reshape(-1, 1, grid_size, grid_size))
        x = self.relu(x)
        # Reshape the output to feed into the RNN layer
        # The output shape is (batch_size, seq_len, cnn_filters * grid_size * grid_size)
        x = x.view(batch_size, seq_len_plus_one - 1, -1)
        # Apply the RNN layer to the sequence
        # The output shape is (batch_size, seq_len_plus_one - 1 , rnn_hidden)
        x, _ = self.rnn(x)
        # Take the last hidden state of the sequence as the input for the output layer
        # The output shape is (batch_size, rnn_hidden)
        x = x[:, -1, :]
        # Apply the output layer to predict the score for the next timestep at a given grid position
        # The output shape is (batch_size ,grid_size * grid_size)
        x = self.linear(x)
        x = self.sigmoid(x)
        # Reshape the output to match the target shape
        # The output shape is (batch_size ,grid_size ,grid_size )
        x = x.view(batch_size, grid_size, grid_size)

        return x


# Create a random numpy array of size x*100*100*100 as input data
x = np.load("reward_array.npy")
x = np.where(x > 1000, 1000, x)
x = (x - np.min(x)) / (np.max(x) - np.min(x))

# Convert the numpy array to a pytorch tensor
x = torch.from_numpy(x).float()

# Create an instance of the network with suitable hyperparameters
# For example: cnn_filters = 16 ,rnn_hidden = 32 ,grid_size = 100
model = GridPredictor(2, 4, 100)

# Print the model summary
print(model)

# Forward pass the input data through the model and print the output shape
output = model(x)

# Define a loss function and an optimizer
# Use MSE loss to compare the output with the target tensor of shape (batch_size ,grid_size ,grid_size )
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Split the data into training and validation sets
train_size = int(0.8 * x.shape[0])
val_size = x.shape[0] - train_size
train_data, val_data = torch.utils.data.random_split(x, [train_size, val_size])

# Create data loaders to load the data in batches
train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False)

# Train the model for some epochs
epochs = 10
for epoch in range(epochs):
    # Set the model to training mode
    model.train()
    # Initialize the training loss and accuracy
    train_loss = 0.0
    train_acc = 0.0
    # Loop over the training batches
    for x_batch in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass the input batch through the model
        output_batch = model(x_batch)
        # Get the target batch from the last timestep of the input batch
        target_batch = x_batch[:, -1, :, :]
        # Compute the loss and accuracy
        loss = criterion(output_batch, target_batch)
        acc = torch.mean(((output_batch > 0.000131727) == (target_batch > 0.000131727)).float())
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        # Update the training loss and accuracy
        train_loss += loss.item()
        train_acc += acc.item()
    # Compute the average training loss and accuracy over an epoch
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    # Print the training statistics
    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

    # Set the model to evaluation mode
    model.eval()
    # Initialize the validation loss and accuracy
    val_loss = 0.0
    val_acc = 0.0
    # Loop over the validation batches
    with torch.no_grad():
        for x_batch in val_loader:
            # Forward pass the input batch through the model
            output_batch = model(x_batch)
            # Get the target batch from the last timestep of the input batch
            target_batch = x_batch[:, -1, :, :]
            # Compute the loss and accuracy
            loss = criterion(output_batch, target_batch)
            acc = torch.mean(((output_batch > 0.5) == (target_batch > 0.5)).float())
            # Update the validation loss and accuracy
            val_loss += loss.item()
            val_acc += acc.item()
    # Compute the average validation loss and accuracy over an epoch
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    # Print the validation statistics
    print(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

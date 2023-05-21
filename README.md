# Different-NNs-on-Lorenz-Equations

### Table of Contents
[Abstract](#Abstract)
<a name="Abstract"/>

[Sec. I. Introduction and Overview](#sec-i-introduction-and-overview)     
<a name="sec-i-introduction-and-overview"/>

[Sec. II. Theoretical Background](#sec-ii-theoretical-background)     
<a name="sec-ii-theoretical-background"/>

[Sec. III. Algorithm Implementation and Development](#sec-iii-algorithm-implementation-and-development)
<a name="sec-iii-algorithm-implementation-and-development"/>

[Sec. IV. Computational Results](#sec-iv-computational-results)
<a name="sec-iv-computational-results"/>

[Sec. V. Summary and Conclusions](#sec-v-summary-and-conclusions)
<a name="sec-v-summary-and-conclusions"/>


### Abstract
In this study, we aim to employ Neural Networks (NNs) for the purpose of predicting future states of the Lorenz equations. Different models such as Feed-Forward Neural Network (FFNN), Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN), and Echo State Network (ESN) are trained to advance the solution from time t to t + ∆t for ρ values of 10, 28, and 40. These models are then evaluated based on their ability to predict future states for ρ = 17 and ρ = 35. The results of these different approaches are compared to gain insights on the performance of each model in handling this complex time-series prediction task.

### Sec. I. Introduction and Overview
#### Introduction:

The Lorenz system of differential equations is a mathematical model for atmospheric convection. The equations describe the rate of change of three variables with respect to time. The Lorenz equations are deterministic, meaning that if we know the current state of the system, we can predict the future state. However, the behavior of the system is highly sensitive to the initial conditions, a phenomenon known as the butterfly effect. The Lorenz system is also nonlinear, which makes it difficult to solve analytically.

In recent years, neural networks have shown their capability to handle a broad range of tasks including the ones that involve nonlinearity and chaos. In particular, different architectures of neural networks such as Feed-Forward, LSTM, RNN, and ESN have shown promising results in handling time-series data, which is often nonlinear and highly dependent on the initial conditions, akin to the Lorenz system.

#### Overview:

In this work, we first provide a detailed explanation of the Lorenz system and the four types of neural networks used. This is followed by a thorough discussion of our methodology for training these networks to solve the Lorenz equations for specific values of ρ. Subsequently, we detail the evaluation methodology of these models based on their performance in predicting future states for other values of ρ.

Then we present our findings, including a comparison of the performance of the different models in handling this task. We conclude with a discussion of the implications of our results and potential future research directions. The ultimate goal is to provide a clear understanding of how different neural network architectures can be used to predict the future states of complex, nonlinear systems like the Lorenz equations.

###  Sec. II. Theoretical Background
#### *The Lorenz Equations*

The Lorenz system is a set of three differential equations originally introduced by Edward Lorenz in 1963. They are used to model atmospheric convection and are often cited as a prime example of how complex, chaotic behavior can arise from non-linear dynamical equations. Here are the equations:

1. **dx/dt = σ(y - x)**
2. **dy/dt = x(ρ - z) - y**
3. **dz/dt = xy - βz**

The system is characterized by three parameters: σ (Prandtl number), ρ (Rayleigh number), and β. When ρ exceeds a critical value (approximately 24.74), the system begins to exhibit chaotic behavior. This means that while the system is deterministic, slight changes to initial conditions can result in vastly different outcomes, making long-term prediction extremely challenging. This sensitivity to initial conditions is known as the butterfly effect.

#### *Neural Networks*

**Feed-Forward Neural Networks (FFNNs)**: FFNNs are the simplest type of artificial neural network. In this type of network, the information moves in only one direction—forward—from the input layer, through the hidden layers, to the output layer. There are no cycles or loops in the network. Although FFNNs can approximate any function, they are not particularly well-suited for time series prediction or for any type of sequence recognition, due to lack of temporal dynamics processing.

**Long Short-Term Memory Networks (LSTMs)**: LSTMs are a type of recurrent neural network capable of learning long-term dependencies in sequence prediction problems. This is a major advantage over traditional feed-forward neural networks and basic RNNs, which do not have this capability. LSTMs contain a structure called a memory cell (which includes an input gate, a neuron with a self-recurrent connection (a connection to itself), a forget gate, and an output gate) that can maintain information in memory for long periods of time.

**Recurrent Neural Networks (RNNs)**: Unlike feed-forward neural networks, RNNs have cyclic connections making them powerful for modeling sequences and lists of inputs/outputs. They utilize their internal state (memory) to process variable length sequences of inputs, which makes them ideal for such tasks. However, they suffer from the vanishing and exploding gradient problems that can make these networks hard to train.

**Echo State Networks (ESNs)**: ESNs are a type of recurrent neural network with a sparsely connected hidden layer (with typically 1% connectivity). The connectivity and weights of hidden neurons are generated randomly and remain unchanged during training. Only the weights of output neurons are updated. This leads to faster and more efficient training.

#### *Implications of Using Neural Networks for Forecasting*

The ability to forecast accurately from nonlinear dynamical systems, like the Lorenz system, is of great interest in many fields, including meteorology, economics, and engineering. Traditional methods often rely on explicit numerical methods which can become unstable or inaccurate when facing chaotic systems or when step sizes are not appropriately small.

Neural networks, with their ability to learn from the underlying data and capture non-linear relationships, offer a promising alternative. However, different types of neural networks offer different advantages and potential pitfalls. Feed-forward networks, for instance, are straightforward to understand and implement, but might struggle with temporal data where past states influence future ones. On the other hand, networks like LSTMs or RNNs, that can use their internal state (memory) to process sequences of inputs, are better suited to these tasks.

In conclusion, understanding these different architectures allows us to make an informed choice about which type of neural network to use, depending on the specific requirements of the problem at hand.

### Sec. III. Algorithm Implementation and Development

We begin by importing the necessary libraries:

```
import torch
import torch.nn as nn
import numpy as np
from scipy import integrate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.autograd as autograd
```

The first part of the code defines the structure of the FFNN model, with a class named `FeedForwardNN`. This model contains three fully connected layers (also known as linear layers), each followed by a ReLU (Rectified Linear Unit) activation function, which is common in deep learning models. This is followed by an output layer, which does not apply any activation function as this is a regression task. The model is initialized with the statement `modelNN = FeedForwardNN()`.

```
class FeedForwardNN(nn.Module):
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(3, 10)  # input layer
        self.fc2 = nn.Linear(10, 10)  # hidden layer 1
        self.fc3 = nn.Linear(10, 10)  # hidden layer 2
        self.fc4 = nn.Linear(10, 3)  # output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # no activation function for the output layer in a regression task
        return x

modelNN = FeedForwardNN()
```

Next, the Lorenz system is defined as a function named `lorenz_deriv`, which takes the current state (`x_y_z`) and time (`t0`) as arguments and returns the derivatives of `x`, `y`, and `z` according to the Lorenz equations.

Subsequent blocks of code set up the initial conditions, time span, and rho values for the Lorenz system. The function `odeint` from the `scipy.integrate` module is used to numerically integrate the Lorenz equations over time, for different rho values. The resulting time series are reshaped and concatenated into arrays suitable for use as inputs and targets for the neural network.

The generated datasets are then normalized using the `StandardScaler` from `sklearn.preprocessing` module. Normalization is an important pre-processing step that makes the training process more stable and efficient.

The code then proceeds to split the dataset into a training set and a validation set using `train_test_split` function from `sklearn.model_selection` module. The split datasets are converted into PyTorch tensors, which can be used directly in the training process.

```
# Define the Lorenz system
def lorenz_deriv(x_y_z, t0, sigma=10, beta=8/3, rho=28):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# Set up time span, initial conditions, and rho values
dt = 0.01
T = 8
t = np.arange(0, T+dt, dt)
np.random.seed(123)
x0 = -15 + 30 * np.random.random((100, 3))
rhos = [10, 28, 40]

# Solve the Lorenz equations
nn_input = []
nn_output = []

for rho in rhos:
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, 8/3, rho)) for x0_j in x0])
    nn_input.append(x_t[:, :-1, :].reshape(-1, 3))  # reshaping the array to the form (samples, features)
    nn_output.append(x_t[:, 1:, :].reshape(-1, 3))

nn_input = np.concatenate(nn_input)
nn_output = np.concatenate(nn_output)

# Normalize the data
scaler_in = StandardScaler()
scaler_out = StandardScaler()
nn_input = scaler_in.fit_transform(nn_input)
nn_output = scaler_out.fit_transform(nn_output)

# Split the dataset into a training set and a validation set
nn_input_train, nn_input_val, nn_output_train, nn_output_val = train_test_split(nn_input, nn_output, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
nn_input_train = torch.from_numpy(nn_input_train).float()
nn_output_train = torch.from_numpy(nn_output_train).float()
nn_input_val = torch.from_numpy(nn_input_val).float()
nn_output_val = torch.from_numpy(nn_output_val).float()
```

The loss function (Mean Squared Error) and the optimizer (Adam) are defined in preparation for the training process. A helper function named `train` is defined to streamline the training process. This function takes the model, optimizer, loss function, inputs, and targets as arguments and trains the model for a specified number of epochs. The function runs the forward pass, computes the loss, performs backpropagation, and updates the weights in each epoch.

The model is then evaluated on the validation set, and the Mean Squared Error of the predictions is calculated and printed.

```
# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(modelNN.parameters(), lr=0.01)

# Define a helper function for the training process
def train(model, optimizer, loss_fn, inputs, targets, n_epochs=100):
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")

# Train the model
train(modelNN, optimizer, loss_fn, nn_input_train, nn_output_train, n_epochs=100)

# Evaluate the model on the validation set
modelNN.eval()
with torch.no_grad():
    predictions = modelNN(nn_input_val)

# Compute the Mean Squared Error of the predictions
mse = loss_fn(predictions, nn_output_val)
print(f"Mean Squared Error (MSE) on the validation set: {mse.item()}")
```

Lastly, the script predicts the dynamics of the Lorenz system for two new rho values (17 and 35) not seen during training. The predictions are evaluated using the Mean Squared Error, providing an indication of how well the model can generalize to unseen conditions.

```
# Solve the Lorenz equations for rho = 17 and rho = 35
test_rhos = [17, 35]
test_input = []
test_output = []

for rho in test_rhos:
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, 8/3, rho)) for x0_j in x0])
    test_input.append(x_t[:, :-1, :].reshape(-1, 3))  
    test_output.append(x_t[:, 1:, :].reshape(-1, 3))

test_input = np.concatenate(test_input)
test_output = np.concatenate(test_output)

# Normalize the test data
test_input = scaler_in.transform(test_input)
test_output = scaler_out.transform(test_output)

# Convert to PyTorch tensors
test_input = torch.from_numpy(test_input).float()
test_output = torch.from_numpy(test_output).float()

# Use the trained model to predict the states at rho = 17 and rho = 35
modelNN.eval()
with torch.no_grad():
    predictions = modelNN(test_input)

# Compute the Mean Squared Error of the predictions
mse_test = loss_fn(predictions, test_output)
print(f"Mean Squared Error (MSE) on the test set: {mse_test.item()}")
```

Next, we began comparing the performance of the feed forward neural network to other neural networks. Firstly, LSTM:

The structure of the LSTM model is defined through a class named LSTMModel. This model includes an LSTM layer with three inputs (corresponding to the three dimensions of the Lorenz system), ten hidden units, and one layer. The LSTM layer is followed by a fully connected layer, which reduces the ten-dimensional output of the LSTM to three dimensions. The forward function orchestrates the passage of input data through these layers.

```
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=10, num_layers=1)
        self.fc = nn.Linear(in_features=10, out_features=3)
        
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

modelLSTM = LSTMModel()
```

Following the model setup, the time span, initial conditions, and rho values for the Lorenz system are set, much like in the previous FFNN model. The Lorenz equations are then solved for these conditions and the results reshaped and concatenated into arrays suitable for training the LSTM model.

```
# Set up time span, initial conditions, and rho values
dt = 0.01
T = 8
t = np.arange(0, T+dt, dt)
np.random.seed(123)
x0 = -15 + 30 * np.random.random((100, 3))
rhos = [10, 28, 40]
```

The LSTM model requires an additional reshaping step, transforming the input data into a 3-dimensional array with dimensions corresponding to (samples, time steps, features). This is because LSTMs expect input in this format, with separate dimensions for the samples in the batch, the time steps in each sample, and the features at each time step.

```
# Solve the Lorenz equations
nn_input_lstm = []
nn_output_lstm = []

for rho in rhos:
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, 8/3, rho)) for x0_j in x0])
    nn_input_lstm.append(x_t[:, :-1, :].reshape(-1, 3))  # reshaping the array to the form (samples, features)
    nn_output_lstm.append(x_t[:, 1:, :].reshape(-1, 3))

nn_input_lstm = np.concatenate(nn_input_lstm)
nn_output_lstm = np.concatenate(nn_output_lstm)
```

We then proceeds with the normalization, splitting the data, and transforming them into PyTorch tensors. The same loss function (MSE) and the optimizer (Adam) from the FFNN part are used for the LSTM model too.

```
# Normalize the data
scaler_in_lstm = StandardScaler()
scaler_out_lstm = StandardScaler()
nn_input_lstm = scaler_in_lstm.fit_transform(nn_input_lstm)
nn_output_lstm = scaler_out_lstm.fit_transform(nn_output_lstm)

# Split the dataset into a training set and a validation set
nn_input_lstm_train, nn_input_lstm_val, nn_output_lstm_train, nn_output_lstm_val = train_test_split(nn_input_lstm, nn_output_lstm, test_size=0.2, random_state=42)

# Reshape the input data to be suitable for LSTM
nn_input_lstm_train = nn_input_lstm_train.reshape(-1, 1, 3)
nn_input_lstm_val = nn_input_lstm_val.reshape(-1, 1, 3)

# Convert to PyTorch tensors
nn_input_lstm_train = torch.from_numpy(nn_input_lstm_train).float()
nn_output_lstm_train = torch.from_numpy(nn_output_lstm_train).float()
nn_input_lstm_val = torch.from_numpy(nn_input_lstm_val).float()
nn_output_lstm_val = torch.from_numpy(nn_output_lstm_val).float()

# Define the loss function
loss_fn_lstm = nn.MSELoss()

# Define the optimizer
optimizer_lstm = optim.Adam(modelLSTM.parameters(), lr=0.01)
```

Training is carried out using a helper function train_lstm, which is similar to the function used for FFNN but has a slight change to accommodate the output shape of LSTM. The .squeeze() function is used to remove the singleton dimension introduced by the LSTM output.

```
# Define a helper function for the training process
def train_lstm(model, optimizer, loss_fn, inputs, targets, n_epochs=100):
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")

# Train the LSTM model
train_lstm(modelLSTM, optimizer_lstm, loss_fn_lstm, nn_input_lstm_train, nn_output_lstm_train, n_epochs=100)
```

After training, the LSTM model is evaluated on the validation set, and the Mean Squared Error of the predictions is computed. The model is then used to forecast the dynamics of the Lorenz system for the new rho values of 17 and 35. The Mean Squared Error of these predictions provides an estimate of how well the LSTM model can predict under unseen conditions.

```
# Evaluate the LSTM model on the validation set
modelLSTM.eval()
with torch.no_grad():
    predictions_lstm = modelLSTM(nn_input_lstm_val)

# Compute the Mean Squared Error of the predictions
mse_lstm = loss_fn_lstm(predictions_lstm.squeeze(), nn_output_lstm_val)
print(f"Mean Squared Error (MSE) on the validation set: {mse_lstm.item()}")

# Solve the Lorenz equations for rho = 17 and rho = 35
test_rhos = [17, 35]
test_input_lstm = []
test_output_lstm = []

for rho in test_rhos:
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, 8/3, rho)) for x0_j in x0])
    test_input_lstm.append(x_t[:, :-1, :].reshape(-1, 3))  
    test_output_lstm.append(x_t[:, 1:, :].reshape(-1, 3))

test_input_lstm = np.concatenate(test_input_lstm)
test_output_lstm = np.concatenate(test_output_lstm)

# Normalize the test data
test_input_lstm = scaler_in_lstm.transform(test_input_lstm)
test_output_lstm = scaler_out_lstm.transform(test_output_lstm)

# Reshape the input data to be suitable for LSTM
test_input_lstm = test_input_lstm.reshape(-1, 1, 3)

# Convert to PyTorch tensors
test_input_lstm = torch.from_numpy(test_input_lstm).float()
test_output_lstm = torch.from_numpy(test_output_lstm).float()

# Use the trained LSTM model to predict the states at rho = 17 and rho = 35
modelLSTM.eval()
with torch.no_grad():
    predictions_lstm = modelLSTM(test_input_lstm)

# Compute the Mean Squared Error of the predictions
mse_test_lstm = loss_fn_lstm(predictions_lstm.squeeze(), test_output_lstm)
print(f"Mean Squared Error (MSE) on the test set: {mse_test_lstm.item()}")
```

Next, the RNN model was trained:

The structure of the RNN model is defined through a class named RNNModel. This model includes an RNN layer with three inputs (corresponding to the three dimensions of the Lorenz system), fifty hidden units, and batch_first set to True to accept inputs in batch size first. The RNN layer is followed by a fully connected layer, which maps the fifty-dimensional output of the RNN to three dimensions. The forward function defines how input data passes through these layers.

```
# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(1))
        out = self.fc(out.squeeze(1))
        return out
```

After the model setup, the time span, initial conditions, and rho values for the Lorenz system are established. Then, the Lorenz equations are solved for these conditions, with the results reshaped and concatenated into arrays suitable for training the RNN model.

```
# Set up time span, initial conditions, and rho values
dt = 0.01
T = 8
t = np.arange(0, T+dt, dt)
np.random.seed(123)
x0 = -15 + 30 * np.random.random((100, 3))
rhos = [10, 28, 40]

# Solve the Lorenz equations
nn_input = []
nn_output = []

for rho in rhos:
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, 8/3, rho)) for x0_j in x0])
    nn_input.append(x_t[:, :-1, :].reshape(-1, 3))  # reshaping the array to the form (samples, features)
    nn_output.append(x_t[:, 1:, :].reshape(-1, 3))

nn_input = np.concatenate(nn_input)
nn_output = np.concatenate(nn_output)
```

Next, the data is normalized, split into training and validation sets, and converted into PyTorch tensors. The loss function (MSE) and the optimizer (Adam) are defined, and the model is trained using the previously defined train function.

```
# Normalize the data
scaler_in = StandardScaler()
scaler_out = StandardScaler()
nn_input = scaler_in.fit_transform(nn_input)
nn_output = scaler_out.fit_transform(nn_output)

# Split the dataset into a training set and a validation set
nn_input_train, nn_input_val, nn_output_train, nn_output_val = train_test_split(nn_input, nn_output, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
nn_input_train = torch.from_numpy(nn_input_train).float()
nn_output_train = torch.from_numpy(nn_output_train).float()
nn_input_val = torch.from_numpy(nn_input_val).float()
nn_output_val = torch.from_numpy(nn_output_val).float()

# Create the model
input_size = 3
hidden_size = 50
output_size = 3
modelNN = RNNModel(input_size, hidden_size, output_size)

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(modelNN.parameters(), lr=0.01)

# Train the model
train(modelNN, optimizer, loss_fn, nn_input_train, nn_output_train, n_epochs=100)
```

After training, the RNN model is evaluated on the validation set, and the Mean Squared Error of the predictions is computed. The model is then used to forecast the dynamics of the Lorenz system for the new rho values of 17 and 35. The Mean Squared Error of these predictions gives an estimate of the model's predictive performance under unseen conditions.

```
# Evaluate the model on the validation set
modelNN.eval()
with torch.no_grad():
  predictions = modelNN(nn_input_val)

# Compute the Mean Squared Error of the predictions
mse = loss_fn(predictions, nn_output_val)
print(f"Mean Squared Error (MSE) on the validation set: {mse.item()}")

# Solve the Lorenz equations for rho = 17 and rho = 35
test_rhos = [17, 35]
test_input = []
test_output = []

for rho in test_rhos:
  x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, 8/3, rho)) for x0_j in x0])
  test_input.append(x_t[:, :-1, :].reshape(-1, 3))
  test_output.append(x_t[:, 1:, :].reshape(-1, 3))

test_input = np.concatenate(test_input)
test_output = np.concatenate(test_output)

# Normalize the test data
test_input = scaler_in.transform(test_input)
test_output = scaler_out.transform(test_output)

# Convert to PyTorch tensors
test_input = torch.from_numpy(test_input).float()
test_output = torch.from_numpy(test_output).float()

# Use the trained model to predict the states at rho = 17 and rho = 35
modelNN.eval()
with torch.no_grad():
  predictions = modelNN(test_input)

# Compute the Mean Squared Error of the predictions
mse_test = loss_fn(predictions, test_output)
print(f"Mean Squared Error (MSE) on the test set: {mse_test.item()}")
```

This process is then repeated for the ESN:

The structure of the ESN model is defined through a class named EchoStateNetwork. This model includes a reservoir with 50 neurons. The reservoir weights are scaled by a spectral radius, which controls the dynamical richness of the reservoir.

The input weights project the three-dimensional input to the reservoir, while the output weights map the reservoir state back to the three-dimensional output. The activation function is the hyperbolic tangent (tanh).

During the forward pass, the model iterates over each time step, updating the reservoir state with the current input and the previous reservoir state. The final output is computed from the last reservoir state.

```
# Define the ESN model
class EchoStateNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, spectral_radius=0.9):
        super(EchoStateNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius

        # Define the ESN reservoir weights (randomly initialized)
        self.reservoir_weights = nn.Parameter(torch.randn(hidden_size, hidden_size))

        # Scale the reservoir weights by the spectral radius
        eigvals = torch.linalg.eigvals(self.reservoir_weights)
        max_eigval = torch.max(torch.abs(eigvals))
        self.reservoir_weights.data *= spectral_radius / max_eigval

        # Define the input-to-reservoir weights (randomly initialized)
        self.input_weights = nn.Parameter(torch.randn(hidden_size, input_size))

        # Define the reservoir-to-output weights (randomly initialized)
        self.output_weights = nn.Parameter(torch.randn(output_size, hidden_size))

        # Define the activation function (tanh)
        self.activation = nn.Tanh()

    def forward(self, x):
        batch_size = x.size(0)
        time_steps = x.size(1)

        # Reshape input tensor
        x = x.view(-1, self.input_size)

        # Initialize the hidden state of the reservoir
        reservoir_state = torch.zeros(batch_size, self.hidden_size)

        # Iterate through time steps
        for t in range(time_steps):
            # Update the reservoir state
            reservoir_state = self.activation(torch.mm(x[t].unsqueeze(0), self.input_weights.t()) + torch.mm(reservoir_state, self.reservoir_weights.t()))

        # Compute the output
        output = torch.mm(reservoir_state, self.output_weights.t())

        return output
```

After the model setup, the time span, initial conditions, and rho values for the Lorenz system are established. Then, the Lorenz equations are solved for these conditions, and the results are reshaped and concatenated into arrays suitable for training the ESN model.

```
# Set up time span, initial conditions, and rho values
dt = 0.01
T = 8
t = np.arange(0, T+dt, dt)
np.random.seed(123)
x0 = -15 + 30 * np.random.random((100, 3))
rhos = [10, 28, 40]

# Solve the Lorenz equations
nn_input = []
nn_output = []

for rho in rhos:
    x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, 8/3, rho)) for x0_j in x0])
    nn_input.append(x_t[:, :-1, :].reshape(-1, 3))  # reshaping the array to the form (samples, features)
    nn_output.append(x_t[:, 1:, :].reshape(-1, 3))

nn_input = np.concatenate(nn_input)
nn_output = np.concatenate(nn_output)
```

The data is normalized, split into training and validation sets, and converted into PyTorch tensors. The loss function (MSE) and the optimizer (Adam) are defined, and the model is trained using the previously defined train function.

```
# Normalize the data
scaler_in = StandardScaler()
scaler_out = StandardScaler()
nn_input = scaler_in.fit_transform(nn_input)
nn_output = scaler_out.fit_transform(nn_output)

# Split the dataset into a training set and a validation set
nn_input_train, nn_input_val, nn_output_train, nn_output_val = train_test_split(nn_input, nn_output, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
nn_input_train = torch.from_numpy(nn_input_train).float()
nn_output_train = torch.from_numpy(nn_output_train).float()
nn_input_val = torch.from_numpy(nn_input_val).float()
nn_output_val = torch.from_numpy(nn_output_val).float()

# Create the ESN model
input_size = 3
hidden_size = 50
output_size = 3
modelESN = EchoStateNetwork(input_size, hidden_size, output_size)

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(modelESN.parameters(), lr=0.01)

# Enable anomaly detection
autograd.set_detect_anomaly(True)

# Train the model
train(modelESN, optimizer, loss_fn, nn_input_train, nn_output_train, n_epochs=100)
```

After training, the ESN model is evaluated on the validation set, and the Mean Squared Error of the predictions is computed. The model is then used to forecast the dynamics of the Lorenz system for the new rho values of 17 and 35. The Mean Squared Error of these predictions gives an estimate of the model's predictive performance under unseen conditions.

```
# Evaluate the model on the validation set
modelESN.eval()
with torch.no_grad():
  predictions = modelESN(nn_input_val)

# Compute the Mean Squared Error of the predictions
mse = loss_fn(predictions, nn_output_val)
print(f"Mean Squared Error (MSE) on the validation set: {mse.item()}")

# Solve the Lorenz equations for rho = 17 and rho = 35
test_rhos = [17, 35]
test_input = []
test_output = []

for rho in test_rhos:
  x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(10, 8/3, rho)) for x0_j in x0])
  test_input.append(x_t[:, :-1, :].reshape(-1, 3))
  test_output.append(x_t[:, 1:, :].reshape(-1, 3))

test_input = np.concatenate(test_input)
test_output = np.concatenate(test_output)

# Normalize the test data
test_input = scaler_in.transform(test_input)
test_output = scaler_out.transform(test_output)

# Convert to PyTorch tensors
test_input = torch.from_numpy(test_input).float()
test_output = torch.from_numpy(test_output).float()

# Use the trained model to predict the states at rho = 17 and rho = 35
modelESN.eval()
with torch.no_grad():
  predictions = modelESN(test_input)

# Compute the Mean Squared Error of the predictions
mse_test = loss_fn(predictions, test_output)
print(f"Mean Squared Error (MSE) on the test set: {mse_test.item()}")
```

### Sec. IV. Computational Results

![image](https://github.com/NajibHaidar/Different-NNs-on-Lorenz-Equations/assets/116219100/182b1a7a-1a1a-4694-bcb1-ca55b4817a39)
*Figure 1: Bar Graph Comparison of Different Model's Validation and Test MSE*

To analyze the performance of the different models, it's important to understand their structures and how they handle time series data.

1. **FFNN (Feedforward Neural Network)**: Feedforward neural networks are the simplest form of artificial neural networks. They map inputs to outputs. They do not have a temporal dimension, meaning they do not have any knowledge about the sequence of data. This may lead to less accurate predictions for complex time series data, especially when there is significant temporal dependency.

2. **LSTM (Long Short-Term Memory)**: LSTMs are a type of recurrent neural network (RNN) designed to remember long-term dependencies in sequence data. They are well-suited for time series prediction tasks. However, they can sometimes be complex to train, and their performance may vary based on the problem complexity, amount of training data, and hyperparameter selection.

3. **RNN (Recurrent Neural Network)**: RNNs are designed to use their internal state (memory) to process sequences of inputs. This makes them good for use with time series data. However, traditional RNNs struggle with the problem of vanishing gradients, making them less effective at capturing long-term dependencies in the data.

4. **ESN (Echo State Network)**: ESNs are a type of recurrent neural network with a sparsely connected hidden layer (the "reservoir"). The special characteristic of ESN is that only the output weights are trained, leaving the reservoir weights unmodified. ESNs can in principle capture long-term dependencies thanks to their recurrent connections, but this is highly dependent on the correct tuning of their hyperparameters. 

Given these considerations, we can make the following observations about the results:

- **FFNN**'s performance is relatively good. It seems that the time series problem at hand may not involve significant temporal dependencies, allowing the FFNN to perform well despite its lack of temporal modeling capabilities.
  
- **LSTM** performs worse than the FFNN and RNN. This could be because the LSTM's ability to capture long-term dependencies is not beneficial for this specific dataset, or it might be due to insufficient training or improper hyperparameter settings.

- **RNN** has the best performance, suggesting that it's able to capture the necessary temporal dependencies in the data efficiently.

- **ESN** performs significantly worse than the other models. This could be due to the "reservoir" not being properly tuned to capture the dynamics of the dataset. Additionally, ESN's performance is highly dependent on the initial state of the reservoir and the input data, which might not be optimal in this case.

It's also worth mentioning that the choice of loss function (MSE in this case) and the amount and quality of data can greatly affect the results. A different loss function might lead to different results, and more or better-quality data could improve the performance of all models.

### Sec. V. Summary and Conclusions
In this study, we explored the application of various neural network models, namely Feedforward Neural Networks (FFNN), Long Short-Term Memory networks (LSTM), Recurrent Neural Networks (RNN), and Echo State Networks (ESN) to predict the evolution of the Lorenz system, a well-known chaotic system. The choice of these models stems from their widespread use in the analysis of time series data and their varying approaches to sequence prediction.

Theoretical considerations suggest that recurrent architectures like LSTM, RNN, and ESN should, in principle, outperform FFNN due to their ability to capture temporal dependencies in the data. However, our computational results painted a slightly different picture. 

FFNN, which has no inherent temporal modeling capability, performed relatively well. This outcome suggests that for this particular problem, substantial long-term dependencies may not be present or necessary to predict accurately. 

Surprisingly, LSTM did not perform as well as expected. Despite their theoretical advantage of capturing long-term dependencies, it appears that this feature did not contribute significantly to the problem at hand. Alternatively, this might indicate that the LSTM model may require further tuning or more extensive training.

RNNs showed the best performance among all models, indicating their efficiency in capturing the necessary short-term dependencies in the data. 

ESN, however, performed significantly worse. Given its unique architecture and reliance on the "reservoir" for dynamics modeling, the poor performance of ESN suggests that the reservoir was not aptly tuned to capture the Lorenz system's dynamics. 

These findings underscore the significance of model choice when dealing with time series prediction tasks. It is not always the case that more complex models, such as LSTM or ESN, will outperform simpler ones like FFNN or RNN. Understanding the underlying characteristics of the data and how different models handle these characteristics is crucial for effective model selection.

Future work could involve the exploration of more advanced and recently developed models for time series prediction, such as Transformer models, which have shown promising results in other domains. Moreover, refining the tuning process for models like ESN could potentially enhance their performance on such tasks. 

In conclusion, this study emphasizes the nuanced performance of various neural network models in predicting complex, chaotic systems.

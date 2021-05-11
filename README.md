# NeuralNetwork_diabetic
Atrificial Neural Network with Backpropagation developed in python  as a learning algorithm to predict if the different cases contain signs of diabetic retinopathy or not.

The Backpropagation algorithm is a supervised learning method for multilayer feed-forward networks from the field of Artificial Neural Networks.

The main dataset (Diabetic_dataset.txt) has been divided in the following way:
* 75% for the training process (dataset_train.csv);
* 15% for the testing process (dataset_test.csv);
* 10% for the validation process (dataset_validation.csv).


ANN with two hidden layers, the first with 13 neurons and the second with 7.
The input neurons are 19, and in the code below are shown all the features defining the names on the dataset:

```python
data = pd.read_csv(path, names=[
                                "Quality",
                                "Pre-Screening",
                                "MA-0.5",
                                "MA-0.6",
                                "MA-0.7",
                                "MA-0.8",
                                "MA-0.9",
                                "MA-1.0",
                                "EX-0.5",
                                "EX-0.6",
                                "EX-0.7",
                                "EX-0.8",
                                "EX-0.9",
                                "EX-1.0",
                                "EX-1.1",
                                "EX-1.2",
                                "ED macula-center of optic disc",
                                "Diameter optic disc",
                                "AM/FM",
                                "Class"])

```

As explained before, the ANN has the following structure:

![GitHub Logo](/img/ANN_structure.png)
Format: ![Alt Text](url)

# Weights initialization
For this algorithm it starts assigning random values to the weights:

```python
        np.random.seed(1)
        self.L1 = 13
        self.L2 = 7
        self.weight_hidden1 = 2 * np.random.rand( 19, self.L1) -1
        self.weight_hidden2 = 2 * np.random.rand(self.L1, self.L2) - 1
        self.weight_output = 2 * np.random.rand(self.L2, 1) - 1
        
```

# Training and weights update
The algorithm works through each layer of our network calculating the outputs for each neuron. All
of the outputs from one layer become inputs to the neurons on the next layer.

The function returns the outputs from the last layer also called the output layer.

Error is calculated between the expected outputs and the outputs forward propagated
from the network. These errors are then propagated backward through the network
from the output layer to the hidden layer, assigning blame for the error and updating
weights as they go.
```python
          def train (self, input_features, target_output, training_iterations, validation_X, validation_Y):
                  for epoch in range(training_iterations):
                   #Forward part
                   # Input and Output for hidden layer1 :
                   input_hidden1 = np.dot(input_features, self.weight_hidden1)
                   output_hidden1 = self.sigmoid(input_hidden1)

                   # Input and Output for hidden layer2 :
                   input_hidden2 = np.dot(output_hidden1, self.weight_hidden2)
                   output_hidden2 = self.sigmoid(input_hidden2)

                   # Input and Output for output layer :
                   input_out = np.dot(output_hidden2, self.weight_output)
                   output_out = self.sigmoid(input_out)

                   #Backward part
                   error_out = target_output - output_out
                   z_delta1 = error_out * self.sigmoid_der(input_out)
                   error_weight_out = np.dot(output_hidden2.T, z_delta1)

                   z_delta2 = np.dot(z_delta1, self.weight_output.T) * output_hidden2 * (1-output_hidden2)
                   error_weight_hidden2 = np.dot(output_hidden1.T, z_delta2)

                   z_delta3 = np.dot(z_delta2, self.weight_hidden2.T) * output_hidden1 * (1-output_hidden1)
                   error_weight_hidden1 = np.dot(input_features.T, z_delta3)

                   self.weight_output += self.lr * error_weight_out
                   self.weight_hidden2 += self.lr * error_weight_hidden2
                   self.weight_hidden1 += self.lr * error_weight_hidden1
```

The output error has been evaluated in the following way:

![equation](./img/formula_error.jpg)


For each neuron we need to calculate the slope using the sigmoid function, calculated
as following:

```python
    # Sigmoid function :
    def sigmoid(self, x):
     return 1/(1+np.exp(-x))
```

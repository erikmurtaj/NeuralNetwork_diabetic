import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Neural Network with two hidden layers, the first with 13 neurons and the second
with 7. The leraning rate is set to 0.02.
The initial weights are generated randomly between -1/1
'''

class NeuralNetwork():
    def __init__(self):
        # random generation of weights
        np.random.seed(1)
        self.L1 = 13
        self.L2 = 7
        self.weight_hidden1 = 2 * np.random.rand( 19, self.L1) -1
        self.weight_hidden2 = 2 * np.random.rand(self.L1, self.L2) - 1
        self.weight_output = 2 * np.random.rand(self.L2, 1) - 1
        #leraning rate
        self.lr = 0.02
        self.accurancies = []

    # Sigmoid function :
    def sigmoid(self, x):
     return 1/(1+np.exp(-x))

    # Derivative of sigmoid function :
    def sigmoid_der(self, x):
     return self.sigmoid(x)*(1-self.sigmoid(x))

    #training function
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

         validation(validation_X, validation_Y)


    def think(self, inputs):
        result1 = np.dot(inputs, self.weight_hidden1)
        result2 = self.sigmoid(result1)
        result3 = np.dot(result2,self.weight_hidden2)
        result4 = self.sigmoid(result3)
        result5 = np.dot(result4, self.weight_output)
        final_result = self.sigmoid(result5)
        return final_result



def validation(validation_X, validation_Y):
    validation_output = neural_network.think(validation_X)
    count_true = 0
    count = 0
    for number in validation_Y:
        if (abs((validation_output[count] - validation_Y[count])) <= 0.5):
            count_true += 1
        count += 1

    accurancy = count_true / (len(validation_Y))
    neural_network.accurancies.append(accurancy)
    print(accurancy)

def load_dataset(path):
  '''
  MA = microaneurysms (each detection field represent the number of MA found
  at the corresponding level of confidence)

  ED = Euclidean Distance
  '''
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


  data_features = data.copy()
  array = np.array(data_features)
  data_label = array[:,[19]]
  data_features.pop("Class")
  arraX = np.array(data_features)
  return arraX, data_label

def draw (x_axes, y_axes):
    plt.figure(figsize=(12, 8))
    plt.plot(x_axes, y_axes)
    plt.show()


if __name__ == "__main__":

    train_X, train_Y = load_dataset("dataset_train.csv")
    validation_X, validation_Y = load_dataset("dataset_validation.csv")
    test_X, test_Y = load_dataset("dataset_test.csv")

    train_X = np.array(train_X)
    validation_X = np.array(validation_X)
    validation_Y = np.array(validation_Y)
    test_X = np.array(test_X)

    neural_network = NeuralNetwork()
    neural_network.train(train_X, train_Y, 1000, validation_X, validation_Y)

    #Accurancy graph
    draw( np.arange(0, 1000, 1), neural_network.accurancies)

    print("test accurancy")
    validation(test_X, test_Y)

# NeuralNetwork_diabetic
Atrificial Neural Network with Backpropagation developed in python  as a learning algorithm to predict if the different cases contain signs of diabetic retinopathy or not.

The Backpropagation algorithm is a supervised learning method for multilayer feed-forward networks from the field of Artificial Neural Networks.

ANN with two hidden layers, the first with 13 neurons and the second with 7.
The input neurons are 19, and in the code below are shown all the features defining the names on the dataset:

```ruby
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

!test(https://github.com/erikmurtaj/NeuralNetwork_diabetic/blob/image.jpg?raw=true)

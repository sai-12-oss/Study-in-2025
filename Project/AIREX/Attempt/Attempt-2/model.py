import numpy as np 
# if using relu function as the activation function for the hidden layers 
def relu(x):
    return np.maximum(0,x)
def relu_derivative(x):
    return 1 if x>0 else 0
# if using sigmoid function as the activation function for the hidden layers 
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
# defining RMS(output activation Layer = linear)
def mean_square_error(predictions, targets):				# loss function
	return np.sum([(predictions[i][0] - targets[i]) ** 2 for i in range(len(predictions))])
# defining cross_entropy(output Activation layer = softmax)
def cross_entropy(predictions, targets):				# loss function
	return np.sum([-targets[i] * np.log(predictions[i]) - (1 - targets[i]) * np.log(1 - predictions[i]) for i in range(predictions.shape[0])])
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0, keepdims=True)
def compute_accuracy(predictions,targets):
    correct = 0
    for i in range(len(targets)):
        if targets[i]==predictions[i]:
            correct += 1
    return (correct/float(len(targets)))*100.00
# class Regretter:
# # here l0,l1,l2,l3,l4,l5 represent the no of nuerons in the initial layer, first hidden layer,second,third,fourth, output layer resp.
# # l5 is the no of nuerons in the output layer 
#     def __init__(self,l0,l1,l2,l3,l4,l5):
#         self.weights_0 = np.random.randn(l0,l1)*0.01	# weights from input layer to first hidden layer
#         self.weights_1 = np.random.randn(l1,l2)*0.01		# weights from first hidden layer to second hidden layer
#         self.weights_2 = np.random.randn(l2,l3)*0.01		# and so on 
#         self.weights_3 = np.random.randn(l3,l4)*0.01
#         self.weights_4 = np.random.randn(l4,l5)*0.01
#         self.sigmoid = np.vectorize(sigmoid)
#         # self.relu = np.vectorize(relu) # for element wise application we vectorize 
# # Training Method 
# # """Algorithm Using : Gradient Descent with backward propagation"""
#     def regress(self, X_train, y_train, learning_rate, num_epochs, verbose):
# 		# forward propagation : computes activation for hidden and output layers 
#         hidden_layer_1 = self.sigmoid(np.dot(X_train, self.weights_0))		# sigmoid (replace sigmoid by relu to use Relu) as activation function
#         hidden_layer_2 = self.sigmoid(np.dot(hidden_layer_1, self.weights_1))  
#         hidden_layer_3 = self.sigmoid(np.dot(hidden_layer_2, self.weights_2))  
#         hidden_layer_4 = self.sigmoid(np.dot(hidden_layer_3, self.weights_3))  
#         output_layer = np.dot(hidden_layer_4, self.weights_4)
#         # define error history to keep track of errors
#         error_history = [mean_square_error(output_layer, y_train)]

#         if verbose:
#             print('Initial Error:', error_history[0])

#         for e in range(num_epochs):
#             if verbose:
#                 print('Training model --> Epoch', e)

#             for p in range(output_layer.shape[0]):

#                 # backward propagation : adjusting the weights based on the error between predicted and auctual outputs 
#                 output_error = output_layer[p] - y_train[p]  # error in output
#                 output_delta = output_error

#                 hidden_4_delta = np.zeros(len(self.weights_3[0]))  # how much 4th hidden layer weights contributed to output error
#                 for i in range(len(self.weights_3[0])):
#                     hidden_4_delta[i] = output_delta * self.weights_4[i][0] * sigmoid_derivative(hidden_layer_4[p][i])

#                 hidden_3_delta = np.zeros(len(self.weights_2[0]))  # how much 3rd hidden layer weights contributed to output error
#                 for i in range(len(self.weights_2[0])):
#                     hidden_3_delta[i] = np.sum([self.weights_3[i][k] * hidden_4_delta * sigmoid_derivative(hidden_layer_3[p][i]) for k in range(len(self.weights_3[0]))])

#                 hidden_2_delta = np.zeros(len(self.weights_1[0]))  # how much 2nd hidden layer weights contributed to output error
#                 for i in range(len(self.weights_1[0])):
#                     hidden_2_delta[i] = np.sum([self.weights_2[i][k] * hidden_3_delta * sigmoid_derivative(hidden_layer_2[p][i]) for k in range(len(self.weights_2[0]))])

#                 hidden_1_delta = np.zeros(len(self.weights_0[0]))  # how much 1st hidden layer weights contributed to output error
#                 for i in range(len(self.weights_0[0])):
#                     hidden_1_delta[i] = np.sum([self.weights_1[i][k] * hidden_2_delta * sigmoid_derivative(hidden_layer_1[p][i]) for k in range(len(self.weights_1[0]))])

#                 for i in range(len(self.weights_4)):  # adjusting (4th hidden --> output) weights
#                     self.weights_4[i][0] -= learning_rate * output_delta * hidden_layer_4[p][i]

#                 for i in range(len(self.weights_3)):  # adjusting (3rd hidden --> 4th hidden) weights
#                     for j in range(len(self.weights_3[0])):
#                         self.weights_3[i][j] -= learning_rate * hidden_4_delta[j] * hidden_layer_3[j][i]

#                 for i in range(len(self.weights_2)):  # adjusting (2nd hidden --> 3rd hidden) weights
#                     for j in range(len(self.weights_2[0])):
#                         self.weights_2[i][j] -= learning_rate * hidden_3_delta[j] * hidden_layer_2[j][i]

#                 for i in range(len(self.weights_1)):  # adjusting (1st hidden --> 2nd hidden) weights
#                     for j in range(len(self.weights_1[0])):
#                         self.weights_1[i][j] -= learning_rate * hidden_2_delta[j] * hidden_layer_1[j][i]

#                 for i in range(len(self.weights_0)):  # adjusting (input --> 1st hidden) weights
#                     for j in range(len(self.weights_0[0])):
#                         self.weights_0[i][j] -= learning_rate * hidden_1_delta[j] * X_train[j][i]

#             # forward propagation: update predictions
#             hidden_layer_1 = self.sigmoid(np.dot(X_train, self.weights_0))
#             hidden_layer_2 = self.sigmoid(np.dot(hidden_layer_1, self.weights_1))
#             hidden_layer_3 = self.sigmoid(np.dot(hidden_layer_2, self.weights_2))
#             hidden_layer_4 = self.sigmoid(np.dot(hidden_layer_3, self.weights_3))
#             output_layer = np.dot(hidden_layer_4, self.weights_4)

#             error_history.append(mean_square_error(output_layer, y_train))
#             if verbose:
#                 print('Error after training for', e, 'epochs:', error_history[-1])

#         return error_history
# # Perform forward propagation on the input data to generate predictions using the sigmoid activation function
#     def predict(self, input_matrix):
#         hidden_layer_1 = self.sigmoid(np.dot(input_matrix, self.weights_0))  # forward propagation
#         hidden_layer_2 = self.sigmoid(np.dot(hidden_layer_1, self.weights_1))
#         hidden_layer_3 = self.sigmoid(np.dot(hidden_layer_2, self.weights_2))
#         hidden_layer_4 = self.sigmoid(np.dot(hidden_layer_3, self.weights_3))
#         return np.dot(hidden_layer_4, self.weights_4)[0]

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

class Regretter:
    def __init__(self, l0, l1, l2, l3, l4, l5):
        self.weights_0 = np.random.randn(l0, l1) * 0.01
        self.weights_1 = np.random.randn(l1, l2) * 0.01
        self.weights_2 = np.random.randn(l2, l3) * 0.01
        self.weights_3 = np.random.randn(l3, l4) * 0.01
        self.weights_4 = np.random.randn(l4, l5) * 0.01

    def forward(self, X):
        self.z1 = np.dot(X, self.weights_0)
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights_1)
        self.a2 = relu(self.z2)
        self.z3 = np.dot(self.a2, self.weights_2)
        self.a3 = relu(self.z3)
        self.z4 = np.dot(self.a3, self.weights_3)
        self.a4 = relu(self.z4)
        self.z5 = np.dot(self.a4, self.weights_4)
        self.a5 = softmax(self.z5)
        return self.a5

    def backward(self, X, y, learning_rate):
        m = y.shape[0]
        output_error = self.a5 - y
        d_weights_4 = np.dot(self.a4.T, output_error) / m

        hidden_4_error = np.dot(output_error, self.weights_4.T) * relu_derivative(self.z4)
        d_weights_3 = np.dot(self.a3.T, hidden_4_error) / m

        hidden_3_error = np.dot(hidden_4_error, self.weights_3.T) * relu_derivative(self.z3)
        d_weights_2 = np.dot(self.a2.T, hidden_3_error) / m

        hidden_2_error = np.dot(hidden_3_error, self.weights_2.T) * relu_derivative(self.z2)
        d_weights_1 = np.dot(self.a1.T, hidden_2_error) / m

        hidden_1_error = np.dot(hidden_2_error, self.weights_1.T) * relu_derivative(self.z1)
        d_weights_0 = np.dot(X.T, hidden_1_error) / m

        self.weights_4 -= learning_rate * d_weights_4
        self.weights_3 -= learning_rate * d_weights_3
        self.weights_2 -= learning_rate * d_weights_2
        self.weights_1 -= learning_rate * d_weights_1
        self.weights_0 -= learning_rate * d_weights_0

    def train(self, X, y, learning_rate, num_epochs, verbose=0):
        for e in range(num_epochs):
            predictions = self.forward(X)
            loss = cross_entropy(predictions, y)
            self.backward(X, y, learning_rate)
            if verbose:
                print(f'Epoch {e + 1}, Loss: {loss}, Accuracy: {compute_accuracy(predictions, y)}%')
import matplotlib.pyplot as plt
import numpy as np
import csv
from sklearn import svm
from cvxopt import matrix, solvers
from scipy.linalg import block_diag
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as CK
from sklearn.gaussian_process.kernels import ExpSineSquared as ESS
from sklearn.gaussian_process.kernels import RationalQuadratic as RQ

# Question 1.a
# Primal SVM
with open('fmnist_train.csv','r') as f: # getting train data
    reader = csv.reader(f)
    data = list(reader)

data_array = np.array(data, dtype=int)
ones = np.ones((data_array.shape[0],1))
data_array = np.append(data_array, ones, axis=1) # augmenting data with 1s

data_x = data_array[:,1:]
data_y = data_array[:,0]

srn = 22205
c0, c1 = np.random.default_rng(srn).choice(range(0,10),size=2,replace=False)
binary_x = data_x[(data_y == c0) | (data_y == c1)]
binary_y = data_y[(data_y == c0) | (data_y == c1)]

for i in range(len(binary_y)):
    if binary_y[i] == c0:
        binary_y[i] = -1
    else:
        binary_y[i] = 1

P1 = np.identity(binary_x.shape[1]-1)
P2 = np.zeros((binary_x.shape[0]+1, binary_x.shape[0]+1))
P = matrix(block_diag(P1, P2))

C = 1 # can be changed
q1 = np.zeros(binary_x.shape[1])
q2 = C*np.ones(binary_x.shape[0])
q = matrix(np.concatenate((q1, q2)))

topleft = np.zeros((binary_x.shape[0], binary_x.shape[1]))
for i in range(binary_x.shape[0]):
    for j in range(binary_x.shape[1]):
        topleft[i][j] = (-1)*binary_y[i]*binary_x[i][j]

topright = (-1)*np.identity(binary_x.shape[0])
bottomleft = np.zeros((binary_x.shape[0], binary_x.shape[1]))
bottomright = (-1)*np.identity(binary_x.shape[0])

G = matrix(np.block([[topleft, topright], [bottomleft, bottomright]]))

h1 = (-1)*np.ones(binary_x.shape[0])
h2 = np.zeros(binary_x.shape[0])
h = matrix(np.concatenate((h1, h2)).reshape(-1,1))

sol = solvers.qp(P, q, G, h)
w = sol['x'][:binary_x.shape[1]]

with open('fmnist_test.csv','r') as f: # getting test data
    reader = csv.reader(f)
    test_data = list(reader)

test_data_array = np.array(test_data, dtype=int)
ones = np.ones((test_data_array.shape[0],1))
test_data_array = np.append(test_data_array, ones, axis=1) # augmenting data with 1s

test_data_x = test_data_array[:,1:]
test_data_y = test_data_array[:,0]

test_binary_x = test_data_x[(test_data_y == c0) | (test_data_y == c1)]
test_binary_y = test_data_y[(test_data_y == c0) | (test_data_y == c1)]

for i in range(len(test_binary_y)):
    if test_binary_y[i] == c0:
        test_binary_y[i] = -1
    else:
        test_binary_y[i] = 1

correct = 0
for i in range(test_binary_x.shape[0]): # testing
    if test_binary_y[i]*np.dot(test_binary_x[i], w) >= 0:
        correct += 1

print("Primal accuracy =", correct/test_binary_x.shape[0])


# Question 1.a
# Dual SVM
P_dual = matrix(np.matmul(topleft,topleft.T))

q_dual = (-1)*matrix(np.ones(binary_x.shape[0]).reshape(-1,1))

G1_dual = (-1)*np.identity(binary_x.shape[0])
G2_dual = np.identity(binary_x.shape[0])
G_dual = matrix(np.concatenate((G1_dual, G2_dual)))

C_dual = 1 # can be changed
h1_dual = np.zeros(binary_x.shape[0])
h2_dual = C_dual*np.ones(binary_x.shape[0])
h_dual = matrix(np.concatenate((h1_dual, h2_dual)).reshape(-1,1))

A_dual = matrix(binary_y.reshape(1,-1))
b_dual = matrix(np.zeros(1))

sol_dual = solvers.qp(P_dual, q_dual, G_dual, h_dual, A_dual, b_dual)
lambda_dual = sol_dual['x']

w_dual = np.zeros(binary_x.shape[1])
for i in range(binary_x.shape[0]): # forming w
    w_dual += lambda_dual[i]*binary_y[i]*binary_x[i]

correct = 0
for i in range(test_binary_x.shape[0]): # testing
    if test_binary_y[i]*np.dot(test_binary_x[i], w_dual) >= 0:
        correct += 1

print("Dual accuracy =",correct/test_binary_x.shape[0])


# Question 1.a
# Kernel SVM
def kernel(x1, x2, sigma):
    r = np.exp((-0.5)*(np.linalg.norm(np.subtract(x1,x2)))**2/(sigma**2))
    if r < 1e-9:
        r = 1e-9
    return r

sigma = 1
topleft_kernel = np.zeros((binary_x.shape[0], binary_x.shape[0]))
for i in range(binary_x.shape[0]):
    for j in range(binary_x.shape[0]):
        topleft_kernel[i][j] = binary_y[i]*binary_y[j]*kernel(binary_x[i], binary_x[j], sigma)

P_kernel = matrix(topleft_kernel)
sol_kernel = solvers.qp(P_kernel, q_dual, G_dual, h_dual, A_dual, b_dual)
lambda_kernel = sol_kernel['x']

correct = 0
sum = 0
for i in range(test_binary_x.shape[0]):
    for j in range(binary_x.shape[0]):
        sum += lambda_kernel[j]*binary_y[j]*kernel(test_binary_x[i], binary_x[j], sigma)
    if  test_binary_y[i]*sum >= 0:
        correct += 1

print("Kernel accuracy =", correct/test_binary_x.shape[0])


# Question 1.a
# Scikit
C = 1 # can be changed
scikit = svm.SVC(C=C, kernel='linear').fit(test_binary_x, test_binary_y) # scikit svm

correct = 0
for i in range(test_binary_x.shape[0]): # testing
    if test_binary_y[i]*scikit.decision_function([test_binary_x[i]]):
        correct += 1

print("Scikit accuracy =", correct/test_binary_x.shape[0])


# Question 2.1
# Linear and ridge regression
house_data = np.genfromtxt('house_price_prediction.csv', delimiter=',', skip_header=1, usecols=range(2,21))

house_train_y = house_data[:int(0.8*house_data.shape[0]), 0] # taking train data
house_test_y = house_data[int(0.8*house_data.shape[0]):, 0]

house_data = house_data[:,1:] # augmenting
ones = np.ones((house_data.shape[0],1))
house_data = np.append(ones, house_data, axis=1)

house_train_x = house_data[:int(0.8*house_data.shape[0])] # taking test data
house_test_x = house_data[int(0.8*house_data.shape[0]):]

w_house_linear = np.matmul(np.matmul(np.linalg.inv(np.matmul(house_train_x.T, house_train_x)), house_train_x.T), house_train_y.T)

lambda_ridge = 1 # can be changed
w_house_ridge = np.matmul(np.matmul(np.linalg.inv(np.matmul(house_train_x.T, house_train_x) + lambda_ridge*np.identity(house_train_x.shape[1])), house_train_x.T), house_train_y.T)

linear_mse = 0
ridge_mse = 0

linear_predictions = np.matmul(house_test_x, w_house_linear)
ridge_predictions = np.matmul(house_test_x, w_house_ridge)

for i in range(house_test_x.shape[0]): # calculating MSE
    linear_mse += (house_test_y[i] - linear_predictions[i])**2
    ridge_mse += (house_test_y[i] - ridge_predictions[i])**2

print("Linear average MSE =", linear_mse/house_test_x.shape[0])
print("Ridge average MSE =", ridge_mse/house_test_x.shape[0])

# Plotting the regression
sqft_living = np.sort(house_test_x[:,3])

plt.scatter(sqft_living, house_test_y, color='blue', label='Actual Price')
plt.plot(sqft_living, linear_predictions, color='red', label='Linear Prediction')
plt.xlabel('sqft_living')
plt.ylabel('Price')
plt.legend()
plt.show() 

plt.scatter(sqft_living, house_test_y, color='blue', label='Actual Price')
plt.plot(sqft_living, ridge_predictions, color='red', label='Ridge Prediction')
plt.xlabel('sqft_living')
plt.ylabel('Price')
plt.legend()
plt.show() 


# Question 2.2
weather_data = np.genfromtxt('weather_data.csv', delimiter=',', skip_header=1)
temps = np.genfromtxt('weather_data.csv', delimiter=',', skip_header=1, usecols=0)

X = np.atleast_2d([float(i) for i in range(1,201)]).T
y = np.atleast_2d(np.linspace(1,200,10000)).T

# Kernel 1
kernel1 = CK()*ESS(length_scale=24, periodicity=1)
gpr1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer=20).fit(X, weather_data)
mean1, std1 = gpr1.predict(y, return_std=True)

plt.title('Kernel 1')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.scatter(np.linspace(1,200,200), temps, color='red', label='True values')
plt.plot(y, mean1[:,0], color='blue', label='Predicted values')
plt.fill_between(y[:,0], mean1[:,0] - 1.96*std1[:,0], mean1[:,0] + 1.96*std1[:,0], color='grey', alpha=0.5, label='95% confidence interval')
plt.legend()
plt.show()

# Kernel 2
kernel2 = CK()*RQ(length_scale=24, alpha=1)
gpr2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=20).fit(X, weather_data)
mean2, std2 = gpr2.predict(y, return_std=True)

plt.title('Kernel 2')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.scatter(np.linspace(1,200,200), temps, color='red', label='True values')
plt.plot(y, mean2[:,0], color='blue', label='Predicted values')
plt.fill_between(y[:,0], mean2[:,0] - 1.96*std2[:,0], mean2[:,0] + 1.96*std2[:,0], color='grey', alpha=0.5, label='95% confidence interval')
plt.legend()
plt.show()

# Kernel 3
kernel3 = CK()*(ESS(length_scale=24, periodicity=1) + RQ(length_scale=24, alpha=0.5))
gpr3 = GaussianProcessRegressor(kernel=kernel3, n_restarts_optimizer=20).fit(X, weather_data)
mean3, std3 = gpr3.predict(y, return_std=True)

plt.title('Kernel 3')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.scatter(np.linspace(1,200,200), temps, color='red', label='True values')
plt.plot(y, mean3[:,0], color='blue', label='Predicted values')
plt.fill_between(y[:,0], mean3[:,0] - 1.96*std3[:,0], mean3[:,0] + 1.96*std3[:,0], color='grey', alpha=0.5, label='95% confidence interval')
plt.legend()
plt.show()

# Kernel 4

kernel4 = CK()*ESS(length_scale=24, periodicity=1)*RQ(length_scale=24, alpha=0.5)
gpr4 = GaussianProcessRegressor(kernel=kernel4, n_restarts_optimizer=20).fit(X, weather_data)
mean4, std4 = gpr4.predict(y, return_std=True)

plt.title('Kernel 4')
plt.xlabel('Timestamp')
plt.ylabel('Temperature')
plt.scatter(np.linspace(1,200,200), temps, color='red', label='True values')
plt.plot(y, mean4[:,0], color='blue', label='Predicted values')
plt.fill_between(y[:,0], mean4[:,0] - 1.96*std4[:,0], mean4[:,0] + 1.96*std4[:,0], color='grey', alpha=0.5, label='95% confidence interval')
plt.legend()
plt.show()

# %%
"""
Question-1 : Support Vector Machine and Perceptron
"""

# %%
"""
Preprocessing 
"""

# %%
import Oracle_Assignment_2
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('default')
train_data,test_data = Oracle_Assignment_2.q1_get_cifar100_train_test(23627)
train_features,train_labels = [],[]
test_features,test_labels = [],[]
for x in train_data:
    train_features.append(x[0])
    train_labels.append(x[1])
train_features = np.array(train_features)
train_labels = np.array(train_labels) # -1 and 1 
for x in test_data:
    test_features.append(x[0])
    test_labels.append(x[1])
test_features = np.array(test_features)
test_labels = np.array(test_labels)
train_mean = np.mean(train_features, axis=0) 
train_std = np.std(train_features, axis=0)   
# print(train_features.shape,train_labels.shape) # Returns : (1000,27) (1000,)
# print(test_features.shape,test_labels.shape) # Returns : (200,27) (200,)

# %%
"""
Task : Run the perceptron algorithm on your data. Report whether it converges, or appears
not to. If it doesn’t seem to converge, make certain that you are reasonably sure.

Question : A plot between misclassification rate and number of iterations for the perceptron algorithm as defined
in Task 1.
"""

# %%
def Perceptron(train_features,train_labels,max_iterations = 20000):
    d = len(train_features[0])
    w = [0.0 for i in range(d)]
    b = 0.0
    misclassification_rates = []
    converged = False
    for t in range(max_iterations):
        misclassified = 0
        for i in range(len(train_features)):
            x = train_features[i]
            y = train_labels[i]
            dot_product = sum([w[j]*x[j] for j in range(d)]) # calculating wT.x
            predicted_label = dot_product + b 
            if predicted_label*y <= 0:
                misclassified += 1
                for j in range(d):
                    w[j] += y*x[j]
                b += y
        misclassification_rate = misclassified/len(train_features)
        misclassification_rates.append(misclassification_rate)
        if misclassified == 0:
            converged = True
            print(f"Converged after {t+1} iterations")
            break
    else:
        print("Did not converge")
    return w,b,misclassification_rates,converged 
w,b,misclassification_rates,converged = Perceptron(train_features,train_labels)
iterations = list(range(1, len(misclassification_rates) + 1))
plt.figure(figsize=(10, 6))
plt.plot(iterations, misclassification_rates, label="Misclassification Rate")
plt.xlabel("No of Iterations")
plt.ylabel("Misclassification Rate")
plt.title("Perceptron Convergence: Misclassification Rate vs. No of Iterations (Task 1)")
plt.grid(True)
plt.legend()
plt.show()

# %%
"""
Task : Linear SVM; Construct a slack support vector machine with the linear kernel. Solve both the primal
and dual versions of the SVM quadratic programs using cvxopt. Use the SVM’s solution to isolate
the sources of non-separability

Question-1: Which, between the primal and dual, is solved faster for Task 2. Report the times taken for running
both, and justify any patterns you see.

Question-2 : Give the indicies which are causing inseparability in the data in a csv file of name inseperable_{sr_number}.csv
Primal SVM :
1. The primal SVM is solved using the cvxopt library.    
2. Returns w,b,xi(slack variables) and time_taken 

Dual SVM :

1. The dual SVM is solved using the cvxopt library.
2. Returns w,b,alphas (Lagrange Multipliers for inequality constraints) and time_taken
"""

# %%
import numpy as np 
import cvxopt 
import time 

class SVM:
    def __init__(self, regularization_parameter=1.0):
        self.C = regularization_parameter  # Regularization parameter (tradeoff between maximizing margin and minimizing slack)
        self.w_primal = None
        self.b_primal = None
        self.slack_values = None
        self.alpha_dual = None
        self.w_dual = None
        self.b_dual = None
        self.primal_time = None
        self.dual_time = None
    def Primal_SVM(self,X,y):
        num_samples, num_features = len(X), len(X[0])
        variable_size = num_features + 1 + num_samples
        P = [[0.0 for _ in range(variable_size)] for _ in range(variable_size)]
        for i in range(num_features):  # For ||w||^2 term
            P[i][i] = 1.0
        q = [0.0 for _ in range(variable_size)]
        q[num_features + 1:] = [float(self.C) for _ in range(num_samples)] # For slack variables
        # Inequality constraints G and h 
        G = np.zeros((2*num_samples, variable_size))
        h = np.zeros(2*num_samples)
        # Constraint 1: y_i (w^T x_i + b) >= 1 - ξ_i
        for i in range(num_samples):
            G[i, :num_features] = -y[i] * X[i]
            G[i, num_features] = -y[i]
            G[i, num_features + 1 + i] = -1
            h[i] = -1
        # Constraint 2: ξ_i >= 0
        for i in range(num_samples):
            G[num_samples + i, num_features + 1 + i] = -1
            h[num_samples + i] = 0
        # Convert to cvxopt format
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        # Solve the QP problem and Measure time taken to solve the primal SVM
        cvxopt.solvers.options['show_progress'] = False
        start_time = time.time()
        solution = cvxopt.solvers.qp(P, q, G, h)
        self.primal_time = time.time() - start_time
        # Extract solution
        primal_solution = np.array(solution['x']).flatten()
        self.w_primal = primal_solution[:num_features]
        self.b_primal = primal_solution[num_features]
        self.slack_values = primal_solution[num_features + 1:]

        # Identify non-separable points (ξ_i > 1)
        non_separable_indices = [] # ξ_i > 1 which implies y_i (w^T x_i + b) < 0
        for i in range(num_samples):
            if self.slack_values[i] > 1:
                non_separable_indices.append(i)
        return self.w_primal, self.b_primal, self.slack_values, non_separable_indices
    def Dual_SVM(self,X,y):
        num_samples = len(X)

        # Compute the kernel matrix (linear kernel: K[i,j] = x_i^T x_j)
        K = [[0.0 for _ in range(num_samples)] for _ in range(num_samples)]
        for i in range(num_samples):
            for j in range(num_samples):
                K[i][j] = sum(X[i][k] * X[j][k] for k in range(len(X[0])))
        # compute P = (y_i * y_j * K_ij)
        P = [[0.0 for _ in range(num_samples)] for _ in range(num_samples)]
        for i in range(num_samples):
            for j in range(num_samples):
                P[i][j] = y[i] * y[j] * K[i][j]
        q = [-1.0 for _ in range(num_samples)]
        # Box constraints: 0 < alpha_i < C
        G = np.vstack((-np.eye(num_samples), np.eye(num_samples)))
        h = np.hstack((np.zeros(num_samples), self.C * np.ones(num_samples)))
        # Equality constraint: A and b
        A = y.reshape(1, -1)
        b = 0.0

        # Convert to cvxopt format
        P = cvxopt.matrix(P)
        q = cvxopt.matrix(q)
        G = cvxopt.matrix(G)
        h = cvxopt.matrix(h)
        A = cvxopt.matrix(A, (1, num_samples), 'd')
        b = cvxopt.matrix(b)

        # Solve the QP problem and measure time
        cvxopt.solvers.options['show_progress'] = False
        start_time = time.time()
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.dual_time = time.time() - start_time

        # Extract dual variables (alpha_i)
        self.alpha_dual = np.array(solution['x']).flatten()

        # Compute w and b
        self.w_dual = np.sum((self.alpha_dual * y).reshape(-1, 1) * X, axis=0)
        support_vector_indices = [i for i in range(num_samples) if 1e-5 < self.alpha_dual[i] < self.C - 1e-5]
        if len(support_vector_indices)!=0:
            self.b_dual = np.mean(y[support_vector_indices] - np.dot(X[support_vector_indices], self.w_dual))
        else:
            self.b_dual = 0

        # Identify misclassified (non-separable) points
        predictions = np.sign(np.dot(X, self.w_dual) + self.b_dual)
        non_separable_indices = [] # y_i (w^T x_i + b) < 0
        for i in range(num_samples):
            if predictions[i] != y[i]:
                non_separable_indices.append(i)
        return self.w_dual, self.b_dual, self.alpha_dual, non_separable_indices
    def Compare_solutions(self):
        print(f"Primal SVM Time: {self.primal_time:.4f} seconds")
        print(f"Dual SVM Time: {self.dual_time:.4f} seconds")
        w_diff = np.linalg.norm(self.w_primal - self.w_dual)
        b_diff = abs(self.b_primal - self.b_dual)
        print(f"Difference in w (L2 norm): {w_diff:.3f}")
        print(f"Difference in b: {b_diff:.3f}")
    def save_non_seperable_indices(self,indices,sr_number):
        file_name = f"inseparable_{sr_number}.csv"
        np.savetxt(file_name,indices,delimiter=",",fmt = "%d")
        print(f"Indices of non-separable points saved to {file_name}")
    
svm = SVM(regularization_parameter=1.0)
w_primal, b_primal, slack_values, primal_non_separable_indices = svm.Primal_SVM(train_features, train_labels)
w_dual, b_dual, alpha_dual, dual_non_separable_indices = svm.Dual_SVM(train_features, train_labels)
print("Primal SVM Results:")
print(f"Number of points with ξ_i > 1: {len(primal_non_separable_indices)}")
print(f"Indices of non-separable points (primal): {primal_non_separable_indices}")
print("\nDual SVM Results:")
print(f"Number of misclassified points: {len(dual_non_separable_indices)}")
print(f"Indices of non-separable points (dual): {dual_non_separable_indices}")
svm.Compare_solutions()
svm.save_non_seperable_indices(primal_non_separable_indices,23627)

# %%
"""
Task : Kernelized SVM; Repeat the previous construction, but with the Gaussian kernel this time. Choose
your hyperparameters such that non-separability is no longer an issue, and your decision boundary is
consistent with the training data’s labels.

Question : The final misclassification rate for the kernelized SVM for Task 3.
"""

# %%
import numpy as np
from cvxopt import matrix, solvers

class Gaussian_Kernel_SVM:
    def __init__(self, C, gamma):
        self.C = C              # Regularization parameter
        self.gamma = gamma      # Gaussian kernel parameter
        self.alphas = None      # Lagrange multipliers
        self.b = None           # Bias term
        self.X_train = None     # Training features
        self.y_train = None     # Training labels
        self.support_vectors = None  # Support vectors
        self.support_indices = None  # Indices of support vectors

    def Gaussian_kernel(self, x1, x2):
        diff = x1 - x2
        return np.exp(-self.gamma * np.dot(diff, diff))

    def Kernel_matrix(self, X):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.Gaussian_kernel(X[i], X[j])
        return K

    def Dual_SVM(self, X, y):
        self.X_train = X
        self.y_train = y
        n = X.shape[0]

        # Compute kernel matrix
        K = self.Kernel_matrix(X)

        # Set up quadratic programming problem
        P = matrix(np.outer(y, y) * K)          # Kernel matrix scaled by labels
        q = matrix(-np.ones(n))                 # Linear term
        G = matrix(np.vstack((-np.eye(n), np.eye(n))))  # Inequality constraints
        h = matrix(np.hstack((np.zeros(n), self.C * np.ones(n))))  
        A = matrix(y, (1, n), 'd')              # Equality constraint (sum alpha_i y_i = 0)
        b = matrix(0.0)                         

        # Solve the QP problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        self.alphas = np.array(solution['x']).flatten()

        # Identify support vectors (alpha_i > threshold)
        self.support_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = X[self.support_indices]

        # Compute bias term using a support vector
        for i in self.support_indices:
            if self.alphas[i] < self.C - 1e-5:
                self.b = y[i] - np.sum(self.alphas[self.support_indices] * y[self.support_indices] * K[self.support_indices, i])
                break
        else:
            self.b = 0  # Fallback if no suitable support vector found
        return self.alphas, self.b

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            kernel_values = np.array([self.Gaussian_kernel(x, sv) for sv in self.support_vectors])
            decision = np.sum(self.alphas[self.support_indices] * self.y_train[self.support_indices] * kernel_values) + self.b
            predictions.append(np.sign(decision))
        return np.array(predictions)

    def misclassification_rate(self, X, y):
        predictions = self.predict(X)
        errors = np.sum(predictions != y)
        return (errors / len(y)) * 100
    
svm = Gaussian_Kernel_SVM(C=1.0, gamma=999)
svm.Dual_SVM(train_features, train_labels)
train_misclassification = svm.misclassification_rate(train_features, train_labels)
print(f"Training misclassification rate: {train_misclassification:.2f}%")
print(f"Optimized Hyper parameters for 0 misclassification rate: C = 1.0, gamma = 999")

# %%
"""
Task : Perceptron, again; Retrain the perceptron, after removing the sources of non-separability isolated
by the linear SVM. Verify that it converges

Question : A plot between misclassification rate and iterations for the perceptron for Task 4.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
svm = SVM(regularization_parameter=1.0)
def Task_4(X, y, C=1.0):
    # Identify the points causing inseparability using the primal solution
    w_primal, b_primal, slack_values, primal_non_separable_indices = svm.Primal_SVM(X, y)
    points_to_remove = X[primal_non_separable_indices]
    print(f"Points to remove: {len(points_to_remove)} ")
    mask = np.ones(len(X), dtype=bool)
    mask[primal_non_separable_indices] = False
    X_modified, y_modified = X[mask], y[mask]
    print(f"Modified dataset: {len(X_modified)} points")
    w, b, misclassification_rates, converged = Perceptron(X_modified, y_modified)
    if converged:
        print("Perceptron converged")
    else:
        print("Perceptron did not converge within max_iterations")
    print(f"Final misclassification rate: {misclassification_rates[-1]:.4f}")
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(misclassification_rates) + 1), misclassification_rates, marker='o', linestyle='-')
    plt.xlabel('No of Iterations')
    plt.ylabel('Misclassification Rate')
    plt.title('Perceptron Misclassification Rate vs. No of Iterations (Task 4)')
    plt.grid(True)
    plt.show()

X, y = train_features, train_labels
Task_4(X, y, C=1.0)
# %%
"""Question-2 : Logistic Regression, MLP, CNN & PCA """
# %%
# Preprocessing and imports 
import Oracle_Assignment_2
import os 
import numpy as np
from PIL import Image
import torch 
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
plt.style.use('default')
data = Oracle_Assignment_2.q2_get_mnist_jpg_subset(23627)
data_directory = "/home/saisandeshk/Study/Assignment-2/q2_data"

# %%
"""
Step-1 : Prepare the data
"""

# %%
# Step 1.1 : Verify the data 
def Verify_data(data_directory):
    total_images = 0
    for class_folder in range(10):
        folder_path = os.path.join(data_directory, str(class_folder))
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder {folder_path} does not exist!")
        num_images = len([f for f in os.listdir(folder_path) if f.endswith(".jpg")])
        print(f"Class {class_folder}: {num_images} images")
        total_images += num_images
    if total_images != 10000:
        raise ValueError(f"Expected 10,000 images, found {total_images}")
    print(f"Total images verified: {total_images}")
Verify_data(data_directory)
# Step 1.2 : Prepare Data for CNN and MLP
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale (1 channel)
    transforms.ToTensor(),                       # Convert to tensor (1x28x28)
    transforms.Normalize((0.5,), (0.5,))         # Normalize to [-1, 1]
])
full_dataset = datasets.ImageFolder(root=data_directory, transform=transform)
# Manual split: 800 train, 200 test per class (10 classes = 8000 train, 2000 test)
train_indices = []
test_indices = []
images_per_class = 1000
train_per_class = 800
test_per_class = 200

for class_idx in range(10):
    start_idx = class_idx * images_per_class
    train_indices.extend(range(start_idx, start_idx + train_per_class))  # 0-799, 1000-1799, etc.
    test_indices.extend(range(start_idx + train_per_class, start_idx + images_per_class))  # 800-999, etc.

# Create training and testing subsets
train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

print(f"PyTorch Train dataset size: {len(train_dataset)}")
print(f"PyTorch Test dataset size: {len(test_dataset)}")

# Verify a sample to ensure correct loading
sample_img, sample_label = train_dataset[0]
print(f"Sample image shape: {sample_img.shape}, Label: {sample_label}")  # Should be torch.Size([1, 28, 28])
# Step 1.3 : Prepare data for Logistic Regression and PCA
# Load and flatten images into NumPy arrays
def load_numpy_data(data_directory):
    X_data = []
    y_data = []
    for class_idx in range(10):
        folder_path = os.path.join(data_directory, str(class_idx))
        for img_file in sorted(os.listdir(folder_path)):  # Sort for consistency
            if img_file.endswith(".jpg"):
                img_path = os.path.join(folder_path, img_file)
                # Open image, convert to grayscale, and flatten
                img = Image.open(img_path).convert("L")  # 'L' mode for grayscale
                img_array = np.array(img).flatten()      # Flatten 28x28 to 784
                img_array = img_array / 255.0            # Normalize to [0, 1]
                X_data.append(img_array)
                y_data.append(class_idx)
    return np.array(X_data), np.array(y_data)

# Load full dataset
X_full_np, y_full_np = load_numpy_data(data_directory)
print(f"Full NumPy data shape: {X_full_np.shape}, Labels shape: {y_full_np.shape}")  # (10000, 784), (10000,)

# Manual split: 800 train, 200 test per class
X_train_np = []
y_train_np = []
X_test_np = []
y_test_np = []

for class_idx in range(10):
    class_data = X_full_np[class_idx * images_per_class:(class_idx + 1) * images_per_class]
    class_labels = y_full_np[class_idx * images_per_class:(class_idx + 1) * images_per_class]
    # Split: first 800 for train, last 200 for test
    X_train_np.append(class_data[:train_per_class])
    y_train_np.append(class_labels[:train_per_class])
    X_test_np.append(class_data[train_per_class:])
    y_test_np.append(class_labels[train_per_class:])

# Concatenate across classes
X_train_np = np.concatenate(X_train_np, axis=0)  # (8000, 784)
y_train_np = np.concatenate(y_train_np, axis=0)  # (8000,)
X_test_np = np.concatenate(X_test_np, axis=0)    # (2000, 784)
y_test_np = np.concatenate(y_test_np, axis=0)    # (2000,)

print(f"NumPy Train data shape: {X_train_np.shape}, Labels shape: {y_train_np.shape}")
print(f"NumPy Test data shape: {X_test_np.shape}, Labels shape: {y_test_np.shape}")

# --- Step 1.4: Validate the Split ---
# Check class distribution in NumPy data
for class_idx in range(10):
    train_count = np.sum(y_train_np == class_idx)
    test_count = np.sum(y_test_np == class_idx)
    print(f"Class {class_idx}: Train = {train_count}, Test = {test_count}")
    assert train_count == 800 and test_count == 200, f"Split failed for class {class_idx}"

# --- Step 1.5: Cache and Organize ---
# Save NumPy arrays for later use
np.save("X_train_np.npy", X_train_np)
np.save("y_train_np.npy", y_train_np)
np.save("X_test_np.npy", X_test_np)
np.save("y_test_np.npy", y_test_np)

# %%
"""
Step-2 : Train the MLP on Flattened Images (Task 1)
"""

# %%
# imports 
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader 

torch.manual_seed(0)  # For reproducibility
# Step 2.1 : Define the MLP Architecture 
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.flatten = nn.Flatten() # Flatten 1*28*28 into 784
        self.layers = nn.Sequential(
            nn.Linear(784, 128), # 784 input, 128 (first hidden layer)
            nn.ReLU(),           # ReLU activation
            nn.Linear(128,64),   # 128 input, 64 (second hidden layer)
            nn.ReLU(),           # ReLU activation
            nn.Linear(64, 10)   # 64 input, 10 output (for 10 classes) no softmax here
        )
    def forward(self, x):
        x = self.flatten(x)  # Convert 1x28x28 to 784
        x = self.layers(x)   # Pass through layers
        return x  # (softmax applied in loss function)
model = MLP()
print("MLP Architecture:")
print(model)
# Step 2.2 : Prepare Data Loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Verify a batch 
for images, labels in train_loader:
    print(f"Batch image shape: {images.shape}, Label shape: {labels.shape}")  # [32, 1, 28, 28], [32]
    break
# Step 2.3 : Train the MLP
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss() # combines nn.LogSoftmax() and nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas = (0.9, 0.999))

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU if available
print(f"Training on: {device}")
for epoch in range(num_epochs):
    model.train()  # Set to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)  # Raw logits
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
# Step 2.4 : Evaluate the MLP
model.eval()  # Set to evaluation mode
# preallocate the Numpy arrays for storing results 
num_test_samples = len(test_dataset)  # 2000 from Step 1
test_predictions = np.zeros(num_test_samples, dtype=int)
test_labels = np.zeros(num_test_samples, dtype=int)
test_probabilities = np.zeros((num_test_samples, 10))  # 10 classes

with torch.no_grad():
    correct = 0
    total = 0
    idx = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)  # Raw logits
        
        # Get probabilities (softmax)
        probs = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)  # Predicted class
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Store results
        test_predictions[idx:idx + batch_size] = predicted.cpu().numpy()
        test_labels[idx:idx + batch_size] = labels.cpu().numpy()
        test_probabilities[idx:idx + batch_size] = probs.cpu().numpy()
        
        idx += batch_size
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

# Convert to NumPy arrays
test_predictions = np.array(test_predictions)  # (2000,)
test_labels = np.array(test_labels)            # (2000,)
test_probabilities = np.array(test_probabilities)  # (2000, 10)

# Step 2.5: Analyse Results
# Analysiing 
true_counts = np.bincount(test_labels, minlength=10)  # Count true labels per class
pred_counts = np.bincount(test_predictions, minlength=10)  # Count predicted labels per class
mistakes = np.array([np.sum((test_predictions != test_labels) & (test_labels == cls)) for cls in range(10)])


print("\nClass-wise analysis:")
for cls in range(10):
    print(f"Class {cls}:")
    print(f"  True count: {true_counts[cls]}")
    print(f"  Predicted count: {pred_counts[cls]}")
    print(f"  Mistakes: {mistakes[cls]}")
# Step 2.6 : PLot graph 
plt.figure(figsize=(10, 6))
classes = np.arange(10)
bar_width = 0.35
plt.bar(classes - bar_width/2, true_counts, bar_width, label="True", color="blue")
plt.bar(classes + bar_width/2, pred_counts, bar_width, label="Predicted", color="orange")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("True vs Predicted Counts per Class")
plt.xticks(classes)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("class_counts_plot.png")
plt.show()
# Step 2.7 : Save results 
torch.save(model.state_dict(), "mlp_model.pth")
print("Model saved to mlp_model.pth")
np.savetxt("mlp_test_predictions.txt", test_predictions, fmt="%d")
np.savetxt("mlp_test_labels.txt", test_labels, fmt="%d")
np.savetxt("mlp_test_probabilities.txt", test_probabilities)


# %%
"""
Step 3 : Perform PCA (Task 3)
"""

# %%
# Step 3.1 : Center the Training data as X need to be centererd(meaning sum(all x_i's) = 0) for applying PCA
# We already have the training data in X_train_np and test data in X_test_np each of shape (8000, 784) and (2000, 784) respectively
n,d = len(X_train_np), len(X_train_np[0])
mean_vector = np.mean(X_train_np, axis=0)
X_train_centered = X_train_np - mean_vector
X_test_centered = X_test_np - mean_vector
# Step 3.2 : Compute PCA via SVD 
U,sigma,Vt = np.linalg.svd(X_train_centered, full_matrices=False) 
V = Vt.T # V: Principal directions, shape (d, d)
lambda_values = sigma**2 / (n-1) # Eigenvalues from singualr values 
# Step 3.3 : Compute the explained variance
Explained_variance_ratio = lambda_values / np.sum(lambda_values)
R_k = np.cumsum(Explained_variance_ratio) # R_k = cummulative Explained variance ratio 
k_idx = np.argmax(R_k >= 0.95) # k = number of principal components to retain 95% of variance
k = k_idx + 1 if k_idx > 0 or R_k[0] >= 0.95 else min(len(lambda_values), d)  # Fallback to max components
print(f"Number of principal components to retain 95% of variance: {k}")
# Step 3.4 :PLot(1) Cummulative Explained Variance 
plt.figure(figsize=(8, 5))
plt.plot(R_k, marker='o', linestyle='--', color='b', label='Cumulative Variance')
plt.axhline(y=0.95, color='r', linestyle='-', label='95% Threshold')
plt.axvline(x=k, color='r', linestyle='-', label=f'k = {k}')
plt.title("Cumulative Explained Variance Ratio ($R_k$)")
plt.xlabel("Number of Components ($k$)")
plt.ylabel("Cumulative Variance ($R_k$)")
plt.legend()
plt.grid(True)
plt.savefig("cumulative_variance.png")
plt.show()
# Step 3.5 : Transform data into low-dimensional space 
W_k = V[:, :k]  # W_k: Projection matrix, shape (d, k)
Y_train = np.matmul(X_train_centered,W_k)  # Y: Low-dimensional representation, shape (n, k)
# Step 3.6 : Transform test data 
Y_test = np.matmul(X_test_centered, W_k)  
# Step 3.7 : High-dimensinal vs low-dimensional representation
sample_idx = 1600  # Pick any sample for visualization
high_dim = X_train_np[sample_idx]  # Original high-dimensional sample
low_dim = Y_train[sample_idx]      # PCA-transformed low-dimensional sample

# High-dimensional plot (as image for MNIST-like data)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
if d == 784:  # Assuming MNIST-like 28x28 images
    plt.imshow(high_dim.reshape(28, 28), cmap='gray')
plt.title("High-Dimensional ($X$), Shape: (d=784)")
plt.axis('off')

# Low-dimensional plot (first 2 components for scatter, if k >= 2)
plt.subplot(1, 2, 2)
if k >= 2:
    plt.scatter(Y_train[:, 0], Y_train[:, 1], c='blue', s=10, alpha=0.5)
    plt.scatter(low_dim[0], low_dim[1], c='red', s=100, label='Sample')
    plt.title(f"Low-Dimensional ($Y$), Shape: (k={k})")
    plt.xlabel("$Y_1$ (First Component)")
    plt.ylabel("$Y_2$ (Second Component)")
    plt.legend()
else:
    plt.text(0.5, 0.5, f"k={k} < 2, no scatter plot", ha='center', va='center')
    plt.title(f"Low-Dimensional ($Y$), Shape: (k={k})")
plt.grid(True)
plt.savefig("high_low_dim.png")
plt.show()

# Step 3.8: Save Results 
np.save("Y_train_pca.npy", Y_train)
np.save("Y_test_pca.npy", Y_test)
np.save("W_k.npy", W_k)
np.save("mu.npy", mean_vector)

print(f"Transformed training data shape: {Y_train.shape}")
print(f"Transformed test data shape: {Y_test.shape}")
print("PCA completed and results saved.")


# %%
"""
Step 4 : Convolution Neural Network: Construct and train a CNN that takes direct image as input and
gives class probabilities as output
"""

# %%
# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)

# Step 4.1: Define the CNN Architecture (LeNet-5 Inspired)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)  # 1x28x28 -> 6x24x24
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 6x24x24 -> 6x12x12
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)  # 6x12x12 -> 16x8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x8x8 -> 16x4x4
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Flatten 16x4x4 (256) -> 120
        self.fc2 = nn.Linear(120, 84)          # 120 -> 84
        self.fc3 = nn.Linear(84, 10)           # 84 -> 10 (output for 10 classes)
    
    def forward(self, x):
        # Input shape: (batch_size, 1, 28, 28)
        x = F.relu(self.conv1(x))  # Apply conv1 and ReLU
        x = self.pool1(x)          # Apply pooling
        x = F.relu(self.conv2(x))  # Apply conv2 and ReLU
        x = self.pool2(x)          # Apply pooling
        
        # Flatten the output for fully connected layers
        x = x.view(-1, 16 * 4 * 4)  # Reshape: (batch_size, 256)
        x = F.relu(self.fc1(x))     # FC1 with ReLU
        x = F.relu(self.fc2(x))     # FC2 with ReLU
        x = self.fc3(x)             # FC3 (output logits, no softmax here)
        return x

# Initialize the model
cnn_model = CNN()
print("CNN Architecture (LeNet-5 Inspired):")
print(cnn_model)

# --- Step 4.2: Prepare Data Loaders ---
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Verify a batch to ensure correct shape
for images, labels in train_loader:
    print(f"Batch image shape: {images.shape}, Label shape: {labels.shape}")  # Expected: [32, 1, 28, 28], [32]
    break

# --- Step 4.3: Train the CNN ---
criterion = nn.CrossEntropyLoss()  # Combines log-softmax and NLL loss
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001, betas=(0.9, 0.999))

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model.to(device)
print(f"Training on: {device}")

print("Starting CNN training...")
for epoch in range(num_epochs):
    cnn_model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# --- Step 4.4: Evaluate the CNN ---
cnn_model.eval()

# Pre-allocate arrays for results
num_test_samples = len(test_dataset)  # 2000
test_predictions = np.zeros(num_test_samples, dtype=int)
test_labels = np.zeros(num_test_samples, dtype=int)
test_probabilities = np.zeros((num_test_samples, 10))

with torch.no_grad():
    correct = 0
    total = 0
    idx = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn_model(images)
        
        _, predicted = torch.max(outputs, 1)
        # Compute probabilities only for storage
        probs = F.softmax(outputs, dim=1)
        
        batch_size = labels.size(0)
        test_predictions[idx:idx + batch_size] = predicted.cpu().numpy()
        test_labels[idx:idx + batch_size] = labels.cpu().numpy()
        test_probabilities[idx:idx + batch_size] = probs.cpu().numpy()
        idx += batch_size
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"CNN Test Accuracy: {accuracy:.2f}%")

print("Test predictions shape:", test_predictions.shape)
print("Test labels shape:", test_labels.shape)
print("Test probabilities shape:", test_probabilities.shape)

# --- Step 4.5: Analyze and Plot Results ---
# Count true and predicted instances per class
true_counts = np.bincount(test_labels, minlength=10)  # True labels per class
pred_counts = np.bincount(test_predictions, minlength=10)  # Predicted labels per class
mistakes = np.array([np.sum((test_predictions != test_labels) & (test_labels == cls)) for cls in range(10)])

print("\nClass-wise analysis:")
for cls in range(10):
    print(f"Class {cls}:")
    print(f"  True count: {true_counts[cls]}")
    print(f"  Predicted count: {pred_counts[cls]}")
    print(f"  Mistakes: {mistakes[cls]}")

# Bar plot: True vs Predicted counts
plt.figure(figsize=(10, 6))
classes = np.arange(10)
bar_width = 0.35
plt.bar(classes - bar_width/2, true_counts, bar_width, label="True", color="blue")
plt.bar(classes + bar_width/2, pred_counts, bar_width, label="Predicted", color="orange")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("True vs Predicted Counts per Class (CNN)")
plt.xticks(classes)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("cnn_class_counts_plot.png")
plt.show()

# --- Step 4.6: Save Results ---
torch.save(cnn_model.state_dict(), "cnn_lenet5_model.pth")
print("CNN model saved to cnn_lenet5_model.pth")

np.savetxt("cnn_test_predictions.txt", test_predictions, fmt="%d")
np.savetxt("cnn_test_labels.txt", test_labels, fmt="%d")
np.savetxt("cnn_test_probabilities.txt", test_probabilities)

# %%
"""
Step-5 : Train the MLP with PCA Features
"""

# %%
# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(0)

# Load PCA-transformed data from Step 3 (replace with your actual file paths if saved)
Y_train_pca = np.load("Y_train_pca.npy")  # shape (8000, k)
Y_test_pca = np.load("Y_test_pca.npy")    # shape (2000, k)
y_train_np = np.load("y_train_np.npy")    # shape (8000,)
y_test_np = np.load("y_test_np.npy")      # shape (2000,)

# --- Step 5.1: Define the MLP Architecture for PCA Features ---
class MLPWithPCA(nn.Module):
    def __init__(self, k):
        super(MLPWithPCA, self).__init__()
        # No flatten needed; input is already 1D PCA features
        self.layers = nn.Sequential(
            nn.Linear(k, 128),   # Input: k PCA features -> 128 (first hidden layer)
            nn.ReLU(),           # ReLU activation
            nn.Linear(128, 64),  # 128 -> 64 (second hidden layer)
            nn.ReLU(),           # ReLU activation
            nn.Linear(64, 10)    # 64 -> 10 (output for 10 classes, no softmax)
        )
    
    def forward(self, x):
        # Input shape: (batch_size, k)
        x = self.layers(x)  # Pass through layers
        return x  # Raw logits (softmax applied in loss function)

# Get k from the PCA data shape
k = Y_train_pca.shape[1]  # Number of PCA components
mlp_pca_model = MLPWithPCA(k)
print(f"MLP with PCA Architecture (input size k={k}):")
print(mlp_pca_model)

# --- Step 5.2: Prepare Data Loaders with PCA Features ---
# Convert NumPy arrays to PyTorch tensors
Y_train_tensor = torch.tensor(Y_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_np, dtype=torch.long)
Y_test_tensor = torch.tensor(Y_test_pca, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_np, dtype=torch.long)

# Create TensorDatasets
train_dataset_pca = TensorDataset(Y_train_tensor, y_train_tensor)
test_dataset_pca = TensorDataset(Y_test_tensor, y_test_tensor)

# Set up DataLoaders
batch_size = 32
train_loader_pca = DataLoader(train_dataset_pca, batch_size=batch_size, shuffle=True)
test_loader_pca = DataLoader(test_dataset_pca, batch_size=batch_size, shuffle=False)

# Verify a batch
for features, labels in train_loader_pca:
    print(f"Batch feature shape: {features.shape}, Label shape: {labels.shape}")  # Expected: [32, k], [32]
    break

# --- Step 5.3: Train the MLP with PCA Features ---
criterion = nn.CrossEntropyLoss()  # Combines log-softmax and NLL loss
optimizer = optim.Adam(mlp_pca_model.parameters(), lr=0.001, betas=(0.9, 0.999))

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp_pca_model.to(device)
print(f"Training on: {device}")

print("Starting MLP with PCA training...")
for epoch in range(num_epochs):
    mlp_pca_model.train()
    running_loss = 0.0
    for features, labels in train_loader_pca:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = mlp_pca_model(features)  # Raw logits
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * features.size(0)
    
    epoch_loss = running_loss / len(train_loader_pca.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# --- Step 5.4: Evaluate the MLP with PCA Features ---
mlp_pca_model.eval()

# Pre-allocate arrays for results
n_test = len(test_dataset_pca)  # 2000
test_predictions = np.zeros(n_test, dtype=int)
test_labels = np.zeros(n_test, dtype=int)
test_probabilities = np.zeros((n_test, 10))

with torch.no_grad():
    correct = 0
    total = 0
    idx = 0
    for features, labels in test_loader_pca:
        features, labels = features.to(device), labels.to(device)
        outputs = mlp_pca_model(features)
        
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        batch_size = labels.size(0)
        test_predictions[idx:idx + batch_size] = predicted.cpu().numpy()
        test_labels[idx:idx + batch_size] = labels.cpu().numpy()
        test_probabilities[idx:idx + batch_size] = probs.cpu().numpy()
        idx += batch_size
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"MLP with PCA Test Accuracy: {accuracy:.2f}%")

print("Test predictions shape:", test_predictions.shape)
print("Test labels shape:", test_labels.shape)
print("Test probabilities shape:", test_probabilities.shape)

# --- Step 5.5: Analyze and Plot Results ---
true_counts = np.bincount(test_labels, minlength=10)  # True labels per class
pred_counts = np.bincount(test_predictions, minlength=10)  # Predicted labels per class
mistakes = np.array([np.sum((test_predictions != test_labels) & (test_labels == cls)) for cls in range(10)])

print("\nClass-wise analysis:")
for cls in range(10):
    print(f"Class {cls}:")
    print(f"  True count: {true_counts[cls]}")
    print(f"  Predicted count: {pred_counts[cls]}")
    print(f"  Mistakes: {mistakes[cls]}")

# Bar plot: True vs Predicted counts
plt.figure(figsize=(10, 6))
classes = np.arange(10)
bar_width = 0.35
plt.bar(classes - bar_width/2, true_counts, bar_width, label="True", color="blue")
plt.bar(classes + bar_width/2, pred_counts, bar_width, label="Predicted", color="orange")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title(f"True vs Predicted Counts per Class (MLP with PCA, k={k})")
plt.xticks(classes)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig("mlp_pca_class_counts_plot.png")
plt.show()

# --- Step 5.6: Save Results ---
torch.save(mlp_pca_model.state_dict(), "mlp_pca_model.pth")
print("MLP with PCA model saved to mlp_pca_model.pth")

np.savetxt("mlp_pca_test_predictions.txt", test_predictions, fmt="%d")
np.savetxt("mlp_pca_test_labels.txt", test_labels, fmt="%d")
np.savetxt("mlp_pca_test_probabilities.txt", test_probabilities)

# %%
"""
Step 6: Logistic Regression with PCA; Train a Logistic Regression model for multi class classification and also train 10 binary classifiers for each class by one vs rest approach using PCA features.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# Load PCA-transformed data and labels from previous steps
Y_train_pca = np.load("Y_train_pca.npy")  # Shape: (8000, k)
Y_test_pca = np.load("Y_test_pca.npy")    # Shape: (2000, k)
y_train_np = np.load("y_train_np.npy")    # Shape: (8000,)
y_test_np = np.load("y_test_np.npy")      # Shape: (2000,)

# --- Step 6.1: Prepare the Data for OvR ---
n_train, k = Y_train_pca.shape  # n_train: 8000 samples, k: PCA features
n_test = Y_test_pca.shape[0]    # n_test: 2000 samples

# Create binary labels for each class (0-9)
binary_labels_train = {}
for cls in range(10):
    binary_labels_train[cls] = (y_train_np == cls).astype(int)  # 1 if class cls, 0 otherwise

# --- Step 6.2: Define Logistic Regression Functions ---
def sigmoid(z):
    """Compute sigmoid function: 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to avoid overflow

def compute_loss(y, y_pred):
    """Binary cross-entropy loss"""
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def gradient_descent(X, y, w, learning_rate, num_iterations):
    """Train binary logistic regression with gradient descent"""
    for _ in range(num_iterations):
        z = X @ w
        y_pred = sigmoid(z)
        gradient = (X.T @ (y_pred - y)) / len(y)
        w -= learning_rate * gradient
    return w

# --- Step 6.3: Train 10 Binary OvR Classifiers ---
learning_rate = 0.01
num_iterations = 10000
weights = {}  # Store weights for each classifier

print("Training 10 OvR Logistic Regression classifiers...")
for cls in range(10):
    # Initialize weights for class cls
    w = np.zeros(k)  # Shape: (k,)
    
    # Train on training data
    w = gradient_descent(Y_train_pca, binary_labels_train[cls], w, learning_rate, num_iterations)
    weights[cls] = w
    
    # Print training progress
    y_pred_train = sigmoid(Y_train_pca @ w)
    loss = compute_loss(binary_labels_train[cls], y_pred_train)
    print(f"Class {cls} trained, Training Loss: {loss:.4f}")

# --- Step 6.4: Evaluate the OvR Model ---
# Predict probabilities for test data
probs_test = np.zeros((n_test, 10))  # Shape: (2000, 10)
for cls in range(10):
    probs_test[:, cls] = sigmoid(Y_test_pca @ weights[cls])

# Get predictions by choosing the class with the highest probability
test_predictions = np.argmax(probs_test, axis=1)  # Shape: (2000,)
test_labels = y_test_np  # Shape: (2000,)

# Calculate overall accuracy
accuracy = np.mean(test_predictions == test_labels) * 100
print(f"\nOvR Logistic Regression Test Accuracy: {accuracy:.2f}%")

# --- Step 6.5: Analyze and Plot Results ---
# Class-wise accuracy
class_accuracies = np.zeros(10)
true_counts = np.bincount(test_labels, minlength=10)  # True counts per class
for cls in range(10):
    class_mask = (test_labels == cls)
    if np.sum(class_mask) > 0:
        class_accuracies[cls] = np.mean(test_predictions[class_mask] == cls) * 100

print("\nClass-wise accuracy:")
for cls in range(10):
    print(f"Class {cls}: {class_accuracies[cls]:.2f}% (True count: {true_counts[cls]})")

# Bar plot: Class-wise accuracy
plt.figure(figsize=(10, 6))
classes = np.arange(10)
plt.bar(classes, class_accuracies, color="teal", label="Accuracy per Class")
plt.xlabel("Class")
plt.ylabel("Accuracy (%)")
plt.title(f"Class-wise Accuracy for OvR Logistic Regression (k={k})")
plt.xticks(classes)
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.savefig("ovr_logreg_class_accuracy.png")
plt.show()

# --- Step 6.6: Save Results ---
# Save weights for each classifier
for cls in range(10):
    np.save(f"ovr_weights_class_{cls}.npy", weights[cls])

# Save predictions and probabilities
np.savetxt("ovr_test_predictions.txt", test_predictions, fmt="%d")
np.savetxt("ovr_test_labels.txt", test_labels, fmt="%d")
np.savetxt("ovr_test_probabilities.txt", probs_test)

# %%
"""
Deliverables
1. Reconstruct an image of your choice using principal components (1,2,3,....) and conclude the results.

"""

# %%
import numpy as np
import matplotlib.pyplot as plt


# Step 2: Recompute PCA with all components
# Center the training data
n_train, d = X_train_np.shape  # n_train: 8000, d: 784
mu = np.mean(X_train_np, axis=0)  # Mean vector, shape: (784,)
X_train_centered = X_train_np - mu  # Shape: (8000, 784)

# Compute SVD
U, Sigma, Vt = np.linalg.svd(X_train_centered, full_matrices=False)  # SVD: X_centered = U * Sigma * V^T
W_k = Vt.T  # W_k: Principal components, shape: (784, 784)

# Step 3: Choose and prepare the image
image_idx = np.random.randint(0, X_test_np.shape[0])  # Any random test image index
original_image = X_test_np[image_idx]  # Shape: (784,)
original_image_2d = original_image.reshape(28, 28)  # Reshape to 28x28

# Step 4: Center the image
centered_image = original_image - mu  # Shape: (784,)

# Step 5: Reconstruct with different numbers of components
k_values = [1, 50, 200, 400, 600, 784]  # Numbers of components to try
reconstructions = []

for k in k_values:
    W_k_partial = W_k[:, :k]  # Shape: (784, k)
    
    # Project the centered image onto k components
    projection = np.matmul(centered_image,W_k_partial)  # Shape: (k,)
    # Reconstruction from k components
    reconstructed_centered = np.matmul(W_k_partial,projection)  # Shape: (784,)
    reconstructed_image = reconstructed_centered + mu  # Shape: (784,)
    
    # Reshape to 2D for visualization
    reconstructed_image_2d = reconstructed_image.reshape(28, 28)
    reconstructions.append(reconstructed_image_2d)

    # Check the difference at k=784
    if k == 784:
        diff = np.abs(original_image - reconstructed_image)
        print(f"Max difference at k=784: {np.max(diff)}")

# Step 6: Visualize the original and reconstructed images
plt.figure(figsize=(12, 6))

# Plot the original image
plt.subplot(2, len(k_values) + 1, 1)
plt.imshow(original_image_2d, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Plot each reconstruction
for i, k in enumerate(k_values):
    plt.subplot(2, len(k_values) + 1, i + 2)
    plt.imshow(reconstructions[i], cmap='gray')
    plt.title(f"k = {k}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("image_reconstruction.png")
plt.show()
# Step 7: Analyze and conclude
print("Analysis of Image Reconstruction with PCA Components:")
print("- k=1: Very blurry, only basic shape visible.")
print("- k=10: Some details emerge, but still fuzzy.")
print("- k=50: Image becomes recognizable as a '0', though edges are rough.")
print("- k=100: Much clearer, close to original with minor distortions.")
print("- k=200: Nearly identical to the original, few differences.")
print("- k=784: Should match the original exactly (all components).")
print("\nConclusion: Around 50-100 components make the image recognizable, while 200+ components yield a near-perfect reconstruction. The difference at k=784 should now be near zero, confirming the PCA process is correct.")

# %%
"""
Deliverables :

2. Print confusion matrix for all the multi class classification models and compare them using following
metrics : Accuracy, precision, recall and F1 score for each class and compare the results
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Gather Predictions and True Labels
# Load predictions and labels for each model (replace with your actual file paths)
models = {
    "MLP": {
        "predictions": np.loadtxt("mlp_test_predictions.txt", dtype=int),
        "labels": np.loadtxt("mlp_test_labels.txt", dtype=int)
    },
    "CNN": {
        "predictions": np.loadtxt("cnn_test_predictions.txt", dtype=int),
        "labels": np.loadtxt("cnn_test_labels.txt", dtype=int)
    },
    "MLP with PCA": {
        "predictions": np.loadtxt("mlp_pca_test_predictions.txt", dtype=int),
        "labels": np.loadtxt("mlp_pca_test_labels.txt", dtype=int)
    },
    "Logistic Regression with PCA": {
        "predictions": np.loadtxt("ovr_test_predictions.txt", dtype=int),
        "labels": np.loadtxt("ovr_test_labels.txt", dtype=int)
    }
}

# Verify shapes (should be (2000,) for each)
for model_name, data in models.items():
    print(f"{model_name} - Predictions shape: {data['predictions'].shape}, Labels shape: {data['labels'].shape}")

# Step 2: Generate Confusion Matrices
confusion_matrices = {}
for model_name, data in models.items():
    confusion_matrix = np.zeros((10, 10), dtype=int)
    predictions = data["predictions"]
    labels = data["labels"]
    # Replace zip with manual nested loops
    for i in range(len(labels)):
        true_class = labels[i]
        predicted_class = predictions[i]
        confusion_matrix[true_class, predicted_class] += 1
    confusion_matrices[model_name] = confusion_matrix

# Step 3: Compute Overall Accuracy
accuracies = {}
for model_name, confusion_matrix in confusion_matrices.items():
    correct = np.sum(np.diag(confusion_matrix))
    total = np.sum(confusion_matrix)
    accuracy = (correct / total) * 100
    accuracies[model_name] = accuracy

# Step 4: Calculate Class-Wise Metrics
metrics = {}
for model_name, confusion_matrix in confusion_matrices.items():
    precision = np.zeros(10)
    recall = np.zeros(10)
    f1 = np.zeros(10)
    class_accuracy = np.zeros(10)  # Class-wise accuracy
    for cls in range(10):
        true_positive = confusion_matrix[cls, cls]
        false_positive = np.sum(confusion_matrix[:, cls]) - true_positive
        false_negative = np.sum(confusion_matrix[cls, :]) - true_positive
        total_class_instances = np.sum(confusion_matrix[cls, :])  # Total instances of class cls
        # Class-wise accuracy = TP / (Total instances of class)
        class_accuracy[cls] = (true_positive / total_class_instances) * 100 if total_class_instances > 0 else 0
        precision[cls] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall[cls] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls]) if (precision[cls] + recall[cls]) > 0 else 0
    metrics[model_name] = {"class_accuracy": class_accuracy, "precision": precision, "recall": recall, "f1": f1}

# Step 5: Visualize Confusion Matrices
for model_name, confusion_matrix in confusion_matrices.items():
    # Visualize as a heatmap using matplotlib with annotations
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    
    # Add annotations (numbers) to each cell
    for i in range(10):
        for j in range(10):
            plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', 
                     color='white' if confusion_matrix[i, j] > (confusion_matrix.max() / 2) else 'black')
    
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.show()
# Step 6: Summarize Metrics in Tables
# Table 1: Overall Accuracy
accuracy_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
accuracy_df["Accuracy"] = accuracy_df["Accuracy"].apply(lambda x: f"{x:.2f}%")
print("\nTable 1: Overall Accuracy")
print(accuracy_df.to_string(index=False))

# Table 2: Class-Wise Metrics
for cls in range(10):
    print(f"\nTable 2.{cls}: Metrics for Class {cls}")
    class_metrics = []
    for model_name in models.keys():
        row = [
            model_name,
            f"{accuracies[model_name]:.2f}%",
            f"{metrics[model_name]['class_accuracy'][cls]:.2f}%",
            f"{metrics[model_name]['precision'][cls]:.3f}",
            f"{metrics[model_name]['recall'][cls]:.3f}",
            f"{metrics[model_name]['f1'][cls]:.3f}"
        ]
        class_metrics.append(row)
    class_df = pd.DataFrame(class_metrics, columns=["Model", "Overall Accuracy", "Class Accuracy", "Precision", "Recall", "F1 Score"])
    print(class_df.to_string(index=False))
# Step 7: Compare and Highlight Differences
print("- **Overall Accuracy**:")
for model_name, acc in accuracies.items():
    print(f"  - {model_name}: {acc:.2f}%")
print("\n- **Class-Wise Observations**:")
for cls in range(10):
    print(f"  Class {cls}:")
    for model_name in models.keys():
        print(f"    - {model_name}: Precision={metrics[model_name]['precision'][cls]:.3f}, "
              f"Recall={metrics[model_name]['recall'][cls]:.3f}, "
              f"F1={metrics[model_name]['f1'][cls]:.3f}",
              f"Accuracy={metrics[model_name]['class_accuracy'][cls]:.3f}")

# %%
"""
Deliverables :

3. Compute average AUC score using ROC curves for each class obtained from 10 binary classifiers
(logistic regression) trained in Task-5
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Load the Necessary Data
# Load true labels and probabilities from Step 6 (Task 5 - OvR Logistic Regression)
y_test_np = np.loadtxt("ovr_test_labels.txt", dtype=int)  # Shape: (2000,)
probs_test = np.loadtxt("ovr_test_probabilities.txt")     # Shape: (2000, 10)

# Verify shapes
print(f"True labels shape: {y_test_np.shape}")
print(f"Probabilities shape: {probs_test.shape}")

# Create binary labels for each class (0-9)
binary_labels = {}
for cls in range(10):
    binary_labels[cls] = (y_test_np == cls).astype(int)  # 1 if class cls, 0 otherwise

# Step 2: Compute ROC Curves and AUC Scores for Each Class
def compute_roc_curve(y_true, y_score):
    # Sort scores and corresponding true labels in descending order
    indices = np.argsort(y_score)[::-1]
    y_true = y_true[indices]
    y_score = y_score[indices]
    
    # Initialize
    thresholds = np.unique(y_score)
    false_positive_rate = []
    true_positive_rate = []
    n_pos = np.sum(y_true == 1)  # Number of positive samples
    n_neg = np.sum(y_true == 0)  # Number of negative samples
    
    # Compute FPR and TPR for each threshold
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        true_positive = np.sum((y_pred == 1) & (y_true == 1))
        false_positive = np.sum((y_pred == 1) & (y_true == 0))
        true_positive_rate.append(true_positive / n_pos if n_pos > 0 else 0)
        false_positive_rate.append(false_positive / n_neg if n_neg > 0 else 0)
    
    # Add endpoints
    false_positive_rate = np.array([0] + false_positive_rate + [1])
    true_positive_rate = np.array([0] + true_positive_rate + [1])
    thresholds = np.array([1.0] + list(thresholds) + [0.0])
    
    return false_positive_rate, true_positive_rate, thresholds

def compute_auc(false_positive_rate, true_positive_rate):
    # Sort by FPR to ensure correct trapezoidal integration
    indices = np.argsort(false_positive_rate)
    false_positive_rate = false_positive_rate[indices]
    true_positive_rate = true_positive_rate[indices]
    auc = 0.0
    for i in range(len(false_positive_rate) - 1):
        auc += (false_positive_rate[i+1] - false_positive_rate[i]) * (true_positive_rate[i] + true_positive_rate[i+1]) / 2
    return auc

# Compute ROC curves and AUC scores for each class
roc_curves = {}
auc_scores = {}
for cls in range(10):
    y_true = binary_labels[cls]  # Binary labels for class cls
    y_score = probs_test[:, cls]  # Probability scores for class cls
    false_positive_rate, true_positive_rate, thresholds = compute_roc_curve(y_true, y_score)
    auc = compute_auc(false_positive_rate, true_positive_rate)
    roc_curves[cls] = (false_positive_rate, true_positive_rate)
    auc_scores[cls] = auc
    print(f"Class {cls} - AUC: {auc:.3f}")

# Step 3: Plot ROC Curves
plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 distinct colors
for cls in range(10):
    false_positive_rate, true_positive_rate = roc_curves[cls]
    plt.plot(false_positive_rate, true_positive_rate, color=colors[cls], 
             label=f"Class {cls} (AUC = {auc_scores[cls]:.3f})")

# Add diagonal line (random guessing)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random (AUC = 0.5)')
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curves for OvR Logistic Regression Classifiers")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("roc_curves.png")
plt.show()

# Step 4: Compute the Average AUC
average_auc = np.mean(list(auc_scores.values()))
print(f"\nAverage AUC across all classes: {average_auc:.3f}")

# Step 5: Summarize and Analyze Results
# Create a table of AUC scores using pandas
auc_table = [[f"Class {cls}", f"{auc_scores[cls]:.3f}"] for cls in range(10)]
auc_table.append(["Average", f"{average_auc:.3f}"])
auc_df = pd.DataFrame(auc_table, columns=["Class", "AUC"])
print("\nAUC Scores Table:")
print(auc_df.to_string(index=False))

# Compare with class-wise accuracies from Step 6
class_accuracies = {
    0: 94.50, 1: 97.50, 2: 78.50, 3: 83.50, 4: 90.00,
    5: 63.00, 6: 91.50, 7: 90.00, 8: 75.00, 9: 79.00
}
print("\nComparison with Class-wise Accuracies from Step 6:")
comparison_table = []
for cls in range(10):
    row = [f"Class {cls}", f"{class_accuracies[cls]:.2f}%", f"{auc_scores[cls]:.3f}"]
    comparison_table.append(row)
comparison_df = pd.DataFrame(comparison_table, columns=["Class", "Accuracy", "AUC"])
print(comparison_df.to_string(index=False))

# Step 6: Return the Average AUC
print(f"\nFinal Deliverable 3 Output - Average AUC: {average_auc:.3f}")
# %%
"""Question-3 : Regression"""
# %%
"""
#### 3.1 Linear Regression - Ordinary Least Squares(OLS) and Ridge Regression(RR)

"""

# %%
"""
Task-1 : Query the oracle to obtain D_train1 , D_train2 , D_test1 , and D_test2
"""

# %%
import Oracle_Assignment_2
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
data_1 = Oracle_Assignment_2.q3_linear_1(23627)
data_2 = Oracle_Assignment_2.q3_linear_2(23627)
(X_train_1,y_train_1,X_test_1,y_test_1) = data_1
(X_train_2,y_train_2,X_test_2,y_test_2) = data_2
X_train_1,y_train_1 = np.array(X_train_1),np.array(y_train_1)
X_test_1,y_test_1 = np.array(X_test_1),np.array(y_test_1)
X_train_2,y_train_2 = np.array(X_train_2),np.array(y_train_2)
X_test_2,y_test_2 = np.array(X_test_2),np.array(y_test_2)
D_1_train = (X_train_1, y_train_1)
D_2_train = (X_train_2, y_train_2)
D_1_test = (X_test_1, y_test_1)
D_2_test = (X_test_2, y_test_2)

# %%
"""
Task-2 : Compute weights using OLS and RR for D_train1 and D_train2 (use Lambda = 1)
"""

# %%
def Solve_OLS(X,y):
    X_T = X.T  # Transpose of X
    X_T_X = np.matmul(X_T, X)  # X^T X using matmul
    X_T_y = np.dot(X_T, y)  # X^T y using dot
    w = np.dot(np.linalg.inv(X_T_X), X_T_y)  # (X^T X)^(-1) (X^T y) using inverse and dot
    return w
def Solve_RR(X, y, lambda_=1.0):
    n, d = X.shape  # Number of samples and features
    X_T = X.T  # Transpose of X
    X_T_X = np.matmul(X_T, X)  # X^T X
    I = np.eye(d)  # d × d identity matrix
    reg_term = n * lambda_ * I  # Regularization term
    X_T_X_reg = X_T_X + reg_term  # X^T X + nλI
    X_T_y = np.dot(X_T, y)  # X^T y
    w = np.dot(np.linalg.inv(X_T_X_reg), X_T_y)  # (X^T X + nλI)^(-1) (X^T y)
    return w
def Compute_Weights(D1_train, D2_train):
    # Unpack datasets
    X1, y1 = D1_train
    X2, y2 = D2_train

    # Compute weights for D1_train
    w1_ols = Solve_OLS(X1, y1)
    w1_rr = Solve_RR(X1, y1, lambda_=1.0)

    # Compute weights for D2_train
    w2_ols = Solve_OLS(X2, y2)
    w2_rr = Solve_RR(X2, y2, lambda_=1.0)

    return w1_ols, w1_rr, w2_ols, w2_rr
w1_ols, w1_rr, w2_ols, w2_rr = Compute_Weights(D_1_train, D_2_train)
print("Shapes of weight vectors:")
print(f"w1_ols: {w1_ols.shape}")
print(f"w1_rr: {w1_rr.shape}")
print(f"w2_ols: {w2_ols.shape}")
print(f"w2_rr: {w2_rr.shape}")

# %%
"""
Task-3 : Calculate MSE for w1_ols and w1_rr using D_1_train and MSE for w2_ols and w2_rr using D_2_train
"""

# %%
def Compute_MSE(X, y, w):
    n = len(y)
    y_pred = np.dot(X, w)  # Predictions: X w
    residuals = y - y_pred  # y - ŷ
    squared_residuals = residuals ** 2  # (y - ŷ)^2
    mse = sum(squared_residuals) / n  # Mean of squared residuals
    return mse

X1, y1 = D_1_train
X2, y2 = D_2_train

# Calculate MSE for each weight vector
mse_w1_ols = Compute_MSE(X1, y1, w1_ols)
mse_w1_rr = Compute_MSE(X1, y1, w1_rr)
mse_w2_ols = Compute_MSE(X2, y2, w2_ols)
mse_w2_rr = Compute_MSE(X2, y2, w2_rr)

# Print the results
print("MSE for w1_ols on D_1_train:", mse_w1_ols)
print("MSE for w1_rr on D_1_train:", mse_w1_rr)
print("MSE for w2_ols on D_2_train:", mse_w2_ols)
print("MSE for w2_rr on D_2_train:", mse_w2_rr)


# %%
"""
Deliverabeles:

2. Report MSE on D_1_train. Report w1_ols and w1_rr in the pdf.
"""

# %%
w1_ols, w1_rr, w2_ols, w2_rr = Compute_Weights(D_1_train, D_2_train)
X = D_1_train[0]
y = D_1_train[1]
w1_ols = Solve_OLS(X,y)
w1_rr = Solve_RR(X,y,lambda_=1.0)
mse_w1_ols = Compute_MSE(X,y,w1_ols)
mse_w1_rr = Compute_MSE(X,y,w1_rr)
print("MSE for w1_ols on D_1_train:", mse_w1_ols)
print("MSE for w1_rr on D_1_train:", mse_w1_rr)
print(w1_ols,w1_rr)

# %%
"""
Deliverables: 

3. Report MSE on Dtrain2 . Attach w2_ols and w2_rr as csv files named w_ols_[five-digit-srnumber].csv and w_rr_[five-digit-srnumber].csv
"""

# %%
X2, y2 = D_2_train
mse_w2_ols = Compute_MSE(X2, y2, w2_ols)
mse_w2_rr = Compute_MSE(X2, y2, w2_rr)

# Print MSE for reporting in PDF
print("MSE for w2_ols on D_2_train:", mse_w2_ols)
print("MSE for w2_rr on D_2_train:", mse_w2_rr)

# Save weights to CSV files
srn = 23627
np.savetxt(f"w_ols_{srn}.csv", w2_ols, delimiter=",")
np.savetxt(f"w_rr_{srn}.csv", w2_rr, delimiter=",")
print(f"Saved weights to w_ols_{srn}.csv and w_rr_{srn}.csv")

# %%
"""
### 3.2 : Support Vector Regression
"""

# %%
import Oracle_Assignment_2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Step 1: Query the oracle to get the stock ticker
Stock_Name = Oracle_Assignment_2.q3_stocknet(23627)
print(f"Assigned stock ticker: {Stock_Name}")  # Returns MRK

# Step 2: Load the stock data from the CSV file
# Assuming the stocknet-dataset has been cloned to the current directory
csv_path = f"MRK.csv"
stock_data = pd.read_csv(csv_path)

# Step 3: Extract closing prices and normalize
# Extract the 'Close' column
closing_prices = stock_data['Close'].values  # Shape: (N,)

# Normalize the closing prices using StandardScaler
scaler = StandardScaler()
closing_prices_normalized = scaler.fit_transform(closing_prices.reshape(-1, 1)).flatten()
N = len(closing_prices_normalized)  # Total number of days
print(f"Total number of days: {N}")

# Step 4 & 5: Create feature matrix X and labels y for each t in {7, 30, 90}
data_splits = {}
for t in [7, 30, 90]:
    # Create X: Each row contains t consecutive days
    X = []
    y = []
    for i in range(N - t):
        # Row i: closing prices from day i to day i+t-1
        X.append(closing_prices_normalized[i:i+t])
        # Label for row i: closing price on day i+t
        y.append(closing_prices_normalized[i+t])
    
    X = np.array(X)  # Shape: (N-t, t)
    y = np.array(y)  # Shape: (N-t,)
    
    # Step 6: Split into train and test sets (first half for training, second half for testing)
    split_idx = (N - t) // 2
    X_train = X[:split_idx]  # Shape: (split_idx, t)
    y_train = y[:split_idx]  # Shape: (split_idx,)
    X_test = X[split_idx:]   # Shape: (N-t-split_idx, t)
    y_test = y[split_idx:]   # Shape: (N-t-split_idx,)
    
    # Store the data split
    data_splits[t] = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler  # Save the scaler for denormalization later
    }
    
    print(f"\nData split for t = {t}:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test shape: {y_test.shape}")

# The data_splits dictionary now contains the preprocessed data for each t

# %%
"""
Train the following SVRs using the train set for t ∈{7,30,90}:

1. Solve the dual of the slack linear support vector regression using cvxopt


2. Solve the dual of the kernelized support vector regression using the RBF kernel for γ = [1,0.1,0.01,0.001]
using cvxopt.
"""

# %%
import numpy as np
from cvxopt import matrix, solvers

class SVR:
    """A class to train Support Vector Regression (SVR) models using the dual formulation with cvxopt."""
    
    def __init__(self, epsilon=0.1, C=1.0):

        self.epsilon = epsilon
        self.C = C
        self.models = {}  # Store trained models for each t and gamma

    def rbf_kernel(self, X1, X2, gamma):

        N1 = len(X1)
        N2 = len(X2)
        K = np.zeros((N1, N2))
        for i in range(N1):
            for j in range(N2):
                diff = X1[i] - X2[j]
                K[i, j] = np.exp(-gamma * np.sum(diff ** 2))
        return K

    def train_linear_svr(self, X_train, y_train):
        N, t = X_train.shape
        K = X_train @ X_train.T  # Linear kernel: K = X @ X.T

        # Dual Formulation Setup
        P = np.block([[K, -K], [-K, K]])
        P = matrix(P)

        q = self.epsilon * np.ones(2 * N)
        q[:N] -= y_train
        q[N:] += y_train
        q = matrix(q)

        A = np.hstack([np.ones(N), -np.ones(N)])
        A = matrix(A, (1, 2 * N))
        b = matrix(0.0)

        G = np.vstack([-np.eye(2 * N), np.eye(2 * N)])
        h = np.hstack([np.zeros(2 * N), self.C * np.ones(2 * N)])
        G = matrix(G)
        h = matrix(h)

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        z = np.array(solution['x']).flatten()

        alphas = z[:N]
        alphas_star = z[N:]

        # Compute w (weights) using all support vectors (nonzero alphas or alphas_star)
        w = np.sum((alphas - alphas_star)[:, np.newaxis] * X_train, axis=0)

        # Identify all support vectors (alpha_i > 0 or alpha_i* > 0)
        support_indices = np.where((alphas > 1e-5) | (alphas_star > 1e-5))[0]

        # For b calculation, use non-bounded support vectors (0 < alpha_i < C or 0 < alpha_i* < C)
        non_bounded_indices = np.where(
            ((alphas > 1e-5) & (alphas < self.C - 1e-5)) |
            ((alphas_star > 1e-5) & (alphas_star < self.C - 1e-5))
        )[0]

        b_values = []
        for i in non_bounded_indices:
            if alphas[i] > 1e-5:  # alpha_i > 0 (and alpha_i* = 0 due to constraint)
                b_i = y_train[i] - np.dot(w, X_train[i]) - self.epsilon
            elif alphas_star[i] > 1e-5:  # alpha_i* > 0 (and alpha_i = 0)
                b_i = y_train[i] - np.dot(w, X_train[i]) + self.epsilon
            b_values.append(b_i)

        b = np.mean(b_values) if b_values else 0.0

        return w, b, alphas, alphas_star

    def train_rbf_svr(self, X_train, y_train, gamma):
        N = X_train.shape[0]
        K = self.rbf_kernel(X_train, X_train, gamma)

        # Dual Formulation Setup
        P = np.block([[K, -K], [-K, K]])
        P = matrix(P)

        q = self.epsilon * np.ones(2 * N)
        q[:N] -= y_train
        q[N:] += y_train
        q = matrix(q)

        A = np.hstack([np.ones(N), -np.ones(N)])
        A = matrix(A, (1, 2 * N))
        b = matrix(0.0)

        G = np.vstack([-np.eye(2 * N), np.eye(2 * N)])
        h = np.hstack([np.zeros(2 * N), self.C * np.ones(2 * N)])
        G = matrix(G)
        h = matrix(h)

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        z = np.array(solution['x']).flatten()

        alphas = z[:N]
        alphas_star = z[N:]

        # Identify all support vectors (alpha_i > 0 or alpha_i* > 0)
        support_indices = np.where((alphas > 1e-5) | (alphas_star > 1e-5))[0]

        # For b calculation, use non-bounded support vectors (0 < alpha_i < C or 0 < alpha_i* < C)
        non_bounded_indices = np.where(
            ((alphas > 1e-5) & (alphas < self.C - 1e-5)) |
            ((alphas_star > 1e-5) & (alphas_star < self.C - 1e-5))
        )[0]

        b_values = []
        for i in non_bounded_indices:
            kernel_row = K[i]
            kernel_sum = np.sum((alphas - alphas_star) * kernel_row)
            if alphas[i] > 1e-5:  # alpha_i > 0 (and alpha_i* = 0 due to constraint)
                b_i = y_train[i] - kernel_sum - self.epsilon
            elif alphas_star[i] > 1e-5:  # alpha_i* > 0 (and alpha_i = 0)
                b_i = y_train[i] - kernel_sum + self.epsilon
            b_values.append(b_i)

        b = np.mean(b_values) if b_values else 0.0

        return alphas, alphas_star, b

    def task_1(self, data_splits):
        for t in [7, 30, 90]:
            X_train = data_splits[t]['X_train']
            y_train = data_splits[t]['y_train']
            
            w, b, alphas, alphas_star = self.train_linear_svr(X_train, y_train)
            
            # Identify support vectors (alpha_i > 0 or alpha_i* > 0)
            support_indices = np.where(((alphas > 1e-5) & (alphas < self.C - 1e-5)) | 
                           ((alphas_star > 1e-5) & (alphas_star < self.C - 1e-5)))[0]
            num_support_vectors = len(support_indices)
            
            # Store the model
            if t not in self.models:
                self.models[t] = {}
            self.models[t]['linear'] = {
                'w': w,
                'b': b,
                'alphas': alphas,
                'alphas_star': alphas_star,
                'support_indices': support_indices,  # Store indices for reference
                'X_train': X_train,
                'y_train': y_train
            }
            
            print(f"Trained Linear SVR for t = {t}:")
            print(f"  Weight vector shape: {w.shape}")
            print(f"  Bias term b: {b:.4f}")
            print(f"  Number of support vectors: {num_support_vectors}\n")

    def task_2(self, data_splits):
        gamma_values = [1, 0.1, 0.01, 0.001]
        print("Task 2: Training RBF SVR Models\n")
        for t in [7, 30, 90]:
            X_train = data_splits[t]['X_train']
            y_train = data_splits[t]['y_train']
            
            for gamma in gamma_values:
                alphas, alphas_star, b = self.train_rbf_svr(X_train, y_train, gamma)
                
                # Identify support vectors (alpha_i > 0 or alpha_i* > 0)
                support_indices = np.where(((alphas > 1e-5) & (alphas < self.C - 1e-5)) | 
                           ((alphas_star > 1e-5) & (alphas_star < self.C - 1e-5)))[0]
                num_support_vectors = len(support_indices)
                
                # Store the model
                if t not in self.models:
                    self.models[t] = {}
                self.models[t][gamma] = {
                    'alphas': alphas,
                    'alphas_star': alphas_star,
                    'b': b,
                    'support_indices': support_indices,  # Store indices for reference
                    'X_train': X_train,
                    'y_train': y_train
                }
                
                print(f"Trained RBF SVR for t = {t}, gamma = {gamma}:")
                print(f"  Number of support vectors: {num_support_vectors}")
                print(f"  Bias term b: {b:.4f}\n")

# %%
"""
Deliverables :

For each SVR trained, plot a graph on the test set containing the following:
1. Predicted closing price value.
2. Actual closing price value.
3. Average price on the previous t days
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(X1, X2, gamma):
    N1 = len(X1)
    N2 = len(X2)
    K = np.zeros((N1, N2))
    for i in range(N1):
        for j in range(N2):
            diff = X1[i] - X2[j]
            K[i, j] = np.exp(-gamma * np.sum(diff ** 2))
    return K

def predict_linear(X, model):
    w = model['w']
    b = model['b']
    y_pred = X @ w + b
    return y_pred

def predict_rbf(X, model, gamma):
    alphas = model['alphas']
    alphas_star = model['alphas_star']
    b = model['b']
    X_train = model['X_train']
    K = rbf_kernel(X, X_train, gamma)
    y_pred = np.sum((alphas - alphas_star) * K, axis=1) + b
    return y_pred

def compute_average_prices(X):
    return np.mean(X, axis=1)

def plot_results(data_splits, svr_trainer):
    gamma_values = [1, 0.1, 0.01, 0.001]
    
    for t in [7, 30, 90]:
        X_test = data_splits[t]['X_test']
        y_test = data_splits[t]['y_test']
        scaler = data_splits[t]['scaler']
        
        # Denormalize the actual prices
        y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Compute the average prices and denormalize
        avg_prices = compute_average_prices(X_test)
        avg_prices_denorm = scaler.inverse_transform(avg_prices.reshape(-1, 1)).flatten()
        
        # Plot for Linear SVR
        model_linear = svr_trainer.models[t]['linear']
        y_pred_linear = predict_linear(X_test, model_linear)
        y_pred_linear_denorm = scaler.inverse_transform(y_pred_linear.reshape(-1, 1)).flatten()
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_pred_linear_denorm, label='Predicted (Linear SVR)', color='blue')
        plt.plot(y_test_denorm, label='Actual', color='green')
        plt.plot(avg_prices_denorm, label=f'Average of previous {t} days', color='orange', linestyle='--')
        plt.title(f'Linear SVR (t = {t})')
        plt.xlabel('Test Sample Index')
        plt.ylabel('Closing Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'linear_svr_t_{t}.png')
        plt.close()
        
        # Plot for each RBF SVR model
        for gamma in gamma_values:
            model_rbf = svr_trainer.models[t][gamma]
            y_pred_rbf = predict_rbf(X_test, model_rbf, gamma)
            y_pred_rbf_denorm = scaler.inverse_transform(y_pred_rbf.reshape(-1, 1)).flatten()
            
            plt.figure(figsize=(10, 6))
            plt.plot(y_pred_rbf_denorm, label=f'Predicted (RBF SVR, gamma={gamma})', color='blue')
            plt.plot(y_test_denorm, label='Actual', color='green')
            plt.plot(avg_prices_denorm, label=f'Average of previous {t} days', color='orange', linestyle='--')
            plt.title(f'RBF SVR (t = {t}, gamma = {gamma})')
            plt.xlabel('Test Sample Index')
            plt.ylabel('Closing Price (USD)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'rbf_svr_t_{t}_gamma_{gamma}.png')
            plt.close()

# Example usage
if __name__ == "__main__":
    # Assume data_splits is available from preprocessing
    svr_trainer = SVR(epsilon=0.1, C=1.0)
    svr_trainer.task_1(data_splits)
    svr_trainer.task_2(data_splits)
    plot_results(data_splits, svr_trainer)
    print("Plots generated for all SVR models.")
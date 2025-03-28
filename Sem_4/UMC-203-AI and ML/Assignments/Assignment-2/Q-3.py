# %%
"""
### Regression 
"""

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
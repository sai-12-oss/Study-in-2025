from utils import *


np.random.seed(2023)
X, y = get_regression_dataset()
X = np.hstack([X, X**2, np.ones((X.shape[0], 1))])

# w = np.linalg.inv(X.T @ X) @ X.T @ y
#
# plt.scatter(X[:, 0], y)
# plt.plot(X[:, 0], X @ w, color='red')
# plt.grid()
# plt.show()

idxs = []
w = np.zeros(shape=(X.shape[1], 1))
M_km1 = np.eye(X.shape[1])
for i, idx in enumerate(np.random.permutation(X.shape[0])):
    idxs.append(idx)
    x_k = X[idx].reshape(-1, 1)
    y_k = y[idx]

    R_k = (x_k @ x_k.T) / (1 + x_k.T @ M_km1 @ x_k)
    M_k = M_km1 - (M_km1 @ R_k @ M_km1)

    w = (np.eye(X.shape[1]) - M_km1 @ R_k) @ (w + y_k * M_km1 @ x_k)

    M_km1 = M_k

    plt.scatter(X[np.array(idxs), 0], y[np.array(idxs)])
    plt.plot(X[:, 0], X @ w, color='red')
    plt.grid()
    plt.show()
    plt.close()

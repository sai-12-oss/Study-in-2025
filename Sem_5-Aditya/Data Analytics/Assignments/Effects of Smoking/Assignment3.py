import numpy as np
import pandas as pd
from scipy.stats import f
from scipy.linalg import pinv
from matplotlib import pyplot as plt


def preprocessing(filepath):
    data = pd.read_csv(filepath, sep="\t")
    
    # Drop the Go column since it is not relevant
    data = data.drop('Go', axis=1)
    
    # Now drop all rows with NaN values
    data = data.dropna()
    
    # Drop the extra details since we are only doing a numerical analysis
    data.drop(['ProbeName', 'GeneSymbol', 'EntrezGeneID'], axis=1, inplace=True)
    data = data.reset_index(drop=True)
    
    # Convert to numpy for ease of use
    data = data.to_numpy(dtype='float64')
    
    # Since values are in log, we need to convert them back to normal values
    data = 2 ** data

    return data


def design_matrix(n):
    D = np.zeros((n, 2*2)) # taking a=2 and b=2

    # Construct the D matrix, to be used in the denominator
    d = int(n/(2*2))
    for i in range(n):
        index = i // d
        D[i, index] = 1

    # Construct the N matrix, to be used in the numerator
    N = np.zeros((n, 2*2))
    for i in range(n):
        index = i // d
        
        x_1 = index // 2
        x_2 = index % 2
        
        N[i, x_1] = 1
        N[i, x_2 + 2] = 1
    
    # Calculate the required numerator and denominator matrices
    num = np.eye(n) - N @ pinv(N.T @ N) @ N.T
    den = np.eye(n) - D @ pinv(D.T @ D) @ D.T
    
    return num, den


def plot(p):
    plt.hist(p, bins=100)
    plt.title('Distribution of p-values for 2 way ANOVA test')
    plt.xlabel('P-value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)


def p_values(data):
    n = data.shape[1]
    p_values = []
    
    # Get the numerator and denominator matrices
    num, den = design_matrix(n)
    
    # Calculate the degrees of freedom
    deg_num = n - 4  # n - rank(D) 
    deg_den = 4 - 3  # rank(D) - rank(N)
    
    for row in data:
        numerator = row.T @ num @ row
        denominator = row.T @ den @ row
        
        if denominator == 0:
            p_values.append(1)
        else:
            # Calculate the F value, and use it to calculate the p-value
            f_val = (deg_num/deg_den) * (numerator / denominator - 1)
            p_values.append(f.sf(f_val, deg_den, deg_num))

    return p_values


if __name__ == "__main__":
    # Preprocess the data to remove NaN values
    data = preprocessing('../data/Raw Data_GeneSpring.txt')
    
    # Calculate the p-values for each row
    p = p_values(data)
    
    # Plot the distribution of p-values
    plot(p)
    plt.show()

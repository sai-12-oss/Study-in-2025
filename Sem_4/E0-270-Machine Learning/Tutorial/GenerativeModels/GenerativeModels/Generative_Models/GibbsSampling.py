import numpy as np


def get_estimated_probs(counts):
    counts = counts + 1
    return np.round(counts / np.sum(counts), 3)


def main():
    # true_probs = np.array([1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3]).reshape(3, 4)
    true_probs = np.array([1, 4, 3, 2]).reshape(2, 2)
    true_probs = true_probs / np.sum(true_probs) # (P(X, Y))
    # print(true_probs)
    
    # import sys; sys.exit()
    
    # Initialize the counts
    counts = np.zeros_like(true_probs) 

    axis = 0
    cell_idx = np.array([0, 0])
    # Gibbs sampling
    for i in range(1_000_000+1):
        # Add one count corresponding to the current cell
        if i > 100_000: # Burn - Time???
            counts[cell_idx[0], cell_idx[1]] += 1

        # Get the next cell
        if axis == 0:  # Row (P(Y|X))
            # Get the conditional probability of the current row
            cond_probs = true_probs[cell_idx[0], :]
            # print(true_probs[cell_idx[0], :])
            
            cond_probs = cond_probs / np.sum(cond_probs)
            # print('I',cond_probs)
            # import sys; sys.exit()
            
           
            # Sample the next column
            cell_idx[1] = np.random.choice(len(cond_probs), p=cond_probs)
            axis = 1
        else:  # Column P (X|Y)
            # Get the conditional probability of the current column
            cond_probs = true_probs[:, cell_idx[1]]
            # print(true_probs[:, cell_idx[1]])
            cond_probs = cond_probs / np.sum(cond_probs)
            # Sample the next row
            cell_idx[0] = np.random.choice(len(cond_probs), p=cond_probs)
            axis = 0

        # Print the estimated probabilities every 10_000 iterations
        if i % 10_000 == 0 and i > 200_000:
            print(f'\nIteration {i}:')
            print(get_estimated_probs(counts))

    print(f'\nTrue probabilities:\n{np.round(true_probs, 3)}')


if __name__ == '__main__':
    main()


# Question-1 
# Load and Preprocess data
print("Question-1")
print("part-1:")
import oracle 
import numpy as np
import matplotlib.pyplot as plt
attributes, train_images, train_labels, test_images, test_labels = oracle.q1_fish_train_test_data(23627)
train_images,train_labels,test_images,test_labels = np.array(train_images),np.array(train_labels),np.array(test_images),np.array(test_labels)
train_images_flat = train_images.reshape(train_images.shape[0], -1)
test_images_flat = test_images.reshape(test_images.shape[0], -1)
print(attributes)
def Preprocessing_data(train_images, train_labels):
    D = train_images 
    L = train_labels
    # class_1 --> 'Attractive' = 0,'Heavy_Makeup' = 0
    # class_2 --> 'Attractive' = 0,'Heavy_Makeup' = 1   
    # class_3 --> 'Attractive' = 1,'Heavy_Makeup' = 0
    # class_4 --> 'Attractive' = 1,'Heavy_Makeup' = 1
    
    # Split data into classes
    class_1, class_2, class_3, class_4 = [], [], [], []
    for i in range(len(D)):
        if L[i] == 0:
            class_1.append(D[i])
        elif L[i] == 1:
            class_2.append(D[i])
        elif L[i] == 2:
            class_3.append(D[i])
        elif L[i] == 3:
            class_4.append(D[i])
    
    # Convert to numpy arrays and flatten
    classes = [
        np.array(class_1).reshape(len(class_1), -1),
        np.array(class_2).reshape(len(class_2), -1),
        np.array(class_3).reshape(len(class_3), -1),
        np.array(class_4).reshape(len(class_4), -1)
    ]
    n_values = [50, 100, 500, 1000, 2000, 4000]
    return classes, n_values
def Computing_norms(classes, n_values):
    mean_vectors = [[] for _ in range(len(n_values))]
    covariances = [[] for _ in range(len(n_values))]
    mean_norms = [[] for _ in range(len(classes))]
    cov_norms = [[] for _ in range(len(classes))]
    
    for i in range(len(classes)):
        class_data = classes[i]  # Already flattened
        for j in range(len(n_values)):
            n = n_values[j]
            # Randomly sample n images without replacement
            indices = np.random.choice(class_data.shape[0], n, replace=False)
            data_n = class_data[indices]
            
            # Mean vector and its L2 norm
            mean_vector = np.mean(data_n, axis=0)
            mean_norm = np.linalg.norm(mean_vector)
            mean_vectors[j].append(mean_vector)
            mean_norms[i].append(mean_norm)
            
            # Covariance matrix and its Frobenius norm
            cov_matrix = np.cov(data_n, rowvar=False)
            cov_norm = np.linalg.norm(cov_matrix, ord='fro')
            covariances[j].append(cov_matrix)
            cov_norms[i].append(cov_norm)
    
    # Plot Mean Norms
    plt.figure(figsize=(10, 6))
    for i in range(len(classes)):
        plt.plot(n_values, mean_norms[i], marker='o', label=f'Class {i+1} (Label {i})')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('L2 Norm of Mean Vector')
    plt.title('L2 Norm of Mean Vectors vs. Number of Samples')
    plt.legend()
    plt.grid(True)
    plt.savefig('mean_norms_plot.png')
    plt.close()
    
    # Plot Covariance Norms
    plt.figure(figsize=(10, 6))
    for i in range(len(classes)):
        plt.plot(n_values, cov_norms[i], marker='o', label=f'Class {i+1} (Label {i})')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Frobenius Norm of Covariance Matrix')
    plt.title('Frobenius Norm of Covariance Matrices vs. Number of Samples')
    plt.legend()
    plt.grid(True)
    plt.savefig('cov_norms_plot.png')
    plt.close()
    
    # Print Mean Norms
    print("\nL2 Norms of Mean Vectors for Each Class:")
    print("n\t", "\t".join([f"Class {i+1}" for i in range(len(classes))]))
    for j in range(len(n_values)):
        row = f"{n_values[j]}\t" + "\t".join([f"{mean_norms[i][j]:.6f}" for i in range(len(classes))])
        print(row)
    
    # Print Covariance Norms
    print("\nFrobenius Norms of Covariance Matrices for Each Class:")
    print("n\t", "\t".join([f"Class {i+1}" for i in range(len(classes))]))
    for j in range(len(n_values)):
        row = f"{n_values[j]}\t" + "\t".join([f"{cov_norms[i][j]:.6f}" for i in range(len(classes))])
        print(row)
    
    return mean_norms, cov_norms, mean_vectors, covariances
classes, n_values = Preprocessing_data(train_images, train_labels)
mean_norms, cov_norms, mean_vectors, covariances = Computing_norms(classes, n_values)
print("part-2:")
print('part-2-a:')
n_classes = len(classes)
# dimesnion of the data
d = classes[0].shape[1] # 3*32*32 = 3072
mean_vectors = np.array(mean_vectors)
overall_mean = np.mean(np.concatenate(classes), axis=0)
def compute_scatter_matrices(selected_classes):
    C = len(selected_classes)
    d = selected_classes[0].shape[1]
    
    # Class means
    mu_i = [np.mean(class_data, axis=0) for class_data in selected_classes]
    
    # Overall mean
    all_data = np.concatenate(selected_classes, axis=0)
    mu = np.mean(all_data, axis=0)
    
    # Computing Between-class scatter matrix S_B
    S_B = np.zeros((d, d))
    for i in range(C):
        N_i = selected_classes[i].shape[0]
        diff = mu_i[i] - mu
        diff = diff.reshape(d, 1)
        S_B = S_B + N_i * np.dot(diff, diff.T)
    
    # Computing Within-class scatter matrix S_W
    S_W = np.zeros((d, d))
    for i in range(C):
        diff = selected_classes[i] - mu_i[i]
        S_W = S_W + np.dot(diff.T, diff)
    
    return S_B, S_W
def Compute_Accuracy(W,selected_train_data,selected_labels,test_images_flat,test_labels):
    # Project training data
    train_projections = np.dot(selected_train_data, W)  # Shape: (n_train, 3)
    
    # Class means in projected space
    C = 4
    mu_proj = [np.mean(train_projections[selected_labels == i], axis=0) for i in range(C)]
    
    # Project test data
    test_projections = np.dot(test_images_flat, W)  # Shape: (1000, 3)
    
    # Predict by nearest centroid
    predictions = []
    for test_proj in test_projections:
        distances = [np.linalg.norm(test_proj - mu) for mu in mu_proj]
        pred_class = np.argmin(distances)
        predictions.append(pred_class)
    
    predictions = np.array(predictions)
    accuracy = np.mean(predictions == test_labels)
    return accuracy
import numpy as np
from scipy.linalg import eigh
import random

n_values = [2500,3500,4000,4500,5000]
np.random.seed(42)
random.seed(42)

objective_values = {n: [] for n in n_values}
fld_weights = {n: [] for n in n_values}
epsilon = 1e-2

for n in n_values:
    num_subsets = 20 if n < 5000 else 1
    
    for subset_idx in range(num_subsets):
        if n < 5000:
            selected_classes = []
            for class_data in classes:
                indices = np.random.choice(class_data.shape[0], n, replace=False)
                selected_classes.append(class_data[indices])
        else:
            selected_classes = classes
        
        S_B, S_W = compute_scatter_matrices(selected_classes)
        S_W_reg = S_W + epsilon * np.eye(d)
        eigvals_S_W_reg = np.linalg.eigvalsh(S_W_reg)
        min_eigval = np.min(eigvals_S_W_reg)
        if min_eigval <= 0:
            print(f"Warning: S_W_reg is not positive definite for n={n}, subset {subset_idx + 1}. Minimum eigenvalue: {min_eigval}")
            S_W_reg += (abs(min_eigval) + 1e-3) * np.eye(d)
            eigvals_S_W_reg = np.linalg.eigvalsh(S_W_reg)
            print(f"After adjustment, minimum eigenvalue: {np.min(eigvals_S_W_reg)}")
        
        try:
            eigenvalues, eigenvectors = eigh(S_B, S_W_reg)
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError for n={n}, subset {subset_idx + 1}: {e}")
            continue
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        W = eigenvectors[:, :3]
        obj_value = np.sum(eigenvalues[:3])
        objective_values[n].append(obj_value)
        fld_weights[n].append(W)
        print(f"Computed FLD for n={n}, subset {subset_idx + 1}/{num_subsets}")
from scipy.linalg import eigh
n_values = [2500,3500,4000,4500,5000]
np.random.seed(42)
objective_values = {n: [] for n in n_values}
fld_weights = {n: [] for n in n_values}
epsilon = 1e-2
for n in n_values:
    num_subsets = 20 if n < 5000 else 1
    for subset_idx in range(num_subsets):
        if n < 5000:
            selected_classes = []
            for class_data in classes:
                indices = np.random.choice(class_data.shape[0], n, replace=False)
                selected_classes.append(class_data[indices])
        else:
            selected_classes = classes
        
        S_B, S_W = compute_scatter_matrices(selected_classes)
        S_W_reg = S_W + epsilon * np.eye(d)
        eigvals_S_W_reg = np.linalg.eigvalsh(S_W_reg)
        min_eigval = np.min(eigvals_S_W_reg)
        if min_eigval <= 0:
            print(f"Warning: S_W_reg is not positive definite for n={n}, subset {subset_idx + 1}. Minimum eigenvalue: {min_eigval}")
            S_W_reg += (abs(min_eigval) + 1e-3) * np.eye(d)
            eigvals_S_W_reg = np.linalg.eigvalsh(S_W_reg)
            print(f"After adjustment, minimum eigenvalue: {np.min(eigvals_S_W_reg)}")
        
        try:
            eigenvalues, eigenvectors = eigh(S_B, S_W_reg)
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError for n={n}, subset {subset_idx + 1}: {e}")
            continue
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        W = eigenvectors[:, :3]
        obj_value = np.sum(eigenvalues[:3])
        objective_values[n].append(obj_value)
        fld_weights[n].append(W)
# - objective_values[n]: list of objective values for each subset at sample size n
# - fld_weights[n]: list of FLD weight matrices (W) for each subset at sample size n
# Example data (replace with your computed values)
n_values = [2500, 3500, 4000, 4500, 5000]
data = [objective_values[n] for n in n_values]

# Plot the box_plot
plt.figure(figsize=(10, 6))
plt.boxplot(data[:4], labels=n_values[:4])
plt.plot([4.5], objective_values[5000], 'ro', label='n=5000 (single value)')
plt.xlabel('Number of Samples per Class (n)')
plt.ylabel('Multi-Class Objective Value (Sum of Top 3 Eigenvalues)')
plt.title('Box Plots of Multi-Class Objective Values vs. Sample Size')
plt.legend()
plt.savefig('boxplot.png')
plt.show()
thresholds_dict = {n: [] for n in n_values}
# Compute thresholds for each n
for n in n_values:
    num_subsets = 20 if n < 5000 else 1
    for subset_idx in range(num_subsets):
        W = fld_weights[n][subset_idx]
        w1 = W[:, 0]  # First discriminant
        projections = np.dot(train_images_flat, w1)
        
        # Compute class means
        class_means = [np.mean(projections[train_labels == i]) for i in range(n_classes)]
        class_means = np.array(class_means)
        
        # Sort and compute thresholds
        sorted_indices = np.argsort(class_means)
        sorted_means = class_means[sorted_indices]
        thresholds = [(sorted_means[i] + sorted_means[i + 1]) / 2 for i in range(len(sorted_means) - 1)]
        thresholds_dict[n].append(thresholds)
        
        # Plot projections for a representative subset (subset 0 or the current one)
        if subset_idx == 0 or n == 5000:  # Plot only the first subset for n < 5000, and the single subset for n = 5000
            plt.figure(figsize=(10, 6))
            colors = ['r', 'g', 'b', 'y']
            for i in range(n_classes):
                idx = train_labels == i
                plt.hist(projections[idx], bins=50, alpha=0.5, label=f'Class {i} (Mean: {class_means[i]:.4f})', color=colors[i])
            for idx, t in enumerate(thresholds):
                plt.axvline(t, color='k', linestyle='--', label=f'Threshold {idx + 1}: {t:.4f}' if idx == 0 else f'Threshold {idx + 1}: {t:.4f}')
            plt.xlabel('Projection onto First Discriminant')
            plt.ylabel('Frequency')
            plt.title(f'Projections onto First Discriminant with Thresholds (n={n})')
            plt.legend()
            plt.savefig(f'projections_n{n}.png')
            plt.close()
# Summarize thresholds for reporting
print("Thresholds Summary:")
for n in n_values:
    print(f"\nn={n}:")
    if n < 5000:
        all_thresholds = np.array(thresholds_dict[n])  # Shape: (20, 3)
        mean_thresholds = np.mean(all_thresholds, axis=0)
        std_thresholds = np.std(all_thresholds, axis=0)
        for i in range(len(mean_thresholds)):
            print(f"  Threshold {i + 1}: {mean_thresholds[i]:.4f} ± {std_thresholds[i]:.4f}")
    else:
        thresholds = thresholds_dict[n][0]
        for i, t in enumerate(thresholds):
            print(f"  Threshold {i + 1}: {t:.4f}")
# Compute accuracies for each threshold
import numpy as np
import matplotlib.pyplot as plt
print("part-2-b:")
# Dictionary to store accuracies for each subset
def compute_and_plot_accuracies(n_values, classes, n_classes,
                                train_images_flat, train_labels,
                                test_images_flat, test_labels,
                                fld_weights, Compute_Accuracy):
    import numpy as np
    import matplotlib.pyplot as plt

    accuracies = {n: [] for n in n_values}
    for n in n_values:
        num_subsets = 20 if n < 5000 else 1
        for subset_idx in range(num_subsets):
            if n < 5000:
                selected_classes = []
                for class_data in classes:
                    indices = np.random.choice(class_data.shape[0], n, replace=False)
                    selected_classes.append(class_data[indices])
                selected_train_data = np.concatenate(selected_classes, axis=0)
                selected_labels = np.concatenate([np.full(n, i) for i in range(n_classes)])
            else:
                selected_train_data = train_images_flat  # Full dataset
                selected_labels = train_labels
            try:
                W = fld_weights[n][subset_idx]  # Shape: (3072, 3)
            except IndexError:
                print(f"Warning: No weight matrix for n={n}, subset {subset_idx + 1}. Skipping.")
                continue
            accuracy = Compute_Accuracy(W, selected_train_data, selected_labels, test_images_flat, test_labels)
            accuracies[n].append(accuracy)

    mean_accuracies = []
    std_accuracies = []
    for n in n_values:
        mean_acc = np.mean(accuracies[n]) if len(accuracies[n]) > 0 else None
        std_acc = np.std(accuracies[n]) if (n < 5000 and len(accuracies[n]) > 0) else None
        mean_accuracies.append(mean_acc)
        std_accuracies.append(std_acc)

    plt.figure(figsize=(10, 6))
    # Plot error bar for n values except when only a single value exists (n=5000)
    if len(n_values) > 1:
        plt.errorbar(n_values[:-1], mean_accuracies[:-1], yerr=std_accuracies[:-1], fmt='o-', capsize=5,
                     color='b', ecolor='white', label='Mean Accuracy ± Std')
        plt.plot(n_values[-1], mean_accuracies[-1], 'ro', label='n=5000 (single value)')
    else:
        plt.plot(n_values, mean_accuracies, 'ro-', label='Mean Accuracy')
    plt.xlabel('Number of Samples per Class (n)')
    plt.ylabel('Accuracy on Test Set')
    plt.title('Test Set Accuracy vs. Sample Size with FLD Classifier')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('part2b_accuracy.png')
    plt.show()

    return accuracies, mean_accuracies, std_accuracies
# Question - 2
print("Question-2")
print("part-1:")
import numpy as np
import oracle
test_data,training_data = oracle.q2_train_test_emnist(23627,'/home/saisandeshk/Desktop/Assignment-1/EMNIST/emnist-balanced-test.csv', 
    '/home/saisandeshk/Desktop/Assignment-1/EMNIST/emnist-balanced-train.csv')
# Preprocess data
X_train = training_data[:, 1:] / 255.0  # Normalize to [0, 1]
y_train = training_data[:, 0]
X_test = test_data[:, 1:] / 255.0
y_test = test_data[:, 0]
labels = np.unique(y_train) # returns 8 and 41
# change the labels 8 to 0 and 41 to 1 
y_train_new = []
for label in y_train:
    if label == 8:
        y_train_new.append(0)
    else:  # Assume any non-8 is 41
        y_train_new.append(1)
y_train = np.array(y_train_new)
y_test_new = []
for label in y_test:
    if label == 8:
        y_test_new.append(0)
    else:  # Assume any non-8 is 41
        y_test_new.append(1)
y_test = np.array(y_test_new)
Binarized_X_train = []
for i in range(len(X_train)):
    row = []
    for j in range(len(X_train[0])):
        if X_train[i][j] > 0.5:
            row.append(1.0)
        else:
            row.append(0.0)
    Binarized_X_train.append(row)
X_train = np.array(Binarized_X_train)  # Convert to NumPy array

Binarized_X_test = []
for i in range(len(X_test)):
    row = []
    for j in range(len(X_test[0])):
        if X_test[i][j] > 0.5:
            row.append(1.0)
        else:
            row.append(0.0)
    Binarized_X_test.append(row)
X_test = np.array(Binarized_X_test)

# Split the data into two classes
class_0 = X_train[y_train == 0]
class_1 = X_train[y_train == 1]

# Compute p1 and p2
p1 = len(class_0) / len(X_train)
p2 = len(class_1) / len(X_train)

# Compute P(X=1|Y=0) and P(X=1|Y=1)
p_x1_y0 = np.mean(class_0, axis=0) + 1e-6  # Add epsilon to avoid division by zero
p_x1_y1 = np.mean(class_1, axis=0) + 1e-6
def clip(array,min,max):
    values = []
    for value in array:
        if value < min:
            values.append(min)
        elif value > max:
            values.append(max)
        else:
            values.append(value)
    return np.array(values)
p_x1_y0 = clip(p_x1_y0,1e-6,1-1e-6)
p_x1_y1 = clip(p_x1_y1,1e-6,1-1e-6)
def Bernoulli(X_test,p_x1_y0,p_x1_y1,p1,p2):
    # log likelihood of class 0
    log_likelihood_class_0 = np.sum(X_test * np.log(p_x1_y0) + (1 - X_test) * np.log(1 - p_x1_y0), axis=1) 
    # log likelihood of class 1 
    log_likelihood_class_1 = np.sum(X_test * np.log(p_x1_y1) + (1 - X_test) * np.log(1 - p_x1_y1), axis=1) 
    log_prior_ratio = np.log(p1/p2)
    delta = log_likelihood_class_0 - log_likelihood_class_1 + log_prior_ratio
    delta = clip(delta,-500,500)
    eta = 1 / (1 + np.exp(-delta))
    return eta
def modified_bayes_classifier(eta,epsilon):
    if eta >= 0.5 + epsilon:
        return 1
    elif eta <= 0.5 - epsilon:
        return 0
    else:
        return -1
def evaluate(predictions, true_labels):
    non_rejected = 0
    for prediction in predictions:
        if prediction != -1:
            non_rejected += 1
    if non_rejected > 0:
        count = 0
        for i in range(len(predictions)):
            if predictions[i] != -1:
                if predictions[i] == true_labels[i]:
                    count += 1
        misclassification_loss = count/non_rejected
    else:
        misclassification_loss = 0
    rejected = 0
    for prediction in predictions:
        if prediction == -1:
            rejected += 1
    return misclassification_loss, rejected
import matplotlib.pyplot as plt 
eta_test = Bernoulli(X_test,p_x1_y0,p_x1_y1,p1,p2)
epsilons = [0.01,0.1,0.25,0.4]
misclassification_losses = []
no_of_rejected_samples = []

for epsilon in epsilons:
    predictions = np.array([modified_bayes_classifier(eta, epsilon) for eta in eta_test])
    misclassification_loss, rejected = evaluate(predictions, y_test)
    misclassification_losses.append(misclassification_loss)
    no_of_rejected_samples.append(rejected)
    print(f"Epsilon: {epsilon}, Misclassification Loss: {misclassification_loss:.4f}, Rejected Samples: {rejected}")
# Plot for misclassification loss
plt.figure(figsize=(10, 6))
plt.plot(epsilons, misclassification_losses, marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Misclassification Loss')
plt.title('Misclassification Loss vs Epsilon')
plt.grid(True)
plt.savefig("misclassification_loss_vs_epsilon.png")
# Plot for number of rejected samples
plt.figure(figsize=(10, 6))
plt.plot(epsilons, no_of_rejected_samples, marker='o')
plt.xlabel('Epsilon')
plt.ylabel('Number of Rejected Samples')
plt.title('Number of Rejected Samples vs Epsilon')
plt.grid(True)
plt.savefig("rejected_samples_vs_epsilon.png")
print("part-2:")
class_0_indices = []
class_1_indices = []
for i in range(len(y_train)):
    if y_train[i] == 0:
        class_0_indices.append(i)
    elif y_train[i] == 1:
        class_1_indices.append(i)
splits = {"60-40": 0.6, "80-20": 0.8, "90-10": 0.9, "99-1": 0.99}
samples_per_class = 2400 
training_sets = {}
for split,p in splits.items():
    no_of_class_0_samples = int(round(p * samples_per_class))
    no_of_class_1_samples = int(round((1 - p) * samples_per_class))
    index_class_0 = np.random.choice(class_0_indices, no_of_class_0_samples, replace=False)
    index_class_1 = np.random.choice(class_1_indices, no_of_class_1_samples, replace=False)
    index = []
    for idx in index_class_0:
        index.append(idx)
    for idx in index_class_1:
        index.append(idx)
    X_new , y_new = [],[]
    for idx in index:
        X_new.append(X_train[idx])
        y_new.append(y_train[idx])
    training_sets[split] = {"X": np.array(X_new), "y": np.array(y_new)}
    # print(f"Split {split} : Class 0 = {len(index_class_0)}, Class 1 = {len(index_class_1)}, Total = {len(index)}")
epsilons = [0.1,0.25,0.4]
results = {split: {'loss': [], 'rejected': []} for split in splits}
for split,data in training_sets.items():
    X_train_split = data["X"]
    y_train_split = data["y"]
    class_0_split = X_train_split[y_train_split == 0]
    class_1_split = X_train_split[y_train_split == 1]
    p1_split = len(class_0_split) / len(X_train_split)
    p2_split = len(class_1_split) / len(X_train_split)
    p_x1_y0_split = np.mean(class_0_split, axis=0) + 1e-6
    p_x1_y1_split = np.mean(class_1_split, axis=0) + 1e-6
    p_x1_y0_split = clip(p_x1_y0_split,1e-6,1-1e-6)
    p_x1_y1_split = clip(p_x1_y1_split,1e-6,1-1e-6)
    eta_test_split = Bernoulli(X_test,p_x1_y0_split,p_x1_y1_split,p1_split,p2_split)
    for epsilon in epsilons:
        predictions = np.array([modified_bayes_classifier(eta, epsilon) for eta in eta_test_split])
        misclassification_loss, rejected = evaluate(predictions, y_test)
        results[split]['loss'].append(misclassification_loss)
        results[split]['rejected'].append(rejected)
        print(f"Split: {split}, Epsilon: {epsilon}, Misclassification Loss: {misclassification_loss:.4f}, Rejected Samples: {rejected}")
# plot of misclassification loss vs epsilon for different splits
plt.figure(figsize=(10, 6))
for split in splits:
    plt.plot(epsilons, results[split]['loss'], marker='o', label=split)
plt.xlabel('Epsilon')
plt.ylabel('Misclassification Loss')
plt.title('Misclassification Loss vs Epsilon for Different Splits')
plt.legend()
plt.grid(True)
plt.savefig("misclassification_loss_vs_epsilon_splits.png")
print("part-3:")
print("part-3-a:")
import numpy as np
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import confusion_matrix
X_train_bin = np.array(X_train)
y_train = np.array(y_train)
X_test_bin = np.array(X_test)
y_test = np.array(y_test)
def compute_metrics(predictions, y_true):
    """Compute metrics for non-rejected samples."""
    non_rejected = predictions != -1
    if np.sum(non_rejected) == 0:
        return 0, 0, 0, 0  # No non-rejected samples
    y_pred_nr = predictions[non_rejected]
    y_true_nr = y_true[non_rejected]
    cm = confusion_matrix(y_true_nr, y_pred_nr)
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return recall, precision, accuracy, f1

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_f1 = -1
best_p_x1_y0 = None
best_p_x1_y1 = None
best_p1 = None
best_p2 = None
best_fold = -1

# Perform cross-validation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_bin, y_train)):
    print(f"\nFold {fold + 1}")
    X_train_fold = X_train_bin[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train_bin[val_idx]
    y_val_fold = y_train[val_idx]
    class_0_fold = X_train_fold[y_train_fold == 0]
    class_1_fold = X_train_fold[y_train_fold == 1]
    p1_fold = len(class_0_fold) / len(X_train_fold)
    p2_fold = len(class_1_fold) / len(X_train_fold)
    p_x1_y0_fold = np.mean(class_0_fold, axis=0) + 1e-6  # Add small constant to avoid log(0)
    p_x1_y1_fold = np.mean(class_1_fold, axis=0) + 1e-6
    p_x1_y0_fold = np.clip(p_x1_y0_fold, 1e-6, 1 - 1e-6)  # Ensure values are in (0, 1)
    p_x1_y1_fold = np.clip(p_x1_y1_fold, 1e-6, 1 - 1e-6)
    eta_val = 1-Bernoulli(X_val_fold,p_x1_y0_fold,p_x1_y1_fold,p1_fold,p2_fold)
    predictions_val = np.array([modified_bayes_classifier(eta, 0.25) for eta in eta_val])
    recall,precision,accuracy,f1 = compute_metrics(predictions_val,y_val_fold)
    print(f"Recall: {recall:.4f}, Precision: {precision:.4f}, Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_p_x1_y0 = p_x1_y0_fold.copy()
        best_p_x1_y1 = p_x1_y1_fold.copy()
        best_p1 = p1_fold
        best_p2 = p2_fold
        best_fold = fold + 1
print("part-3-b:")
print(f"\nBest fold: {best_fold} with F1-Score: {best_f1:.4f}")
# Apply best fold's classifier to test set
eta_test = 1-Bernoulli(X_test_bin, best_p_x1_y0, best_p_x1_y1, best_p1, best_p2)
predictions_test = np.array([modified_bayes_classifier(eta, 0.25) for eta in eta_test])
n_rejected = np.sum(predictions_test == -1)
non_rejected_idx = predictions_test != -1
if np.sum(non_rejected_idx) > 0:
    y_test_nr = y_test[non_rejected_idx]
    predictions_test_nr = predictions_test[non_rejected_idx]
    misclassification_loss = np.mean(predictions_test_nr != y_test_nr)
else:
    misclassification_loss = 0  # No non-rejected samples
print(f"\nTest Set Results using Best Fold's Classifier:")
print(f"Number of rejected samples: {n_rejected}")
print(f"Misclassification loss among non-rejected samples: {misclassification_loss:.4f}")
print("Question-3:")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
data_path ="/home/saisandeshk/Desktop/Assignment-1/heart+disease/processed.cleveland.data"
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]
data = pd.read_csv(data_path, names=columns, na_values='?')
data.dropna(inplace=True)
# Convert target to binary (0 = No Disease, 1 = Disease)
data['target'] = np.where(data['target'] > 0, 1, 0)
# Split into features (X) and labels (y)
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
import oracle
criterion,splitter,max_depth = oracle.q3_hyper(23627)

Tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,random_state=18)
Tree.fit(X_train,y_train)
print("part-1:")
import matplotlib.pyplot as plt
from dtreeviz.trees import model 
viz = model(
    model=Tree,
    X_train = X_train,
    y_train = y_train,
    target_name='Heart Disease',
    feature_names=X.columns,
    class_names=['No Disease', 'Disease']
)
viz.view()
def values(y_true, y_pred):
    True_Positives = np.sum((y_true == 1) & (y_pred == 1))
    True_Negatives = np.sum((y_true == 0) & (y_pred == 0))
    False_Positives = np.sum((y_true == 0) & (y_pred == 1))
    False_Neagtives = np.sum((y_true == 1) & (y_pred == 0))
    return True_Positives, True_Negatives, False_Positives, False_Neagtives
def compute_metrics(y_true, y_pred):
    True_Positives, True_Negatives, False_Positives, False_Negatives = values(y_true, y_pred)
    recall = True_Positives / (True_Positives + False_Negatives)
    precision = True_Positives / (True_Positives + False_Positives)
    accuracy = (True_Positives + True_Negatives) / (True_Positives + True_Negatives + False_Positives + False_Negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return recall, precision, accuracy, f1
y_pred = Tree.predict(X_test)
recall, precision, accuracy, f1 = compute_metrics(y_pred, y_test)
print("part-2")
print(f"Recall: {recall:.2f}, Precision: {precision:.2f}, Accuracy: {accuracy:.2f}, F1: {f1:.2f}")
most_important_feature = X.columns[np.argmax(Tree.feature_importances_)]
print("part-3")
print(f"Most important feature: {most_important_feature}")
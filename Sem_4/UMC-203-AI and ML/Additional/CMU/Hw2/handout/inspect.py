# program to calculate the label entropy of the root (i.e entropy of labels before 
# any splits)
import csv 
import numpy as np
import sys
def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        labels = [row[-1][-1] for row in data[1:]] # getting the labels
    return labels,data[1:], data[0] # returning labels, data and header itself 
def Entropy(Y):
    # using the formula = -sigma sv/s log*(sv/s) for all v belongs to V(Y)
    # where V(Y) is the set of all unique labels in Y
    labels = set(Y)
    entropy = 0
    for v in labels:
        # set of all labels with value v
        S_v = [y for y in Y if y == v]
        s_v = set(S_v)
        entropy -= (len(s_v)/len(labels)) * np.log2(len(s_v)/len(labels))
    return entropy

def Majority_Vote(Y):
    """Find the majority label in the training data"""
    return max(set(Y), key=Y.count)
def Error(Y):
    # by using Majority Vote as the prediction for all the labels
    # we can calculate the error rate
    v = Majority_Vote(Y)
    incorrect = 0
    for i in range(len(Y)):
        if Y[i] != v:
            incorrect += 1
    return incorrect / len(data)

def write_metrics(entropy, error, output_path):
    """Write the metrics to the output file"""
    with open(output_path, "w") as f:
        f.write(f"entropy: {entropy:.6f}\n")
        f.write(f"error: {error:.6f}\n")


if __name__ == "__main__":
    train_input = sys.argv[1]
    inspect = sys.argv[2]
    labels, data, header = read_csv(train_input)
    # print(labels)
    # print(data)
    Entropy(labels)
    Error(labels)
    write_metrics(Entropy(labels), Error(labels), inspect)

### COMPLETED ###
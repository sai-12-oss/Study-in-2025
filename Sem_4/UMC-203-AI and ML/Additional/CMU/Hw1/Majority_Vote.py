import csv 
import numpy as np
import sys

def read_tsv(file_path):
    """Read the tsv file and return the contents of the data and the header"""
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        data = [row for row in reader]
        return data[1:], data[0]  # returning contents and the header itself

def train(data):
    """Train the model using the data provided"""
    # store v = Majority_Vote(D) the class that appears most frequently in D
    def Majority_Vote(data):
        """Find the majority label in the training data"""
        label_index = len(data[0]) - 1
        labels = [row[label_index] for row in data]
        return max(set(labels), key=labels.count)
    
    global v
    v = Majority_Vote(data)

def h(x):
    """Predict the majority label for the given row"""
    return v

def predict(data):
    """Predict the labels for the given data"""
    predictions = []
    for row in data:
        predictions.append(h(row))
    return predictions

def error(data, predictions):
    """Calculate the error rate of the predictions"""
    true_labels = [row[-1] for row in data]
    incorrect = 0
    for i in range(len(data)):
        if true_labels[i] != predictions[i]:
            incorrect += 1
    return incorrect / len(data)

def write_predictions(predictions, output_path): #functions to write the predictions and metrics to the output file 
    # specified by the path as the python command
    """Write the predictions to the output file"""
    with open(output_path, "w") as f:
        for prediction in predictions:
            f.write(prediction + "\n")

def write_metrics(train_error, test_error, output_path):
    """Write the metrics to the output file"""
    with open(output_path, "w") as f:
        f.write(f"error(train): {train_error:.6f}\n")
        f.write(f"error(test): {test_error:.6f}\n")

if __name__ == "__main__":
    # Example usage
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_output = sys.argv[3]
    test_output = sys.argv[4]
    metrics_output = sys.argv[5]
    train_data,train_header = read_tsv(train_input)
    test_data,test_header = read_tsv(test_input)
    train(train_data)
    train(test_data)
    test_predictions = predict(test_data)
    train_predictions = predict(train_data)
    # print(train_predictions)
    # print(test_predictions)
    train_error = error(train_data, train_predictions)
    test_error = error(test_data, test_predictions)
    # print(train_error)
    # print(test_error)
    write_predictions(train_predictions, train_output)
    write_predictions(test_predictions, test_output)
    write_metrics(train_error, test_error, metrics_output)

"""-----------------------------------------------------------------------------------------------------------------"""
"""Another Method: Basically  the same as the above code but with some changes in the functions and the way they are called 
combining train and h(x) into one fucntion and adding a new_label function to find the majority label
import sys
from collections import Counter
import csv

def read_tsv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        data = [row for row in reader]
    return data[1:], data[0]  # Return the data (excluding the header) and the header itself

def find_majority_label(data, label_index):
    # Count the frequency of each label
    label_counts = Counter()
    for row in data:
        label_counts[row[label_index]] += 1

    # Find the majority label manually
    max_count = -1
    majority_label = None
    for label, count in label_counts.items():
        if count > max_count or (count == max_count and label > majority_label):
            max_count = count
            majority_label = label
    return majority_label

def predict(data, majority_label):
    predictions = []
    for _ in data:
        predictions.append(majority_label)
    return predictions

def calculate_error(data, predictions, label_index):
    true_labels = [row[label_index] for row in data]
    incorrect = 0
    for i in range(len(data)):
        if true_labels[i] != predictions[i]:
            incorrect += 1
    return incorrect / len(data)

def write_predictions(predictions, output_path):"
    with open(output_path, 'w') as f:
        for prediction in predictions:
            f.write(prediction + '\n')

def write_metrics(train_error, test_error, output_path):
    with open(output_path, 'w') as f:
        f.write(f"error(train): {train_error:.6f}\n")
        f.write(f"error(test): {test_error:.6f}\n")

if __name__ == "__main__":
    # Step 1: Parse command-line arguments
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_output = sys.argv[3]
    test_output = sys.argv[4]
    metrics_output = sys.argv[5]
    # Step 2: Load the training and test datasets
    train_data, train_header = read_tsv(train_input)
    test_data, test_header = read_tsv(test_input)

    # Step 3: Find the majority label in the training data
    label_index = -1  # Assuming the label is in the last column
    majority_label = find_majority_label(train_data, label_index)

    # Step 4: Predict labels for training and test data
    train_predictions = predict(train_data, majority_label)
    test_predictions = predict(test_data, majority_label)

    # Step 5: Calculate error rates
    train_error = calculate_error(train_data, train_predictions, label_index)
    test_error = calculate_error(test_data, test_predictions, label_index)
    print(f"error(train): {train_error:.6f}")
    print(f"error(test): {test_error:.6f}")

    # Step 6: Write predictions and metrics
    write_predictions(train_predictions, train_output)
    write_predictions(test_predictions, test_output)
    write_metrics(train_error, test_error, metrics_output)
"""

import numpy as np
import pickle


def predict(X, p_x_given_c, pc):
    """
    X: (num_sentences, vocab_size) binary vectors
    p_x_given_c: List of numpy arrays of size (vocab_size, 1)
                 where the i^th entry contains
                 P(word = j | class = i) for j = 1, ..., vocab_size
    pc: List containing prior probabilities of classes

    returns:
        preds: (num_sentences, 1) labels
    """

    # Counts
    num_sentences = X.shape[0]
    num_classes = len(p_x_given_c)

    # Compute predictions
    posterior = np.zeros((num_sentences, num_classes))
    for c in range(num_classes):
        conditional = p_x_given_c[c].T.repeat(num_sentences, axis=0)
        probs = X * conditional + (1 - X) * (1 - conditional)
        probs = np.exp(np.log(1e-12 + probs).sum(axis=1))
        posterior[:, c] = probs * pc[c]

    # Assign the labels
    preds = np.argmax(posterior, axis=1).reshape((-1, 1))

    return preds


def compute_probs(X, y):
    """
    X: (num_sentences, vocab_size) binary vectors
    y: (num_sentences, 1) labels

    returns:
        p_x_given_c: A list of numpy arrays used by predict function
        pc: A list of numpy arrays used by predict function
    """

    # Find the number of classes
    num_classes = int(np.max(y)) + 1

    # Compute the probabilities
    pc = []
    p_x_given_c = []
    for c in range(num_classes):
        idx = y[:, 0] == c

        # Compute the prior
        pc.append(np.mean(idx))

        # Compute the conditional
        # if idx == spam
        #   X[idx, :] : (num_spam, vocab_size) -> (1, vocab_size) -> (vocab_size, 1)
        p_x_given_c.append(X[idx, :].mean(axis=0).reshape((-1, 1)))

    return p_x_given_c, pc


# Read the data
X = np.loadtxt('./Data/messages.txt')
y = np.loadtxt('./Data/labels.txt').reshape((-1, 1))
idx2word = pickle.load(open('./Data/idx2word.pkl', 'rb'))

print('Vocabulary')
print(idx2word)


# Split data into train and test set
num_examples = X.shape[0]
idx = list(range(num_examples))
np.random.shuffle(idx)
train_idx = idx[:int(0.8 * num_examples)]
test_idx = idx[int(0.8 * num_examples):]
trainX = X[train_idx, :]
trainy = y[train_idx, :]
testX = X[test_idx, :]
testy = y[test_idx, :]


# Compute the probabilities needed by Naive Bayes
p_x_given_c, pc = compute_probs(trainX, trainy)


# Compute accuracy, precision, recall and f1-score on training set
pred_train = predict(trainX, p_x_given_c, pc)
print('Training accuracy:', (pred_train.reshape((-1,)) == trainy.reshape((-1,))).mean())
training_precision = np.mean(trainy[pred_train == 1] == 1)
training_recall = np.mean(pred_train[trainy == 1] == 1)
print('Training precision:', training_precision)
print('Training recall:', training_recall)
print('Training f1-score:', 2 * training_precision * training_recall / (training_precision + training_recall))


# Compute accuracy on test set
pred_test = predict(testX, p_x_given_c, pc)
print('Test accuracy:', (pred_test.reshape((-1,)) == testy.reshape((-1,))).mean())
test_precision = np.mean(testy[pred_test == 1] == 1)
test_recall = np.mean(pred_test[testy == 1] == 1)
print('Test precision:', test_precision)
print('Test recall:', test_recall)
print('Test f1-score:', 2 * test_precision * test_recall / (test_precision + test_recall))


# Find top 10 non-spam words
sorted_idx = np.argsort(p_x_given_c[0].reshape((-1,)) / (1e-12 + p_x_given_c[1].reshape((-1,))))
print('\n\nWords with high probability of occurrence in non-spam messages:')
for j in range(-1, -11, -1):
    print(idx2word[sorted_idx[j]])


# Find top 10 spam words
sorted_idx = np.argsort(p_x_given_c[1].reshape((-1,)) / (1e-12 + p_x_given_c[0].reshape((-1,))))
print('\n\nWords with high probability of occurrence in spam messages:')
for j in range(-1, -11, -1):
    print(idx2word[sorted_idx[j]])

import numpy as np
import pickle

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

train_data = np.append(trainX, trainy, axis=1)
test_data = np.append(testX, testy, axis=1)

np.save('./Data_npy/spam_train.npy', train_data)
np.save('./Data_npy/spam_test.npy', test_data)
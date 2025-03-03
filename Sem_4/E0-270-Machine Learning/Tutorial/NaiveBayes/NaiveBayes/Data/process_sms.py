import pickle
import numpy as np


# Download data from: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection


# Read the data
with open('SMSSpamCollection') as f:
    msgs = []
    labels = []
    for line in f:
        label, msg = line.split('\t')
        labels.append(int(label == 'spam'))
        msgs.append(msg)


# Get the dictionary
dictionary = dict()
processed_msgs = []
for msg in msgs:
    msg = msg.lower()
    processed_msg = ''
    for character in msg:
        if character in 'abcdefghijklmnopqrstuvwxyz $0123456789':
            processed_msg += character
    
    msg = []
    for token in processed_msg.split(' '):
        token = token.strip()
        if len(token) == 0:
            continue

        if token not in dictionary:
            dictionary[token] = 0
        
        dictionary[token] += 1

        msg.append(token)
    processed_msgs.append(msg)


# Get the rare words
rare = []
for word in dictionary:
    if dictionary[word] <= 10:
        rare.append(word)


# Remove the rare words
num_messages = len(processed_msgs)
for i in range(num_messages):
    msg = processed_msgs[i]
    num_words = len(msg)
    for j in range(num_words):
        if msg[j] in rare:
            msg[j] = '__rare__'


# Create the dictionary again
dictionary = dict()
idx2word = dict()
word2idx = dict()
idx = 0
msgs = []
for msg in processed_msgs:
    curr_msg = []
    for word in msg:
        if word not in dictionary:
            dictionary[word] = 0
            idx2word[idx] = word
            word2idx[word] = idx
            idx += 1
        
        dictionary[word] += 1

        curr_msg.append(word2idx[word])
    msgs.append(curr_msg)


# Convert messages to binary vectors
num_messages = len(msgs)
num_words = len(dictionary)
messages = np.zeros((num_messages, num_words), dtype=int)
for i in range(num_messages):
    for idx in msgs[i]:
        messages[i, idx] = 1


# Save the data
np.savetxt('messages.txt', messages, fmt='%d')
np.savetxt('labels.txt', np.asarray(labels, dtype=int), fmt='%d')

with open('word2idx.pkl', 'wb') as f:
    pickle.dump(word2idx, f)

with open('idx2word.pkl', 'wb') as f:
    pickle.dump(idx2word, f)

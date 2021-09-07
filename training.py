
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics
import random
from random import randint
import time
import os
import FishDebug
# print(tf.version.VERSION)

# Preparing dataset-------------------------------------------------
# Useful Constants

# Output classes to learn how to classify
LABELS = [    
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"
] 
# LABELS = [
#     "hunger",
#     "normal",
#     "rest"
# ]
DATASET_PATH = "RNN-HAR-2D-Pose-database/"
X_train_path = DATASET_PATH + "X_train.txt"
X_test_path = DATASET_PATH + "X_test.txt"

y_train_path = DATASET_PATH + "Y_train.txt"
y_test_path = DATASET_PATH + "Y_test.txt"

n_steps = 32 # 32 timesteps per series

# Load the networks inputs
def load_X(X_path):
    with open(X_path, 'r') as file:
        X_ = np.array(
            [elem for elem in [
                row.split(',') for row in file
            ]], 
            dtype=np.float32
        )
    blocks = int(len(X_) / n_steps)
    
    X_ = np.array(np.split(X_,blocks))

    return X_ 

# Load the networks outputs

def load_y(y_path):
    with open(y_path, 'r') as file:
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]], 
            dtype=np.int32
        )
    
    # for 0-based indexing 
    return y_ - 1

X_train = load_X(X_train_path)
X_test = load_X(X_test_path)
#print X_test

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)
# proof that it actually works for the skeptical: replace labelled classes with random classes to train on
#for i in range(len(y_train)):
#    y_train[i] = randint(0, 5)

# Set Parameters ------------------------------------------------------------
# Input Data 

training_data_count = len(X_train)  # 4519 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 1197 test series
n_input = len(X_train[0][0])  # num input parameters per timestep

n_hidden = 34 # Hidden layer num of features
n_classes = 6 

#updated for learning-rate decay
# calculated as: decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
decaying_learning_rate = True
learning_rate = 0.0025 #used if decaying_learning_rate set to False
init_learning_rate = 0.005
decay_rate = 0.96 #the base of the exponential in the decay
decay_steps = 100000 #used in decay every 60000 steps with a base of 0.96

global_step = tf.Variable(0, trainable=False)
lambda_loss_amount = 0.0015

training_iters = training_data_count *100  # Loop 100 times on the dataset, ie 100 epochs
batch_size = 4096
display_iter = batch_size*8  # To show test set accuracy during training

print_msg = {
    "X shape" : X_train.shape,
    "y shape" : y_test.shape,
    "every X's mean" : np.mean(X_test),
    "every X's standard deviation" : np.std(X_test),
    "The dataset has not been preprocessed, is not normalised etc." : ""
}
FishDebug.writeLog({"lineNum":104, "funName":False, "fileName":"./training.py"},'SetParameters',print_msg)
'''
(X shape, y shape, every X's mean, every X's standard deviation)
((22625, 32, 36), (5751, 1), 251.01117, 126.12204)

The dataset has not been preprocessed, is not normalised etc
'''

# Utility functions for training----------------------------------------------------------
def LSTM_RNN(_X, _weights, _biases):
    # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input])   
    # Rectifies Linear Unit activation function used
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
    lstm_last_output = outputs[-1]
    
    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, _labels, _unsampled, batch_size):
    print()
    # Fetch a "batch_size" amount of data and labels from "(X|y)_train" data. 
    # Elements of each batch are chosen randomly, without replacement, from X_train with corresponding label from Y_train
    # unsampled_indices keeps track of sampled data ensuring non-replacement. Resets when remaining datapoints < batch_size    
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    batch_labels = np.empty((batch_size,1)) 

    # debug by Polly in 2021/6/10    
    index = random.choice(_unsampled)
    prints = {
        "\n_train": _train,
        '\n_labels': _labels,
        '\n_unsampled': _unsampled,
        '\nbatch_size': batch_size,
        "\n\nshape":shape,
        '\nshape[0]': shape[0],
        '\nbatch_s': batch_s,
        '\nbatch_labels': batch_labels,
        "\n\nindex":random.choice(_unsampled),
        '\nbatch_s[0]':_train[index],
        '\nbatch_labels[0]':_labels[index]
    }
    FishDebug.writeLog({"lineNum":174, "funName":"extract_batch_size", "fileName":"C:\Users\88692\Desktop\Hey-Fish-What-are-you-doing\training.py"},'extract_batch_size154__',prints)

    _unsampled = list(_unsampled) # ['range' object has no attribute 'remove'] add by Polly in 2021/6/10
    for i in range(batch_size):
        # Loop index
        # index = random sample from _unsampled (indices)
        index = random.choice(_unsampled)
        batch_s[i] = _train[index] 
        batch_labels[i] = _labels[index]
        _unsampled.remove(index)     

    return batch_s, batch_labels, _unsampled


def one_hot(y_):
    # One hot encoding of the network outputs
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

# Build the network------------------------------------------------------------------
# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2 # Softmax loss
if decaying_learning_rate:
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True)


#decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps) #exponentially decayed learning rate
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,global_step=global_step) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Train the network-------------------------------------------------------------------------
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of data at each loop. 
# Elements of each batch are chosen randomly, without replacement, from X_train, 
# restarting when remaining datapoints < batch_size
step = 1
time_start = time.time()
unsampled_indices = range(0,len(X_train))

while step * batch_size <= training_iters:
    #print (sess.run(learning_rate)) #decaying learning rate
    #print (sess.run(global_step)) # global number of iterations
    if len(unsampled_indices) < batch_size:
        unsampled_indices = range(0,len(X_train)) 
    batch_xs, raw_labels, unsampled_indicies = extract_batch_size(X_train, y_train, unsampled_indices, batch_size)
    batch_ys = one_hot(raw_labels)
    # check that encoded output is same length as num_classes, if not, pad it 
    if len(batch_ys[0]) < n_classes:
        temp_ys = np.zeros((batch_size, n_classes))
        temp_ys[:batch_ys.shape[0],:batch_ys.shape[1]] = batch_ys
        batch_ys = temp_ys
       
    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs, 
            y: batch_ys
        }
    )
    train_losses.append(loss)
    train_accuracies.append(acc)
    
    # Evaluate network only at some steps for faster training: 
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        # To not spam console, show training accuracy/loss in this "if"
        with open('lstm_log/TrainTheNetwork', 'a') as f:
            f.write("Iter #" + str(step*batch_size) + \
              ":  Learning rate = " + "{:.6f}".format(sess.run(learning_rate)) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}\n".format(acc))
        # print("Iter #" + str(step*batch_size) + \
        #       ":  Learning rate = " + "{:.6f}".format(sess.run(learning_rate)) + \
        #       ":   Batch Loss = " + "{:.6f}".format(loss) + \
        #       ", Accuracy = {}".format(acc))
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        with open('lstm_log/TrainTheNetwork', 'a') as f:
            f.write("\nPERFORMANCE ON TEST SET:             " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}\n".format(acc))
        # print("PERFORMANCE ON TEST SET:             " + \
        #       "Batch Loss = {}".format(loss) + \
        #       ", Accuracy = {}".format(acc))
        # print_msg = {
        #     "Iter #"        : step * batch_size,
        #     "Learning rate" : sess.run(learning_rate),
        #     "Batch Loss"    : loss, 
        #     "Accuracy"      : acc,    
        #     "PERFORMANCE ON TEST SET": "",            
        #     "Batch Loss"    : loss, 
        #     "Accuracy"      : acc
        # }
        # FishDebug.writeLog({"lineNum":267, "funName":False, "fileName":"./training.py"},'TrainTheNetwork',print_msg)

    step += 1
with open('lstm_log/TrainTheNetwork', 'a') as f:
    f.write("\nOptimization Finished!\n")

# Accuracy for test data

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)
with open('lstm_log/TrainTheNetwork', 'a') as f:
    f.write("\nFINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}\n".format(accuracy))
# print("FINAL RESULT: " + \
#       "Batch Loss = {}".format(final_loss) + \
#       ", Accuracy = {}".format(accuracy))
time_stop = time.time()
with open('lstm_log/TrainTheNetwork', 'a') as f:
    f.write("\nTOTAL TIME:  {}\n".format(time_stop - time_start))

# print("TOTAL TIME:  {}".format(time_stop - time_start))
# print("\n\n\n\n\n\n")

# Results---------------------------------------------------------------
# (Inline plots: )
# %matplotlib inline

font = {
    'family' : 'Bitstream Vera Sans',
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)

width = 12
height = 12
plt.figure(figsize=(width, height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
#plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
#plt.plot(indep_test_axis, np.array(test_losses), "b-", linewidth=2.0, label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "b-", linewidth=2.0, label="Test accuracies")

with open('lstm_log/Results', 'w') as f:
    f.write("test_accuracies len: {}".format(len(test_accuracies)))
    f.write("train_accuracies len: {}\n\n".format(len(train_accuracies)))
# print("\n\ntest_accuracies len: {}".format(len(test_accuracies)))
# print("train_accuracies len: {}\n\n".format(len(train_accuracies)))

plt.title("Training session's Accuracy over Iterations")
plt.legend(loc='lower right', shadow=True)
plt.ylabel('Training Accuracy')
plt.xlabel('Training Iteration')

plt.show()

# Results

predictions = one_hot_predictions.argmax(1)

with open('lstm_log/Results', 'a') as f:
    f.write("Testing Accuracy: {}%".format(100*accuracy))
    f.write("\n\nPrecision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
    f.write("\nRecall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
    f.write("\nf1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))
    f.write("\n\nConfusion Matrix:")
    f.write("\nCreated using test set of {} datapoints, normalised to % of each class in the test dataset\n".format(len(y_test)))

# print("Testing Accuracy: {}%".format(100*accuracy))

# print("")
# print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
# print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
# print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

# print("")
# print("Confusion Matrix:")
# print("Created using test set of {} datapoints, normalised to % of each class in the test dataset".format(len(y_test)))
confusion_matrix = metrics.confusion_matrix(y_test, predictions)


#print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.Blues
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
# -
sess.close()
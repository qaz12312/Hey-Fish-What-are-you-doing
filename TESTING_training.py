# !/usr/bin/python3.6
"""
(測試用)訓練模型並測試.
"""
from dotenv import load_dotenv
from os import getenv, makedirs, environ
from os.path import isdir
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf # Version 1.0.0 (some previous versions are used in past commits)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import random
from random import randint  # 證明此網路架構真的有運作: 用隨機類替換標記類以進行訓練
import time
import argparse
import sys
from traceback import extract_tb
import FishLog
import FishDebug

load_dotenv()
parser = argparse.ArgumentParser(description="Run lstm for fish.")
parser.add_argument('--hidden', help='N_HIDDEN_CELLS',type=int, default=34)
parser.add_argument('--steps', help='N_STEPS',type=int, default=30)
args = parser.parse_args()
run_time_start = time.time()
# -----------------------------------------------------------
# Preparing dataset
# -----------------------------------------------------------
DATASET_PATH = "convertTo_txt/"
X_TRAIN_PATH = DATASET_PATH + "X_train.txt"
X_TEST_PATH = DATASET_PATH + "X_test.txt"
Y_TRAIN_PATH = DATASET_PATH + "Y_train.txt"
Y_TEST_PATH = DATASET_PATH + "Y_test.txt"
# -----------------------------------------------------------
# Set some hyperparameters and parameters
# -----------------------------------------------------------
LABEL_LIST = [
    "normal",
    "hunger",
    "rest",
    # "exception",
]
N_CLASSES = len(LABEL_LIST)
'''
共有幾個分類.
'''
N_STEPS = args.steps
'''
幾個frame是一個動作
'''
N_INPUT = 26
'''
一個frame裡有幾個座標點(x+y)。n input/timestep(frame).
'''
N_HIDDEN_CELLS = args.hidden
'''
number of neurons in hidden layer(自己假設)
'''
# learning rate
DECAYING_LEARNING_RATE = True
'''
是否對學習率learning_rate應用指數衰減

decayed_learning_rate = INIT_LEARNING_RATE * DECAY_RATE ^ (global_step / N_DECAY_STEPS)
'''
learning_rate = 0.0025
'''
學習速率。更新參數的步幅.
- 固定的學習率總是顯得笨拙：太小速度太慢，太大又擔心得不到最優解。一個很直接的想法就是隨著訓練的進行，動態設定學習率——隨著訓練次數增加，學習率逐步減小
- used if `DECAYING_LEARNING_RATE` = False
'''
INIT_LEARNING_RATE = 0.005
'''
若要進行指數衰減
'''
DECAY_RATE = 0.96
'''
衰減中指數的基數 the base of the exponential in the decay
'''
N_DECAY_STEPS = 100000
'''
used in decay every N_DECAY_STEPS steps with a base of 0.96
'''
# loss
LAMBDA_LOSS_AMOUNT = 0.0015
'''
(in 損失函數)懲罰項的倍率
- 當 λ=0 時，則權重衰減不會發生；當 λ 越大時，懲罰的比率較高，權重衰減的程度也就跟著變大
'''
# train
N_EPOCHS = 600
'''
迭代次數.
- Loop n times on the dataset(在數據集上循環 n 次)
'''
BATCH_SIZE = 1000
'''
批量大小。每次的迭代，送入類神經網路的資料數量.
'''
DISPLAY_ITERATION = BATCH_SIZE * 20
'''
在訓練期間顯示測試集的準確性 To show test set accuracy during training
'''


# -----------------------------------------------------------
# Load the networks inputs
# -----------------------------------------------------------
def load_X(X_path):
    '''
    Parameters
    ----------
    X_path : `str`
        file has n(x,y) coords.
    
    Returns
    -------
    `ndarray` (dtype = object, element type = float32)
        3D, 每一 dim 的元素個數 = [x筆動作, frame數, n*2(x,y)座標點].
    '''
    with open(X_path, 'r') as file:
        X_arr = np.array(
            [elem for elem in [row.split(',') for row in file]],
            dtype=np.float32
        )
    if len(X_arr) < N_STEPS :
        print("資料量太少 Failed")
        raise ValueError('Failed: Not enough data')
    blocks = int(len(X_arr) / N_STEPS)
    return np.array(np.split(X_arr, blocks))


# -----------------------------------------------------------
# Load the networks outputs
# -----------------------------------------------------------
def load_Y(Y_path):
    '''
    Parameters
    ----------
    Y_path : `str`
        file has labels.
    
    Returns
    -------
    `ndarray` (dtype = object, element type = float32)
        2D, 每一 dim 的元素個數 = [x 筆動作, 結果].
    '''
    with open(Y_path, 'r') as file:
        Y_arr = np.array(
            [elem for elem in [row.replace('  ', ' ').strip().split(' ') for row in file]],
            dtype=np.int32
        )
    return Y_arr - 1  # 因為 index 從 0 開始算


# -----------------------------------------------------------
# 證明此網路架構真的有運作: 用隨機類替換標記類以進行訓練
# -----------------------------------------------------------
def random_Y(Y_arr):
    '''
    Parameters
    ----------
    Y_arr : `ndarray`
        2D, Content is labels.
    
    Returns
    -------
    `ndarray` (dtype = object, element type = float32)
        2D, 每一 dim 的元素個數 = [x 筆動作, random結果].
    '''
    for i in range(len(Y_arr)):
        Y_arr[i] = randint(0, N_CLASSES-1)
    return Y_arr

# -----------------------------------------------------------
# 運算流程 Utility functions for training
# -----------------------------------------------------------
def LSTM_RNN(_X, _weights, _biases):
    '''
    RNN 總共有 3 個組成部分 ( input ---> |cell| ----> output )
    '''
    # model architecture based on "guillaume-chevalier" and "aymericdamien" under the MIT license.

    #############################################
    # hidden layer for input to cell
    #############################################
    _X = tf.transpose(_X, [1, 0, 2])  # [N_STEPS, x筆資料, N_INPUT]
    # 變成 2D [N_STEPS * x筆資料, N_INPUT]，才可以使用weights 的矩陣乘法
    _X = tf.reshape(_X, [-1, N_INPUT])
    # 修正線性單元激活函數 Rectifies Linear Unit activation function used
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, N_STEPS, 0)  # [N_STEPS, x筆資料, N_INPUT]

    #############################################
    # Define two stacked LSTM cells (two recurrent layers deep)
    # 2-layer LSTM, each layer has N_HIDDEN_CELLS units
    #############################################
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_CELLS, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_CELLS, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # generate prediction (Get lstm cell output)
    outputs, final_state = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    #############################################
    # hidden layer for output as the final results
    #############################################
    # A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
    # 有 N_STEPS 輸出但我們只想要最後一個輸出
    lstm_last_output = outputs[-1]

    # 線性激活 Linear activation, using rnn inner loop last output
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


# -----------------------------------------------------------
# 提取批次資料
# -----------------------------------------------------------
def extract_batch_size(X_data, Y_labels, unsampled_range, batch_size):
    '''
    Fetch a "batch_size" amount of data and labels from "(X|Y)_train" data.\n
    Elements of each batch are chosen randomly, without replacement, from X_train with corresponding label from Y_train

    Parameters
    ----------
    X_data : `ndarray`
        3D, 每一 dim 的元素個數 = [x筆動作, frame數, n*2(x,y)座標點].
    Y_labels : `ndarray`
        2D, 每一 dim 的元素個數 = [x筆動作, 結果].
    unsampled_range : `range`
        0 ~ x
    batch_size : `int`
        b 筆動作 (b < x)
    
    Returns
    -------
    batch_data : `ndarray`
        3D, 每一 dim 的元素個數 = [`batch_size` 筆動作, frame數, n*2(x,y)座標點].
    batch_labels : `ndarray`
        2D, 每一 dim 的元素個數 = [`batch_size` 筆動作結果, 1].
    unsampled_range : `list`
        剩下還沒抓到的資料indices
    '''
    X_shape = list(X_data.shape)
    X_shape[0] = batch_size
    batch_data = np.empty(X_shape)
    batch_labels = np.empty((batch_size, 1))

    unsampled_range = list(unsampled_range)
    for i in range(batch_size):
        index = random.choice(unsampled_range) # random sample from unsampled_range (indices)
        batch_data[i] = X_data[index]
        batch_labels[i] = Y_labels[index]
        unsampled_range.remove(index)
    return batch_data, batch_labels, unsampled_range


# -----------------------------------------------------------
# One hot encoding of the network outputs
# -----------------------------------------------------------
def one_hot(y_):
    '''
    將 LABEL 以 (0,1) 的方式表示.

    Examples
    ---
    >>> one_hot([[5], [0], [3]])
    [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    '''
    y_ = y_.reshape(len(y_))
    # n_values = int(np.max(y_)) + 1
    # if n_values != N_CLASSES:
    #     print("[Now batch values]= {}".format(n_values))
    return np.eye(N_CLASSES)[np.array(y_, dtype=np.int32)]


try:
    # -----------------------------------------------------------
    # 拿訓練、測試資料
    # -----------------------------------------------------------
    X_train = load_X(X_TRAIN_PATH)
    X_test = load_X(X_TEST_PATH)
    Y_train = load_Y(Y_TRAIN_PATH)
    Y_test = load_Y(Y_TEST_PATH)
    # -----------------------------------------------------------
    # Set Hyperparameters & Parameters
    # -----------------------------------------------------------
    # Input Data
    n_training_data = len(X_train)
    '''
    共有幾筆動作.
    '''
    n_test_data = len(X_test)
    '''
    共有幾筆動作.
    '''
    if len(X_train[0][0]) != N_INPUT:
        print('資料特徵數不符合設定的特徵數 Failed')
        raise ValueError('Failed: len(X_train[0][0]) != N_INPUT')
    if n_training_data<BATCH_SIZE or n_test_data < 1 :
        print("資料量太少 Failed")
        
        raise ValueError('Failed: Not enough data')
    global_step = tf.Variable(0, trainable=False)
    '''
    紀錄迭代的總次數.
    - 可用於更改 learning rate 或其他超參數
    - 在100次迭代后停止训练，然后第二天恢复模型并再运行100次迭代。现在`global_step`=200，但是第二次运行的局部迭代数是1到100
    '''
    training_iterations = n_training_data * N_EPOCHS
    '''
    訓練次數.train step 上限
    '''
    # -----------------------------------------------------------
    # 定義類神經網路模型 Build the network
    # -----------------------------------------------------------
    # x = tf.placeholder(tf.float32, [None, N_STEPS, N_INPUT])
    x = tf.compat.v1.placeholder(tf.float32, [None, N_STEPS, N_INPUT])
    '''
    Input.
    3D=[x筆動作, frame數, 座標點數]
    '''
    # y = tf.placeholder(tf.float32, [None, N_CLASSES])
    y = tf.compat.v1.placeholder(tf.float32, [None, N_CLASSES])
    '''
    Output.
    2D=[x筆動作, label數]
    '''
    # weights = {
    #     'hidden': tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN_CELLS])), # Hidden layer
    #     'out': tf.Variable(tf.random_normal([N_HIDDEN_CELLS, N_CLASSES], mean=1.0))
    # }
    weights = {
        'hidden': tf.Variable(tf.random.normal([N_INPUT, N_HIDDEN_CELLS])), # Hidden layer
        'out': tf.Variable(tf.random.normal([N_HIDDEN_CELLS, N_CLASSES], mean=1.0))
    }
    '''
    權重.
    '''
    # biases = {
    #     'hidden': tf.Variable(tf.random_normal([N_HIDDEN_CELLS])),
    #     'out': tf.Variable(tf.random_normal([N_CLASSES]))
    # }
    biases = {
        'hidden': tf.Variable(tf.random.normal([N_HIDDEN_CELLS])),
        'out': tf.Variable(tf.random.normal([N_CLASSES]))
    }
    '''
    偏差值.
    '''
    pred = LSTM_RNN(x, weights, biases)
    '''
    Graph.
    '''
    # l2 = LAMBDA_LOSS_AMOUNT * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    l2 = LAMBDA_LOSS_AMOUNT * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.compat.v1.trainable_variables())
    '''
    Loss function 的懲罰項
    - 對抗 Overfitting: Weight decay -- 不讓模型 fit 時過度依賴一些 weights
    '''
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2
    '''
    Loss function.
    - `softmax_cross_entropy_with_logits`: 結合 Softmax & Cross Entropy 的函式
    '''
    if DECAYING_LEARNING_RATE:  # exponentially decayed learning rate
        learning_rate = tf.compat.v1.train.exponential_decay(INIT_LEARNING_RATE, global_step*BATCH_SIZE, N_DECAY_STEPS, DECAY_RATE, staircase=True)
        # learning_rate = tf.train.exponential_decay(INIT_LEARNING_RATE, global_step*BATCH_SIZE, N_DECAY_STEPS, DECAY_RATE, staircase=True)
        # decayed_learning_rate = INIT_LEARNING_RATE * DECAY_RATE ^ (global_step / N_DECAY_STEPS) 
        # DECAY_RATE = 100: 每100步衰減一次(stair=True時)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)  # Adam Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)  # Adam Optimizer
    '''
    優化器(optimizer)
    - Adam Optimizer: 會在訓練時動態更新 learning_rate
    '''
    # 定義計算準確度的運算
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    '''
    [預測值,真實值]是否一樣，會是一個 boolean array
    '''
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    '''
    計算平均(True=1, False=0)
    '''
    # -----------------------------------------------------------
    # 訓練神經網路 Train the network
    # -----------------------------------------------------------
    print_log_list = list()
    '''
    紀錄 log 用.
    '''
    train_loss_list = []
    train_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []
    # 啟動 Session
    sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    # sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.compat.v1.global_variables_initializer()
    # init = tf.global_variables_initializer()
    sess.run(init)

    train_time_start = time.time()
    '''
    計算訓練網路所需的時間
    '''
    unsampled_indices = range(0, n_training_data)
    '''
    訓練資料的range
    '''
    one_hot_Y_test = one_hot(Y_test)
    step = 1
    while step * BATCH_SIZE <= training_iterations:
        # 當剩餘數據點 < BATCH_SIZE 時重新啟動
        if len(unsampled_indices) < BATCH_SIZE:
            unsampled_indices = range(0, n_training_data)
        
        batch_xs, raw_labels, unsampled_indicies = extract_batch_size(X_train, Y_train, unsampled_indices, BATCH_SIZE)
        batch_ys = one_hot(raw_labels)
        
        # 如果 encoded_y 的長度 != 總分類數，就用 0 填充到一樣大小
        if len(batch_ys[0]) < N_CLASSES:
            temp_ys = np.zeros((BATCH_SIZE, N_CLASSES))
            temp_ys[:batch_ys.shape[0], :batch_ys.shape[1]] = batch_ys
            batch_ys = temp_ys

        # 使用批量數據進行擬合訓練 Fit training using batch data
        _, train_loss, train_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_xs, y: batch_ys})
        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_acc)

        # Evaluate network only at some steps for faster training: 要展示時 / 一開始 / 最後一次
        current_iteration = step*BATCH_SIZE
        if (current_iteration % DISPLAY_ITERATION == 0) or (step == 1) or (current_iteration > training_iterations):
            # 對測試集的評估（這裡沒有學習 - 只是診斷評估） Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: X_test, y: one_hot_Y_test})
            test_loss_list.append(test_loss)
            test_accuracy_list.append(test_acc)
            current_info_str = "Iter #{:<8d}/ Global_iter #{:<5d}: Learning rate={:.6f} Batch Loss={:.3f}, Acc={:.3f} [test data]=> Loss= {:.3f}, Acc= {:.3f}".format(current_iteration, sess.run(global_step), sess.run(learning_rate), train_loss, train_acc, test_loss, test_acc)
            print("\n"+current_info_str)
            print_log_list.append(current_info_str)

        step += 1
    train_time = time.time() - train_time_start
    # -----------------------------------------------------------
    # Accuracy for test data
    # -----------------------------------------------------------
    one_hot_predictions, final_accuracy, final_loss = sess.run([pred, accuracy, cost], feed_dict={x: X_test, y: one_hot_Y_test})
    test_loss_list.append(final_loss)
    test_accuracy_list.append(final_accuracy)

    # debug
    print_log_list.append("\nTrain network time: {}\nFinal result: Loss= {:.3f}, Acc= {:.3f}\n".format(train_time, final_loss, final_accuracy))
    FishLog.writeLog(FishLog.formatLog(20, "TESTING_training.py", "line 456", "Train & Test The Network.", "Train network time: {}".format(train_time)))


    # -----------------------------------------------------------
    # Visualization of results
    # -----------------------------------------------------------
    # fig = plt.figure()
    iters_of_run ={
        'train': np.array(range(BATCH_SIZE, (len(train_loss_list)+1)*BATCH_SIZE, BATCH_SIZE)),
        'test': np.append(
            np.array(range(BATCH_SIZE, len(test_loss_list) * DISPLAY_ITERATION, DISPLAY_ITERATION)[:-1]),
            [training_iterations]
        )
    }
    # # Show 損失率
    # fig.add_subplot(221)
    # plt.plot(iters_of_run['train'], np.array(train_loss_list), "-g", label="train losses")
    # plt.plot(iters_of_run['test'], np.array(test_loss_list), "-m", linewidth=2.0, label="test losses")
    # plt.title("Each iteration loss")
    # plt.legend(loc=1, shadow=True)
    # plt.xlabel('iteration')
    # plt.ylabel('loss')
    # # Show 準確度
    # fig.add_subplot(222)
    # plt.plot(iters_of_run['train'], np.array(train_accuracy_list), "-c", label="train accuracy")
    # plt.plot(iters_of_run['test'], np.array(test_accuracy_list), "-r", linewidth=2.0, label="test accuracy")
    # plt.title("Each iteration accuracy")
    # plt.legend(loc='lower right', shadow=True)
    # plt.xlabel('iteration')
    # plt.ylabel('accuracy')
    # Show 預測值 & 真實值
    predictions = one_hot_predictions.argmax(1)
    '''
    # 每一筆測試動作的預測結果.
    2D=[x筆動作, 1]
    '''
    confusion_matrix = metrics.confusion_matrix(Y_test, predictions)
    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
    '''
    經正規化.
    - 0~100 (單位:%)
    '''
    # fig.add_subplot(223)
    # plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title("Confusion matrix (normalised to %) of total test data")
    # plt.colorbar()
    tick_marks = np.arange(N_CLASSES)
    # plt.xticks(tick_marks, LABEL_LIST, rotation=90)
    # plt.yticks(tick_marks, LABEL_LIST)
    # plt.tight_layout()
    # plt.xlabel('Predicted label')
    # plt.ylabel('True label')
    # plt.show()

    # 關閉 Session
    sess.close()
except Exception as e:
    print(e)
    cl, exc, tb = sys.exc_info()  # 取得Call Stack
    FishLog.writeLog(FishLog.formatException(e.args[0], extract_tb(tb)[0], "Run LSTM code.", "N_HIDDEN_CELLS={:<3d}, N_STEPS={:<2d}".format(N_HIDDEN_CELLS, N_STEPS)))
else:
    run_time = time.time() - run_time_start
    # log
    FishLog.writeLog(FishLog.formatLog(20, "TESTING_training.py", "line 518", "Run LSTM code.", "run total time:={} / N_HIDDEN_CELLS={:<3d}, N_STEPS={:<2d}".format(run_time, N_HIDDEN_CELLS, N_STEPS)))
    # debug
    if final_accuracy > 0.95: # STEPS_Hidden_trainingData_testData_trainTime.txt
        FishDebug.writeLog({"lineNum": 522, "funName": False, "fileName": "TESTING_training.py"}, "1000_{}steps_{}hidden_{}_{}_{}".format(N_STEPS, N_HIDDEN_CELLS, n_training_data, n_test_data, train_time), {
            "X_train shape": X_train.shape,
            "Y_train shape": Y_train.shape,
            "(X_train) training data count": n_training_data,
            "\nX_test shape": X_test.shape,
            "Y_test shape": Y_test.shape,
            "(X_test) test data count": n_test_data,
            "\nn_input": N_INPUT,
            "N_HIDDEN_CELLS": N_HIDDEN_CELLS,
            "\ndecaying_learning_rate": DECAYING_LEARNING_RATE,
            "learning_rate": learning_rate,
            "INIT_LEARNING_RATE": INIT_LEARNING_RATE,
            "DECAY_RATE": DECAY_RATE,
            "N_DECAY_STEPS": N_DECAY_STEPS,
            "global_step": global_step,
            "LAMBDA_LOSS_AMOUNT": LAMBDA_LOSS_AMOUNT,
            "training_iterations": training_iterations,
            "BATCH_SIZE": BATCH_SIZE,
            "\n---\nDisplay Train & Run The Network": print_log_list,
            "\n---\ntrain (loss, accuracy, iters_of_run['train']) len": (len(train_loss_list), len(train_accuracy_list), len(iters_of_run['train'])),
            "test  (loss, accuracy, iters_of_run['test'] ) len": (len(test_loss_list), len(test_accuracy_list), len(iters_of_run['test'])),
            "\ntrain(20 actions)": {'losses':train_loss_list[::20], 'accuracies':train_accuracy_list[::20]},
            "test ": {'losses':test_loss_list, 'accuracies':test_accuracy_list},
            "ONLY VALUE": "\n\n[Confusion Matrix] test set: {} actions".format(len(Y_test)),
            "Testing Accuracy": final_accuracy,
            "\nPrecision": metrics.precision_score(Y_test, predictions, average="weighted"),
            "Recall": metrics.recall_score(Y_test, predictions, average="weighted"),
            "f1_score": metrics.f1_score(Y_test, predictions, average="weighted"), 
            "\nconfusion_matrix": confusion_matrix,
            "normalised_confusion_matrix": normalised_confusion_matrix, })
    elif final_accuracy > 0.9:
        with open("log/debug/1000_Acc9_result.txt", 'a') as f:
            f.write("[{:<2d}] Acc={:.3f}, Loss={:.3f} / Train time={}\n".format(N_HIDDEN_CELLS, final_accuracy, final_loss, train_time))
    elif final_accuracy >= 0.85:
        with open("log/debug/1000_Acc8_result.txt", 'a') as f:
            f.write("[{:<2d}] Acc={:.3f}, Loss={:.3f} / Train time={}\n".format(N_HIDDEN_CELLS, final_accuracy, final_loss, train_time))
    # record
    temp_dir = 'log/records/'
    if not isdir(temp_dir):
        makedirs(temp_dir)

    with open(temp_dir + "1000_all_result.txt", 'a') as f:
        f.write("\t[{:<2d}] Acc={:.3f}, Loss={:.3f} / Train time={}\n".format(N_HIDDEN_CELLS, final_accuracy, final_loss, train_time))

    with open(temp_dir + "1000_all_array.txt", 'a') as f:
        f.write("\t[{:<2d}] acc={} / loss={}\n".format(N_HIDDEN_CELLS, test_accuracy_list, test_loss_list))
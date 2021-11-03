"""
訓練模型並測試.
"""
import tensorflow as tf # Version 1.0.0 (some previous versions are used in past commits)
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import random
from random import randint  # 證明此網路架構真的有運作: 用隨機類替換標記類以進行訓練
import time
import FishLog
import FishDebug

run_time_start = time.time()
FishLog.writeLog(FishLog.formatLog(20, "training.py", "line 16", "Start running the file.", "tensorflow version={}".format(tf.version.VERSION)))
load_dotenv()
# -----------------------------------------------------------
# Preparing dataset
# -----------------------------------------------------------
LABELS = [
    "normal",  # 原本 "JUMPING",
    "hunger",  # 原本 "JUMPING_JACKS",
    "rest",  # 原本 "BOXING",
    "exception",  # "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"
]
n_classes = len(LABELS)
'''
共有幾個分類.
'''
DATASET_PATH = "RNN-HAR-2D-Pose-database/"  # "convertTo_txt/"
X_train_path = DATASET_PATH + "X_train.txt"
X_test_path = DATASET_PATH + "X_test.txt"
y_train_path = DATASET_PATH + "Y_train.txt"
y_test_path = DATASET_PATH + "Y_test.txt"

n_steps = 32 # n_steps = int(getenv('N_STEPS'))
'''
幾個frame是一個動作
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
    blocks = int(len(X_arr) / n_steps)
    X_arr = np.array(np.split(X_arr, blocks))
    return X_arr


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
        1D, 每一 dim 的元素個數 = [x 筆動作結果].
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
        1D, Content is labels.
    
    Returns
    -------
    `ndarray` (dtype = object, element type = float32)
        1D, 每一 dim 的元素個數 = [x 筆動作random結果].
    '''
    for i in range(len(Y_arr)):
        Y_arr[i] = randint(0, n_classes-1)
    return Y_arr


# -----------------------------------------------------------
# 拿訓練、測試資料
# -----------------------------------------------------------
X_train = load_X(X_train_path)
X_test = load_X(X_test_path)
Y_train = load_Y(y_train_path)
Y_test = load_Y(y_test_path)

# -----------------------------------------------------------
# Set Hyperparameters & Parameters
# -----------------------------------------------------------
# Input Data
training_data_count = len(X_train) # 4519 training series(action) (每筆action有 50% 重疊)
'''
共有幾筆動作.
'''
test_data_count = len(X_test)  # 1197 test series
'''
共有幾筆動作.
'''
n_input = len(X_train[0][0])
'''
一個frame裡有幾個座標點(x+y)。n input/timestep(frame).
'''
n_hidden = 34
'''
number of neurons in hidden layer(自己假設)
'''
# learning rate
decaying_learning_rate = True
'''
是否更新學習率衰減 updated for learning-rate decay
decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
'''
learning_rate = 0.0025
'''
學習速率。更新參數的步幅.
- used if `decaying_learning_rate` = False
'''
init_learning_rate = 0.005
'''

'''
decay_rate = 0.96
'''
衰減中指數的基數 the base of the exponential in the decay
'''
decay_steps = 100000
'''
used in decay every 60000 steps with a base of 0.96
'''
global_step = tf.Variable(0, trainable=False)
'''
紀錄迭代的總次數.
- 可用於更改 learning rate 或其他超參數
- 在100次迭代后停止训练，然后第二天恢复模型并再运行100次迭代。现在`global_step`=200，但是第二次运行的局部迭代数是1到100
'''
# loss
lambda_loss_amount = 0.0015
'''
(in 損失函數)懲罰項的倍率
- 當 λ=0 時，則權重衰減不會發生；當 λ 越大時，懲罰的比率較高，權重衰減的程度也就跟著變大
'''
# train
epochs = 200
'''
迭代次數.
- Loop n times on the dataset(在數據集上循環 n 次)
'''
training_iterations = training_data_count * epochs
'''
訓練次數.train step 上限
'''
batch_size = 4096
'''
批量大小。每次的迭代，送入類神經網路的資料數量.
'''
display_iteration = batch_size*20
'''
在訓練期間顯示測試集的準確性 To show test set accuracy during training
'''
FishLog.writeLog(FishLog.formatLog(20, "training.py", "line 78", "Set Hyperparameters & Parameters."))
FishDebug.writeLog({"lineNum": 78, "funName": False, "fileName": "training.py"}, "training/v1/0_Set_Parameters", {
    "X_train shape": X_train.shape,
    "Y_train shape": Y_train.shape,
    "X_test shape": X_test.shape,
    "Y_test shape": Y_test.shape,
    "ONLY VALUE": "The dataset has not been preprocessed, is not normalised etc",
    "(X_train) training data count": training_data_count,
    "(X_test) test data count": test_data_count,
    "n_input": n_input,
    "n_hidden": n_hidden,
    "decaying_learning_rate": decaying_learning_rate,
    "learning_rate": learning_rate,
    "init_learning_rate": init_learning_rate,
    "decay_rate": decay_rate,
    "decay_steps": decay_steps,
    "global_step": global_step,
    "lambda_loss_amount": lambda_loss_amount,
    "training_iterations": training_iterations,
    "batch_size": batch_size,
    "display_iteration": display_iteration, })


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
    _X = tf.transpose(_X, [1, 0, 2])  # [n_steps, x筆資料, n_input]
    # 變成 2D [n_steps * x筆資料, n_input]，才可以使用weights 的矩陣乘法
    _X = tf.reshape(_X, [-1, n_input])
    # 修正線性單元激活函數 Rectifies Linear Unit activation function used
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)  # [n_steps, x筆資料, n_input]

    #############################################
    # Define two stacked LSTM cells (two recurrent layers deep)
    # 2-layer LSTM, each layer has n_hidden units
    #############################################
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # generate prediction (Get lstm cell output)
    outputs, final_state = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)
    #############################################
    # hidden layer for output as the final results
    #############################################
    # A single output is produced, in style of "many to one" classifier, refer to http://karpathy.github.io/2015/05/21/rnn-effectiveness/ for details
    # 有 n_input 輸出但我們只想要最後一個輸出
    lstm_last_output = outputs[-1]

    FishLog.writeLog(FishLog.formatLog(20, "training.py", "line 117", "Create graph."))
    FishDebug.writeLog({"lineNum": 144, "funName": "extract_batch_size", "fileName": "training.py"}, "training/v1/1_create_graph", {
        'lstm_last_output': lstm_last_output,
        'outputs': outputs,
        'final_state': final_state, })
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
        1D, 每一 dim 的元素個數 = [x 筆動作結果].
    unsampled_range : `range`
        0 ~ x
    batch_size : `int`
        n 筆動作 (n < x)
    
    Returns
    -------
    batch_data : `ndarray`
        3D, 每一 dim 的元素個數 = [n 筆動作, frame數, n*2(x,y)座標點].
    batch_labels : `ndarray`
        1D, 每一 dim 的元素個數 = [n 筆動作結果].
    unsampled_range : `list`
        剩下還沒抓到的資料indices
    '''
    X_shape = list(X_data.shape)
    X_shape[0] = batch_size
    batch_data = np.empty(X_shape)
    batch_labels = np.empty((batch_size, 1))


    # debug by Polly in 2021/10/9
    index = random.choice(unsampled_range) # [原本]
    FishDebug.writeLog({"lineNum": 144, "funName": "extract_batch_size", "fileName": "training.py"}, "training/v1/2_extract_batch_size", {
        "\nX_shape": X_shape,
        'batch_data\'s shape': batch_data.shape,
        'batch_labels\'s shape': batch_labels.shape,
        # -------------------------------------
        "\nindex": index,
        'batch_data[index]\'s shape': X_data[index].shape,
        'batch_labels[index]\'s shape': Y_labels[index].shape,
        'batch_data[index]': X_data[index],
        'batch_labels[index]': Y_labels[index], })

    unsampled_range = list(unsampled_range)
    for i in range(batch_size):
        index = random.choice(unsampled_range) # random sample from unsampled_range (indices)
        batch_data[i] = X_data[index]
        batch_labels[i] = Y_labels[index]
        unsampled_range.remove(index)
    FishLog.writeLog(FishLog.formatLog(20, "training.py","line 144", "extract_batch_size", "unsampled_range = {}".format(unsampled_range)))
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
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


# -----------------------------------------------------------
# 定義類神經網路模型 Build the network
# -----------------------------------------------------------
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
'''
Input.
3D=[x筆動作, frame數, 座標點數]
'''
y = tf.placeholder(tf.float32, [None, n_classes])
'''
Output.
2D=[x筆動作, label數]
'''
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
'''
權重.
'''
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
'''
偏差值.
'''
pred = LSTM_RNN(x, weights, biases)
'''
Graph.
'''
l2 = lambda_loss_amount * sum(tf.nn.l2(tf_var) for tf_var in tf.trainable_variables())
'''
Loss function 的懲罰項
- 對抗 Overfitting: Weight decay -- 不讓模型 fit 時過度依賴一些 weights
'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2
'''
Loss function.
- `softmax_cross_entropy_with_logits`: 結合 Softmax & Cross Entropy 的函式
'''
if decaying_learning_rate:  # exponentially decayed learning rate
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step*batch_size, decay_steps, decay_rate, staircase=True) # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps) 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)  # Adam Optimizer
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
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

time_start = time.time()
'''
計算訓練網路所需的時間
'''
unsampled_indices = range(0, training_data_count)
'''
訓練資料的range
'''
one_hot_Y_test = one_hot(Y_test)
step = 1
while step * batch_size <= training_iterations:
    # 當剩餘數據點 < batch_size 時重新啟動
    if len(unsampled_indices) < batch_size:
        unsampled_indices = range(0, training_data_count)
    
    batch_xs, raw_labels, unsampled_indicies = extract_batch_size(X_train, Y_train, unsampled_indices, batch_size)
    batch_ys = one_hot(raw_labels)
    
    # 如果 encoded_y 的長度 != 總分類數，就用 0 填充到一樣大小
    if len(batch_ys[0]) < n_classes:
        temp_ys = np.zeros((batch_size, n_classes))
        temp_ys[:batch_ys.shape[0], :batch_ys.shape[1]] = batch_ys
        batch_ys = temp_ys

    # 使用批量數據進行擬合訓練 Fit training using batch data
    _, train_loss, train_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_xs, y: batch_ys})
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_acc)

    # Evaluate network only at some steps for faster training: 要展示時 / 一開始 / 最後一次
    current_iteration = step*batch_size
    if (current_iteration % display_iteration == 0) or (step == 1) or (current_iteration > training_iterations):
        # 對測試集的評估（這裡沒有學習 - 只是診斷評估） Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: X_test, y: one_hot_Y_test})
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_acc)
        print("Iter #{}: global number of iterations={}".format(current_iteration, sess.run(global_step)))
        tf.logging.log_every_n(tf.logging.INFO, "np.mean(loss_evl)= %f at step %d", 100, np.mean(
            test_loss), sess.run(tf.train.get_global_step()))
        print_log_list.append("Iter #{}: Learning rate= {:.6f} => Batch Loss= {:.3f}, Acc= {:.3f} [test data]=> Loss= {:.3f}, Acc= {:.3f}".format(current_iteration, sess.run(learning_rate), train_loss, train_acc, test_loss, test_acc))

    step += 1

# -----------------------------------------------------------
# Accuracy for test data
# -----------------------------------------------------------
one_hot_predictions, final_accuracy, final_loss = sess.run([pred, accuracy, cost], feed_dict={x: X_test, y: one_hot_Y_test})
test_loss_list.append(final_loss)
test_accuracy_list.append(final_accuracy)

# debug by Polly in 2021/10/9
print_log_list.append("\nTrain network time: {}\nFinal result: Loss= {:.3f}, Acc= {:.3f}\n".format(time.time() - time_start, final_loss, final_accuracy))
FishLog.writeLog(FishLog.formatLog(20, "training.py", "line 233", "Train & Run The Network.", "Train network time: {}".format(time.time() - time_start)))
FishDebug.writeLog({"lineNum": 233, "funName": False, "fileName": "training.py"}, "training/v1/3_Train_Run_The_Network", print_log_list)


# -----------------------------------------------------------
# Visualization of results
# -----------------------------------------------------------
# Show 準確度
plt.figure()
data_x_axis ={
    'train': np.array(range(batch_size, (len(train_loss_list)+1)*batch_size, batch_size)),
    'test': np.append(
        np.array(range(batch_size, len(test_loss_list) * display_iteration, display_iteration)[:-1]),
        [training_iterations]
    )
}
plt.plot(data_x_axis['train'], np.array(train_loss_list), ".g", label="train losses")
plt.plot(data_x_axis['train'], np.array(train_accuracy_list), "-c", label="train accuracy")
plt.plot(data_x_axis['test'], np.array(test_loss_list), ".m", linewidth=2.0, label="test losses")
plt.plot(data_x_axis['test'], np.array(test_accuracy_list), "-r", linewidth=2.0, label="test accuracy")
plt.title("session's each iteration accuracy and loss")
plt.legend(loc='lower right', shadow=True)
plt.xlabel('iteration')
plt.ylabel('accuracy')
plt.show()
# debug by Polly in 2021/10/9
FishLog.writeLog(FishLog.formatLog(20, "training.py", "line 317", "Show accuracy & loss"))
FishDebug.writeLog({"lineNum": 317, "funName": False, "fileName": "training.py"}, "training/v1/4_Visualization_Result1", {
    "train loss      len": len(train_loss_list),
    "train accuracy  len": len(train_accuracy_list),
    "test  loss      len": len(test_loss_list),
    "test  accuracy  len": len(test_accuracy_list),
    "data_x_axis['train']len": len(data_x_axis['train']),
    "data_x_axis['test'] len": len(data_x_axis['test']),
    "data_x_axis['train']": data_x_axis['train'],
    "data_x_axis['test'] ": data_x_axis['test'],
    "train losses       ": train_loss_list,
    "train accuracies   ": train_accuracy_list,
    "test  losses       ": test_loss_list,
    "test  accuracies   ": test_accuracy_list,
    })


# Show 
predictions = one_hot_predictions.argmax(1)
confusion_matrix = metrics.confusion_matrix(Y_test, predictions)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
plt.figure()
plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix \n(normalised to \% of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

FishLog.writeLog(FishLog.formatLog(20, "training.py", "line 354", "Result picture 2."))
FishDebug.writeLog({"lineNum": 354, "funName": False, "fileName": "training.py"}, "training/v1/5_Visualization_Result2", {
    "Testing Accuracy": 100*accuracy,
    "\nPrecision": 100*metrics.precision_score(Y_test, predictions, average="weighted"),
    "Recall": 100*metrics.recall_score(Y_test, predictions, average="weighted"),
    "f1_score": 100*metrics.f1_score(Y_test, predictions, average="weighted"),
    "ONLY VALUE": "\nConfusion Matrix:\nCreated using test set of {} datapoints, normalised to \% of each class in the test dataset\n".format(len(Y_test)),
    "predictions": predictions,
    "confusion_matrix": confusion_matrix,
    "normalised_confusion_matrix": normalised_confusion_matrix, })

# 關閉 Session
sess.close()
# debug by Polly in 2021/10/9
FishLog.writeLog(FishLog.formatLog(20, "training.py", "line 391", "Finish running the file.", "Total time: {}".format(time.time() - run_time_start)))

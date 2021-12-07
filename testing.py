# !/usr/bin/python3.6
"""
預測結果.
"""
import tensorflow as tf # Version 1.0.0 (some previous versions are used in past commits)
import numpy as np
from dotenv import load_dotenv
from os import getenv, environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from traceback import extract_tb
import FishLog

load_dotenv()
DATASET_PATH = "convertTo_txt/"
X_TRAIN_PATH = DATASET_PATH + "X_test.txt"
LABEL_LIST = [
    "normal",
    "hunger",
    "rest  ",
    # "exception",
]
N_STEPS = int(getenv('N_STEPS'))
MODEL = 'LSTM_model/6485.ckpt'
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
    if len(X_arr) < N_STEPS :
        print('資料量太少 Failed')
        raise ValueError('Failed: Not enough data')
        
    blocks = int(len(X_arr) / N_STEPS)
    return np.array(np.split(X_arr, blocks))


try:
    X_train = load_X(X_TRAIN_PATH)
    with tf.compat.v1.Session() as sess:
        new_saver = tf.compat.v1.train.import_meta_graph(MODEL+'.meta')
        # 会将已经保存的变量值resotre到变量中
        new_saver.restore(sess,MODEL)
        y = tf.compat.v1.get_collection('pred_network')
        y = tf.argmax(y,2)+1
        # 獲取當前默認計算圖
        graph = tf.compat.v1.get_default_graph()
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

        # 預測輸出
        ans_list=sess.run(y, feed_dict={input_x:X_train, keep_prob:1.0})[0]
        total_data_part = 0
        for data_part_ans in ans_list:
            total_data_part += 1
            # print('No.',total_data_part,':',data_part_ans)
        
        vals, counts = np.unique(ans_list, return_counts=True)
        # ans
        index = np.argmax(counts)
        print("\n\nThe final prediction ans is {}".format(LABEL_LIST[vals[index]-1]))
        for i in range(len(LABEL_LIST)):
            if (i+1) in vals:
                print("\tThis is {} with a {:.6f} probability".format(LABEL_LIST[i],counts[i]/total_data_part))
            else:
                print("\tThis is {} with a {:.6f} probability".format(LABEL_LIST[i],0.00000000000000))
except Exception as e:
    cl, exc, tb = sys.exc_info()  # 取得Call Stack
    FishLog.writeLog(FishLog.formatException(e.args[0], extract_tb(tb)[0], "Convert data."))
else:
    # log
    FishLog.writeLog(FishLog.formatLog(20, "testing.py", "line 89", "Testing video."))
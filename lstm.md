# training.py

## Modules
+ tensorflow: 深度學習的框架
+ numpy: 處理多維度資料比 list 快
+ matplotlib: 資料視覺化套件
+ sklearn: 
+ random: 打亂資料的工具

---

+ Softmax 回歸: 使用 Softmax 運算使得最後一層輸出的機率分布總和=1 ( sum(資料結果為每一種label的機率)=1 )

## Loss function -- 交叉熵(Cross Entropy)
+ 判斷誤差值
- Cross Entropy: 評估 2 個機率分配有多接近，若很接近，result趨近於0，反之則趨近於1

## 優化器(optimizer) -- Adam Optimizer
+ 反向傳播誤差。幫助神經網路調整參數，使 Loss 越小越好
- Adam Optimizer: 會在訓練時動態更新 learning_rate

## Activation function -- ReLU
+ 線性整流函數（Rectified Linear Unit, ReLU），又稱修正線性單元

## Code.
```python=
import tensorflow as tf

tf.Variable(0, trainable=False) # 關閉梯度
tf.transpose(arr,[1,0,2]) # 交換維度。把對應的輸入張量的對應的維度對應交換
tf.reshape(arr, [-1, 32]) # 將 arr.shape 變成 ?*32
product = tf.matmul(matrix1,matrix2) # 矩陣相乘(matrix1*matrix2)
tf.nn.relu(a) # 將輸入<0 的值設為0(負數變為0)，輸入>0 的值不變
```
## [RNN 用法彙整](https://www.twblogs.net/a/5ca59985bd9eee5b1a072277)
### 最基本的LSTM循環神經網絡單元
+ `tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0, state_is_tuple=True, activation=None, reuse=None, name=None)`
    + 參數
        + `num_units`: LSTM cell層中的單元數
        + `forget_bias`: forget gates中的偏置
        + `state_is_tuple`: 設置爲True, 返回 (c_state , m_state)的二元組
        + `activation`: 狀態之間轉移的激活函數
        + `reuse`: Python布爾值, 描述是否重用現有作用域中的變量
        + `name`：操作的名稱
    + return 一個隱層神經元數量爲 num_units 的 LSTM 基本單元
    + 如果我們處理的是分類問題，那麼我們還需要對new_h添加單獨的Softmax層才能得到最後的分類概率輸出
        + function def
        ```
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=dtypes.int32)
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

        gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), self._kernel)
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)

        forget_bias_tensor = constant_op.constant(self._forget_bias, dtype=f.dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        multiply = math_ops.multiply
        new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))),multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = array_ops.concat([new_c, new_h], 1)
        return new_h, new_state
        ```
### 堆疊RNNCell：MultiRNNCell
+ 對RNNCell進行堆疊
+ `tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)`
    + `tf.nn.rnn_cell`貌似要弃用了。将所有`tf.nn.rnn_cell`更改为`tf.contrib.rnn`
    + 參數
        + cells: RNNCells的列表，RNN網絡將會按照這個順序搭建.
            + 用於神經網絡的RNN神經元,如BasicRNNCell,BasicLSTMCell
        + state_is_tuple: 如果爲True, 接受和返回的狀態將爲一個n元組，其中n = len(cells) 
            + False: 這n個元素將會被concat到一起組成states.爲False的情況將會被廢棄
### 使用指定的RNN神經元創建循環神經網絡
+ `tf.contrib.rnn.static_rnn(cell, inputs, initial_state=None, dtype=None, sequence_length=None, scope=None)`
    + 參數
        + `cell`: 用於神經網絡的RNN神經元,如BasicRNNCell,BasicLSTMCell
        + `inputs`: 一個長度為T的list,list中的每個元素為一個Tensor，Tensor形如：[batch_size,input_size]
        + `initial_state`: RNN的初始狀態，如果cell.state_size是一個整數，則它必須是適當類型和形如[batch_size,cell.state_size]的張量。如cell.state_size是一個元組，那麼它應該是一個張量元組，對於cell.state_size中的s,應該是具有形如[batch_size,s]的張量的元組。
        + `dtype`: 初始狀態和預期輸出的數據類型。可選參數。
        + `sequence_length`: 指定每個輸入的序列的長度。大小為batch_size的向量。
        + `scope`: 變量範圍
    + `dynamic_rnn` v.s `static_rnn`
        + static_rnn不同的batch序列长度必须是相同的，都必须是 n_steps  ，dynamic_rnn不用
        + dynamic_rnn 比 static_rnn 快的原因是：dynamic_rnn运行到序列长度后自动停止，不再运行，而static_rnn必须运行完 n_steps 才停止
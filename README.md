# Hey! Fish,  What are you doing
2021 大四專題

## 使用 /usr/bin/python3.6 training.py 去跑

## Code
+ `convertData.py` : 將 Deeplabcut 的 log 檔中的資料轉成 LSTM 所需格式
    + `convertData_fun.py`: 將 code 拆成獨立 function
+ `training.py` : 訓練 LSTM 資料
+ `rebot.py` : 可用來進行行為分析
+ `FishLog.py` : log 格式
    + 查看指令格式 `python .\FishLog.py`
    + 查看 function arg `python .\FishLog.py --help`
    + 執行測試 function `python .\FishLog.py --test`
+ `FishDebug.py` : debug 的 log 格式
    + 查看指令格式 `python .\FishDebug.py`
    + 查看 function arg `python .\FishDebug.py -s`
    + 執行測試 function `python .\FishDebug.py -t`

### coding style
+ 變數命名
    + ...`_arr` : 是 ndarray
    + ...`_list` : 是 list
    + ...`_dict` : 是 dict
    + ...`_iter` : 可以迭代


## Directory
+ `tests/Unit` : unit test
    + `TestConvertData.py`

+ `CSV` : Deeplabcut 取得的資料
    + `DLC0818.csv` : /media/ntou501/4f2b9da8-a755-49a3-afea-60704f1a7d00/merge/mergeFish0816-cse509-2021-08-16/test/test2DLC_resnet50_mergeFish0816Aug16shuffle1_50000.csv
        + 08/18 第一次訓練模型
    + `DLC0830.csv` : /media/ntou501/4f2b9da8-a755-49a3-afea-60704f1a7d00/merge/mergeFish0816-cse509-2021-08-16/test/test1DLC_resnet50_mergeFish0816Aug16shuffle1_50000.csv
        + 08/30 第二次訓練模型
    + `GH010241.csv` : /media/ntou501/4f2b9da8-a755-49a3-afea-60704f1a7d00/merge/mergeFish0816-cse509-2021-08-16/test/GH010241croppedDLC_resnet50_mergeFish0816Aug16shuffle1_50000.csv
        + 08/30 第二次訓練模型

+ `convertTo_txt` : LSTM 測試用資料
+ `log` : 依執行時間落log
+ `debug` : 測試執行結果
+ `RNN-HAR-2D-Pose-database` : 原始訓練、測試資料
    + only in ntou501
    + `X_test.txt`  : 測試資料, 座標點 (`n` keypoints per line, `n_steps` lines per datapoint)
    + `X_train.txt` : 訓練資料, 座標點
    + `X_val.txt`   : Placeholder for a single inference after training
    + `X_val2.txt`  :
    + `Y_test.txt`  : 測試資料, 分類labels
    + `Y_train.txt` : 訓練資料, 分類labels


## Usage
+ test `.env`
    ```python
    from os import getenv
    from dotenv import load_dotenv

    load_dotenv()
    PROJECT_PATH = getenv('PROJECT_PATH')
    print("{}--{}--{}--{}".format(PROJECT_PATH,TOTAL_POINTS,TRANSLATION_POINT,MIRROR_POINT))
    ```

+ `requirements.txt`
    + 存入 `pip freeze > requirements.txt`
    + 安裝 `pip install -r requirements.txt`
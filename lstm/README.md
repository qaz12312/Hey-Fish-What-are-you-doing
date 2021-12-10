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
+ `convertTo_txt` : LSTM 測試用資料
    + `X_train.txt` : 訓練資料, 座標點 (`n` keypoints per line, `n_steps` lines per datapoint)
    + `X_validate.txt` : 驗證資料, 座標點
    + `X_test.txt`  : 測試資料, 座標點
    + `Y_train.txt` : 訓練資料, 分類labels
    + `Y_validate.txt` : 驗證資料, 分類labels
+ `CSV` : Deeplabcut 取得的資料
    + `test` : 測試資料
    + `train` : 訓練資料
    + `vaildate` : 驗證資料
+ `Docs` : 存放轉存的excel
+ `LSTM_model` : 模型
+ `log` : 日誌 & 除錯 & 紀錄
    + 依執行時間落log
    + `debug` : 檢查執行過程 & 結果
    + `records` : 紀錄準確度等等資料
+ `TESTING` : 測試環境下的上述檔案
+ `tests/Unit` : unit test



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
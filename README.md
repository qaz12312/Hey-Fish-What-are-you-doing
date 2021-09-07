# Hey! Fish,  What are you doing
2021 大四專題

## 使用 /usr/bin/python3.6 training.py 去跑

## code
+ `convertData.py` : 從 Deeplabcut 取得的資料轉成 LSTM 所要用的格式
+ `training.py` : 訓練 LSTM 資料
+ `rebot.py` : 可用來進行行為分析
+ `FishLog.py` : log 格式
    + 查看指令格式 `python .\FishLog.py`
    + 查看 function arg `python .\FishLog.py --help`
    + 執行測試 function `python .\FishLog.py --test`
+ `FishDebug.py` : debug 的 log 格式
    + 查看指令格式 `python .\FishDebug.py`
    + 查看 function arg `python .\FishDebug.py --help`
    + 執行測試 function `python .\FishDebug.py --test`


## Folder
+ `testCSV` : Deeplabcut 取得的資料
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


## message
+ 09/06 (一)
    + feat: log 格式
+ 09/07 (二)
    + update: add commands for FishLog.py
    + feat: debug log 格式
    <!-- + feat: 資料正規劃 -->

# Hey-Fish-What-are-you-doing
2021 大四專題

# 使用/usr/bin/python3.6 training.py去跑

## code
+ convertData.py
    + 從 Deeplabcut 取得的資料轉成 lstm 所要用的格式
+ training.py
    + 訓練 lstm 資料
+ rebot.py
    + 可用來進行行為分析
+ FishLog.py
    + log 格式
    + 查看指令格式 `python .\FishLog.py`
    + 查看 function arg `python .\FishLog.py --help`
    + 執行測試 function `python .\FishLog.py --test`

## Folder
+ testCSV
    + Deeplabcut 取得的資料
+ convertTo_txt
    + lstm 測試用資料
+ debug
    + 測試執行結果
+ log
    + 依執行時間落log
+ RNN-HAR-2D-Pose-database
    + 原始訓練、測試資料

## message
+ 09/06 (一)
    + feat: log 格式
+ 09/07 (二)
    + update: add commands for FishLog.py
    <!-- + feat: 資料正規劃 -->

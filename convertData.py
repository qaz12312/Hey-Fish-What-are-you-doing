# 將 dlc 的 log 檔資料轉成 lstm 測試格式

import csv

# -----------------------------------------------------------
# Read Deeplabcut's log file (.csv)
# -----------------------------------------------------------
def readFile(filePath):
    training_x = []
    with open(filePath, newline='') as f:
        # 讀取 CSV 檔案內容
        rows = csv.reader(f)
        i = 0
        # 以迴圈輸出每一列
        for row in rows:
            i += 1
            if i < 4 : # 略過前3行
                continue
            # result_row = ",".join(x for x in row[1:]) # 第一個(xxx.png)不需要
            # print(result_row)
            # training_x.append(result_row)
            training_x.append(row[1:])
    return training_x


# -----------------------------------------------------------
# Write coords_array into file (.txt)
# -----------------------------------------------------------
def writeData(filePath,data):
    with open(filePath, 'w') as f:
        for item in data:
            f.write("{}\n".format(item))
    return True


# -----------------------------------------------------------
# 移除正面、背面的frame & 弄成同一測
# -----------------------------------------------------------
def removeUnusableData(data_arr):
    pass


# -----------------------------------------------------------
# 將座標點補齊 - 使用內插法
# -----------------------------------------------------------
def fillData(data_arr):
    pass


# -----------------------------------------------------------
# 將座標點正規化 - 讓frame1為(0,0)，其他的frame根據frame1來改變座標
# -----------------------------------------------------------
def resetOrigin(data_arr):
    pass


if __name__ == '__main__':
    print("test readFile().")
    print("test removeUnusableData().")
    print("test fillData().")
    print("test resetOrigin().")
    print("test writeData().")

    # Need to be converted to a new format
    csvfile = ['fish_0602','GH010139','GH010144']

    for file_name in csvfile:
        data_arr = readFile('testCSV/' + file_name + '.csv')
        data_arr = removeUnusableData(data_arr)
        data_arr = fillData(data_arr)
        data_arr = resetOrigin(data_arr)
        # write into new file
        if writeData('convertTo_txt/' + file_name + '_X_train.txt',data_arr) :
            print(file_name + ': ok')
        else :
            print(file_name + ': error')
    
    print("finish.")


    

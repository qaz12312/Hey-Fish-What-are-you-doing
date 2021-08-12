import csv

csvfile = ['fish_0602','GH010139','GH010144']
for file_name in csvfile:
    training_x = []
    with open('testCSV/' + file_name + '.csv', newline='') as f:
        # 讀取 CSV 檔案內容
        rows = csv.reader(f)
        i = 0
        # 以迴圈輸出每一列
        for row in rows:
            i += 1
            if i < 4 :
                continue
            result_row = ",".join(x for x in row[1:])
            # print(result_row)
            training_x.append(result_row)
    # print(type(training_x))
    # print(training_x)
    with open('convertTo_txt/' + file_name + '_X_train.txt', 'w') as f:
        for item in training_x:
            f.write("{}\n".format(item))
        print(file_name + ' ok')

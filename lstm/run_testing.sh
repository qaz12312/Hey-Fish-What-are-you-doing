#!/bin/bash

# cd pwd
cd `dirname $0`

# dlc的測試影片存放地
test_dir_path='/home/ntou501/Desktop/video/normal/'

echo -e '將載入 Deeplabcut 的全部檔案至 CSV/test 資料夾下'

read -p '是否清空在 CSV/test 資料夾裡的先前資料? (type:y/n) ' is_clear
while [ "$is_clear" != 'y' ] && [ "$is_clear" != 'n' ]
do
    echo "$(tput setaf 5)[!警告]只能輸入 y 或是 n$(tput sgr 0)"
    read -p '是否清空在 CSV/test 資料夾裡的先前資料? (type:y/n) ' is_clear
done
if [ "$is_clear" = 'y' ]
then
    rm -rf ./CSV/test/*
fi

read -p "要載入的資料夾路徑是否為: ${test_dir_path}? (type:y/n) " is_right
while [ "$is_right" != 'y' ] && [ "$is_right" != 'n' ]
do
    echo "$(tput setaf 5)[!警告]只能輸入 y 或是 n$(tput sgr 0)"
    read -p "要載入的資料夾路徑是否為: ${test_dir_path}? (type:y/n) " is_right
done
if [ "$is_right" = 'n' ]
then
    while [ "$is_right" != 'y' ]
    do
        read -p "請輸入要載入的資料夾路徑: " test_dir_path
        read -p "要載入的資料夾路徑是否為: ${test_dir_path}? (type:y/n) " is_right
    done
fi

echo -e '\033[30m\e[1;43m正在載入資料中.....\e[0m'
if [ ! -d ./CSV/test ];then
    mkdir ./CSV/test
fi
if [ ! -d "$test_dir_path" ]  # 檢查是否為目錄
then
    echo "$(tput setaf 5)[!警告] $test_dir_path 目錄不存在$(tput sgr 0) 請重新進行操作"
    exit 0
fi

file_count=0
file_names=($(ls -d "$test_dir_path"*.csv))
for file in "${file_names[@]}"
do
    echo -e "\n**********************************************************"
    cp "$file" ./CSV/test # copy file
    echo "$test_dir_path$file.....載入成功"
    if [ -f "./convertTo_txt/X_test.txt" ]
    then
        truncate -s 0 "convertTo_txt/X_test.txt" # empty content of the file
        echo "清除 convertTo_txt/X_test.txt.....成功"
    fi

    echo -e "\033[30m\e[1;43m進行資料處理.....\e[0m"
    /usr/bin/python3.6 convertData.py --type 1

    echo -e "\033[30m\e[1;43m開始預測.....\e[0m"
    /usr/bin/python3.6 testing.py

    file_count=$(($file_count + 1))
    rm -rf ./CSV/test/*
    sleep 2
done
echo "**********************************************************"
echo "$(tput setaf 0)$(tput setab 2) 結束 $(tput sgr 0)"

exit 0
